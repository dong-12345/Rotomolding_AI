from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .mesh import MeshGraph, build_mesh_graph, build_mesh_graph_from_arrays


DEFAULT_PROCESS_SCHEMA = [
    {"name": "main_rpm", "default": 0.0, "mean": 0.0, "std": 1.0},
    {"name": "sub_rpm", "default": 0.0, "mean": 0.0, "std": 1.0},
    {"name": "time_s", "default": 0.0, "mean": 0.0, "std": 1.0},
    {"name": "powder_mass_g", "default": 1.0, "mean": 1.0, "std": 1.0},
    {"name": "oven_temp_c", "default": 1.0, "mean": 1.0, "std": 1.0},
    {"name": "heating_time_s", "default": 1.0, "mean": 1.0, "std": 1.0},
    {"name": "cooling_time_s", "default": 1.0, "mean": 1.0, "std": 1.0},
    {"name": "material_code", "default": 1.0, "mean": 1.0, "std": 1.0},
]


def load_process_schema(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_PROCESS_SCHEMA
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("Process schema must be a non-empty JSON list")
    for item in value:
        if "name" not in item:
            raise ValueError("Every process feature needs a name")
    return value


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            for required in ("mesh", "thickness"):
                if required not in record:
                    raise ValueError(f"Manifest line {line_number} misses '{required}'")
            records.append(record)
    if not records:
        raise ValueError("Manifest is empty")
    return records


def split_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    explicit = all("split" in record for record in records)
    if explicit:
        train = [record for record in records if record["split"] == "train"]
        val = [record for record in records if record["split"] == "val"]
        test = [record for record in records if record["split"] == "test"]
        return train, val, test

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("mesh_id", record["mesh"]))
        groups.setdefault(key, []).append(record)
    keys = list(groups)
    rng.shuffle(keys)

    if len(keys) >= 3:
        n_test = max(1, round(len(keys) * test_fraction))
        n_val = max(1, round(len(keys) * val_fraction))
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test : n_test + n_val])
        train = [r for k, rows in groups.items() if k not in test_keys | val_keys for r in rows]
        val = [r for k, rows in groups.items() if k in val_keys for r in rows]
        test = [r for k, rows in groups.items() if k in test_keys for r in rows]
        return train, val, test

    rng.shuffle(records)
    if len(records) == 1:
        return records, [], []
    n_test = 1 if len(records) >= 3 else 0
    n_val = 1 if len(records) >= 2 else 0
    test = records[:n_test]
    val = records[n_test : n_test + n_val]
    train = records[n_test + n_val :]
    return train, val, test


@dataclass
class GraphItem:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    process: torch.Tensor
    target: torch.Tensor
    mesh_id: str


class RotomoldingDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        process_schema: list[dict[str, Any]],
        mesh_root: str | Path = ".",
        axes: dict | None = None,
        thickness_scale: float = 1.0,
    ) -> None:
        self.records = records
        self.schema = process_schema
        self.mesh_root = Path(mesh_root)
        self.axes = axes
        self.thickness_scale = float(thickness_scale)
        self._mesh_cache: dict[str, MeshGraph] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_mesh(self, path: str) -> MeshGraph:
        resolved = str((self.mesh_root / path).resolve())
        if resolved not in self._mesh_cache:
            if Path(resolved).suffix.lower() == ".npz":
                with np.load(resolved) as archive:
                    vertices = archive["vertices"]
                    faces = archive["faces"]
                self._mesh_cache[resolved] = build_mesh_graph_from_arrays(vertices, faces, self.axes)
            else:
                self._mesh_cache[resolved] = build_mesh_graph(resolved, self.axes)
        return self._mesh_cache[resolved]

    def _process_vector(self, record: dict[str, Any]) -> torch.Tensor:
        params = dict(record.get("params", {}))
        for key in ("main_rpm", "sub_rpm", "time_s"):
            if key in record:
                params[key] = record[key]
        values = []
        for item in self.schema:
            name = item["name"]
            value = float(params.get(name, item.get("default", 0.0)))
            mean = float(item.get("mean", 0.0))
            std = max(float(item.get("std", 1.0)), 1e-8)
            values.append((value - mean) / std)
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index: int) -> GraphItem:
        record = self.records[index]
        mesh = self._load_mesh(record["mesh"])
        target = np.load(self.mesh_root / record["thickness"]).astype(np.float32)
        if target.ndim != 1 or len(target) != len(mesh.vertices):
            raise ValueError(
                f"Target vertex count mismatch for {record['thickness']}: "
                f"expected {len(mesh.vertices)}, got {target.shape}"
            )
        return GraphItem(
            node_features=mesh.node_features,
            edge_index=mesh.edge_index,
            edge_features=mesh.edge_features,
            process=self._process_vector(record),
            target=torch.from_numpy(target / self.thickness_scale),
            mesh_id=str(record.get("mesh_id", record["mesh"])),
        )


def collate_graphs(items: list[GraphItem]) -> dict[str, torch.Tensor | list[str]]:
    node_features = []
    edge_features = []
    edge_indices = []
    processes = []
    targets = []
    batches = []
    offset = 0
    for batch_index, item in enumerate(items):
        node_features.append(item.node_features)
        edge_features.append(item.edge_features)
        edge_indices.append(item.edge_index + offset)
        processes.append(item.process)
        targets.append(item.target)
        batches.append(torch.full((item.node_features.shape[0],), batch_index, dtype=torch.long))
        offset += item.node_features.shape[0]
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_features": torch.cat(edge_features, dim=0),
        "process": torch.stack(processes, dim=0),
        "target": torch.cat(targets, dim=0),
        "batch": torch.cat(batches, dim=0),
        "mesh_id": [item.mesh_id for item in items],
    }
