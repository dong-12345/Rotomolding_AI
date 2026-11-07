from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotomolding_ai.color import colors_to_values, load_color_map  # noqa: E402
from rotomolding_ai.colored_obj import load_colored_obj, write_uncolored_obj  # noqa: E402
from rotomolding_ai.data import DEFAULT_PROCESS_SCHEMA, load_process_schema  # noqa: E402


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "mesh"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    required = {"colored_obj", "main_rpm", "sub_rpm", "time_s"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"CSV misses required columns: {sorted(missing)}")
    return rows


def _build_schema(rows: list[dict[str, str]], schema_path: str | None) -> list[dict]:
    base = load_process_schema(schema_path) if schema_path else DEFAULT_PROCESS_SCHEMA
    result = []
    for item in base:
        name = item["name"]
        values = []
        for row in rows:
            value = _number(row.get(name))
            if value is not None:
                values.append(value)
        if values:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
            default = mean
        else:
            default = float(item.get("default", 1.0))
            mean = float(item.get("mean", default))
            std = max(float(item.get("std", 1.0)), 1e-6)
        result.append({"name": name, "default": default, "mean": mean, "std": std})
    return result


def prepare(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_root)
    output_value = args.output if hasattr(args, "output") else args.prepared_output
    output_root = Path(output_value)
    mesh_root = output_root / "meshes"
    label_root = output_root / "labels"
    mesh_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(source_root / args.csv)
    map_values, map_colors = load_color_map(args.color_map)
    schema = _build_schema(rows, args.schema)
    schema_path = output_root / "process_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(rows):
            colored_path = source_root / row["colored_obj"]
            mesh_id = row.get("mesh_id") or colored_path.stem
            mesh_name = _safe_name(mesh_id)
            colored = load_colored_obj(colored_path)
            if mesh_name not in mesh_cache:
                mesh_cache[mesh_name] = (colored.vertices, colored.faces)
                write_uncolored_obj(mesh_root / f"{mesh_name}.obj", colored.vertices, colored.faces)
            else:
                reference_vertices, reference_faces = mesh_cache[mesh_name]
                if not np.allclose(reference_vertices, colored.vertices, atol=1e-6) or not np.array_equal(reference_faces, colored.faces):
                    raise ValueError(f"Geometry mismatch inside mesh_id={mesh_id}: {colored_path}")
            thickness = colors_to_values(colored.vertex_colors, map_values, map_colors).astype(np.float32)
            label_name = f"sample_{index:06d}.npy"
            np.save(label_root / label_name, thickness)
            params = {}
            for key, value in row.items():
                if key in {"colored_obj", "mesh_id", "main_rpm", "sub_rpm", "time_s", "sample_id"} or value is None or value == "":
                    continue
                if key == "params_json":
                    params.update(json.loads(value))
                else:
                    params[key] = float(value)
            record = {
                "mesh": f"meshes/{mesh_name}.obj",
                "thickness": f"labels/{label_name}",
                "mesh_id": mesh_name,
                "main_rpm": float(row["main_rpm"]),
                "sub_rpm": float(row["sub_rpm"]),
                "time_s": float(row["time_s"]),
                "params": params,
            }
            if row.get("split"):
                record["split"] = row["split"]
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_root / "conversion_report.json").write_text(
        json.dumps({"samples": len(rows), "meshes": len(mesh_cache), "color_map": str(args.color_map)}, indent=2),
        encoding="utf-8",
    )
    print(f"prepared {len(rows)} samples and {len(mesh_cache)} meshes under {output_root}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert colored OBJ simulation results into training data")
    parser.add_argument("--source-root", required=True, help="directory containing colored OBJs and CSV")
    parser.add_argument("--csv", required=True, help="CSV path relative to source-root")
    parser.add_argument("--output", required=True, help="prepared dataset directory")
    parser.add_argument("--color-map", default=None, help="JSON color legend; values are thickness units")
    parser.add_argument("--schema", default=None, help="optional process schema JSON")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
