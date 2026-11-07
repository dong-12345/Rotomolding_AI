from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class MeshGraph:
    """A surface mesh plus graph features used by the GNN."""

    vertices: np.ndarray
    faces: np.ndarray
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor


def _parse_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    value = int(raw)
    return value - 1 if value > 0 else vertex_count + value


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertex positions and triangulated faces from a Wavefront OBJ."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                polygon = [_parse_index(token, len(vertices)) for token in parts[1:]]
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ must contain vertices and faces: {path}")
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.min() < 0 or faces_array.max() >= len(vertices_array):
        raise ValueError(f"OBJ contains an invalid face index: {path}")
    return vertices_array, faces_array


def _normalize_axis(axis: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(axis), dtype=np.float32)
    norm = np.linalg.norm(value)
    if norm < 1e-8:
        raise ValueError("Axis direction must be non-zero")
    return value / norm


def _axis_features(
    vertices: np.ndarray,
    origin: Iterable[float],
    direction: Iterable[float],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin_array = np.asarray(list(origin), dtype=np.float32)
    direction_array = _normalize_axis(direction)
    relative = vertices - origin_array
    projection = relative @ direction_array
    radial = np.linalg.norm(relative - projection[:, None] * direction_array[None, :], axis=1)
    return radial / scale, projection / scale


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals[faces[:, 0]] += face_normals
    normals[faces[:, 1]] += face_normals
    normals[faces[:, 2]] += face_normals
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-8)


def _edges_from_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    undirected: set[tuple[int, int]] = set()
    for a, b, c in faces.tolist():
        for first, second in ((a, b), (b, c), (c, a)):
            undirected.add((min(first, second), max(first, second)))

    directed: list[tuple[int, int]] = []
    for first, second in sorted(undirected):
        directed.extend(((first, second), (second, first)))
    edge_index = np.asarray(directed, dtype=np.int64).T
    source = vertices[edge_index[0]]
    destination = vertices[edge_index[1]]
    relative = destination - source
    lengths = np.linalg.norm(relative, axis=1, keepdims=True)
    edge_features = np.concatenate([relative, lengths], axis=1).astype(np.float32)
    return edge_index, edge_features


def deduplicate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    decimals: int = 6,
    max_vertices: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Merge OBJ face-corner vertices that share the same rounded position."""

    rounded = np.round(np.asarray(vertices, dtype=np.float32), decimals=decimals)
    unique_vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    remapped_faces = inverse[np.asarray(faces, dtype=np.int64)]
    valid = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 0] != remapped_faces[:, 2])
    )
    remapped_faces = remapped_faces[valid]
    merged_color_values = None
    if colors is not None:
        source_colors = np.asarray(colors, dtype=np.float32)
        merged_color_values = np.zeros((len(unique_vertices), source_colors.shape[1]), dtype=np.float32)
        counts = np.zeros(len(unique_vertices), dtype=np.float32)
        np.add.at(merged_color_values, inverse, source_colors)
        np.add.at(counts, inverse, 1.0)
        merged_color_values /= np.maximum(counts[:, None], 1.0)

    if max_vertices and len(unique_vertices) > max_vertices:
        return coarsen_mesh(unique_vertices, remapped_faces, merged_color_values, max_vertices)
    if merged_color_values is None:
        return unique_vertices.astype(np.float32), remapped_faces.astype(np.int64), None
    return unique_vertices.astype(np.float32), remapped_faces.astype(np.int64), merged_color_values


def coarsen_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Deterministically merge nearby surface vertices into a coarse mesh."""

    if len(vertices) <= max_vertices:
        return vertices, faces, colors
    extent = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
    cell_size = float(np.max(extent) / max(np.sqrt(max_vertices), 1.0))
    minimum = vertices.min(axis=0)
    for _ in range(12):
        buckets = np.floor((vertices - minimum) / max(cell_size, 1e-8)).astype(np.int64)
        _, inverse = np.unique(buckets, axis=0, return_inverse=True)
        if inverse.max(initial=0) + 1 <= max_vertices:
            break
        cell_size *= 1.25

    count = int(inverse.max()) + 1
    reduced_vertices = np.zeros((count, 3), dtype=np.float32)
    counts = np.bincount(inverse, minlength=count).astype(np.float32)
    np.add.at(reduced_vertices, inverse, vertices)
    reduced_vertices /= np.maximum(counts[:, None], 1.0)
    reduced_colors = None
    if colors is not None:
        reduced_colors = np.zeros((count, colors.shape[1]), dtype=np.float32)
        np.add.at(reduced_colors, inverse, colors)
        reduced_colors /= np.maximum(counts[:, None], 1.0)
    reduced_faces = inverse[faces]
    valid = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
        & (reduced_faces[:, 0] != reduced_faces[:, 2])
    )
    reduced_faces = reduced_faces[valid]
    return reduced_vertices, reduced_faces.astype(np.int64), reduced_colors


def build_mesh_graph_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    axes: dict | None = None,
) -> MeshGraph:
    """Build a graph from an already parsed, optionally deduplicated mesh."""

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    scale = float(np.max(maximum - minimum))
    scale = max(scale, 1e-6)
    normalized = (vertices - center) / scale
    normals = _vertex_normals(vertices, faces)

    axis_config = axes or {}
    main = axis_config.get("main", {"origin": [0, 0, 0], "direction": [0, 0, 1]})
    sub = axis_config.get("sub", {"origin": [0, 0, 0], "direction": [1, 0, 0]})
    main_radial, main_projection = _axis_features(vertices, main["origin"], main["direction"], scale)
    sub_radial, sub_projection = _axis_features(vertices, sub["origin"], sub["direction"], scale)
    node_features = np.concatenate(
        [
            normalized,
            normals,
            main_radial[:, None],
            sub_radial[:, None],
            main_projection[:, None],
            sub_projection[:, None],
            np.full((len(vertices), 1), np.log1p(scale), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    edge_index, edge_features = _edges_from_faces(normalized, faces)
    return MeshGraph(
        vertices=vertices,
        faces=faces,
        node_features=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_features=torch.from_numpy(edge_features),
    )


def build_mesh_graph(
    path: str | Path,
    axes: dict | None = None,
) -> MeshGraph:
    """Build normalized node/edge features from an OBJ.

    Axes are expressed in the original OBJ coordinate system. Defaults match a
    canonical machine frame: main axis Z and sub axis X.
    """

    vertices, faces = load_obj(path)
    return build_mesh_graph_from_arrays(vertices, faces, axes)
