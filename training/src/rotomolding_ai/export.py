from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_STOPS = np.asarray(
    [
        [0.05, 0.15, 0.80],
        [0.00, 0.75, 0.95],
        [0.10, 0.75, 0.20],
        [1.00, 0.85, 0.00],
        [0.90, 0.05, 0.03],
    ],
    dtype=np.float32,
)


def thickness_colors(values: np.ndarray, minimum: float | None = None, maximum: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = float(values.min()) if minimum is None else float(minimum)
    high = float(values.max()) if maximum is None else float(maximum)
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    positions = normalized * (len(_STOPS) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(_STOPS) - 1)
    fraction = (positions - left)[:, None]
    colors = _STOPS[left] * (1.0 - fraction) + _STOPS[right] * fraction
    return colors


def write_colored_obj(
    output_obj: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: np.ndarray,
    color_min: float | None = None,
    color_max: float | None = None,
    bins: int = 64,
) -> None:
    output_obj = Path(output_obj)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_obj.with_suffix(".mtl")
    values = np.asarray(thickness, dtype=np.float32)
    colors = thickness_colors(values, color_min, color_max)
    low = float(values.min()) if color_min is None else float(color_min)
    high = float(values.max()) if color_max is None else float(color_max)
    face_values = values[faces].mean(axis=1)
    face_bins = np.clip(((face_values - low) / max(high - low, 1e-8) * (bins - 1)).round(), 0, bins - 1).astype(int)

    with mtl_path.open("w", encoding="utf-8") as handle:
        for index in range(bins):
            color = thickness_colors(np.asarray([low + (high - low) * index / max(bins - 1, 1)]), low, high)[0]
            handle.write(f"newmtl thickness_{index:03d}\n")
            handle.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            handle.write("Ka 0.000000 0.000000 0.000000\nKs 0.000000 0.000000 0.000000\n\n")

    with output_obj.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write("# Vertex colors are RGB values in the v records; MTL provides fallback face colors.\n")
        for vertex, color in zip(vertices, colors):
            handle.write(
                f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )
        current_material = None
        for face, material_bin in zip(faces, face_bins):
            material = f"thickness_{material_bin:03d}"
            if material != current_material:
                handle.write(f"usemtl {material}\n")
                current_material = material
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    metadata = output_obj.with_name(output_obj.stem + "_thickness.json")
    metadata.write_text(
        json.dumps(
            {
                "unit": "mm (or source unit)",
                "color_min": low,
                "color_max": high,
                "thickness": values.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
