from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_COLOR_MAP = {
    "values": [0.0, 0.25, 0.5, 0.75, 1.0],
    "colors": [
        [0.05, 0.15, 0.80],
        [0.00, 0.75, 0.95],
        [0.10, 0.75, 0.20],
        [1.00, 0.85, 0.00],
        [0.90, 0.05, 0.03],
    ],
}


def load_color_map(path: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        raw = DEFAULT_COLOR_MAP
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    colors = np.asarray(raw["colors"], dtype=np.float32)
    if colors.max(initial=0.0) > 1.5:
        colors = colors / 255.0
    if "values" in raw:
        values = np.asarray(raw["values"], dtype=np.float32)
    else:
        values = np.linspace(float(raw["min"]), float(raw["max"]), len(colors), dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) < 2:
        raise ValueError("Color map colors must be an Nx3 array with at least two stops")
    if values.ndim != 1 or len(values) != len(colors):
        raise ValueError("Color map values must have the same length as colors")
    return values, np.clip(colors, 0.0, 1.0)


def colors_to_values(colors: np.ndarray, map_values: np.ndarray, map_colors: np.ndarray) -> np.ndarray:
    """Invert a piecewise linear RGB color map by nearest segment projection."""

    colors = np.asarray(colors, dtype=np.float32)
    colors = colors / 255.0 if colors.max(initial=0.0) > 1.5 else colors
    result = np.empty(len(colors), dtype=np.float32)
    for index, color in enumerate(colors):
        best_distance = float("inf")
        best_value = float(map_values[0])
        for stop in range(len(map_colors) - 1):
            start = map_colors[stop]
            end = map_colors[stop + 1]
            delta = end - start
            denominator = float(np.dot(delta, delta))
            fraction = 0.0 if denominator < 1e-12 else float(np.dot(color - start, delta) / denominator)
            fraction = float(np.clip(fraction, 0.0, 1.0))
            projected = start + fraction * delta
            distance = float(np.sum((color - projected) ** 2))
            if distance < best_distance:
                best_distance = distance
                best_value = float(map_values[stop] + fraction * (map_values[stop + 1] - map_values[stop]))
        result[index] = best_value
    return result
