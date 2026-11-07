from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .export import write_colored_obj
from .mesh import build_mesh_graph
from .model import RotomoldingMeshGNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict thickness and export colored OBJ")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--params", default=None, help="JSON object or JSON file with process parameters")
    parser.add_argument("--main-rpm", type=float, default=None)
    parser.add_argument("--sub-rpm", type=float, default=None)
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--color-min", type=float, default=None)
    parser.add_argument("--color-max", type=float, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    device = torch.device(device_name)
    schema = checkpoint["schema"]
    axes = checkpoint.get("axes")
    graph = build_mesh_graph(args.obj, axes)
    params = {}
    if args.params:
        candidate = Path(args.params)
        params = json.loads(candidate.read_text(encoding="utf-8") if candidate.exists() else args.params)
    if args.main_rpm is not None:
        params["main_rpm"] = args.main_rpm
    if args.sub_rpm is not None:
        params["sub_rpm"] = args.sub_rpm
    if args.time_s is not None:
        params["time_s"] = args.time_s
    process_values = []
    for item in schema:
        value = float(params.get(item["name"], item.get("default", 0.0)))
        process_values.append((value - float(item.get("mean", 0.0))) / max(float(item.get("std", 1.0)), 1e-8))
    model = RotomoldingMeshGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        prediction = model(
            graph.node_features.to(device),
            graph.edge_index.to(device),
            graph.edge_features.to(device),
            torch.tensor([process_values], dtype=torch.float32, device=device),
            torch.zeros(len(graph.vertices), dtype=torch.long, device=device),
        ).cpu().numpy()
    thickness = prediction * float(checkpoint.get("thickness_scale", 1.0))
    write_colored_obj(args.output, graph.vertices, np.asarray(graph.faces), thickness, args.color_min, args.color_max)


if __name__ == "__main__":
    main()
