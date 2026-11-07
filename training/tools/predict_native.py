from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict rotomolding thickness from an OBJ and export a colored OBJ"
    )
    parser.add_argument("--obj", required=True, help="input OBJ mesh")
    parser.add_argument(
        "--main-rpm",
        required=True,
        type=float,
        help="main-axis rotation speed in the same unit used by training data",
    )
    parser.add_argument(
        "--sub-rpm",
        required=True,
        type=float,
        help="secondary-axis rotation speed in the same unit used by training data",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/native_full/best.pt",
        help="model checkpoint; defaults to the best completed training model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output colored OBJ; defaults to <input>_prediction.obj",
    )
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument("--params", default=None, help="JSON object or JSON file with extra process parameters")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--color-min", type=float, default=None)
    parser.add_argument("--color-max", type=float, default=None)
    args = parser.parse_args()

    obj = Path(args.obj).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not obj.exists():
        raise FileNotFoundError(f"Input OBJ does not exist: {obj}")
    if obj.suffix.lower() != ".obj":
        raise ValueError(f"Input must be an OBJ file: {obj}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    output = (
        Path(args.output).resolve()
        if args.output
        else obj.with_name(f"{obj.stem}_prediction.obj")
    )
    command = [
        sys.executable,
        "-m",
        "rotomolding_ai.predict",
        "--checkpoint",
        str(checkpoint),
        "--obj",
        str(obj),
        "--output",
        str(output),
        "--main-rpm",
        str(args.main_rpm),
        "--sub-rpm",
        str(args.sub_rpm),
        "--device",
        args.device,
    ]
    if args.time_s is not None:
        command.extend(["--time-s", str(args.time_s)])
    if args.params is not None:
        command.extend(["--params", args.params])
    if args.color_min is not None:
        command.extend(["--color-min", str(args.color_min)])
    if args.color_max is not None:
        command.extend(["--color-max", str(args.color_max)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"colored OBJ: {output}", flush=True)
    print(f"material file: {output.with_suffix('.mtl')}", flush=True)
    print(f"thickness data: {output.with_name(output.stem + '_thickness.json')}", flush=True)


if __name__ == "__main__":
    main()
