from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_colored_dataset import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare colored OBJ data and train in one command")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prepared-output", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--color-map", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--axes", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    args = parser.parse_args()
    prepare(args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "rotomolding_ai.train",
        "--manifest",
        str(Path(args.prepared_output) / "manifest.jsonl"),
        "--mesh-root",
        str(args.prepared_output),
        "--schema",
        str(Path(args.prepared_output) / "process_schema.json"),
        "--output",
        args.run_output,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--blocks",
        str(args.blocks),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.learning_rate),
        "--thickness-scale",
        str(args.thickness_scale),
    ]
    if args.axes:
        command.extend(["--axes", args.axes])
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
