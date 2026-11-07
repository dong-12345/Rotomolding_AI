from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _prepare(args: argparse.Namespace, env: dict[str, str]) -> Path:
    prepared = Path(args.prepared_output).resolve()
    manifest = prepared / "manifest.jsonl"
    schema = prepared / "process_schema.json"

    if args.skip_preprocess:
        if not manifest.exists() or not schema.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset files: {manifest} and/or {schema}"
            )
        print(f"reusing prepared dataset: {prepared}", flush=True)
        return prepared

    if manifest.exists() and schema.exists() and not args.force_preprocess and not args.force_molds:
        print(f"reusing prepared dataset: {prepared}", flush=True)
        return prepared
    if manifest.exists() and schema.exists() and args.force_molds and not args.force_preprocess:
        print(f"reprocessing selected molds in: {prepared}", flush=True)

    command = [
        sys.executable,
        "tools/prepare_native_dataset.py",
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--output",
        str(prepared),
        "--max-vertices",
        str(args.max_vertices),
        "--time-s",
        str(args.time_s),
        "--val-molds",
        str(args.val_molds),
        "--workers",
        str(args.prep_workers),
    ]
    if args.color_map:
        command.extend(["--color-map", str(Path(args.color_map).resolve())])
    if args.limit_molds is not None:
        command.extend(["--limit-molds", str(args.limit_molds)])
    if args.max_simulations is not None:
        command.extend(["--max-simulations", str(args.max_simulations)])
    if args.allow_single_train_mold:
        command.append("--allow-single-train-mold")
    if args.force_molds:
        command.extend(["--force-molds", *args.force_molds])
    if args.force_preprocess:
        command.append("--force")
    _run(command, env)

    if not manifest.exists() or not schema.exists():
        raise RuntimeError("Preprocessing completed without manifest.jsonl/process_schema.json")
    return prepared


def _train(args: argparse.Namespace, prepared: Path, env: dict[str, str]) -> None:
    run_output = Path(args.run_output).resolve()
    command = [
        sys.executable,
        "-m",
        "rotomolding_ai.train",
        "--manifest",
        str(prepared / "manifest.jsonl"),
        "--mesh-root",
        str(prepared),
        "--schema",
        str(prepared / "process_schema.json"),
        "--output",
        str(run_output),
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
        "--weight-decay",
        str(args.weight_decay),
        "--smooth-weight",
        str(args.smooth_weight),
        "--thickness-scale",
        str(args.thickness_scale),
        "--seed",
        str(args.seed),
    ]
    if args.axes:
        command.extend(["--axes", str(Path(args.axes).resolve())])
    if args.test_limit is not None:
        command.extend(["--test-limit", str(args.test_limit)])

    if args.resume is not None:
        checkpoint = run_output / "last.pt" if args.resume == "auto" else Path(args.resume).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        command.extend(["--resume", str(checkpoint)])
        print(f"resuming training from {checkpoint}", flush=True)
    else:
        print(f"starting training from scratch in {run_output}", flush=True)
    _run(command, env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the native dataset and train the rotomolding Mesh GNN"
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--prepared-output", default="dataset_prepared")
    parser.add_argument("--run-output", default="runs/native_full")
    parser.add_argument("--color-map", default=None)
    parser.add_argument("--axes", default=None)
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument("--max-vertices", type=int, default=12000)
    parser.add_argument("--time-s", type=float, default=1.0)
    parser.add_argument("--val-molds", type=int, default=20)
    parser.add_argument("--limit-molds", type=int, default=None)
    parser.add_argument("--max-simulations", type=int, default=None)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument(
        "--force-molds",
        nargs="+",
        default=[],
        help="force raw reprocessing for selected molds while reusing other caches",
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--allow-single-train-mold", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from --run-output/last.pt, or provide a checkpoint path",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="start from scratch even if the run directory already contains checkpoints",
    )
    args = parser.parse_args()

    if args.resume is not None and args.restart:
        parser.error("--resume and --restart are mutually exclusive")
    if args.skip_preprocess and args.force_preprocess:
        parser.error("--skip-preprocess and --force-preprocess are mutually exclusive")
    if args.prep_workers < 1:
        parser.error("--prep-workers must be at least 1")
    if args.max_vertices < 1:
        parser.error("--max-vertices must be positive")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    prepared = _prepare(args, env)
    if args.preprocess_only:
        print("preprocessing complete", flush=True)
        return
    _train(args, prepared, env)


if __name__ == "__main__":
    main()
