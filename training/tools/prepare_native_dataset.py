from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Each worker owns one mold. Prevent NumPy/PyTorch from creating another
# thread pool inside every worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotomolding_ai.color import colors_to_values, load_color_map  # noqa: E402
from rotomolding_ai.colored_obj import load_colored_obj, write_uncolored_obj  # noqa: E402
from rotomolding_ai.data import DEFAULT_PROCESS_SCHEMA  # noqa: E402
from rotomolding_ai.mesh import deduplicate_mesh  # noqa: E402


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "mesh"


def _load_rotation_parameters(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(key).zfill(2): {
            "main_rpm": float(value["rotationSpeed"]),
            "sub_rpm": float(value["rotationSpeed2"]),
        }
        for key, value in raw["results"].items()
    }


def _simulation_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("simulation_*.obj"), key=lambda path: int(path.stem.split("_")[-1]))


def _make_schema(records: list[dict], train_records: list[dict]) -> list[dict]:
    schema = []
    for item in DEFAULT_PROCESS_SCHEMA:
        name = item["name"]
        values = [float(record[name]) for record in train_records if name in record]
        if values:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
            default = mean
        else:
            default = float(item.get("default", 1.0))
            mean = float(item.get("mean", default))
            std = max(float(item.get("std", 1.0)), 1e-6)
        schema.append({"name": name, "default": default, "mean": mean, "std": std})
    return schema


def _process_one_mesh(
    mold_dir: Path,
    split: str,
    output_root: Path,
    rotation: dict[str, dict[str, float]],
    map_values: np.ndarray,
    map_colors: np.ndarray,
    decimals: int,
    max_vertices: int | None,
    time_s: float,
    max_simulations: int | None,
    force: bool,
) -> tuple[list[dict], list[dict]]:
    result_dir = mold_dir / "simulation_results"
    files = _simulation_files(result_dir)
    if max_simulations:
        files = files[:max_simulations]
    if not files:
        raise ValueError(f"No simulation_XX.obj files found in {result_dir}")

    mesh_name = _safe_name(f"{split}_{mold_dir.name}")
    mesh_npz = output_root / "meshes" / f"{mesh_name}.npz"
    mesh_obj = output_root / "meshes" / f"{mesh_name}.obj"
    canonical_vertices = None
    canonical_faces = None
    if not force and mesh_npz.exists():
        with np.load(mesh_npz) as cached:
            canonical_vertices = cached["vertices"]
            canonical_faces = cached["faces"]
        if not mesh_obj.exists():
            write_uncolored_obj(mesh_obj, canonical_vertices, canonical_faces)

    def cached_label_is_valid(path: Path) -> bool:
        if force or canonical_vertices is None or not path.exists():
            return False
        try:
            label = np.load(path, mmap_mode="r")
            return label.ndim == 1 and label.shape[0] == canonical_vertices.shape[0]
        except (OSError, ValueError):
            return False

    records = []
    skipped = []
    label_dir = output_root / "labels" / mesh_name
    label_dir.mkdir(parents=True, exist_ok=True)

    def skip_simulation(source_path: Path, reason: str) -> None:
        item = {
            "mesh_id": mesh_name,
            "split": split,
            "source_obj": str(source_path),
            "reason": reason,
        }
        skipped.append(item)
        print(
            f"skipping {source_path}: {reason}",
            flush=True,
        )

    for source_path in files:
        simulation_id = source_path.stem.split("_")[-1].zfill(2)
        if simulation_id not in rotation:
            skip_simulation(
                source_path,
                f"No rotation parameters for simulation_{simulation_id}",
            )
            continue
        label_path = label_dir / f"simulation_{simulation_id}.npy"
        if cached_label_is_valid(label_path):
            records.append(
                {
                    "mesh": f"meshes/{mesh_name}.npz",
                    "thickness": f"labels/{mesh_name}/simulation_{simulation_id}.npy",
                    "mesh_id": mesh_name,
                    "main_rpm": rotation[simulation_id]["main_rpm"],
                    "sub_rpm": rotation[simulation_id]["sub_rpm"],
                    "time_s": time_s,
                    "split": split,
                    "source_obj": str(source_path),
                }
            )
            continue
        try:
            colored = load_colored_obj(source_path)
            vertices, faces, colors = deduplicate_mesh(
                colored.vertices,
                colored.faces,
                colored.vertex_colors,
                decimals=decimals,
                max_vertices=max_vertices,
            )
            thickness = colors_to_values(colors, map_values, map_colors).astype(np.float32)
            if canonical_vertices is None:
                np.savez_compressed(
                    mesh_npz,
                    vertices=vertices,
                    faces=faces,
                )
                write_uncolored_obj(mesh_obj, vertices, faces)
                canonical_vertices, canonical_faces = vertices, faces
            elif (
                not np.allclose(canonical_vertices, vertices, atol=10 ** (-decimals))
                or not np.array_equal(canonical_faces, faces)
            ):
                raise ValueError(f"Mesh topology differs between simulations in {mold_dir}")

            np.save(label_path, thickness)
        except (OSError, ValueError, IndexError, FloatingPointError) as exc:
            skip_simulation(source_path, f"{type(exc).__name__}: {exc}")
            continue

        records.append(
            {
                "mesh": f"meshes/{mesh_name}.npz",
                "thickness": f"labels/{mesh_name}/simulation_{simulation_id}.npy",
                "mesh_id": mesh_name,
                "main_rpm": rotation[simulation_id]["main_rpm"],
                "sub_rpm": rotation[simulation_id]["sub_rpm"],
                "time_s": time_s,
                "split": split,
                "source_obj": str(source_path),
            }
        )
    return records, skipped


def _process_job(job: tuple) -> tuple[list[dict], list[dict]]:
    """Process one mold in a spawn-safe worker process."""
    torch.set_num_threads(1)
    return _process_one_mesh(*job)


def prepare(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output)
    (output_root / "meshes").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)
    rotation = _load_rotation_parameters(dataset_root / "rotation_parameters_01_to_25.json")
    map_values, map_colors = load_color_map(args.color_map)
    force_molds = set(args.force_molds or [])

    train_dirs = sorted((dataset_root / "training_set").iterdir())
    train_dirs = [path for path in train_dirs if path.is_dir()]
    val_dirs = sorted((dataset_root / "validation_set").iterdir()) if (dataset_root / "validation_set").exists() else []
    val_dirs = [path for path in val_dirs if path.is_dir()]
    if args.limit_molds:
        train_dirs = train_dirs[: args.limit_molds]
        val_dirs = val_dirs[: args.limit_molds]
    if len(train_dirs) < 2 and not args.allow_single_train_mold:
        raise ValueError("Need at least two training molds; use --allow-single-train-mold only for a smoke test")

    internal_val_count = min(max(args.val_molds, 0), max(len(train_dirs) - 1, 0))
    internal_val_names = {path.name for path in train_dirs[-internal_val_count:]} if internal_val_count else set()
    jobs = []
    for mold_dir in train_dirs:
        split = "val" if mold_dir.name in internal_val_names else "train"
        force_mold = args.force or mold_dir.name in force_molds or f"{split}_{mold_dir.name}" in force_molds
        jobs.append(
            (
                mold_dir,
                split,
                output_root,
                rotation,
                map_values,
                map_colors,
                args.decimals,
                args.max_vertices,
                args.time_s,
                args.max_simulations,
                force_mold,
            )
        )
    for mold_dir in val_dirs:
        force_mold = args.force or mold_dir.name in force_molds or f"test_{mold_dir.name}" in force_molds
        jobs.append(
            (
                mold_dir,
                "test",
                output_root,
                rotation,
                map_values,
                map_colors,
                args.decimals,
                args.max_vertices,
                args.time_s,
                args.max_simulations,
                force_mold,
            )
        )

    records = []
    skipped = []
    if args.workers <= 1:
        for index, job in enumerate(jobs, start=1):
            job_records, job_skipped = _process_job(job)
            records.extend(job_records)
            skipped.extend(job_skipped)
            print(f"processed_molds={index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_job, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                job_records, job_skipped = future.result()
                records.extend(job_records)
                skipped.extend(job_skipped)
                print(f"processed_molds={index}/{len(jobs)}", flush=True)

    train_records = [record for record in records if record["split"] == "train"]
    schema = _make_schema(records, train_records)
    (output_root / "process_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "train_molds": len(train_dirs) - internal_val_count,
        "internal_validation_molds": internal_val_count,
        "external_validation_molds": len(val_dirs),
        "samples": len(records),
        "skipped_samples": len(skipped),
        "skipped": skipped,
        "force_molds": sorted(force_molds),
        "time_s_fixed": args.time_s,
        "label_definition": "normalized scalar reconstructed from RGB heatmap; provide a calibrated color map for physical thickness units",
        "color_map": str(args.color_map) if args.color_map else "default blue-cyan-green-yellow-red normalized map",
    }
    (output_root / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the native Rotomolding_AI dataset layout")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output", default="dataset_prepared")
    parser.add_argument("--color-map", default=None)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--max-vertices", type=int, default=None, help="coarsen each mesh to this many vertices")
    parser.add_argument("--time-s", type=float, default=1.0, help="fixed final simulation time because source data has no time field")
    parser.add_argument("--val-molds", type=int, default=20)
    parser.add_argument("--limit-molds", type=int, default=None, help="limit each split for a smoke test")
    parser.add_argument("--max-simulations", type=int, default=None, help="limit cases per mold for a smoke test")
    parser.add_argument("--workers", type=int, default=1, help="parallel mold workers; 4-8 is usually suitable")
    parser.add_argument("--force", action="store_true", help="ignore existing mesh/label caches and rebuild everything")
    parser.add_argument(
        "--force-molds",
        nargs="+",
        default=[],
        help="force raw reprocessing for selected mold names, while reusing other caches",
    )
    parser.add_argument("--allow-single-train-mold", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
