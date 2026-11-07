from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import RotomoldingDataset, collate_graphs, load_process_schema, read_manifest, split_records
from .losses import thickness_loss
from .model import RotomoldingMeshGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, optimizer, device, scaler, smooth_weight, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_features"],
                batch["process"],
                batch["batch"],
            )
            loss, _ = thickness_loss(prediction, batch["target"], batch["edge_index"], smooth_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def _checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    schema,
    axes,
    thickness_scale: float,
    best_val_loss: float,
    best_epoch: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> dict:
    state = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "schema": schema,
        "axes": axes,
        "thickness_scale": thickness_scale,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(checkpoint: dict) -> None:
    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a rotomolding thickness Mesh GNN")
    parser.add_argument("--manifest", required=True, help="JSONL manifest")
    parser.add_argument("--mesh-root", default=".")
    parser.add_argument("--schema", default=None, help="process schema JSON")
    parser.add_argument("--axes", default=None, help="axes JSON, optional")
    parser.add_argument("--output", default="runs/rotomolding")
    parser.add_argument("--resume", default=None, help="checkpoint path, normally runs/.../last.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    parser.add_argument("--test-limit", type=int, default=None, help="limit external test samples for a quick run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.schema is None:
            schema = resume_checkpoint["schema"]
        else:
            schema = load_process_schema(args.schema)
            if schema != resume_checkpoint.get("schema"):
                raise ValueError("Resume schema differs from the checkpoint schema")
    else:
        schema = load_process_schema(args.schema)
    axes = None
    if args.axes:
        axes = json.loads(Path(args.axes).read_text(encoding="utf-8"))
    elif resume_checkpoint:
        axes = resume_checkpoint.get("axes")
    thickness_scale = float(
        resume_checkpoint.get("thickness_scale", args.thickness_scale) if resume_checkpoint else args.thickness_scale
    )
    records = read_manifest(args.manifest)
    train_records, val_records, test_records = split_records(records, args.seed)
    print(f"samples: train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    train_set = RotomoldingDataset(train_records, schema, args.mesh_root, axes, thickness_scale)
    val_set = RotomoldingDataset(val_records, schema, args.mesh_root, axes, thickness_scale)
    test_set = RotomoldingDataset(test_records, schema, args.mesh_root, axes, thickness_scale)
    if args.test_limit is not None:
        test_set.records = test_set.records[: args.test_limit]
    if len(train_set) == 0:
        raise RuntimeError("Training split is empty")
    first = train_set[0]
    model_config = resume_checkpoint["model_config"] if resume_checkpoint else {
        "node_dim": first.node_features.shape[1],
        "edge_dim": first.edge_features.shape[1],
        "process_dim": len(schema),
        "hidden_dim": args.hidden_dim,
        "blocks": args.blocks,
        "dropout": args.dropout,
    }
    if (
        model_config["node_dim"] != first.node_features.shape[1]
        or model_config["edge_dim"] != first.edge_features.shape[1]
        or model_config["process_dim"] != len(schema)
    ):
        raise ValueError("Mesh feature dimensions differ from the checkpoint")
    model = RotomoldingMeshGNN(**model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model_parameters={parameter_count:,}")

    loader_kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_value = float("inf")
    best_epoch = 0
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model_state"])
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if "scaler_state" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_value = float(resume_checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        _restore_random_state(resume_checkpoint)
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    patience = 30
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            train_loss = run_epoch(model, train_loader, optimizer, device, scaler, args.smooth_weight, True)
            val_loss = run_epoch(model, val_loader, optimizer, device, scaler, args.smooth_weight, False) if len(val_set) else train_loss
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            scheduler.step(val_loss)
            epoch_seconds = time.perf_counter() - epoch_started
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            print(
                f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} seconds={epoch_seconds:.2f} "
                f"peak_vram_mib={peak_memory:.0f}"
            )
            if val_loss < best_value:
                best_value = val_loss
                best_epoch = epoch
            state = _checkpoint_state(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                schema,
                axes,
                thickness_scale,
                best_value,
                best_epoch,
                train_loss,
                val_loss,
            )
            torch.save(state, output / "last.pt")
            if epoch == best_epoch:
                torch.save(state, output / "best.pt")
            if epoch - best_epoch >= patience:
                print(f"early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print("training interrupted; last completed epoch is saved in last.pt")
        return

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = run_epoch(model, test_loader, optimizer, device, scaler, args.smooth_weight, False) if len(test_set) else float("nan")
    print(f"test_loss={test_loss:.6f}")
    (output / "training_summary.json").write_text(
        json.dumps({"best_val_loss": best_value, "test_loss": test_loss, "best_epoch": best_epoch}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
