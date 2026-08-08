import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import PintleDynamicMeshDataset
from model_batgno_lite import (
    BATGNOLite,
    MetricAccumulator,
    masked_edge_gradient_loss,
    masked_mse_loss,
)
from model_mgn_t import MeshGraphNetT

# Recommended edge-gradient experiments:
#   python train.py --dynamic-edge-attr --grad-loss-weight 0.01 --pressure-grad-weight 2.0 --save-dir checkpoints/batgno_lite_dynamic_grad001
#   python train.py --dynamic-edge-attr --grad-loss-weight 0.05 --pressure-grad-weight 2.0 --save-dir checkpoints/batgno_lite_dynamic_grad005
#   python train.py --dynamic-edge-attr --grad-loss-weight 0.1  --pressure-grad-weight 2.0 --save-dir checkpoints/batgno_lite_dynamic_grad01


def parse_args():
    parser = argparse.ArgumentParser(description="Train BA-TGNO-Lite single-step predictor.")
    parser.add_argument("--model", choices=("batgno_lite", "mgn_t"), default="batgno_lite")
    parser.add_argument("--dataset", default="pintle_nozzle_dynamic_dataset.npz")
    parser.add_argument("--graph", default="pintle_nozzle_graph_k8.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=30,
        help="Stop after this many epochs without a meaningful validation improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1.0e-6,
        help="Minimum absolute validation-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--dynamic-edge-attr",
        action="store_true",
        help="Use per-sample dynamic mesh edge attributes [E,9] instead of ref edge_attr [E,3].",
    )
    parser.add_argument(
        "--grad-loss-weight",
        type=float,
        default=0.0,
        help="Weight for edge-gradient loss. Recommended: 0.01, 0.05, or 0.1.",
    )
    parser.add_argument("--pressure-grad-weight", type=float, default=2.0)
    parser.add_argument("--temperature-grad-weight", type=float, default=1.0)
    parser.add_argument("--velocity-grad-weight", type=float, default=1.0)
    return parser.parse_args()


def squeeze_batch(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def gradient_variable_weights(args):
    return [
        args.temperature_grad_weight,
        args.velocity_grad_weight,
        args.pressure_grad_weight,
    ]


def compute_losses(pred, batch, args):
    field_loss = masked_mse_loss(pred, batch["target"], batch["target_mask"])
    if args.model == "batgno_lite" and args.grad_loss_weight > 0.0:
        edge_grad_loss = masked_edge_gradient_loss(
            pred=pred,
            target=batch["target"],
            mask=batch["target_mask"],
            edge_index=batch["edge_index"],
            variable_weights=gradient_variable_weights(args),
        )
        total_loss = field_loss + args.grad_loss_weight * edge_grad_loss
    else:
        edge_grad_loss = field_loss.new_zeros(())
        total_loss = field_loss
    return total_loss, field_loss, edge_grad_loss


def configure_model_args(args):
    if args.save_dir is None:
        args.save_dir = "checkpoints/mgn_t" if args.model == "mgn_t" else "checkpoints/batgno_lite"

    if args.model == "mgn_t":
        args.dynamic_edge_attr = False
        args.grad_loss_weight = 0.0
        args.edge_attr_dim = 3
    else:
        args.edge_attr_dim = 9 if args.dynamic_edge_attr else 3
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(args):
    if args.model == "mgn_t":
        return MeshGraphNetT(
            node_dim=args.node_dim,
            edge_attr_dim=args.edge_attr_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=3,
            chunk_size=args.chunk_size,
        )
    if args.model == "batgno_lite":
        return BATGNOLite(
            input_dim=args.node_dim,
            edge_attr_dim=args.edge_attr_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=3,
            chunk_size=args.chunk_size,
        )
    raise ValueError(f"Unknown model: {args.model}")


def check_finite_loss(total_loss, field_loss, edge_grad_loss, batch, phase):
    if not torch.isfinite(total_loss):
        time_index = batch["time_index"].detach().cpu().item()
        raise FloatingPointError(
            f"{phase} loss is NaN/Inf at time_index={time_index}, "
            f"total_loss={total_loss.item()}, field_loss={field_loss.item()}, "
            f"edge_grad_loss={edge_grad_loss.item()}"
        )


def run_train_epoch(model, loader, optimizer, scaler, device, use_amp, grad_clip, args):
    model.train()
    total_loss_sum = 0.0
    field_loss_sum = 0.0
    grad_loss_sum = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch(squeeze_batch(batch), device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            pred, _ = model(
                batch["node_x"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["current_field"],
                batch["target_mask"],
            )
            loss, field_loss, edge_grad_loss = compute_losses(pred, batch, args)

        check_finite_loss(loss, field_loss, edge_grad_loss, batch, "train")
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += float(loss.detach().cpu())
        field_loss_sum += float(field_loss.detach().cpu())
        grad_loss_sum += float(edge_grad_loss.detach().cpu())
        total_samples += 1

    denom = max(total_samples, 1)
    return (
        total_loss_sum / denom,
        field_loss_sum / denom,
        grad_loss_sum / denom,
    )


@torch.no_grad()
def run_eval_epoch(model, loader, device, use_amp, args):
    model.eval()
    total_loss_sum = 0.0
    field_loss_sum = 0.0
    grad_loss_sum = 0.0
    total_samples = 0
    metrics = MetricAccumulator()

    for batch in loader:
        batch = move_batch(squeeze_batch(batch), device)
        with autocast(device_type=device.type, enabled=use_amp):
            pred, _ = model(
                batch["node_x"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["current_field"],
                batch["target_mask"],
            )
            loss, field_loss, edge_grad_loss = compute_losses(pred, batch, args)

        check_finite_loss(loss, field_loss, edge_grad_loss, batch, "val")
        metrics.update(pred, batch["target"], batch["target_mask"])
        total_loss_sum += float(loss.detach().cpu())
        field_loss_sum += float(field_loss.detach().cpu())
        grad_loss_sum += float(edge_grad_loss.detach().cpu())
        total_samples += 1

    denom = max(total_samples, 1)
    return (
        total_loss_sum / denom,
        field_loss_sum / denom,
        grad_loss_sum / denom,
        metrics.compute(),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    epochs_without_improvement,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This first implementation expects --batch-size 1 for large graphs.")
    args = configure_model_args(args)
    resume_node_dim = None

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        resume_args = resume_ckpt.get("args", {}) if isinstance(resume_ckpt, dict) else {}
        if "model" in resume_args:
            args.model = str(resume_args["model"])
        if "node_dim" in resume_args:
            resume_node_dim = int(resume_args["node_dim"])
        if "dynamic_edge_attr" in resume_args:
            args.dynamic_edge_attr = bool(resume_args["dynamic_edge_attr"])
        if "edge_attr_dim" in resume_args:
            args.edge_attr_dim = int(resume_args["edge_attr_dim"])
        else:
            args.edge_attr_dim = 9 if args.dynamic_edge_attr else 3
        if args.model == "mgn_t":
            args.dynamic_edge_attr = False
            args.grad_loss_weight = 0.0
            args.edge_attr_dim = 3
        for name in (
            "grad_loss_weight",
            "temperature_grad_weight",
            "velocity_grad_weight",
            "pressure_grad_weight",
        ):
            if name in resume_args and args.model == "batgno_lite":
                setattr(args, name, float(resume_args[name]))

    seed_everything(args.seed)
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PintleDynamicMeshDataset(
        args.dataset,
        args.graph,
        split="train",
        history_steps=args.history_steps,
        dynamic_edge_attr=args.dynamic_edge_attr,
    )
    val_ds = PintleDynamicMeshDataset(
        args.dataset,
        args.graph,
        split="val",
        history_steps=args.history_steps,
        dynamic_edge_attr=args.dynamic_edge_attr,
    )
    if train_ds.node_feature_dim != val_ds.node_feature_dim:
        raise ValueError(
            "Train/validation node feature dimensions differ: "
            f"{train_ds.node_feature_dim} vs {val_ds.node_feature_dim}"
        )
    args.node_dim = int(train_ds.node_feature_dim)
    if resume_node_dim is not None and resume_node_dim != args.node_dim:
        raise ValueError(
            f"Checkpoint node_dim={resume_node_dim}, but this dataset produces "
            f"node_dim={args.node_dim}. Use a checkpoint trained on the same schema."
        )
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    model = build_model(args).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        epochs_without_improvement = int(
            ckpt.get("epochs_without_improvement", 0)
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"model name: {args.model}")
    print(f"node dim: {args.node_dim}")
    print(f"device: {device}")
    print(f"amp: {use_amp}")
    print(f"dynamic edge attr: {args.dynamic_edge_attr}")
    print(f"edge attr dim: {args.edge_attr_dim}")
    print(f"hidden dim: {args.hidden_dim}")
    print(f"num layers: {args.num_layers}")
    print(f"trainable parameters: {trainable_parameters:,}")
    print(f"seed: {args.seed}")
    print(
        "early stopping: "
        f"patience={args.early_stop_patience}, "
        f"min_delta={args.early_stop_min_delta:.3e}"
    )
    print(f"explicit window schema: {train_ds.is_explicit_window_schema}")
    print(f"condition features used: {train_ds.use_condition_features}")
    print(f"gradient loss used: {args.model == 'batgno_lite' and args.grad_loss_weight > 0.0}")
    print(f"grad loss weight: {args.grad_loss_weight}")
    print(
        "grad variable weights: "
        f"T={args.temperature_grad_weight}, V={args.velocity_grad_weight}, "
        f"p={args.pressure_grad_weight}"
    )
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    print(f"num nodes: {train_ds.num_nodes}, num edges: {train_ds.edge_index.shape[1]}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_total, train_field, train_grad = run_train_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, args.grad_clip, args
        )
        val_total, val_field, val_grad, val_metrics = run_eval_epoch(
            model, val_loader, device, use_amp, args
        )
        scheduler.step(val_total)

        improved = val_total < best_val_loss - args.early_stop_min_delta
        if improved:
            best_val_loss = val_total
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        rel = val_metrics["rel_l2"]
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:04d} | lr {lr:.3e} | "
            f"train total {train_total:.6e} | train field {train_field:.6e} | "
            f"train grad {train_grad:.6e} | "
            f"val total {val_total:.6e} | val field {val_field:.6e} | "
            f"val grad {val_grad:.6e} | "
            f"relL2 T {rel['temperature']:.6e} | "
            f"V {rel['velocity']:.6e} | p {rel['pressure']:.6e} | "
            f"overall {rel['overall']:.6e}"
        )

        save_checkpoint(
            save_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            epochs_without_improvement,
            args,
        )
        if improved:
            save_checkpoint(
                save_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                epochs_without_improvement,
                args,
            )
            print(f"saved best checkpoint: {save_dir / 'best.pt'}")
        elif args.early_stop_patience > 0:
            print(
                "early-stop counter: "
                f"{epochs_without_improvement}/{args.early_stop_patience} "
                f"(best val {best_val_loss:.6e})"
            )

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"early stopping at epoch {epoch}: no validation improvement "
                f"larger than {args.early_stop_min_delta:.3e} for "
                f"{epochs_without_improvement} epochs; best val "
                f"{best_val_loss:.6e}"
            )
            break


if __name__ == "__main__":
    main()
# python train.py --model mgn_t --save-dir checkpoints/mgn_t --amp
