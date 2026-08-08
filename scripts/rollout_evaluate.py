import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import (
    build_node_features as assemble_node_features,
    compute_boundary_dynamic_edge_attr,
    compute_condition_statistics,
    compute_dynamic_edge_attr,
    normalized_condition_values,
)
from model_batgno_lite import BATGNOLite
from model_mgn_t import MeshGraphNetT


VAR_NAMES = ("Temperature", "Velocity", "Pressure")
CSV_COLUMNS = (
    "rollout_step",
    "time_index",
    "Temperature_MAE",
    "Temperature_RMSE",
    "Temperature_RelL2",
    "Velocity_MAE",
    "Velocity_RMSE",
    "Velocity_RelL2",
    "Pressure_MAE",
    "Pressure_RMSE",
    "Pressure_RelL2",
    "Overall_MAE",
    "Overall_RMSE",
    "Overall_RelL2",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Autoregressive rollout evaluation for trained BA-TGNO-Lite / "
            "DM-BA-TGNO checkpoints."
        )
    )
    parser.add_argument("--dataset", default="pintle_nozzle_dynamic_dataset.npz")
    parser.add_argument("--graph", default="pintle_nozzle_graph_k8.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("batgno_lite", "mgn_t"), default="batgno_lite")
    parser.add_argument("--output", default="rollout_results.npz")
    parser.add_argument("--start-time", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dynamic-edge-attr",
        action="store_true",
        help="Use dynamic [E,9] edge attributes if the checkpoint has no saved config.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save every k-th rollout prediction. Metrics are still computed every step.",
    )
    parser.add_argument(
        "--mask-solid",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, pass target masks to the model and keep solid-region rollout values zero.",
    )
    parser.add_argument(
        "--metric-space",
        choices=("physical", "norm"),
        default="physical",
        help="Print and save step metrics in physical or normalized space.",
    )
    return parser.parse_args()


def read_scalar(npz, key, default=None):
    if key not in npz:
        if default is None:
            raise KeyError(f"Missing required key: {key}")
        return int(default)
    value = npz[key]
    if np.ndim(value) == 0:
        return int(value.item())
    return int(np.asarray(value).reshape(-1)[0])


def load_dataset_arrays(dataset_path):
    with np.load(dataset_path, allow_pickle=False) as data:
        required = ("fields_norm", "mask", "mean", "std")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Dataset is missing required arrays: {missing}")

        coords = (
            np.asarray(data["coords"], dtype=np.float32) if "coords" in data else None
        )
        ref_coords = (
            np.asarray(data["ref_coords"], dtype=np.float32)
            if "ref_coords" in data
            else None
        )
        if coords is None and ref_coords is None:
            raise KeyError("Dataset must contain either 'coords' or 'ref_coords'")
        fields_norm = np.asarray(data["fields_norm"], dtype=np.float32)
        mask = np.asarray(data["mask"])
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)

        num_frames = fields_norm.shape[0]
        if "test_frame_indices" in data:
            test_frame_indices = np.asarray(data["test_frame_indices"], dtype=np.int64)
        else:
            default_test_start = int(0.85 * num_frames)
            test_start = read_scalar(data, "test_start", default_test_start)
            test_end = read_scalar(data, "test_end", num_frames)
            test_frame_indices = np.arange(test_start, test_end, dtype=np.int64)

        rollout_history_indices = (
            np.asarray(data["test_rollout_history_indices"], dtype=np.int64)
            if "test_rollout_history_indices" in data
            else None
        )
        rollout_target_indices = (
            np.asarray(data["test_rollout_target_indices"], dtype=np.int64)
            if "test_rollout_target_indices" in data
            else None
        )

        has_condition_metadata = all(
            key in data
            for key in (
                "pintle_position_x_m",
                "speed_mps",
                "flow_time_s",
                "train_window_history_indices",
                "train_window_target_indices",
            )
        )
        if has_condition_metadata:
            pintle_position_x_m = np.asarray(
                data["pintle_position_x_m"], dtype=np.float32
            )
            speed_mps = np.asarray(data["speed_mps"], dtype=np.float32)
            flow_time_s = np.asarray(data["flow_time_s"], dtype=np.float64)
            train_history = np.asarray(
                data["train_window_history_indices"], dtype=np.int64
            )
            train_targets = np.asarray(
                data["train_window_target_indices"], dtype=np.int64
            )
            condition_mean, condition_std = compute_condition_statistics(
                pintle_position_x_m,
                speed_mps,
                flow_time_s,
                train_history,
                train_targets,
            )
        else:
            pintle_position_x_m = None
            speed_mps = None
            flow_time_s = None
            condition_mean = None
            condition_std = None

    validate_dataset_arrays(coords, ref_coords, fields_norm, mask, mean, std)
    test_frame_indices = np.unique(test_frame_indices)
    if test_frame_indices.size == 0:
        raise ValueError("Test split is empty")
    if test_frame_indices.min() < 0 or test_frame_indices.max() >= fields_norm.shape[0]:
        raise ValueError("test_frame_indices fall outside the dataset")

    return {
        "coords": coords,
        "ref_coords": ref_coords,
        "fields_norm": fields_norm,
        "mask": mask,
        "mean": mean,
        "std": std,
        "test_frame_indices": test_frame_indices,
        "rollout_history_indices": rollout_history_indices,
        "rollout_target_indices": rollout_target_indices,
        "has_condition_metadata": has_condition_metadata,
        "pintle_position_x_m": pintle_position_x_m,
        "speed_mps": speed_mps,
        "flow_time_s": flow_time_s,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
    }


def validate_dataset_arrays(coords, ref_coords, fields_norm, mask, mean, std):
    if fields_norm.ndim != 3 or fields_norm.shape[-1] != 3:
        raise ValueError(f"fields_norm must have shape [T,N,3], got {fields_norm.shape}")
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError(f"mask must have shape [T,N,1], got {mask.shape}")
    if mask.shape[:2] != fields_norm.shape[:2]:
        raise ValueError("mask shape is inconsistent with fields_norm")
    if coords is not None:
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must have shape [T,N,2], got {coords.shape}")
        if fields_norm.shape[:2] != coords.shape[:2]:
            raise ValueError("fields_norm shape is inconsistent with coords")
    else:
        if ref_coords.ndim != 2 or ref_coords.shape[-1] != 2:
            raise ValueError(f"ref_coords must have shape [N,2], got {ref_coords.shape}")
        if ref_coords.shape[0] != fields_norm.shape[1]:
            raise ValueError("ref_coords shape is inconsistent with fields_norm")
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(f"mean/std must have shape [3], got mean={mean.shape}, std={std.shape}")
    for name, array in (
        ("fields_norm", fields_norm),
        ("mask", mask),
        ("mean", mean),
        ("std", std),
        ("coordinates", coords if coords is not None else ref_coords),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if np.any(std == 0):
        raise ValueError("std contains zero values, cannot denormalize safely")


def load_graph_arrays(graph_path):
    with np.load(graph_path, allow_pickle=False) as graph:
        missing = [key for key in ("edge_index", "edge_attr") if key not in graph]
        if missing:
            raise KeyError(f"Graph file is missing required arrays: {missing}")
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(graph["edge_attr"], dtype=np.float32)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {edge_index.shape}")
    if edge_attr.ndim != 2 or edge_attr.shape[1] != 3:
        raise ValueError(f"edge_attr must have shape [E,3], got {edge_attr.shape}")
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError("edge_index and edge_attr contain different edge counts")
    if not np.isfinite(edge_index).all() or not np.isfinite(edge_attr).all():
        raise ValueError("graph arrays contain NaN or Inf")

    return edge_index, edge_attr


def extract_state_dict(checkpoint):
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError(
        "Could not find model weights in checkpoint. Expected 'model_state', "
        "'model_state_dict', 'state_dict', or a raw state_dict."
    )


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def flag_was_passed(flag):
    return flag in sys.argv


def resolve_checkpoint_args(checkpoint):
    return checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = resolve_checkpoint_args(checkpoint)

    model_name = str(ckpt_args.get("model", args.model))
    node_dim = int(ckpt_args.get("node_dim", 21))
    if model_name == "mgn_t":
        dynamic_edge_attr = False
        edge_attr_dim = int(ckpt_args.get("edge_attr_dim", 3))
    else:
        dynamic_edge_attr = bool(ckpt_args.get("dynamic_edge_attr", args.dynamic_edge_attr))
        edge_attr_dim = int(ckpt_args.get("edge_attr_dim", 9 if dynamic_edge_attr else 3))
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    num_layers = int(ckpt_args.get("num_layers", args.num_layers))
    chunk_size = int(ckpt_args.get("chunk_size", args.chunk_size))

    if flag_was_passed("--chunk-size") and "chunk_size" not in ckpt_args:
        chunk_size = int(args.chunk_size)

    if model_name == "mgn_t":
        model = MeshGraphNetT(
            node_dim=node_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3,
            chunk_size=chunk_size,
        ).to(device)
    elif model_name == "batgno_lite":
        model = BATGNOLite(
            input_dim=node_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3,
            chunk_size=chunk_size,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.load_state_dict(strip_module_prefix(extract_state_dict(checkpoint)))
    model.eval()

    meta = {
        "model": model_name,
        "node_dim": node_dim,
        "dynamic_edge_attr": dynamic_edge_attr,
        "edge_attr_dim": edge_attr_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "chunk_size": chunk_size,
        "epoch": checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
        "grad_loss_weight": ckpt_args.get("grad_loss_weight", None),
    }
    return model, meta


def validate_rollout_window(args, data):
    if args.history_steps != 3:
        raise ValueError(
            "This model interface supports history_steps=3. "
            f"Got history_steps={args.history_steps}."
        )
    if args.rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be positive, got {args.rollout_steps}")
    if args.save_every <= 0:
        raise ValueError(f"save_every must be positive, got {args.save_every}")

    if (
        args.start_time is None
        and data["rollout_history_indices"] is not None
        and data["rollout_target_indices"] is not None
    ):
        history_indices = data["rollout_history_indices"].astype(np.int64, copy=True)
        available_targets = data["rollout_target_indices"]
        if args.rollout_steps > available_targets.size:
            raise ValueError(
                f"Requested {args.rollout_steps} rollout steps, but the dataset "
                f"defines only {available_targets.size}"
            )
        target_indices = available_targets[: args.rollout_steps].astype(
            np.int64, copy=True
        )
    else:
        start_time = (
            int(args.start_time)
            if args.start_time is not None
            else int(data["test_frame_indices"].min()) + args.history_steps - 1
        )
        history_indices = np.arange(
            start_time - args.history_steps + 1,
            start_time + 1,
            dtype=np.int64,
        )
        target_indices = np.arange(
            start_time + 1,
            start_time + args.rollout_steps + 1,
            dtype=np.int64,
        )

    if history_indices.shape != (args.history_steps,):
        raise ValueError(
            f"Rollout history must have shape [{args.history_steps}], "
            f"got {history_indices.shape}"
        )
    if target_indices.shape != (args.rollout_steps,):
        raise ValueError("Rollout target count does not match --rollout-steps")
    combined = np.concatenate([history_indices, target_indices])
    if np.any(combined[1:] != combined[:-1] + 1):
        raise ValueError("Rollout history and targets must form one contiguous sequence")
    test_frames = set(data["test_frame_indices"].tolist())
    outside = [int(index) for index in combined if int(index) not in test_frames]
    if outside:
        raise ValueError(
            "Rollout window must be fully contained in the held-out test frames. "
            f"Outside indices: {outside[:10]}"
        )
    return history_indices, target_indices


def coords_at(data, time_index):
    if data["coords"] is not None:
        return data["coords"][time_index]
    return data["ref_coords"]


def build_rollout_node_features(data, fields_history, history_times, target_time):
    condition_values = None
    if data["has_condition_metadata"]:
        condition_values = normalized_condition_values(
            data["pintle_position_x_m"],
            data["speed_mps"],
            data["flow_time_s"],
            int(history_times[-1]),
            int(target_time),
            data["condition_mean"],
            data["condition_std"],
        )
    return assemble_node_features(
        [coords_at(data, int(index)) for index in history_times],
        coords_at(data, int(target_time)),
        fields_history,
        [
            data["mask"][int(index)].astype(np.float32, copy=False)
            for index in history_times
        ],
        data["mask"][int(target_time)].astype(np.float32, copy=False),
        condition_values=condition_values,
    )


def denormalize(array, mean, std):
    return array * std.reshape(1, 3) + mean.reshape(1, 3)


def compute_step_metrics(pred_norm, target_norm, mask, mean, std, metric_space):
    if metric_space == "physical":
        pred = denormalize(pred_norm, mean, std)
        target = denormalize(target_norm, mean, std)
    elif metric_space == "norm":
        pred = pred_norm
        target = target_norm
    else:
        raise ValueError(f"Unknown metric_space={metric_space}")

    fluid = mask.reshape(-1) > 0.5
    if not np.any(fluid):
        raise ValueError("Target mask has no fluid nodes (mask > 0.5)")

    diff = pred[fluid] - target[fluid]
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2
    mae = abs_diff.mean(axis=0)
    rmse = np.sqrt(sq_diff.mean(axis=0))
    rel_l2 = np.sqrt(sq_diff.sum(axis=0)) / (np.sqrt((target[fluid] ** 2).sum(axis=0)) + 1.0e-8)

    overall_mae = float(abs_diff.mean())
    overall_rmse = float(np.sqrt(sq_diff.mean()))
    overall_rel_l2 = float(
        np.sqrt(sq_diff.sum()) / (np.sqrt((target[fluid] ** 2).sum()) + 1.0e-8)
    )

    return {
        "mae": mae.astype(np.float64),
        "rmse": rmse.astype(np.float64),
        "rel_l2": rel_l2.astype(np.float64),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_rel_l2": overall_rel_l2,
    }


def metrics_to_csv_row(step, time_index, metrics):
    row = {"rollout_step": step, "time_index": time_index}
    for idx, name in enumerate(VAR_NAMES):
        row[f"{name}_MAE"] = float(metrics["mae"][idx])
        row[f"{name}_RMSE"] = float(metrics["rmse"][idx])
        row[f"{name}_RelL2"] = float(metrics["rel_l2"][idx])
    row["Overall_MAE"] = float(metrics["overall_mae"])
    row["Overall_RMSE"] = float(metrics["overall_rmse"])
    row["Overall_RelL2"] = float(metrics["overall_rel_l2"])
    return row


def print_step_metrics(step, time_index, metrics, metric_space):
    prefix = f"step {step:04d} | time {time_index:04d} | {metric_space}"
    pieces = []
    for idx, name in enumerate(VAR_NAMES):
        pieces.append(
            f"{name}: MAE {metrics['mae'][idx]:.6e}, "
            f"RMSE {metrics['rmse'][idx]:.6e}, RelL2 {metrics['rel_l2'][idx]:.6e}"
        )
    pieces.append(
        f"Overall: MAE {metrics['overall_mae']:.6e}, "
        f"RMSE {metrics['overall_rmse']:.6e}, RelL2 {metrics['overall_rel_l2']:.6e}"
    )
    print(prefix + " | " + " | ".join(pieces))


def derive_metrics_path(output_path):
    output_path = Path(output_path)
    stem = output_path.stem
    if stem.startswith("rollout_results"):
        metrics_stem = "rollout_metrics" + stem[len("rollout_results") :]
    else:
        metrics_stem = stem + "_metrics"
    return output_path.with_name(metrics_stem + ".csv")


@torch.no_grad()
def run_rollout(args):
    device = torch.device(args.device)
    data = load_dataset_arrays(args.dataset)
    edge_index_np, edge_attr_np = load_graph_arrays(args.graph)
    num_frames, num_nodes = data["fields_norm"].shape[:2]
    if edge_index_np.max(initial=0) >= num_nodes or edge_index_np.min(initial=0) < 0:
        raise ValueError("edge_index contains node ids outside the dataset node range")

    history_times, target_times = validate_rollout_window(args, data)
    model, model_meta = load_model(args, device)

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    static_edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device)

    fields_history = [
        data["fields_norm"][int(time_idx)].astype(np.float32, copy=True)
        for time_idx in history_times
    ]
    if args.mask_solid:
        for idx, time_idx in enumerate(history_times):
            fields_history[idx] *= data["mask"][int(time_idx)]

    first_node_x = build_rollout_node_features(
        data, fields_history, history_times, int(target_times[0])
    )
    if first_node_x.shape[1] != model_meta["node_dim"]:
        raise ValueError(
            f"Checkpoint expects node_dim={model_meta['node_dim']}, but this "
            f"dataset produces node_dim={first_node_x.shape[1]}. "
            "Use a checkpoint trained with the same dataset schema."
        )

    print(f"dataset: {args.dataset}")
    print(f"graph: {args.graph}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {model_meta['epoch']}")
    print(
        "model config: "
        f"model={model_meta['model']}, "
        f"node_dim={model_meta['node_dim']}, "
        f"dynamic_edge_attr={model_meta['dynamic_edge_attr']}, "
        f"edge_attr_dim={model_meta['edge_attr_dim']}, "
        f"hidden_dim={model_meta['hidden_dim']}, "
        f"num_layers={model_meta['num_layers']}, "
        f"chunk_size={model_meta['chunk_size']}, "
        f"grad_loss_weight={model_meta['grad_loss_weight']}"
    )
    print(
        f"rollout window: history={history_times.tolist()}, "
        f"first_target={int(target_times[0])}, "
        f"last_target={int(target_times[-1])}, rollout_steps={args.rollout_steps}"
    )
    print(
        f"num_frames={num_frames}, num_nodes={num_nodes}, "
        f"num_edges={edge_index_np.shape[1]}, save_every={args.save_every}, "
        f"mask_solid={args.mask_solid}, metric_space={args.metric_space}"
    )
    print(f"condition features used: {data['has_condition_metadata']}")
    print(f"coordinate representation: {'dynamic' if data['coords'] is not None else 'fixed reference'}")

    saved_pred = []
    saved_target = []
    saved_coords = []
    saved_mask = []
    saved_time_indices = []
    saved_rollout_steps = []
    metric_rows = []

    history_times = history_times.tolist()
    for step, target_time_value in enumerate(target_times, start=1):
        current_time = int(history_times[-1])
        target_time = int(target_time_value)

        node_x_np = build_rollout_node_features(
            data, fields_history, history_times, target_time
        )
        node_x = torch.as_tensor(node_x_np, dtype=torch.float32, device=device)
        current_field = torch.as_tensor(fields_history[-1], dtype=torch.float32, device=device)
        target_mask = torch.as_tensor(data["mask"][target_time], dtype=torch.float32, device=device)

        if model_meta["dynamic_edge_attr"]:
            if data["coords"] is None:
                edge_attr_np = compute_boundary_dynamic_edge_attr(
                    data["ref_coords"],
                    data["mask"][current_time],
                    data["mask"][target_time],
                    edge_index_np,
                )
            else:
                edge_attr_np = compute_dynamic_edge_attr(
                    data["coords"][current_time],
                    data["coords"][target_time],
                    edge_index_np,
                )
            edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device)
        else:
            edge_attr = static_edge_attr

        pred_norm, _ = model(
            node_x,
            edge_index,
            edge_attr,
            current_field,
            target_mask if args.mask_solid else None,
        )
        if not torch.isfinite(pred_norm).all():
            raise FloatingPointError(f"Prediction contains NaN or Inf at rollout step {step}")

        pred_norm_np = pred_norm.detach().cpu().numpy().astype(np.float32)
        if args.mask_solid:
            pred_norm_np *= data["mask"][target_time]

        target_norm_np = data["fields_norm"][target_time].astype(np.float32, copy=False)
        metrics = compute_step_metrics(
            pred_norm_np,
            target_norm_np,
            data["mask"][target_time],
            data["mean"],
            data["std"],
            args.metric_space,
        )
        metric_rows.append(metrics_to_csv_row(step, target_time, metrics))
        print_step_metrics(step, target_time, metrics, args.metric_space)

        if step % args.save_every == 0:
            saved_pred.append(pred_norm_np.astype(np.float32, copy=False))
            saved_target.append(target_norm_np.astype(np.float32, copy=False))
            saved_coords.append(coords_at(data, target_time).astype(np.float32, copy=False))
            saved_mask.append(data["mask"][target_time].astype(np.float32, copy=False))
            saved_time_indices.append(target_time)
            saved_rollout_steps.append(step)

        fields_history = fields_history[1:] + [pred_norm_np]
        history_times = history_times[1:] + [target_time]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = derive_metrics_path(output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        pred_norm=np.stack(saved_pred, axis=0).astype(np.float32)
        if saved_pred
        else np.empty((0, num_nodes, 3), dtype=np.float32),
        target_norm=np.stack(saved_target, axis=0).astype(np.float32)
        if saved_target
        else np.empty((0, num_nodes, 3), dtype=np.float32),
        coords=np.stack(saved_coords, axis=0).astype(np.float32)
        if saved_coords
        else np.empty((0, num_nodes, 2), dtype=np.float32),
        mask=np.stack(saved_mask, axis=0).astype(np.float32)
        if saved_mask
        else np.empty((0, num_nodes, 1), dtype=np.float32),
        time_indices=np.asarray(saved_time_indices, dtype=np.int64),
        rollout_steps=np.asarray(saved_rollout_steps, dtype=np.int64),
        mean=data["mean"].astype(np.float32),
        std=data["std"].astype(np.float32),
    )

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    print(f"saved rollout results: {output_path}")
    print(f"saved rollout metrics: {metrics_path}")


def main():
    args = parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
