import torch
import torch.nn as nn


FIELD_NAMES = ("temperature", "velocity", "pressure")


def make_mlp(input_dim, hidden_dim, output_dim, num_layers=2, activation=nn.SiLU):
    layers = []
    dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(activation())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim=128, edge_attr_dim=3, chunk_size=200_000):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.edge_encoder = make_mlp(edge_attr_dim, hidden_dim, hidden_dim, num_layers=2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        num_edges = src.numel()
        agg = h.new_zeros(h.shape)

        for start in range(0, num_edges, self.chunk_size):
            end = min(start + self.chunk_size, num_edges)
            src_c = src[start:end]
            dst_c = dst[start:end]
            edge_emb = self.edge_encoder(edge_attr[start:end])
            msg_in = torch.cat([h[src_c], h[dst_c], edge_emb], dim=-1)
            msg = self.edge_mlp(msg_in)
            # Autocast may produce FP16 messages while LayerNorm keeps h/agg
            # in FP32. index_add_ requires identical dtypes, so accumulate in
            # the buffer dtype (FP32 under AMP is also numerically preferable).
            agg.index_add_(0, dst_c, msg.to(dtype=agg.dtype))

        update = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.node_norm(h + update)
        return h


class BATGNOLite(nn.Module):
    def __init__(
        self,
        input_dim=21,
        edge_attr_dim=3,
        hidden_dim=128,
        num_layers=4,
        output_dim=3,
        chunk_size=200_000,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                GraphBlock(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    chunk_size=chunk_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_x, edge_index, edge_attr, current_field, target_mask=None):
        node_x = node_x.to(torch.float32)
        edge_attr = edge_attr.to(torch.float32)
        current_field = current_field.to(torch.float32)

        h = self.encoder(node_x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)

        delta = self.decoder(h)
        pred = current_field + delta
        if target_mask is not None:
            pred = pred * target_mask.to(pred.dtype)
        return pred, delta


def masked_mse_loss(pred, target, mask, eps=1.0e-8):
    mask = mask.to(pred.dtype)
    diff2 = (pred - target) ** 2
    loss = (diff2 * mask).sum() / (mask.sum() * pred.shape[-1] + eps)
    return loss


def masked_edge_gradient_loss(
    pred,
    target,
    mask,
    edge_index,
    variable_weights=None,
    eps=1.0e-8,
):
    """Match graph-edge field differences on fluid-fluid edges only."""
    src = edge_index[0].long()
    dst = edge_index[1].long()

    pred_diff = pred[dst] - pred[src]
    target_diff = target[dst] - target[src]
    edge_mask = (mask[src] * mask[dst]).to(pred_diff.dtype)

    diff_sq = (pred_diff - target_diff) ** 2
    if variable_weights is not None:
        weights = torch.as_tensor(
            variable_weights,
            dtype=diff_sq.dtype,
            device=diff_sq.device,
        ).view(1, -1)
        if weights.shape[1] != diff_sq.shape[1]:
            raise ValueError(
                f"variable_weights must have {diff_sq.shape[1]} values, got {weights.shape[1]}"
            )
        diff_sq = diff_sq * weights

    loss = (edge_mask * diff_sq).sum() / (edge_mask.sum() * pred.shape[-1] + eps)
    return loss


@torch.no_grad()
def masked_metrics(pred, target, mask, eps=1.0e-8):
    """Return masked MAE, RMSE and relative L2 per variable plus overall."""
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)

    diff = (pred - target) * mask
    abs_diff = diff.abs()
    sq_diff = diff.pow(2)
    denom = mask.sum().clamp_min(eps)

    mae_vars = abs_diff.sum(dim=0) / denom
    rmse_vars = torch.sqrt(sq_diff.sum(dim=0) / denom)
    rel_l2_vars = torch.sqrt(sq_diff.sum(dim=0)) / (
        torch.sqrt(((target * mask) ** 2).sum(dim=0)) + eps
    )

    overall_mae = abs_diff.sum() / (denom * pred.shape[-1])
    overall_rmse = torch.sqrt(sq_diff.sum() / (denom * pred.shape[-1]))
    overall_rel_l2 = torch.sqrt(sq_diff.sum()) / (torch.sqrt(((target * mask) ** 2).sum()) + eps)

    return {
        "mae": {
            FIELD_NAMES[i]: float(mae_vars[i].detach().cpu()) for i in range(len(FIELD_NAMES))
        }
        | {"overall": float(overall_mae.detach().cpu())},
        "rmse": {
            FIELD_NAMES[i]: float(rmse_vars[i].detach().cpu()) for i in range(len(FIELD_NAMES))
        }
        | {"overall": float(overall_rmse.detach().cpu())},
        "rel_l2": {
            FIELD_NAMES[i]: float(rel_l2_vars[i].detach().cpu()) for i in range(len(FIELD_NAMES))
        }
        | {"overall": float(overall_rel_l2.detach().cpu())},
    }


class MetricAccumulator:
    def __init__(self, eps=1.0e-8):
        self.eps = eps
        self.abs_sum = torch.zeros(3, dtype=torch.float64)
        self.sq_sum = torch.zeros(3, dtype=torch.float64)
        self.target_sq_sum = torch.zeros(3, dtype=torch.float64)
        self.mask_count = 0.0

    @torch.no_grad()
    def update(self, pred, target, mask):
        pred = pred.detach().to(torch.float64).cpu()
        target = target.detach().to(torch.float64).cpu()
        mask = mask.detach().to(torch.float64).cpu()
        diff = (pred - target) * mask
        self.abs_sum += diff.abs().sum(dim=0)
        self.sq_sum += diff.pow(2).sum(dim=0)
        self.target_sq_sum += ((target * mask) ** 2).sum(dim=0)
        self.mask_count += float(mask.sum().item())

    def compute(self):
        denom = max(self.mask_count, self.eps)
        mae = self.abs_sum / denom
        rmse = torch.sqrt(self.sq_sum / denom)
        rel_l2 = torch.sqrt(self.sq_sum) / (torch.sqrt(self.target_sq_sum) + self.eps)

        total_count = denom * 3.0
        overall_mae = self.abs_sum.sum() / total_count
        overall_rmse = torch.sqrt(self.sq_sum.sum() / total_count)
        overall_rel_l2 = torch.sqrt(self.sq_sum.sum()) / (
            torch.sqrt(self.target_sq_sum.sum()) + self.eps
        )

        return {
            "mae": {FIELD_NAMES[i]: float(mae[i]) for i in range(3)}
            | {"overall": float(overall_mae)},
            "rmse": {FIELD_NAMES[i]: float(rmse[i]) for i in range(3)}
            | {"overall": float(overall_rmse)},
            "rel_l2": {FIELD_NAMES[i]: float(rel_l2[i]) for i in range(3)}
            | {"overall": float(overall_rel_l2)},
        }
