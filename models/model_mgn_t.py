import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        activate_final=False,
        layer_norm=True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        dim = int(input_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, output_dim))
        if activate_final:
            layers.append(nn.SiLU())
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.edge_mlp = MLP(
            input_dim=hidden_dim * 3,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activate_final=False,
            layer_norm=False,
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_attr, h_src, h_dst):
        edge_update = self.edge_mlp(torch.cat([edge_attr, h_src, h_dst], dim=-1))
        return self.edge_norm(edge_attr + edge_update)


class NodeModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.node_mlp = MLP(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activate_final=False,
            layer_norm=False,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, aggregated_messages):
        node_update = self.node_mlp(torch.cat([h, aggregated_messages], dim=-1))
        return self.node_norm(h + node_update)


class MeshGraphNetBlock(nn.Module):
    def __init__(self, hidden_dim=128, chunk_size=200_000):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.chunk_size = int(chunk_size)
        self.edge_model = EdgeModel(hidden_dim=hidden_dim)
        self.node_model = NodeModel(hidden_dim=hidden_dim)

    def forward(self, h, edge_attr, edge_index):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        num_edges = src.numel()
        updated_edges = []
        aggregated = h.new_zeros(h.shape)

        for start in range(0, num_edges, self.chunk_size):
            end = min(start + self.chunk_size, num_edges)
            src_c = src[start:end]
            dst_c = dst[start:end]
            edge_c = self.edge_model(edge_attr[start:end], h[src_c], h[dst_c])
            updated_edges.append(edge_c)
            # Keep scatter accumulation compatible with CUDA autocast. The
            # message tensor can be FP16 while the node buffer remains FP32.
            aggregated.index_add_(0, dst_c, edge_c.to(dtype=aggregated.dtype))

        edge_attr = torch.cat(updated_edges, dim=0)
        h = self.node_model(h, aggregated)
        return h, edge_attr


class MeshGraphNetT(nn.Module):
    """MeshGraphNet-style temporal baseline for the dynamic pintle-nozzle data."""

    def __init__(
        self,
        node_dim=21,
        edge_attr_dim=3,
        hidden_dim=128,
        num_layers=4,
        output_dim=3,
        chunk_size=200_000,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.chunk_size = int(chunk_size)

        self.node_encoder = MLP(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activate_final=True,
            layer_norm=True,
        )
        self.edge_encoder = MLP(
            input_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activate_final=True,
            layer_norm=True,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetBlock(hidden_dim=hidden_dim, chunk_size=chunk_size)
                for _ in range(num_layers)
            ]
        )
        self.decoder = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            activate_final=False,
            layer_norm=False,
        )

    def forward(self, node_x, edge_index, edge_attr, current_field, target_mask=None):
        node_x = node_x.to(torch.float32)
        edge_attr = edge_attr.to(torch.float32)
        current_field = current_field.to(torch.float32)

        h = self.node_encoder(node_x)
        e = self.edge_encoder(edge_attr)
        for block in self.processor:
            h, e = block(h, e, edge_index)

        delta = self.decoder(h)
        pred = current_field + delta
        if target_mask is not None:
            pred = pred * target_mask.to(pred.dtype)
        return pred, delta
