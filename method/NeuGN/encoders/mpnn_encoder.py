from torch import nn
import torch
import torch.nn.functional as F

from NeuGN.pt_graph import global_max_pool


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index

        loop_idx = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, loop_idx], dim=0)
        dst = torch.cat([dst, loop_idx], dim=0)

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

        msg = x[src] * (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        return self.linear(out)


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        out = out + x
        return self.mlp(out)


class GNN(nn.Module):
    def __init__(self, params):
        super(GNN, self).__init__()
        self.in_dim = params.encoder_config.graph_feature_dim
        self.out_dim = params.decoder_config.dim
        self.num_layers = params.encoder_config.encoder_layers
        encoder_name = params.encoder_config.encoder_name.lower()
        self.convs = nn.ModuleList()

        if encoder_name == 'gcn':
            self.layer_type = GCNLayer
        elif encoder_name == 'gin':
            self.layer_type = GINLayer
        else:
            raise ValueError(f"Unsupported encoder name: {params.encoder_config.encoder_name}")

        self.convs.append(self.layer_type(self.in_dim, self.out_dim))
        self.value_embedding = nn.Embedding(params.encoder_config.graph_value_num, self.in_dim)

        for _ in range(1, self.num_layers):
            self.convs.append(self.layer_type(self.out_dim, self.out_dim))

    def forward(self, batched_graph):
        h = self.value_embedding(batched_graph.feat_id)
        edge_index = batched_graph.edge_index
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        features = global_max_pool(h, batched_graph.batch)
        return features
