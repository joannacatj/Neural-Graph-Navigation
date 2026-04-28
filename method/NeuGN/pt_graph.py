from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class PTGraph:
    edge_index: torch.Tensor
    num_nodes: int
    feat_id: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    edge_feat: Optional[torch.Tensor] = None
    n_id: Optional[torch.Tensor] = None

    def to(self, device):
        return PTGraph(
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            feat_id=self.feat_id.to(device) if self.feat_id is not None else None,
            feat=self.feat.to(device) if self.feat is not None else None,
            edge_feat=self.edge_feat.to(device) if self.edge_feat is not None else None,
            n_id=self.n_id.to(device) if self.n_id is not None else None,
        )


class PTBatch:
    def __init__(self, edge_index, num_nodes, batch, ptr, feat_id=None, feat=None, edge_feat=None, n_id=None, graphs=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch = batch
        self.ptr = ptr
        self.feat_id = feat_id
        self.feat = feat
        self.edge_feat = edge_feat
        self.n_id = n_id
        self._graphs = graphs or []

    @staticmethod
    def from_data_list(graphs: List[PTGraph]):
        edge_indexes = []
        feat_ids = []
        feats = []
        edge_feats = []
        n_ids = []
        batch = []
        ptr = [0]
        node_offset = 0

        for i, g in enumerate(graphs):
            edge_indexes.append(g.edge_index + node_offset)
            if g.feat_id is not None:
                feat_ids.append(g.feat_id)
            if g.feat is not None:
                feats.append(g.feat)
            if g.edge_feat is not None:
                edge_feats.append(g.edge_feat)
            if g.n_id is not None:
                n_ids.append(g.n_id)
            batch.append(torch.full((g.num_nodes,), i, dtype=torch.long))
            node_offset += g.num_nodes
            ptr.append(node_offset)

        edge_index = torch.cat(edge_indexes, dim=1) if edge_indexes else torch.empty((2, 0), dtype=torch.long)
        feat_id = torch.cat(feat_ids, dim=0) if feat_ids else None
        feat = torch.cat(feats, dim=0) if feats else None
        edge_feat = torch.cat(edge_feats, dim=0) if edge_feats else None
        n_id = torch.cat(n_ids, dim=0) if n_ids else None

        return PTBatch(
            edge_index=edge_index,
            num_nodes=node_offset,
            batch=torch.cat(batch, dim=0) if batch else torch.empty((0,), dtype=torch.long),
            ptr=torch.tensor(ptr, dtype=torch.long),
            feat_id=feat_id,
            feat=feat,
            edge_feat=edge_feat,
            n_id=n_id,
            graphs=graphs,
        )

    def to(self, device):
        return PTBatch(
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            feat_id=self.feat_id.to(device) if self.feat_id is not None else None,
            feat=self.feat.to(device) if self.feat is not None else None,
            edge_feat=self.edge_feat.to(device) if self.edge_feat is not None else None,
            n_id=self.n_id.to(device) if self.n_id is not None else None,
            graphs=[g.to(device) for g in self._graphs],
        )

    def to_data_list(self):
        return self._graphs


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
    out = []
    for i in range(num_graphs):
        mask = batch == i
        out.append(x[mask].max(dim=0).values)
    return torch.stack(out, dim=0) if out else torch.empty((0, x.size(-1)), device=x.device)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
    out = []
    for i in range(num_graphs):
        mask = batch == i
        out.append(x[mask].mean(dim=0))
    return torch.stack(out, dim=0) if out else torch.empty((0, x.size(-1)), device=x.device)
