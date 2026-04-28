import random
from typing import List, Sequence

import torch
from NeuGN.pt_graph import PTGraph


class GraphTokenizer:
    """Tokenizer for graph nodes with pure PyTorch random walk/sampling utilities."""

    def __init__(self, graph: PTGraph):
        self.graph = graph
        self.node_num = self.graph.num_nodes

        self.padding_id = self.node_num
        self.sos_id = self.padding_id + 1

        self.node_to_token = {node_id: idx for idx, node_id in enumerate(range(self.node_num))}
        self.node_to_token[self.padding_id] = self.padding_id
        self.token_to_node = {idx: node_id for node_id, idx in self.node_to_token.items()}

        src, dst = self.graph.edge_index
        self._neighbors = [[] for _ in range(self.node_num)]
        for s, d in zip(src.tolist(), dst.tolist()):
            self._neighbors[s].append(d)

    def node_nums(self):
        return self.node_num

    def token_nums(self):
        return self.node_num + 2

    def random_walk(self, start_node: int, length: int) -> List[int]:
        walk = [int(start_node)]
        current = int(start_node)
        for _ in range(length):
            next_candidates = self._neighbors[current]
            if not next_candidates:
                walk.append(-1)
            else:
                current = random.choice(next_candidates)
                walk.append(current)
        return walk

    def neighborhoods_sampling(self, start_nodes, fanouts_max):
        seed_nodes_list = []
        for node in start_nodes:
            node = int(node)
            if len(self._neighbors[node]) == 0:
                continue

            out_len = random.randint(1, len(fanouts_max))
            sampled_nodes = {node}
            frontier = {node}
            for i in range(out_len):
                fanout = random.randint(1, fanouts_max[i])
                next_frontier = set()
                for fnode in frontier:
                    nbrs = self._neighbors[fnode]
                    if not nbrs:
                        continue
                    pick_num = min(fanout, len(nbrs))
                    sampled = random.sample(nbrs, pick_num)
                    next_frontier.update(sampled)
                sampled_nodes.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
            seed_nodes_list.append(torch.tensor(list(sampled_nodes), dtype=torch.long))
        return seed_nodes_list

    def random_walks(self, start_nodes: Sequence[int], length: int) -> torch.Tensor:
        walks = [self.random_walk(int(node), int(length)) for node in start_nodes]
        return torch.tensor(walks, dtype=torch.long)

    def encode_walks(self, walks: List[List[int]]) -> List[List[int]]:
        return [[self.node_to_token[node] for node in walk] for walk in walks]

    def encode_walk(self, walk: List[int]) -> List[int]:
        return [self.node_to_token[node] for node in walk]

    def decode_walks(self, token_sequences: List[List[int]]) -> List[List[int]]:
        return [[self.token_to_node[token] for token in sequence] for sequence in token_sequences]
