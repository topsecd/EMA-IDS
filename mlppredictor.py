import torch.nn as nn
import torch


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes, edge_feats):
        super().__init__()
        self.W = nn.Linear(in_features * 2 + edge_feats, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        attr = edges.data['attr']
        score = self.W(torch.cat([h_u, h_v, attr], 1))
        return {'score': score}

    def forward(self, graph, h, edge_attr):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.edata['attr'] = edge_attr
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

    def reset_parameters(self):
        self.W.reset_parameters()

