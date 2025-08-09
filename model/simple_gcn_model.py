import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EdgeDecoder(torch.nn.Module):
    """Predict citation existence of two node embeddings."""

    def __init__(self, in_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels * 2, 1)

    def forward(self, z, edge_index):
        row, col = edge_index
        # Concatenate the embeddings of the two nodes
        z_cat = torch.cat([z[row], z[col]], dim=-1)
        return self.linear(z_cat).squeeze(-1)


class SimpleGCN(torch.nn.Module):
    """Include encoder and decoder part. Encoder creates embedding for given node and decoder predict link existence between node embeddings."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z, edge_label_index):
        # We pass the edge_label_index to the decoder, which contains both pos and neg edges
        return self.decoder(z, edge_label_index)
