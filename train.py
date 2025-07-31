import torch
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import pandas as pd
import torch_geometric
import os
import subprocess
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader


batch_size = 128


# download dataset
# import torch.serialization
#
# with torch.serialization.safe_globals(
#     [
#         torch_geometric.data.data.DataTensorAttr,
#         torch_geometric.data.data.DataEdgeAttr,
#         torch_geometric.data.storage.GlobalStorage,
#     ]
# ):
dataset = PygNodePropPredDataset(name="ogbn-arxiv")
data = dataset[0]
# print(dataset.root)


# download abstracts
target_dir = os.path.join(dataset.root, "mapping")
tsv_path = os.path.join(target_dir, "titleabs.tsv")
gz_path = tsv_path + ".gz"

if not os.path.exists(tsv_path):
    url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run(["wget", "-P", target_dir, url], check=True)
    subprocess.run(["gunzip", gz_path], check=True)
    print(f"File downloaded and extracted to: {target_dir}")
else:
    print("File already downloaded")

# read abstracts to df
try:
    df_text = pd.read_csv(
        os.path.join(dataset.root, "mapping", "titleabs.tsv"),
        sep="\t",
        header=None,
        names=["paper_id", "title", "abstract"],
        # usecols=['title', 'abstract'], # Keep 'paper_id' to verify alignment if needed
        lineterminator="\n",
        low_memory=False,
    )
    print("Abstracts loaded successfully")
except FileNotFoundError:
    print("Error: titleabs.tsv not found.")
    df_text = None

# concat abs with title
if df_text is not None:
    df_text_aligned = df_text.reset_index(drop=True)
    corpus = (
        df_text_aligned["title"].fillna("")
        + "\n "
        + df_text_aligned["abstract"].fillna("")
    ).tolist()
    print(f"Corpus created with {len(corpus)} documents.")
else:
    corpus = []
    print("Could not create corpus because df_text was not loaded.")


# data train/test splits
# Calculate the number of edges for validation and testing based on ratios
num_edges = data.edge_index.size(1)
num_val_edges = int(num_edges * 0.1)
num_test_edges = int(num_edges * 0.2)

# Apply RandomLinkSplit using num_val and num_test
transform = RandomLinkSplit(
    num_val=num_val_edges,
    num_test=num_test_edges,
    is_undirected=False,
    add_negative_train_samples=False,  # LinkNeighborLoader handles negative sampling for training
)
train_data, val_data, test_data = transform(data)


train_loader = LinkNeighborLoader(
    train_data,
    num_neighbors=[-1, -1],  # Use all neighbors
    neg_sampling_ratio=1.0,  # 1 negative sample per positive edge
    edge_label_index=train_data.edge_label_index,
    edge_label=train_data.edge_label,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

val_loader = LinkNeighborLoader(
    val_data,
    num_neighbors=[-1, -1],
    neg_sampling_ratio=0.0,  # RandomLinkSplit already added negative edges
    edge_label_index=val_data.edge_label_index,
    edge_label=val_data.edge_label,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

test_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[-1, -1],
    neg_sampling_ratio=0.0,
    edge_label_index=test_data.edge_label_index,
    edge_label=test_data.edge_label,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels * 2, 1)

    def forward(self, z, edge_index):
        row, col = edge_index
        # Concatenate the embeddings of the two nodes
        z_cat = torch.cat([z[row], z[col]], dim=-1)
        # Apply the linear layer and sigmoid
        return self.linear(z_cat).squeeze(-1)


class SimpleGCN(torch.nn.Module):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleGCN(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=128,  # Node embedding dimension
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


# train/test loops


from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def train(train_loader):
    model.train()
    total_loss = 0
    scaler = torch.GradScaler()

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):  # mixed precission training
            z = model(batch.x, batch.edge_index)
            out = model.decode(z, batch.edge_label_index)
            labels = batch.edge_label.float()

            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader):
    model.eval()
    all_scores = []
    all_labels = []

    pbar = tqdm(loader, desc="Testing")
    for batch in pbar:
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16
        ):  # faster evaluation
            z = model(batch.x, batch.edge_index)
            # Decode edge scores for the edges in the batch
            out = model.decode(z, batch.edge_label_index)

        scores = torch.sigmoid(out).cpu().numpy()
        labels = batch.edge_label.cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return roc_auc_score(all_labels, all_scores), accuracy_score(
        all_labels, all_scores > 0.5
    )


num_epochs = 20
best_val_auc = 0
final_test_auc_at_best_val = 0

for epoch in range(1, num_epochs + 1):
    loss = train(train_loader)

    val_auc, val_acc = test(val_loader)
    test_auc, test_acc = test(test_loader)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc_at_best_val = test_auc
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val acc: {val_acc:.4f} Test AUC: {test_auc:.4f}, Test acc: {test_acc:.4f} (New best!)"
        )
    else:
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val acc: {val_acc:.4f} Test AUC: {test_auc:.4f}, Test acc: {test_acc:.4f}"
        )

print("-" * 30)
print(f"Final Test AUC at best validation AUC: {final_test_auc_at_best_val:.4f}")
