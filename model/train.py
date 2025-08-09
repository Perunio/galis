import torch
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from model.simple_gcn_model import SimpleGCN
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset


BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
dataset = OGBNLinkPredDataset(val_size=0.1, test_size=0.2)
train_data, val_data, test_data = dataset.get_splits()

train_loader = LinkNeighborLoader(
    train_data,
    num_neighbors=[-1, -1],  # Use all neighbors
    neg_sampling_ratio=1.0,  # 1 negative sample per positive edge
    edge_label_index=train_data.edge_label_index,
    edge_label=train_data.edge_label,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

val_loader = LinkNeighborLoader(
    val_data,
    num_neighbors=[-1, -1],
    neg_sampling_ratio=0.0,  # RandomLinkSplit already added negative edges
    edge_label_index=val_data.edge_label_index,
    edge_label=val_data.edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

test_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[-1, -1],
    neg_sampling_ratio=0.0,
    edge_label_index=test_data.edge_label_index,
    edge_label=test_data.edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

# model
model = SimpleGCN(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=128,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()


# training
def train(train_loader, epoch):
    model.train()
    total_loss = 0
    scaler = torch.GradScaler()

    pbar = tqdm(train_loader, desc=f"Training Epoch: {epoch}")
    for batch in pbar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
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
def calc_metrics(loader):
    model.eval()
    all_scores = []
    all_labels = []

    pbar = tqdm(loader, desc="Testing")
    for batch in pbar:
        batch = batch.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            z = model(batch.x, batch.edge_index)
            out = model.decode(z, batch.edge_label_index)

        scores = torch.sigmoid(out).float().cpu().numpy()
        labels = batch.edge_label.cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return roc_auc_score(all_labels, all_scores), accuracy_score(
        all_labels, all_scores > 0.5
    )


if __name__ == "__main__":
    best_val_auc = 0
    best_auc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(train_loader, epoch)
        val_auc, val_acc = calc_metrics(val_loader)


        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val acc: {val_acc:.4f}",
            end=" ",
        )
        if val_auc > best_val_auc:
            print("New best")
            best_val_auc = val_auc
            best_auc = val_auc
            torch.save(model.state_dict(), "model.pth")

    test_auc, test_acc = calc_metrics(test_loader)

    print("-" * 30)
    print(f"Best validation AUC: {best_auc:.4f}")
