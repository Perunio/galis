import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset, OGBNLinkPredNegDataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-neg", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bert-embed", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


args = parse_args()
USE_CUSTOM_NEG = args.custom_neg
USE_BERT_EMBED = args.bert_embed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
NUM_EPOCHS = 10

# --- Load dataset + frozen embeddings ---
if USE_CUSTOM_NEG:
    print("using hard negatives")
    dataset = OGBNLinkPredNegDataset(val_size=0.1, test_size=0.2)
else:
    print("using random negatives")
    dataset = OGBNLinkPredDataset(val_size=0.1, test_size=0.2)
if USE_BERT_EMBED:
    print("using BERT embeds")
    if Path("model/embeddings.pth").exists():
        emb = torch.load("model/embeddings.pth", map_location=DEVICE)
    else:
        st = SentenceTransformer("bongsoo/kpf-sbert-128d-v1", device=DEVICE)
        emb = st.encode(dataset.corpus, convert_to_tensor=True, show_progress_bar=True)
        Path("model").mkdir(parents=True, exist_ok=True)
        torch.save(emb, "model/embeddings.pth")
    emb = F.normalize(emb.to(DEVICE), p=2, dim=1)
else:
    print("using skipgram embeds")
    emb = dataset.data.x

train_data, val_data, test_data = dataset.get_splits()


# --- Feature builder ---
def edge_features(emb, ei):
    u, v = ei
    eu, ev = emb[u], emb[v]
    return torch.cat([eu * ev, torch.abs(eu - ev)], dim=1)


# --- Simple MLP ---
class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


model = PairMLP(emb.size(1) * 2).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


# --- Training loop ---
def run_epoch(data, train=True):
    model.train(train)
    total_loss = 0
    idx = (
        torch.randperm(data.edge_label.size(0))
        if train
        else torch.arange(data.edge_label.size(0))
    )
    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i : i + BATCH_SIZE]
        feats = edge_features(emb, data.edge_label_index[:, batch_idx]).to(DEVICE)
        labels = data.edge_label[batch_idx].float().to(DEVICE)
        scores = model(feats)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(batch_idx)
    return total_loss / len(idx)


@torch.no_grad()
def evaluate(data):
    scores_all, labels_all = [], []
    for i in range(0, data.edge_label.size(0), BATCH_SIZE):
        feats = edge_features(emb, data.edge_label_index[:, i : i + BATCH_SIZE]).to(
            DEVICE
        )
        labels = data.edge_label[i : i + BATCH_SIZE]
        scores = torch.sigmoid(model(feats)).cpu().numpy()
        scores_all.append(scores)
        labels_all.append(labels.numpy())
    y_scores = np.concatenate(scores_all)
    y_true = np.concatenate(labels_all)
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)


# --- Train ---
for epoch in range(NUM_EPOCHS):
    loss = run_epoch(train_data, train=True)
    val_roc, val_ap = evaluate(val_data)
    print(
        f"Epoch {epoch + 1} | Loss {loss:.4f} | Val ROC {val_roc:.4f} | Val AP {val_ap:.4f}"
    )

# --- Final test ---
test_roc, test_ap = evaluate(test_data)
print(f"Test ROC {test_roc:.4f} | Test AP {test_ap:.4f}")
