from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sentence_transformers import SentenceTransformer
import argparse
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset, OGBNLinkPredNegDataset

BATCH_SIZE_EDGES = 100_000  # edge batching for scoring


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-neg", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bert-embed", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


@torch.no_grad()
def eval_edges_cos(global_emb, edge_index, edge_label, batch_size=BATCH_SIZE_EDGES):
    # edge_index shape: [2, M] with GLOBAL node ids; edge_label: [M] in {0,1}
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_index.max() < global_emb.size(0), "Edge node id out of range."
    assert (edge_label == 0).any() and (edge_label == 1).any(), "Need both classes."

    scores_list, labels_list = [], []
    M = edge_index.size(1)
    for i in range(0, M, batch_size):
        j = min(i + batch_size, M)
        src = edge_index[0, i:j].to(global_emb.device)
        dst = edge_index[1, i:j].to(global_emb.device)
        scores = (global_emb[src] * global_emb[dst]).sum(
            dim=1
        )  # cosine (L2-normalized)
        scores_list.append(scores.float().cpu().numpy())
        labels_list.append(edge_label[i:j].cpu().numpy())
    y_scores = np.concatenate(scores_list)
    y_true = np.concatenate(labels_list)
    roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return roc, ap


if __name__ == "__main__":
    args = parse_args()
    USE_CUSTOM_NEG = args.custom_neg
    USE_BERT_EMBED = args.bert_embed
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            emb = st.encode(
                dataset.corpus, convert_to_tensor=True, show_progress_bar=True
            )
            Path("model").mkdir(parents=True, exist_ok=True)
            torch.save(emb, "model/embeddings.pth")
        emb = F.normalize(emb.to(DEVICE), p=2, dim=1)
    else:
        print("using skipgram embeds")
        emb = dataset.data.x

    train_data, val_data, test_data = dataset.get_splits()

    # Sanity checks
    for split_name, data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        assert data.edge_label_index.size(1) == data.edge_label.size(0), (
            f"{split_name} size mismatch"
        )
        assert (data.edge_label == 0).any() and (data.edge_label == 1).any(), (
            f"{split_name} lacks negatives"
        )
        assert data.edge_label_index.max() < emb.size(0), (
            f"{split_name} has node ids >= num_nodes"
        )

    val_roc, val_ap = eval_edges_cos(
        emb, val_data.edge_label_index, val_data.edge_label
    )
    test_roc, test_ap = eval_edges_cos(
        emb, test_data.edge_label_index, test_data.edge_label
    )

    print(f"Val ROC-AUC:  {val_roc:.4f}, Val AP:  {val_ap:.4f}")
    print(f"Test ROC-AUC: {test_roc:.4f}, Test AP: {test_ap:.4f}")
