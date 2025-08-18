import torch
from torch import nn
import torch.nn.functional as F
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset
from pathlib import Path
import structlog
from sentence_transformers import SentenceTransformer


def edge_features(emb, ei):
    u, v = ei
    eu = emb[u].view(1, -1)  # Use view to add batch dimension
    ev = emb[v].view(1, -1)
    return torch.cat([eu * ev, torch.abs(eu - ev)], dim=1)


class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(indent=4, sort_keys=True),
    ]
)
logger = structlog.get_logger()


def abstract_to_vector(
    title: str, abstract_text: str, st_model: SentenceTransformer
) -> torch.Tensor:
    text = title + "\n" + abstract_text
    with torch.no_grad():
        vector = st_model.encode(text, convert_to_tensor=True, device=DEVICE)
    return vector


def get_citation_predictions(
    vector: torch.Tensor,
    model: PairMLP,
    z_all: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Modified to use PairMLP instead of SimpleGCN but keeps same interface
    """
    model.eval()

    with torch.no_grad():
        # Create combined embeddings: [new_vector, all_corpus_embeddings]
        combined_embeddings = torch.cat([vector[None, :], z_all], dim=0)

        # Create edge indices: new node (idx 0) to all existing nodes (1 to num_nodes)
        new_node_idx = 0
        existing_nodes = torch.arange(1, num_nodes + 1, device=DEVICE)
        edge_indices = torch.stack(
            [torch.full_like(existing_nodes, new_node_idx), existing_nodes], dim=0
        )

        # Compute edge features and predictions
        scores = []
        for i in range(num_nodes):
            feat = edge_features(combined_embeddings, torch.tensor([0, i + 1])).to(
                DEVICE
            )
            score = torch.sigmoid(model(feat))
            scores.append(score)

        return torch.stack(scores).squeeze()


def format_top_k_predictions(
    probs: torch.Tensor, dataset: OGBNLinkPredDataset, top_k=10, show_prob=False
) -> str:
    """Same as before - no changes needed"""
    probs = probs.cpu()
    top_probs, top_indices = torch.topk(probs, k=top_k)
    output_lines = []
    header = f"Top {top_k} Citation Predictions:"
    output_lines.append(header)

    for i in range(top_k):
        paper_idx = top_indices[i].item()
        prob = top_probs[i].item()
        paper_info = dataset.corpus[paper_idx]
        paper_title = paper_info.split("\n")[0]
        if show_prob:
            line = f"  - Title: '{paper_title.strip()}', Probability: {prob:.4f}"
        else:
            line = f"  - Title: '{paper_title.strip()}'"
        output_lines.append(line)

    return "\n".join(output_lines)


def prepare_system(model_path: Path):
    """
    Modified to load PairMLP and corpus embeddings instead of SimpleGCN
    """
    logger.info("system_preparation.start")

    dataset = OGBNLinkPredDataset()
    logger.info("dataset.load.success")

    model_name = "bongsoo/kpf-sbert-128d-v1"
    logger.info(
        "model.load.start", model_type="SentenceTransformer", model_name=model_name
    )
    st_model = SentenceTransformer(model_name, device=DEVICE)
    logger.info("model.load.success", model_type="SentenceTransformer")

    # Load corpus embeddings
    if Path("model/embeddings.pth").exists():
        corpus_embeddings = torch.load("model/embeddings.pth", map_location=DEVICE)
        logger.info("embeddings.load.success")
    else:
        logger.info("embeddings.calculation.start")
        corpus_embeddings = st_model.encode(
            dataset.corpus, convert_to_tensor=True, show_progress_bar=True
        )
        Path("model").mkdir(parents=True, exist_ok=True)
        torch.save(corpus_embeddings, "model/embeddings.pth")
        logger.info("embeddings.calculation.success")

    corpus_embeddings = F.normalize(corpus_embeddings.to(DEVICE), p=2, dim=1)

    # Initialize PairMLP
    embedding_dim = corpus_embeddings.size(1)
    pair_mlp = PairMLP(embedding_dim * 2).to(DEVICE)

    if model_path.exists():
        pair_mlp.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logger.info("model.load.success", model_type="PairMLP", path=str(model_path))
    else:
        logger.warning(
            "model.load.failure",
            model_type="PairMLP",
            path=str(model_path),
            reason="File not found, using random weights.",
        )

    pair_mlp.eval()

    logger.info(
        "embeddings.calculation.success",
        embedding_name="corpus_embeddings",
        shape=list(corpus_embeddings.shape),
    )
    logger.info("system_preparation.finish", status="ready_for_predictions")

    return pair_mlp, st_model, dataset, corpus_embeddings


if __name__ == "__main__":
    MODEL_PATH = Path("model_roc0_90.pth")  # Use your trained model
    pair_model, st_model, dataset, corpus_embeddings = prepare_system(MODEL_PATH)

    my_title = "A Survey of Graph Neural Networks for Link Prediction"
    my_abstract = """Link prediction is a critical task in graph analysis. 
                   In this paper, we review various GNN architectures like GCN and GraphSAGE for predicting edges."""

    new_vector = abstract_to_vector(my_title, my_abstract, st_model)
    new_vector = F.normalize(
        new_vector.view(1, -1), p=2, dim=1
    )  # Normalize like corpus embeddings

    probabilities = get_citation_predictions(
        vector=new_vector,
        model=pair_model,
        z_all=corpus_embeddings,
        num_nodes=dataset.data.num_nodes,
    )

    references = format_top_k_predictions(
        probabilities, dataset, top_k=5, show_prob=True
    )
    print(references)
