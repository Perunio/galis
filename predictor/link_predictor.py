from pathlib import Path
import torch
import structlog

from sentence_transformers import SentenceTransformer # type: ignore
from model.simple_gcn_model import SimpleGCN
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset


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
    text: str = title + "\n" + abstract_text
    with torch.no_grad():
        vector = st_model.encode(text, convert_to_tensor=True, device=DEVICE)
    return vector.unsqueeze(0)


def get_citation_predictions(
    vector: torch.Tensor, model: SimpleGCN, z_all: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        empty_edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
        h1_new = model.conv1(vector, edge_index=empty_edge_index).relu()
        z_new = model.conv2(h1_new, edge_index=empty_edge_index)

    new_node_idx = num_nodes
    row = torch.full((num_nodes,), fill_value=new_node_idx, device=DEVICE)
    col = torch.arange(num_nodes, device=DEVICE)
    edge_label_index_to_check = torch.stack([row, col], dim=0)

    z_combined = torch.cat([z_all, z_new], dim=0)

    with torch.no_grad():
        logits = model.decode(z_combined, edge_label_index_to_check)

    return torch.sigmoid(logits)


def format_top_k_predictions(
    probs: torch.Tensor, dataset: OGBNLinkPredDataset, top_k: int =10, show_prob: bool =False
) -> str:
    """
    Formats the top K predictions into a single string for display.

    Args:
        probs (torch.Tensor): The tensor of probabilities for all potential links.
        dataset (OGBNLinkPredDataset): The dataset object containing the corpus.
        top_k (int): The number of top predictions to format.

    Returns:
        str: A formatted string with the top K predictions.
    """
    probs = probs.cpu()
    top_probs, top_indices = torch.topk(probs, k=top_k)

    output_lines = []

    header = f"Top {top_k} Citation Predictions:"
    output_lines.append(header)

    for i in range(top_k):
        paper_idx = int(top_indices[i].item()) # top_indices[i].item()
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
    Performs all one-time, expensive operations to prepare the system.
    Initializes models, loads data, and pre-calculates embeddings using structured logging.
    """
    logger.info("system_preparation.start")

    dataset = OGBNLinkPredDataset()
    data = dataset.data.to(DEVICE)
    logger.info("dataset.load.success")

    model_name = "bongsoo/kpf-sbert-128d-v1"
    logger.info(
        "model.load.start", model_type="SentenceTransformer", model_name=model_name
    )
    st_model = SentenceTransformer(model_name, device=DEVICE)
    logger.info("model.load.success", model_type="SentenceTransformer")

    gcn_model = SimpleGCN(
        in_channels=dataset.num_features, hidden_channels=256, out_channels=128
    ).to(DEVICE)

    if model_path.exists():
        gcn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logger.info("model.load.success", model_type="GCN", path=str(model_path))
    else:
        logger.warning(
            "model.load.failure",
            model_type="GCN",
            path=str(model_path),
            reason="File not found, using random weights.",
        )
    gcn_model.eval()

    logger.info("embeddings.calculation.start", embedding_name="z_all")
    with torch.no_grad():
        z_all = gcn_model(data.x, data.edge_index)

    logger.info(
        "embeddings.calculation.success",
        embedding_name="z_all",
        shape=list(z_all.shape),
    )

    logger.info("system_preparation.finish", status="ready_for_predictions")
    return gcn_model, st_model, dataset, z_all


if __name__ == "__main__":
    MODEL_PATH = Path("model.pth")

    gcn_model, st_model, dataset, z_all = prepare_system(MODEL_PATH)

    my_title = "A Survey of Graph Neural Networks for Link Prediction"
    my_abstract = """Link predictor is a critical task in graph analysis. "
                   "In this paper, we review various GNN architectures like GCN and GraphSAGE for predicting edges.
                   """

    new_vector = abstract_to_vector(my_title, my_abstract, st_model)

    probabilities = get_citation_predictions(
        vector=new_vector,
        model=gcn_model,
        z_all=z_all,
        num_nodes=dataset.data.num_nodes,
    )

    references = format_top_k_predictions(probabilities, dataset, top_k=5)
    print(references)
