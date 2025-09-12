import os
import random
from torch_sparse import SparseTensor
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

import requests
import gzip
import shutil


class OGBNLinkPredDataset:
    def __init__(
        self, root_dir: str = "data", val_size: float = 0.1, test_size: float = 0.2
    ):
        self._base_dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root_dir)
        self.data = self._base_dataset[0]
        self.root = self._base_dataset.root
        self.num_features = self._base_dataset.num_features

        self._download_abstracts()
        self.corpus = self._load_corpus()

        self.val_size = val_size
        self.test_size = test_size

    def _download_abstracts(self):
        target_dir = os.path.join(self.root, "mapping")
        tsv_path = os.path.join(target_dir, "titleabs.tsv")

        if not os.path.exists(tsv_path):
            print("Downloading title and abstract information...")
            gz_path = tsv_path + ".gz"
            url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
            os.makedirs(target_dir, exist_ok=True)

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                with gzip.open(gz_path, "rb") as f_in:
                    with open(tsv_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(gz_path)

            except requests.exceptions.RequestException as e:
                print(f"Error downloading file: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        else:
            print("Title and abstract file already exists.")

    def _load_corpus(self) -> list[str]:
        tsv_path = os.path.join(self.root, "mapping", "titleabs.tsv")
        try:
            df_text = pd.read_csv(
                tsv_path,
                sep="\t",
                header=None,
                names=["paper_id", "title", "abstract"],
                lineterminator="\n",
                low_memory=False,
            )
            df_text_aligned = df_text.reset_index(drop=True)
            corpus = (
                df_text_aligned["title"].fillna("")
                + "\n "
                + df_text_aligned["abstract"].fillna("")
            ).tolist()
            return corpus
        except FileNotFoundError:
            print("Error: titleabs.tsv not found. Could not create corpus.")
            return []

    def get_splits(self) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=self.val_size,
            num_test=self.test_size,
            is_undirected=False,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )
        train_split, val_split, test_split = transform(self.data)
        return train_split, val_split, test_split


class OGBNLinkPredNegDataset(OGBNLinkPredDataset):
    """Degree similar hard negatives sampling"""

    def __init__(
        self, root_dir: str = "data", val_size: float = 0.1, test_size: float = 0.2
    ):
        super().__init__(root_dir, val_size, test_size)
        self.degree_tol = 0

    def get_splits(self) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=self.val_size,
            num_test=self.test_size,
            is_undirected=False,
            add_negative_train_samples=False,
            neg_sampling_ratio=0.0,
        )
        train_split, val_split, test_split = transform(self.data)

        print("Generating hard negatives...")
        adj_matrix = SparseTensor.from_edge_index(
            train_split.edge_index,  # only from train_split
            sparse_sizes=(self.data.num_nodes, self.data.num_nodes),
        )
        self.degrees = adj_matrix.sum(dim=0).to(torch.long)
        # to prevent creating negative edges that are positive in other split
        self.all_edge_set = set(zip(*self.data.edge_index.tolist()))
        train_split = self._add_balanced_negs(train_split)
        val_split = self._add_balanced_negs(val_split)
        test_split = self._add_balanced_negs(test_split)
        return train_split, val_split, test_split

    def _add_balanced_negs(self, split_data):
        assert (split_data.edge_label == 1).all(), "Expected only positive edges"

        pos_edges = split_data.edge_label_index
        pos_list = pos_edges.t().tolist()
        num_negs = pos_edges.size(1)

        negs = []
        for _ in range(num_negs):
            u, v_orig = random.choice(pos_list)
            target_deg = int(self.degrees[v_orig])

            found = False
            for _ in range(20):
                w = random.randrange(self.data.num_nodes)
                if (
                    (u, w) not in self.all_edge_set
                    and w != u
                    and abs(int(self.degrees[w]) - target_deg) <= self.degree_tol
                ):
                    negs.append((u, w))
                    found = True
                    break

            if not found:
                while True:
                    w = random.randrange(self.data.num_nodes)
                    if (u, w) not in self.all_edge_set and w != u:
                        negs.append((u, w))
                        break

        neg_edges = torch.tensor(negs, dtype=torch.long).t()
        N = pos_edges.size(1)

        split_data.edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
        split_data.edge_label = torch.cat(
            [
                torch.ones(N, dtype=torch.long, device=pos_edges.device),
                torch.zeros(N, dtype=torch.long, device=pos_edges.device),
            ]
        )

        return split_data


if __name__ == "__main__":
    dataset = OGBNLinkPredDataset()
    train, val, test = dataset.get_splits()

    def extract_pos_neg_edges(split):
        pos = split.edge_label_index[:, split.edge_label == 1]
        neg = split.edge_label_index[:, split.edge_label == 0]
        return pos, neg

    for name, split in [("train", train), ("val", val), ("test", test)]:
        assert split.edge_label_index.shape[0] == 2, (
            f"{name}: edge_label_index must have 2 rows"
        )
        assert split.edge_label_index.shape[1] == split.edge_label.shape[0], (
            f"{name}: label/index shape mismatch"
        )
        assert torch.all(0 <= split.edge_label) and torch.all(split.edge_label <= 1), (
            f"{name}: labels not 0/1"
        )

        pos, neg = extract_pos_neg_edges(split)
        assert pos.size(1) == neg.size(1), f"{name}: pos/neg count mismatch"

        pos_set = set(tuple(e) for e in pos.t().tolist())
        neg_set = set(tuple(e) for e in neg.t().tolist())
        assert pos_set.isdisjoint(neg_set), f"{name}: pos/neg overlap"

        assert all(u != v for u, v in pos_set), f"{name}: pos self-loops"
        assert all(u != v for u, v in neg_set), f"{name}: neg self-loops"

        assert len(pos_set) == pos.size(1), f"{name}: pos duplicates"
        assert len(neg_set) == neg.size(1), f"{name}: neg duplicates"

        assert pos.size(1) / neg.size(1) == 1.0 if neg.size(1) > 0 else True, (
            f"{name}: ratio not 1.0"
        )

    print("All asserts passed!")
