import os
import torch.nn.functional as F
import random
from torch_sparse import SparseTensor
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
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

        self.train_data, self.val_data, self.test_data = self._split_data(
            val_size, test_size
        )

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

    def _split_data(self, val_size: float, test_size: float) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=False,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )
        train_split, val_split, test_split = transform(self.data)
        return train_split, val_split, test_split

    def get_splits(self) -> tuple[Data, Data, Data]:
        return self.train_data, self.val_data, self.test_data


class OGBNLinkPredNegDataset(OGBNLinkPredDataset):
    """Degree similar hard negatives sampling"""

    def __init__(
        self, root_dir: str = "data", val_size: float = 0.1, test_size: float = 0.2
    ):
        self._base_dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root_dir)
        self.data = self._base_dataset[0]
        self.root = self._base_dataset.root
        self.num_features = self._base_dataset.num_features

        self._download_abstracts()
        self.corpus = self._load_corpus()

        self.train_data, self.val_data, self.test_data = self._split_data(
            val_size, test_size
        )
        self.degree_tol = 0

        # Setup degree and edge statistics from train only
        train_adj = SparseTensor.from_edge_index(
            self.train_data.edge_index,
            sparse_sizes=(self.data.num_nodes, self.data.num_nodes),
        )
        self.train_in_deg = train_adj.sum(dim=0).to(torch.long)

        # Get ALL edges to avoid false negatives
        all_edges = torch.cat(
            [
                self.train_data.edge_index,
                self.val_data.edge_index,
                self.test_data.edge_index,
            ],
            dim=1,
        )
        self.all_edge_set = set(zip(*all_edges.tolist()))

        # Add negatives to all splits
        print("Generating hard negatives...")
        self.train_data = self._add_balanced_negs(self.train_data)
        self.val_data = self._add_balanced_negs(self.val_data)
        self.test_data = self._add_balanced_negs(self.test_data)

    def _split_data(self, val_size: float, test_size: float) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=False,
            add_negative_train_samples=False,
            neg_sampling_ratio=0.0,
        )
        train_split, val_split, test_split = transform(self.data)
        return train_split, val_split, test_split

    def _add_balanced_negs(self, split_data):
        assert (split_data.edge_label == 1).all(), "Expected only positive edges"

        pos_edges = split_data.edge_label_index
        pos_list = pos_edges.t().tolist()
        num_negs = pos_edges.size(1)

        negs = []
        for _ in range(num_negs):
            u, v_orig = random.choice(pos_list)
            target_deg = int(self.train_in_deg[v_orig])

            found = False
            for _ in range(20):
                w = random.randrange(self.data.num_nodes)
                if (
                    (u, w) not in self.all_edge_set
                    and w != u
                    and abs(int(self.train_in_deg[w]) - target_deg) <= self.degree_tol
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


class OGBNLinkPredNegDataset2(OGBNLinkPredDataset):
    """Degree and semantically similar hard negatives sampling"""

    def __init__(self, root_dir="data", val_size=0.1, test_size=0.2):
        super().__init__(root_dir, val_size, test_size)

        self.degree_tol = 0
        train_adj = SparseTensor.from_edge_index(
            self.train_data.edge_index,
            sparse_sizes=(self.data.num_nodes, self.data.num_nodes),
        )
        self.train_in_deg = train_adj.sum(dim=0).to(torch.long)

        all_edges = torch.cat(
            [
                self.train_data.edge_index,
                self.val_data.edge_index,
                self.test_data.edge_index,
            ],
            dim=1,
        )
        self.all_edge_set = set(zip(*all_edges.tolist()))

    def _split_data(self, val_size: float, test_size: float) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=False,
            add_negative_train_samples=False,
            neg_sampling_ratio=0.0,
        )
        train_split, val_split, test_split = transform(self.data)

        print("Generating semantic hard negatives...")
        train_split = self._add_balanced_negs(train_split)
        val_split = self._add_balanced_negs(val_split)
        test_split = self._add_balanced_negs(test_split)
        return train_split, val_split, test_split

    def _add_balanced_negs(self, split_data):
        assert (split_data.edge_label == 1).all(), "Expected only positive edges"

        BS = 10_000
        B = self.data.x.to("cuda", dtype=torch.bfloat16)  # (num_nodes, dim)
        B = F.normalize(B, p=2, dim=1)
        K = 10

        pos_edges = split_data.edge_label_index
        adj_matrix = SparseTensor.from_edge_index(
            split_data.edge_index,
            sparse_sizes=(self.data.num_nodes, self.data.num_nodes),
        )
        degrees = adj_matrix.sum(dim=0).to("cuda")

        topk_val = torch.empty((BS, K), dtype=torch.bfloat16, device="cuda")
        topk_idx = torch.empty((BS, K), dtype=torch.int64, device="cuda")

        neg_edges = []

        for i in range(0, pos_edges.shape[1], BS):
            batch_end = min(i + BS, pos_edges.shape[1])
            src_idx = pos_edges[0, i:batch_end]  # (batch_size,)
            dst_idx = pos_edges[1, i:batch_end]  # (batch_size,)

            A = B[src_idx]  # (batch_size, dim)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sim = torch.mm(A, B.t())  # equivalent to cos-sim

                # mask for similarity with itself and existing edges
                sim[torch.arange(len(A)), dst_idx] = -1
                sim[torch.arange(len(A)), src_idx] = -1
                # TODO: exclude edges from val&test sets FUCK!!!!

                torch.topk(sim, K, out=(topk_val, topk_idx))
                topk_idx2 = topk_idx[: len(A)]

                # sample degree matched negs
                topk_deg = degrees[topk_idx2]
                src_deg = degrees[src_idx]

                deg_diffs = torch.abs(topk_deg - src_deg.unsqueeze(1))
                closest_idx = torch.argmin(deg_diffs, dim=1)  # (batch_size,)
                sampled_negs = topk_idx[
                    torch.arange(len(A), device="cuda"), closest_idx
                ]
                neg_edges.append(sampled_negs)

        neg_dsts = torch.cat(neg_edges, dim=0).to("cpu")
        neg_edge_index = torch.stack([pos_edges[0], neg_dsts], dim=0)
        assert len(neg_dsts) == len(pos_edges.T), (
            "Expected same amount of positive and negative edges"
        )
        edge_label_index = torch.cat([pos_edges, neg_edge_index], dim=1)
        edge_label = torch.cat(
            [split_data.edge_label, torch.zeros(len(neg_edges))], dim=0
        )
        print(edge_label.shape, edge_label_index.shape)
        return Data(
            x=split_data.x,
            edge_index=split_data.edge_index,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
        )
