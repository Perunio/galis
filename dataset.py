import os
import subprocess
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
                print(f"Downloading from {url}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"File downloaded to: {gz_path}")

                print(f"Decompressing {gz_path}...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(tsv_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"File extracted to: {tsv_path}")

                os.remove(gz_path)
                print(f"Removed temporary file: {gz_path}")

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
            print(f"Corpus created with {len(corpus)} documents.")
            return corpus
        except FileNotFoundError:
            print("Error: titleabs.tsv not found. Could not create corpus.")
            return []

    def _split_data(self, val_size: float, test_size: float) -> tuple[Data, Data, Data]:
        transform = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=False,
            add_negative_train_samples=False,
        )
        train_split, val_split, test_split = transform(self.data)
        print("Data successfully split into train, validation, and test sets.")
        return train_split, val_split, test_split

    def get_splits(self) -> tuple[Data, Data, Data]:
        return self.train_data, self.val_data, self.test_data
