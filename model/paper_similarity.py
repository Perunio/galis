from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re


class PaperSimilarityFinder:
    """Extension to find most similar papers based on title and abstract"""

    def __init__(self, dataset, method="tfidf", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the similarity finder

        Args:
            dataset: Your OGBNLinkPredDataset instance
            method: 'tfidf' or 'sentence_transformer'
            model_name: For sentence transformer method
        """
        self.dataset = dataset
        self.method = method
        self.corpus = dataset.corpus

        self._load_citations()

        if method == "tfidf":
            self._setup_tfidf()
        elif method == "sentence_transformer":
            self.model = SentenceTransformer(model_name)
            self._setup_sentence_embeddings()
        else:
            raise ValueError("Method must be 'tfidf' or 'sentence_transformer'")

    def _load_citations(self):
        """Load citation information from the dataset"""
        self.citations = {}
        self.cited_by = {}

        edge_index = self.dataset.data.edge_index

        for i in range(edge_index.shape[1]):
            citing_paper = edge_index[0, i].item()
            cited_paper = edge_index[1, i].item()

            if citing_paper not in self.citations:
                self.citations[citing_paper] = []
            self.citations[citing_paper].append(cited_paper)

            if cited_paper not in self.cited_by:
                self.cited_by[cited_paper] = []
            self.cited_by[cited_paper].append(citing_paper)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Basic text preprocessing"""
        text = re.sub(r"\s+", " ", text.strip())
        return text.lower()

    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer and compute corpus vectors"""
        print("Setting up TF-IDF vectorization...")

        processed_corpus = [self._preprocess_text(doc) for doc in self.corpus]

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )

        self.corpus_vectors = self.vectorizer.fit_transform(processed_corpus)
        print(f"TF-IDF setup complete. Corpus shape: {self.corpus_vectors.shape}")

    def _setup_sentence_embeddings(self):
        """Setup sentence transformer and compute corpus embeddings"""
        print("Computing sentence embeddings for corpus...")

        batch_size = 100
        embeddings = []

        for i in range(0, len(self.corpus), batch_size):
            batch = self.corpus[i : i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)

        self.corpus_embeddings = np.vstack(embeddings)
        print(f"Sentence embeddings complete. Shape: {self.corpus_embeddings.shape}")

    def find_similar_papers(
        self, title: str, abstract: str, top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Find most similar papers to given title and abstract

        Args:
            title: Title of your paper
            abstract: Abstract of your paper
            top_k: Number of top similar papers to return

        Returns:
            List of tuples: (paper_index, similarity_score, paper_text)
        """
        query_text = f"{title}\n {abstract}"

        if self.method == "tfidf":
            return self._find_similar_tfidf(query_text, top_k)
        elif self.method == "sentence_transformer":
            return self._find_similar_sentence_transformer(query_text, top_k)

    def _find_similar_tfidf(
        self, query_text: str, top_k: int
    ) -> List[Tuple[int, float, str]]:
        """Find similar papers using TF-IDF"""
        processed_query = self._preprocess_text(query_text)

        query_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((idx, similarities[idx], self.corpus[idx]))

        return results

    def _find_similar_sentence_transformer(
        self, query_text: str, top_k: int
    ) -> List[Tuple[int, float, str]]:
        """Find similar papers using sentence transformers"""
        query_embedding = self.model.encode([query_text])

        similarities = cosine_similarity(
            query_embedding, self.corpus_embeddings
        ).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((idx, similarities[idx], self.corpus[idx]))

        return results

    def get_paper_citations(self, paper_idx: int) -> Tuple[List[int], List[int]]:
        """
        Get citations for a specific paper

        Args:
            paper_idx: Index of the paper in the dataset

        Returns:
            Tuple of (papers_this_cites, papers_that_cite_this)
        """
        papers_cited = self.citations.get(paper_idx, [])
        papers_citing = self.cited_by.get(paper_idx, [])

        return papers_cited, papers_citing

    def find_most_similar_with_citations(self, title: str, abstract: str) -> dict:
        """
        Find the most similar paper and return its citation information

        Args:
            title: Title of your paper
            abstract: Abstract of your paper

        Returns:
            Dictionary with similarity info and citations
        """
        similar_papers = self.find_similar_papers(title, abstract, top_k=1)

        if not similar_papers:
            return {"error": "No similar papers found"}

        most_similar_idx, similarity_score, paper_text = similar_papers[0]

        papers_cited, papers_citing = self.get_paper_citations(most_similar_idx)

        cited_papers_text = []
        for cited_idx in papers_cited[:5]:
            if cited_idx < len(self.corpus):
                cited_papers_text.append(
                    {
                        "index": cited_idx,
                        "text": self.corpus[cited_idx][:200] + "...",
                    }
                )

        return {
            "most_similar_paper": {
                "index": most_similar_idx,
                "similarity_score": float(similarity_score),
                "text": paper_text,
            },
            "citation_stats": {
                "num_papers_this_cites": len(papers_cited),
                "num_papers_citing_this": len(papers_citing),
                "total_citation_network_size": len(papers_cited) + len(papers_citing),
            },
            "papers_this_cites": papers_cited,
            "papers_citing_this": papers_citing,
            "sample_cited_papers": cited_papers_text,
        }


if __name__ == "__main__":
    dataset = OGBNLinkPredDataset()

    similarity_finder = PaperSimilarityFinder(
        dataset, method="tfidf"
    )

    my_title = "Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders"
    my_abstract = """
        Machine Learning as a Service (MLaaS) APIs provide ready-to-use and highutility 
        encoders that generate vector representations for given inputs. Since these
        encoders are very costly to train, they become lucrative targets for model stealing
        attacks during which an adversary leverages query access to the API to replicate
        the encoder locally at a fraction of the original training costs. We propose Bucks
        for Buckets (B4B), the first active defense that prevents stealing while the attack is
        happening without degrading representation quality for legitimate API users. Our
        defense relies on the observation that the representations returned to adversaries
        who try to steal the encoder’s functionality cover a significantly larger fraction
        of the embedding space than representations of legitimate users who utilize the
        encoder to solve a particular downstream task. B4B leverages this to adaptively
        adjust the utility of the returned representations according to a user’s coverage of
        the embedding space. To prevent adaptive adversaries from eluding our defense by
        simply creating multiple user accounts (sybils), B4B also individually transforms
        each user’s representations. This prevents the adversary from directly aggregating
        representations over multiple accounts to create their stolen encoder copy. Our
        active defense opens a new path towards securely sharing and democratizing
        encoders over public APIs.
    """

    top_k = 10
    print(f"\nTop {top_k} Citation Predictions:\n")

    top_papers = similarity_finder.find_similar_papers(
        my_title, my_abstract, top_k=top_k
    )

    for idx, score, text in top_papers:
        title = text.split('\n')[0].strip()
        print(f"Title: '{title}'")