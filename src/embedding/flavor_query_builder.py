import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple

from src.embedding.embedding_hub import EmbeddingHub


class FlavorQueryBuilder:
    def __init__(self, embedding_hub: EmbeddingHub):
        self.embedding_hub = embedding_hub

        self.axes = [
            ("sweet", "bitter"),
            ("acid", "smooth"),
            ("fruit", "nut"),
            ("citrus", "chocolate"),
            ("floral", "woody")
        ]

        self.axis_labels = [f"{left}_{right}" for left, right in self.axes]
        self.term_to_embedding = self._precompute_term_embeddings()

    def _precompute_term_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Precompute and store the embeddings for each individual term in the axes.
        """
        unique_terms = set()
        for left, right in self.axes:
            unique_terms.update([left, right])
        return {term: self.embedding_hub.embed(term, True) for term in unique_terms}

    def build_query_embedding(
        self, preferences: Dict[str, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds a weighted query embedding from user slider preferences.

        Returns:
            (positive_embedding, negative_embedding)
        """
        emb_dim = self.embedding_hub.model.get_sentence_embedding_dimension()
        pos_vector = torch.zeros(emb_dim)
        neg_vector = torch.zeros(emb_dim)
        pos_count, neg_count = 0, 0

        for (left, right) in self.axes:
            axis_label = f"{left}_{right}"
            weight = preferences.get(axis_label, 0.0)
            if weight == 0.0:
                continue

            if weight > 0:
                pos_vector += self.term_to_embedding[right] * weight
                pos_count += 1
            elif weight < 0:
                neg_vector += self.term_to_embedding[left] * abs(weight)
                neg_count += 1

        if pos_count > 0:
            pos_vector = pos_vector / torch.norm(pos_vector)
        if neg_count > 0:
            neg_vector = neg_vector / torch.norm(neg_vector)

        return pos_vector, neg_vector
