from typing import List, Tuple, Dict, Union
from sentence_transformers import SentenceTransformer, util
import torch


class EmbeddingHub:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

        self.flavor_terms: List[str] = [
            "sweet", "bitter", "acid", "smooth",
            "fruit", "nut", "citrus", "chocolate",
            "floral", "wood"
        ]

        self.precompute_flavor_embeddings()

    def precompute_flavor_embeddings(self):
        """Precompute and store embeddings for individual flavor terms."""
        self.flavor_term_embeddings = self.embed(self.flavor_terms, to_tensor=True)

    def embed(self, text: Union[str, List[str]], to_tensor=True):
        """Embeds a string or list of strings."""
        return self.model.encode(text, convert_to_tensor=to_tensor)

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Computes cosine similarity between two tensors."""
        return util.cos_sim(emb1, emb2).item()

    def batch_cosine_similarities(self, query_emb: torch.Tensor, candidates: torch.Tensor) -> List[float]:
        """Computes similarity of query to each candidate (e.g., for exclusions)."""
        sims = util.cos_sim(query_emb, candidates)
        return sims[0].tolist()

    def get_flavor_vector(self, text: str) -> Dict[str, float]:
        """
        Computes a flavor strength score ∈ [0,1] for each individual flavor term.

        Returns:
            Dict[str, float]: e.g. { "sweet": 0.64, "acid": 0.29, ... }
        """
        doc_emb = self.embed(text, to_tensor=True)
        similarities = self.batch_cosine_similarities(doc_emb, self.flavor_term_embeddings)
        return {
            flavor: round((sim + 1) / 2, 3)
            for flavor, sim in zip(self.flavor_terms, similarities)
        }
