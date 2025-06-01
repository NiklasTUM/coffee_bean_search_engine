import re
from typing import List, Dict
from langchain_core.documents import Document
from rapidfuzz import fuzz

from src.embedding.embedding_hub import EmbeddingHub


class FlavorRadar:
    def __init__(
        self,
        embedding_hub: EmbeddingHub,
        boost_per_match: float = 0.1,
        max_lexical_boost: float = 0.3,
        fuzz_threshold: int = 85,
    ):
        """
        Args:
            embedding_hub (EmbeddingHub): Shared embedding logic.
            boost_per_match (float): Amount added per lexical match.
            max_lexical_boost (float): Cap for total lexical boost per term.
            fuzz_threshold (int): Minimum fuzzy match score for token matching.
        """
        self.embedding_hub = embedding_hub
        self.boost_per_match = boost_per_match
        self.max_lexical_boost = max_lexical_boost
        self.fuzz_threshold = fuzz_threshold
        self.flavor_terms = embedding_hub.flavor_terms

    def _count_fuzzy_matches(self, text: str, term: str) -> int:
        """
        Counts fuzzy matches of a term in the input text using word-level comparison.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(fuzz.ratio(word, term.lower()) >= self.fuzz_threshold for word in words)

    def _lexical_bias(self, text: str, term: str) -> float:
        """
        Returns a proportional boost based on number of fuzzy matches.
        """
        count = self._count_fuzzy_matches(text, term)
        return min(count * self.boost_per_match, self.max_lexical_boost)

    def enhance_flavor_contrast(self, flavor_scores: Dict[str, float], factor: float = 0.2) -> Dict[str, float]:
        """
        Enhances contrast between paired flavors (e.g., sweet vs bitter).

        Args:
            flavor_scores: Dict of normalized scores ∈ [0, 1]
            factor: Scaling factor for contrast exaggeration ∈ [0, 1]

        Returns:
            New dict with contrast-adjusted scores.
        """
        enhanced = flavor_scores.copy()

        flavor_pairs = [
            ("sweet", "bitter"),
            ("acid", "smooth"),
            ("fruit", "nut"),
            ("citrus", "chocolate"),
            ("floral", "wood"),
        ]

        for left, right in flavor_pairs:
            l_val = enhanced.get(left, 0.0)
            r_val = enhanced.get(right, 0.0)

            if l_val == r_val:
                continue  # no contrast to enhance

            stronger, weaker = (left, right) if l_val > r_val else (right, left)
            delta = abs(l_val - r_val)
            shift = delta * factor

            enhanced[stronger] = min(enhanced[stronger] + shift, 1.0)
            enhanced[weaker] = max(enhanced[weaker] - shift, 0.0)

        return {k: round(v, 3) for k, v in enhanced.items()}

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Computes flavor strength vector with semantic + proportional lexical boost.
        Returns values ∈ [0, 1].
        """
        base_scores = self.embedding_hub.get_flavor_vector(text)
        boosted_scores = {}

        for flavor, base in base_scores.items():
            boost = self._lexical_bias(text, flavor)
            boosted_scores[flavor] = round(min(base + boost, 1.0), 3)

        return boosted_scores

    def analyze_document(self, doc: Document) -> Document:
        scores = self.analyze_text(doc.page_content)
        enhanced_scores = self.enhance_flavor_contrast(scores, 1)
        doc.metadata["flavor_vector"] = enhanced_scores
        print(doc)
        return doc

    def generate_flavor_radar_batch(self, docs: List[Document]) -> List[Document]:
        return [self.analyze_document(doc) for doc in docs]
