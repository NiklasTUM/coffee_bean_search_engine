import re
from langchain_core.documents import Document
from typing import List
from rapidfuzz import fuzz

from src.embedding.embedding_hub import EmbeddingHub


class Filter:
    def __init__(
        self,
        embedding_hub: EmbeddingHub,
        similarity_threshold: float = 0.6,
        fuzz_threshold: int = 85
    ):
        """
        Args:
            embedding_hub (EmbeddingHub): Shared instance for embedding and similarity.
            similarity_threshold (float): Max cosine similarity allowed before exclusion.
            fuzz_threshold (int): Fuzzy string match threshold (0–100) for lexical exclusion.
        """
        self.embedding_hub = embedding_hub
        self.similarity_threshold = similarity_threshold
        self.fuzz_threshold = fuzz_threshold

    def _contains_fuzzy_term(self, text: str, term: str) -> bool:
        """
        Checks if any word in the text fuzzily matches the term.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return any(fuzz.ratio(word, term.lower()) >= self.fuzz_threshold for word in words)

    def filter_by_lexical_and_semantic_exclusion(
        self,
        docs: List[Document],
        negative_terms: List[str]
    ) -> List[Document]:
        """
        Filters out documents whose page_content is either semantically similar
        or fuzzily matches any of the negative terms.

        Args:
            docs (List[Document]): Documents to filter.
            negative_terms (List[str]): Terms to exclude.

        Returns:
            List[Document]: Filtered list.
        """
        if not negative_terms:
            return docs

        filtered_docs = []
        negative_embeddings = self.embedding_hub.embed(negative_terms, to_tensor=True)

        for doc in docs:
            text = doc.page_content

            # Lexical fuzzy exclusion
            lexically_blocked = any(
                self._contains_fuzzy_term(text, term)
                for term in negative_terms
            )

            if lexically_blocked:
                continue

            # Semantic exclusion
            doc_embedding = self.embedding_hub.embed(text, to_tensor=True)
            similarities = self.embedding_hub.batch_cosine_similarities(doc_embedding, negative_embeddings)

            if all(sim < self.similarity_threshold for sim in similarities):
                filtered_docs.append(doc)

        return filtered_docs
