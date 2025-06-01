import os
from typing import List

import torch
from langchain_core.documents import Document

from src.constants import constants
from src.embedding.embedding_hub import EmbeddingHub
from src.embedding.flavor_query_builder import FlavorQueryBuilder
from src.embedding.flavor_radar import FlavorRadar
from src.filter.filter import Filter
from src.index.index import Index
from src.inference.llm_inference import LLMInference
from src.logger.custom_logger import CustomLogger
from src.prompt_builder.prompt_builder import PromptBuilder
from src.retrieve.retriever import Retriever
from src.translator.translator import Translator


class SearchEngine:
    """
    A class to manage the entire search engine process,
    including retrieving relevant documents, generating answers, and updating the index.

    Attributes:
        logger (logger): logger instance for logging information and errors.
        retriever (Retriever): Instance of the Retriever class for document retrieval.
        llm_inference (LLMInference): Instance of LLMInference for generating answers.
        index (Index): Instance of the Index class for managing the document index.
    """

    def __init__(self):
        """
        Initializes the SearchEngine class, setting up the logger, retrieve,
        llm inference, system prompt, and index.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, "logs.log").logger
        self.logger.info("Initializing Search Engine...")

        self.index = Index()
        self.index.index_documents()
        self.retriever = Retriever(self.index)
        self.prompt_builder = PromptBuilder()
        self.llm_inference = LLMInference(
            system_instruction=self.prompt_builder.get_system_prompt(),
            model_name="gemini-2.0-flash"
        )
        self.translator = Translator()
        self.embedding_hub = EmbeddingHub()
        self.flavor_builder = FlavorQueryBuilder(embedding_hub=self.embedding_hub)
        self.filter = Filter(embedding_hub=self.embedding_hub)
        self.flavor_radar = FlavorRadar(embedding_hub=self.embedding_hub)
        self.num_of_search_results = 10
        self.num_of_unfiltered_search_results = 200
        self.user_language = "en"
        self.logger.info("Search Engine initialized successfully.")

    def search(self, query: str, filters: dict[str, str], negative_terms: list) -> list[Document]:
        try:
            self.logger.info(f"Processing query: {query}")
            translation_dict = self.translator.translate_text(query, "en")
            query = translation_dict["translated_text"]
            self.user_language = translation_dict["detected_source_language"]
            search_results = self.retriever.ensemble_retriever.invoke(
                query,
                config={"k": self.num_of_unfiltered_search_results}
            )

            # remove duplicate search results
            unique_results = self.remove_duplicate_results(search_results)

            # filter search results
            filtered_results = self.filter_results(unique_results, filters)
            filtered_results = self.filter.filter_by_lexical_and_semantic_exclusion(filtered_results, negative_terms)

            return filtered_results[:self.num_of_search_results]

        except Exception as e:
            self.logger.error(f"Error during the search process: {e}")
            raise

    def search_by_flavor_preferences(
            self,
            preferences: dict[str, float],
            filters: dict[str, str],
            negative_terms: list[str]
    ) -> list[Document]:
        """
        Searches based on slider-defined flavor preferences using vector embeddings.

        Args:
            preferences: Dict mapping axis labels (e.g., "citrusy ↔ chocolatey") to slider values ∈ [-1, 1]
            filters: Dict of metadata filters (e.g. roast="dark")
            negative_terms: List of semantic exclusions (e.g. ["honey", "vanilla"])

        Returns:
            List[Document]: Ranked and filtered search results.
        """
        try:
            self.logger.info(f"Processing flavor preference search...")
            # Step 1: Build query + rejection embedding
            query_vector, rejection_vector = self.flavor_builder.build_query_embedding(preferences)
            query_vector_np = query_vector.detach().cpu().numpy()

            # Step 2: Retrieve using vector search (e.g., from Pinecone)
            results = self.index.vector_store.similarity_search_by_vector(
                embedding=query_vector_np.to_list(),
                k=self.num_of_unfiltered_search_results
            )

            # Step 3: Remove duplicates
            unique_results = self.remove_duplicate_results(results)

            # Step 4: Apply filter-by-metadata and semantic exclusions
            filtered_results = self.filter_results(unique_results, filters)
            filtered_results = self.filter.filter_by_lexical_and_semantic_exclusion(filtered_results, negative_terms)

            # Step 5: Optional rerank by rejection vector similarity
            if rejection_vector is not None and torch.norm(rejection_vector) > 0:
                for doc in filtered_results:
                    doc_emb = self.embedding_hub.embed(doc.page_content, to_tensor=True)
                    penalty = self.embedding_hub.cosine_similarity(doc_emb, rejection_vector)
                    doc.metadata["rejection_penalty"] = penalty

                filtered_results = sorted(
                    filtered_results,
                    key=lambda d: d.metadata.get("score", 0.0) - 0.5 * d.metadata.get("rejection_penalty", 0.0),
                    reverse=True
                )

            return filtered_results[:self.num_of_search_results]

        except Exception as e:
            self.logger.error(f"Error during flavor preference search: {e}")
            raise

    def filter_results(self, results: List[Document], filters: dict[str, str]) -> List[Document]:
        filtered_results = []

        for doc in results:
            keep = True
            for filter_key, filter_value in filters.items():
                doc_value = doc.metadata.get(filter_key)
                if doc_value is None or str(doc_value).lower() != str(filter_value).lower():
                    keep = False
                    break
            if keep:
                filtered_results.append(doc)

        return filtered_results

    def remove_duplicate_results(self, results: list[Document]) -> list[Document]:
        seen_ids = set()
        deduped = []
        for doc in results:
            if len(deduped) >= self.num_of_search_results:
                break

            doc_id = doc.metadata.get("source")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduped.append(doc)

        return deduped

    def explain_result(self, query: str, search_result: Document) -> str:
        prompt = self.prompt_builder.create_user_content(query=query, search_result=search_result)
        explanation = self.llm_inference.inference(prompt)
        translation_dict = self.translator.translate_text(explanation, target_language=self.user_language)
        translated_explanation = translation_dict["translated_text"]
        return translated_explanation

    def update_index(self):
        """
        Updates the document index by reloading the data.
        """
        try:
            self.logger.info("Updating the document index...")
            self.index.index_documents()
            self.logger.info("Document index updated successfully.")
        except Exception as e:
            self.logger.error(f"Error updating the document index: {e}")
            raise


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "walnut"
    filters = {
        "roast": "Medium-Light",
    }

    negative_terms = ["peach"]

    results = search_engine.search(query, filters, negative_terms)
    for result in results:
        print(result)
    flavor_radar_chart = search_engine.flavor_radar.analyze_document(result)
    explanation = search_engine.explain_result(query, results[0])
    translation_dict = search_engine.translator.translate_text(explanation, target_language=search_engine.user_language)
    print(translation_dict["translated_text"])
