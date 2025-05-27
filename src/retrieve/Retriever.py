from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from logging import Logger
import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.prompt_values import StringPromptValue

from src.inference.LLMInference import LLMInference
from src.logger.custom_logger import CustomLogger
from src.index.index import Index


class Retriever:
    def __init__(self, logger: Logger = None, index: Index = None, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        Hybrid Retriever combining semantic vector search (Pinecone) and lexical BM25 search.
        """
        self.logger = logger or RAGLogger('../../logs', 'RAG.log').logger
        self.logger.info("Initializing Retriever class.")

        self.index = index or Index()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.vector_retriever = self.index.vector_store.as_retriever()
        self.bm25_retriever = BM25Retriever.from_documents(self.index.chunks)
        self.ensemble_retriever = self.initialize_ensemble_retriever()


    def initialize_ensemble_retriever(self):
        """
        Combine vector and BM25 retrievers using weighted ensemble.
        """
        return EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight]
        )

    def retrieve(self, query: str):
        """
        Perform a hybrid retrieval on the given query.
        """
        return self.ensemble_retriever.invoke(query)


if __name__ == '__main__':
    # index = Index()
    # retriever_instance = Retriever(index=index)
    # # retriever = retriever_instance.ensemble_retriever
    # retriever = retriever_instance.initialize_ensemble_retriever(0.6, 0.4)
    # retrieved_passages = retriever.invoke(full_query)
    # print(retrieved_passages)

    retriever = Retriever()
    while True:
        user_query = input("Enter a flavor profile or adjective query: ")
        results = retriever.retrieve(user_query)
        print("\nTop matching documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:150]}...")  # print preview