import logging
import os

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.constants import constants
from src.index.index import Index
from src.logger.custom_logger import CustomLogger
from src.retrieve.bm25_elastic_search import ElasticBM25Retriever


class Retriever:
    def __init__(self, index: Index = None):
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, 'logs.log').logger
        self.logger.info("Initializing Retriever class.")
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        self.index = index
        self.semantic_retriever = self.index.vector_store.as_retriever()

        self.bm25_retriever = ElasticBM25Retriever(index.elastic_search, constants.es_index_name)

        self.ensemble_retriever = self.initialize_ensemble_retriever()

    def initialize_ensemble_retriever(self, vector_weight=0.7, bm25_weight=0.3):
        def wrap_bm25(bm25_retriever):
            return RunnableLambda(lambda query, config: [
                Document(page_content=doc["flavor_description"], metadata=doc)
                for doc in bm25_retriever.invoke(
                    query,
                    k=config.get("k", 100)
                )
            ])

        def wrap_semantic(semantic_retriever):
            return RunnableLambda(lambda query, config: semantic_retriever.invoke(
                query,
                config={"k": config.get("k", 100)}
            ))

        return EnsembleRetriever(
            retrievers=[
                wrap_semantic(self.semantic_retriever),
                wrap_bm25(self.bm25_retriever)
            ],
            weights=[vector_weight, bm25_weight]
        )


