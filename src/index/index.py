from logging import Logger
import os
from pathlib import Path

from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from src.logger.custom_logger import CustomLogger
from src.constants import constants
from src.index.data_loader import DataLoader
from src.index.vector_store import VectorStore
from elasticsearch import Elasticsearch, helpers


class Index:
    """
    A class to manage the index of documents, including loading data,
    splitting markdown files into chunks, and adding those chunks to an index.

    Attributes:
        log_dir (str): Directory where logs are stored.
        logger (CustomLogger): logger instance for logging information and errors.
        record_manager (SQLRecordManager): Manages the records in the SQL database.
        vector_store (PineconeVectorStore): Vector store instance for managing embeddings.
        data_loader (DataLoader): Instance to load data from the data directory.
    """

    def __init__(self):
        """
        Initializes the Index class, setting up the logger, record manager, vector store,
        data loader.


        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, "logs.log").logger
        self.logger.info("Initializing Index class...")
        self.data_loader = DataLoader()

        self.elastic_search = Elasticsearch(
            constants.es_url,
            api_key=constants.es_api_key
        )
        self.es_index_name = constants.es_index_name

        self.create_es_index_if_missing()

        self.record_manager = self.initialize_record_manager(self.logger)
        self.vector_store = VectorStore().create_vectorstore()
        self.logger.info("Index class initialized successfully.")

    @staticmethod
    def initialize_record_manager(logger: Logger) -> SQLRecordManager:
        """
        Initializes the SQLRecordManager with a specified namespace and database URL.

        Returns:
            SQLRecordManager: The initialized SQLRecordManager instance.
        """
        namespace = "coffee_beans_large"
        db_dir = os.path.join(constants.root_dir, "db", "record_manager_cache_large.sql")
        db_path = Path(db_dir).as_posix()
        db_url = f'sqlite:///{db_path}'
        record_manager = SQLRecordManager(
            namespace, db_url=db_url
        )

        record_manager.create_schema()
        logger.info(f"SQLRecordManager initialized with namespace: {namespace}")
        return record_manager

    def create_es_index_if_missing(self):
        if not self.elastic_search.indices.exists(index=self.es_index_name):
            self.logger.info(f"Creating ElasticSearch index: {self.es_index_name}")

            mappings = {
                "mappings": {
                    "properties": {
                        "flavor_description": {"type": "text"},
                        "desc_2": {"type": "text"},
                        "desc_3": {"type": "text"},
                        "name": {"type": "keyword"},
                        "roaster": {"type": "keyword"},
                        "roast": {"type": "keyword"},
                        "loc_country": {"type": "keyword"},
                        "origin_1": {"type": "keyword"},
                        "origin_2": {"type": "keyword"},
                        "100g_USD": {"type": "float"},
                        "rating": {"type": "float"},
                        "review_date": {"type": "keyword"},
                        "source": {"type": "keyword"},
                    }
                }
            }

            self.elastic_search.indices.create(index=self.es_index_name, body=mappings)
            self.logger.info(f"Index and mappings created for: {self.es_index_name}")

    def row_to_document(self, row, idx=None) -> Document:
        row_clean = row.fillna("")

        page_content = str(row_clean["desc_1"]).strip()
        metadata = row_clean.drop("desc_1").to_dict()
        metadata["source"] = f"review_{idx}"

        return Document(page_content=page_content, metadata=metadata)

    def add_to_elasticsearch(self, documents: list[Document]):
        actions = []

        for doc in documents:
            doc_id = doc.metadata.get("source")
            if not doc_id:
                continue

            doc_body = {
                "flavor_description": doc.page_content,
                "desc_2": doc.metadata.get("desc_2", ""),
                "desc_3": doc.metadata.get("desc_3", ""),
                "name": doc.metadata.get("name", ""),
                "roaster": doc.metadata.get("roaster", ""),
                "roast": doc.metadata.get("roast", ""),
                "loc_country": doc.metadata.get("loc_country", ""),
                "origin_1": doc.metadata.get("origin_1", ""),
                "origin_2": doc.metadata.get("origin_2", ""),
                "100g_USD": doc.metadata.get("100g_USD", 0.0),
                "rating": doc.metadata.get("rating", 0.0),
                "review_date": doc.metadata.get("review_date", ""),
                "source": doc_id,
            }

            actions.append({
                "_index": self.es_index_name,
                "_id": doc_id,
                "_source": doc_body,
            })

        if actions:
            helpers.bulk(self.elastic_search, actions)
            self.logger.info(f"Indexed {len(actions)} structured docs into Elastic Cloud.")

    def add_chunk_to_index(self, chunk_doc: list[Document]):
        """
        Adds a list of document chunks to the index.

        Args:
            chunk_doc (list[Document]): A list of Document objects to be added to the index.
        """
        try:
            self.logger.info("Adding document chunks to the index...")
            index(
                chunk_doc,
                self.record_manager,
                self.vector_store,
                cleanup="incremental",
                source_id_key="source"
            )
            self.logger.info("Document chunks added to the index successfully.")
        except Exception as e:
            self.logger.error(f"Failed to add chunks to the index: {e}")
            raise

    def index_documents(self):
        """
        Loads coffee data, converts rows to Documents (embedding only the review),
        and indexes them using the vector store and record manager.
        """
        try:
            self.logger.info("Starting the coffee review index process...")
            df = self.data_loader.load_coffee_data()

            documents = [
                self.row_to_document(row, idx)
                for idx, (_, row) in enumerate(df.iterrows())
            ]

            self.chunks = documents
            self.add_chunk_to_index(documents)
            self.add_to_elasticsearch(documents)  # NEW

            self.logger.info("Indexing completed successfully.")
        except Exception as e:
            self.logger.error("Error during indexing.")
            self.logger.error(f"{e}")
            raise


if __name__ == '__main__':
    my_index = Index()

    # Perform the index
    try:
        my_index.index_documents()
    except Exception as exc:
        print(f"Error during the document index process: {exc}")
