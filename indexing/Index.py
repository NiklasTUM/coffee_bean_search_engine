import logging
import os
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from RAGLogger import RAGLogger
from constants import constants
from indexing.DataLoader import DataLoader
from indexing.MarkDownSplitter import MarkDownSplitter
from indexing.VectorStore import VectorStore


class Index:
    """
    A class to manage the indexing of documents, including loading data,
    splitting markdown files into chunks, and adding those chunks to an index.

    Attributes:
        log_dir (str): Directory where logs are stored.
        logger (RAGLogger): Logger instance for logging information and errors.
        record_manager (SQLRecordManager): Manages the records in the SQL database.
        vector_store (PineconeVectorStore): Vector store instance for managing embeddings.
        data_path (str): Path to the data directory.
        data_loader (DataLoader): Instance to load data from the data directory.
        markdown_splitter (MarkDownSplitter): Instance to split markdown documents into chunks.
    """

    def __init__(self, logger: RAGLogger = None):
        """
        Initializes the Index class, setting up the logger, record manager, vector store,
        data loader, and markdown splitter.

        Args:
            logger (RAGLogger, optional): Logger instance for logging.
                                          If not provided, a default logger is set up.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        self.logger.info("Initializing Index class...")

        self.record_manager = self.initialize_record_manager(self.logger)
        self.vector_store = VectorStore().create_vectorstore()
        self.data_path = os.path.join(constants.root_dir, "data")
        self.data_loader = DataLoader(self.data_path, self.logger)
        self.markdown_splitter = MarkDownSplitter(self.logger)

        self.logger.info("Index class initialized successfully.")

    @staticmethod
    def initialize_record_manager(logger: logging.Logger) -> SQLRecordManager:
        """
        Initializes the SQLRecordManager with a specified namespace and database URL.

        Returns:
            SQLRecordManager: The initialized SQLRecordManager instance.
        """
        namespace = "RAGChallenge"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )

        record_manager.create_schema()
        logger.info(f"SQLRecordManager initialized with namespace: {namespace}")
        return record_manager

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
        Loads Markdown documents, splits them into chunks, and indexes those chunks.
        """
        try:
            self.logger.info("Starting the document indexing process...")
            loaded_documents = self.data_loader.load_data()
            chunks = []
            for doc in loaded_documents:
                chunks.extend(self.markdown_splitter.hybrid_split(doc))

            self.add_chunk_to_index(chunks)
            self.logger.info("Document indexing process completed successfully.")
        except Exception as e:
            self.logger.error("An error occurred during the indexing process.")
            self.logger.error(f"Error: {e}")
            raise


if __name__ == '__main__':
    rag_logger = RAGLogger().logger
    data_path = os.path.join(constants.root_dir, "data")
    data_loader = DataLoader(data_path, logger=rag_logger)
    markdown_splitter = MarkDownSplitter(rag_logger)
    my_index = Index(rag_logger)

    # Perform the indexing
    try:
        my_index.index_documents()
    except Exception as e:
        rag_logger.error(f"Error during the document indexing process: {e}")
