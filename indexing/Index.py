import os

import langchain.indexes
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from RAGLogger import RAGLogger
from constants import constants
from indexing.DataLoader import DataLoader
from indexing.MarkDownSplitter import MarkDownSplitter
from indexing.VectorStore import VectorStore


class Index:
    def __init__(self, logger: RAGLogger):
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        self.record_manager = self.initialize_record_manager()
        self.vector_store = VectorStore().create_vectorstore()
        self.data_path = os.path.join(constants.root_dir, "data")
        self.data_loader = DataLoader(self.data_path, self.logger)
        self.markdown_splitter = MarkDownSplitter(self.logger)

    @staticmethod
    def initialize_record_manager():
        namespace = "RAGChallenge"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )

        record_manager.create_schema()

        return record_manager

    def add_chunk_to_index(self, chunk_doc: list[Document]):
        index(
            chunk_doc,
            self.record_manager,
            self.vector_store,
            cleanup="incremental",
            source_id_key="source"
        )


    def index_documents(self):
        loaded_documents = self.data_loader.load_data()
        chunks = []
        for doc in loaded_documents:
            chunks.extend(self.markdown_splitter.hybrid_split(doc))

        self.add_chunk_to_index(chunks)


if __name__ == '__main__':
    rag_logger = RAGLogger().logger
    data_path = os.path.join(constants.root_dir, "data")
    data_loader = DataLoader(data_path, logger=rag_logger)
    markdown_splitter = MarkDownSplitter(rag_logger)
    my_index = Index(rag_logger)

    data = data_loader.load_data()

    # Perform the splitting
    try:
        my_index.index_documents()
    except Exception as e:
        rag_logger.error(f"Error: {e}")
