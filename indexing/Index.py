import time

from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore as pc
from pinecone import ServerlessSpec

from RAGLogger import RAGLogger
from indexing.DataLoader import DataLoader
from indexing.MarkDownSplitter import MarkDownSplitter
from indexing.VectorStore import VectorStore


class Index:
    def __init__(self):
        self.record_manager = self.initialize_record_manager()
        self.vector_store = VectorStore().create_vectorstore()

    def initialize_record_manager(self):
        namespace = "RAGChallenge"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )

        record_manager.create_schema()

        return record_manager

    def add_chunk_to_index(self, chunk: Document):
        index(
            [chunk],
            self.record_manager,
            self.vector_store,
            cleanup="incremental",
            source_id_key="source"
        )


if __name__=='__main__':
    rag_logger = RAGLogger().logger
    data_loader = DataLoader('../data', rag_logger)
    markdown_splitter = MarkDownSplitter(rag_logger)
    my_index = Index()

    data = data_loader.load_data()
    md_doc = data[0]

    # Perform the splitting
    try:
        final_chunks = markdown_splitter.hybrid_split(md_doc)
        for chunk in final_chunks:
            my_index.add_chunk_to_index(chunk)
            break

        print(index)
    except Exception as e:
        rag_logger.error(f"Error: {e}")









