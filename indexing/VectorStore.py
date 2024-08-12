import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from constants import constants


class VectorStore:
    def __init__(self):
        self.pinecone_api_key = constants.pinecone_api_key
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = self.create_vectorstore()

    def initialize_index(self):
        index_name = "rag-challenge-index"

        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        index = self.pc.Index(index_name)
        return index

    def create_vectorstore(self) -> PineconeVectorStore:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vs_index_name = self.initialize_index()
        vector_store = PineconeVectorStore(index=vs_index_name, embedding=embeddings)

        return vector_store


