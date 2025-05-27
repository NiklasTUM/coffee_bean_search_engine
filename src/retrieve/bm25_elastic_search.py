from elasticsearch import Elasticsearch


class ElasticBM25Retriever:
    def __init__(self, es_client: Elasticsearch, index_name: str):
        self.client = es_client
        self.index = index_name

    def invoke(self, query: str, k: int = 10):
        response = self.client.search(
            index=self.index,
            query={
                "match": {
                    "flavor_description": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            size=k
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
