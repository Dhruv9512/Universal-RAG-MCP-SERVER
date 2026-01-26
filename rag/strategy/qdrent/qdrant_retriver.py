
from typing import Dict, Any, List, Optional
from rag.dbClient import dbClient


class qdrentVectorStrategy():
    
    # Normal retrival using search
    def SimilaritySearchRetriveDocuments(
        self,
        query: str,
        connection_config: Dict[str, Any],
        collection_name: str,
        retrieval_config: Dict[str, Any],
    ) -> str:
        try:
            client = dbClient().qdrant_client(connection_config)
                    
            search_result=client.query(
                collection_name=collection_name,
                query_text=query,
                limit=retrieval_config["top_k"]
            )
            if search_result:
                return search_result
        except Exception as e:
            return f"Error during Qdrant search: {e}"
    
    def MMRRetriveDocuments(self):
        pass