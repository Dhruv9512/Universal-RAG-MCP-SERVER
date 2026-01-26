from .qdrent.qdrant_retriver import qdrentVectorStrategy
from typing import Dict, Any, Optional, List

class Strategy:
    def retriver(self, query: str, connection_config: Dict[str, Any], collection_name: str, retrieval_config: Dict[str, Any], db_type: str):
        if db_type.lower() == "qdrant":
            return qdrentVectorStrategy().SimilaritySearchRetriveDocuments(query, connection_config, collection_name, retrieval_config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        