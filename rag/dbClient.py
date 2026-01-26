from typing import Dict, Any, Optional, List
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore




# --------------------------------------------------
# DB Client
# --------------------------------------------------
class dbClient:
    
    # Creating Qdrent Client Method
    def qdrant_client(
        self,
        connection: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        qdrantClient = QdrantClient(
            url=connection.get("url"),
            api_key=connection.get("api_key"),
        )
        
        return qdrantClient