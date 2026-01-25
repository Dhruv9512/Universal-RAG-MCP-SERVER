import asyncio
import os
import logging
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from huggingface_hub import InferenceClient
from pydantic import Field
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from qdrant_client import QdrantClient

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP Server
mcp = FastMCP("Universal RAG Server")

# Global placeholder for the engine
_rag_engine_instance = None

class RAGEngine:
    def __init__(self):
        logger.info("Initializing RAG Engine & Loading Models...")
        
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            logger.error("Missing HUGGINGFACEHUB_API_TOKEN environment variable.")
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set.")

        self.embedder = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=api_token
        )
        self.reranker = InferenceClient(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            token=api_token
        )
        logger.info("Models loaded successfully.")

    def _handle_qdrant_sync(self, connection: Dict, collection: str, vector: List[float], config: Dict) -> List[Dict]:
        """Synchronous Qdrant logic to be run in a thread."""
        url = connection.get("url")
        api_key = connection.get("api_key")
        
        client = QdrantClient(url=url, api_key=api_key)

        limit = config.get("k", 10)
        filter_expr = config.get("filter", None) 
        
        hits = client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=filter_expr,
            limit=limit
        )
        
        return [{
            "content": hit.payload.get("text", ""),
            "metadata": hit.payload,
            "score": hit.score
        } for hit in hits]

    async def execute_pipeline(
        self,
        query: str,
        db_type: str,
        connection_config: Dict[str, Any],
        collection_name: str,
        retrieval_config: Dict[str, Any],
        rerank_config: Dict[str, Any],
        query_vector: Optional[List[float]] = None
    ) -> str:
        try:
            # 1. Embedding (Threaded)
            final_vector = []
            if query_vector and len(query_vector) > 0:
                final_vector = query_vector
            else:
                final_vector = await asyncio.to_thread(self.embedder.embed_query, query)
                if hasattr(final_vector, 'tolist'):
                    final_vector = final_vector.tolist()

            # 2. Retrieval (Threaded)
            raw_results = []
            if db_type.lower() == "qdrant":
                raw_results = await asyncio.to_thread(
                    self._handle_qdrant_sync, 
                    connection_config, 
                    collection_name, 
                    final_vector, 
                    retrieval_config
                )
            else:
                return f"Error: Unsupported db_type '{db_type}'"
                
            if not raw_results:
                return "No documents found."

            # 3. Reranking (Threaded)
            rerank_method = rerank_config.get("method", "none")
            top_n = rerank_config.get("top_n", 3)
            final_results = []
            
            if rerank_method == "cross_encoder":
                pairs = [[query, doc["content"]] for doc in raw_results]
                scores = await asyncio.to_thread(self.reranker.text_classification, pairs)
                
                for i, doc in enumerate(raw_results):
                    doc["rerank_score"] = float(scores[i])
                final_results = sorted(raw_results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
            else:
                final_results = raw_results[:top_n]

            # 4. Format
            output = f"Strategy: {db_type} -> {rerank_method}\n"
            output += "(Used external vector)\n\n" if query_vector else "(Used internal embedding)\n\n"
            for i, doc in enumerate(final_results):
                score = doc.get("rerank_score", doc["score"])
                output += f"--- Result {i+1} (Score: {score:.4f}) ---\n{doc['content']}\n\n"
            return output

        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            return f"Pipeline Error: {str(e)}"

# --- Helper to get/init the engine ---
def get_engine():
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance

# --- Register the Tool ---
@mcp.tool()
async def universal_rag_executor(
    query: str,
    db_type: str,
    connection_config: Dict[str, Any],
    collection_name: str,
    retrieval_config: Dict[str, Any],
    rerank_config: Dict[str, Any],
    query_vector: Optional[List[float]] = Field(default=None, description="Pre-computed embedding vector.")
) -> str:
    """Executes a RAG pipeline asynchronously."""
    try:
        engine = get_engine()
        return await engine.execute_pipeline(
            query, db_type, connection_config, collection_name, retrieval_config, rerank_config, query_vector
        )
    except Exception as e:
        return f"System Error: {str(e)}"

# if __name__ == "__main__":
#     mcp.run()
