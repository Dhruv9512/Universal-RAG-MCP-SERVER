import asyncio
import os
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from huggingface_hub import InferenceClient
from pydantic import Field
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
# Initialize the MCP Server
mcp = FastMCP("Universal RAG Server")

class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine & Loading Models... (This may take a moment)")
        self.embedder = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        self.reranker = InferenceClient(model="cross-encoder/ms-marco-MiniLM-L-6-v2",token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        print("Models loaded successfully.")

    def _handle_qdrant(self, connection: Dict, collection: str, vector: List[float], config: Dict) -> List[Dict]:
        """Internal helper to handle Qdrant connections"""
        from qdrant_client import QdrantClient
        
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
        """
        Executes the RAG pipeline logic.
        """
        try:
            # 1. Determine which Vector to use
            final_vector = []
            
            if query_vector is not None and len(query_vector) > 0:
                final_vector = query_vector
            else:
                # Run embedding in thread pool to avoid blocking async event loop
                final_vector = self.embedder.embed_query(query)
                final_vector = final_vector.tolist()

            # 2. Route to DB
            raw_results = []
            if db_type.lower() == "qdrant":
                raw_results = self._handle_qdrant(connection_config, collection_name, final_vector, retrieval_config)
            else:
                return f"Error: Unsupported db_type '{db_type}'"
                
            if not raw_results:
                return "No documents found."

            # 3. Reranking
            rerank_method = rerank_config.get("method", "none")
            top_n = rerank_config.get("top_n", 3)
            final_results = []
            
            if rerank_method == "cross_encoder":
                pairs = [[query, doc["content"]] for doc in raw_results]
                
                # Run reranking in thread pool
                scores = self.reranker.text_classification(pairs)
                
                for i, doc in enumerate(raw_results):
                    doc["rerank_score"] = float(scores[i])
                final_results = sorted(raw_results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
            else:
                final_results = raw_results[:top_n]

            # 4. Format Output
            output = f"Strategy: {db_type} -> {rerank_method}\n"
            output += "(Used external vector)\n\n" if query_vector else "(Used internal embedding)\n\n"
                
            for i, doc in enumerate(final_results):
                score = doc.get("rerank_score", doc["score"])
                output += f"--- Result {i+1} (Score: {score:.4f}) ---\n{doc['content']}\n\n"
                
            return output

        except Exception as e:
            return f"Pipeline Error: {str(e)}"

# --- Instantiate the Engine ---
rag_engine = RAGEngine()

# --- Register the Tool with MCP ---
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
    """
    Executes a RAG pipeline asynchronously.
    
    Args:
        query: The text query to search for.
        db_type: The database type ('qdrant' or 'milvus').
        connection_config: Dictionary containing 'url', 'api_key', etc.
        collection_name: The name of the vector collection.
        retrieval_config: Config for retrieval (e.g., {'k': 10}).
        rerank_config: Config for reranking (e.g., {'method': 'cross_encoder', 'top_n': 3}).
        query_vector: Optional pre-computed vector.
    """
    return await rag_engine.execute_pipeline(
        query, db_type, connection_config, collection_name, retrieval_config, rerank_config, query_vector
    )

if __name__ == "__main__":
    # mcp.run()
    mcp.run(transport="sse", host="0.0.0.0", port=8000)