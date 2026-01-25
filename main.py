import asyncio
import os
import logging
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from huggingface_hub import InferenceClient
from pydantic import Field
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import QdrantClient


# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# MCP Server
# --------------------------------------------------
mcp = FastMCP("Universal RAG Server")


# --------------------------------------------------
# RAG Engine
# --------------------------------------------------
class RAGEngine:
    def __init__(self):
        logger.info("Initializing RAG Engine & Loading Models...")

        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set")

        self.embedder = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=api_token,
        )

        # HF Inference API cross-encoder (REMOTE, blocking HTTP)
        self.reranker = InferenceClient(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            token=api_token,
        )

        logger.info("Models loaded successfully.")

    # -----------------------------
    # Sync Qdrant helper
    # -----------------------------
    def _qdrant_search_sync(
        self,
        connection: Dict[str, Any],
        collection: str,
        vector: List[float],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        client = QdrantClient(
            url=connection.get("url"),
            api_key=connection.get("api_key"),
        )

        hits = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=config.get("k", 10),
            query_filter=config.get("filter"),
        )

        return [
            {
                "content": hit.payload.get("text", ""),
                "metadata": hit.payload,
                "score": hit.score,
            }
            for hit in hits
        ]

    # -----------------------------
    # Async pipeline
    # -----------------------------
    async def execute_pipeline(
        self,
        query: str,
        db_type: str,
        connection_config: Dict[str, Any],
        collection_name: str,
        retrieval_config: Dict[str, Any],
        rerank_config: Dict[str, Any],
        query_vector: Optional[List[float]] = None,
    ) -> str:
        try:
            # 1. Embedding
            if query_vector:
                vector = query_vector
            else:
                vector = await asyncio.to_thread(
                    self.embedder.embed_query, query
                )
                vector = vector.tolist()

            # 2. Retrieval
            if db_type.lower() != "qdrant":
                return f"Unsupported db_type: {db_type}"

            raw_docs = await asyncio.to_thread(
                self._qdrant_search_sync,
                connection_config,
                collection_name,
                vector,
                retrieval_config,
            )

            if not raw_docs:
                return "No documents found."

            # 3. Reranking
            rerank_method = rerank_config.get("method", "none")
            top_n = rerank_config.get("top_n", 3)

            if rerank_method == "cross_encoder":
                pairs = [[query, d["content"]] for d in raw_docs]

                scores = await asyncio.to_thread(
                    self.reranker.text_classification,
                    pairs,
                )

                for i, d in enumerate(raw_docs):
                    d["rerank_score"] = float(scores[i])

                raw_docs.sort(
                    key=lambda x: x["rerank_score"],
                    reverse=True,
                )

            final_docs = raw_docs[:top_n]

            # 4. Output formatting
            output = f"Strategy: {db_type} -> {rerank_method}\n"
            output += "(Used external vector)\n\n" if query_vector else "(Used internal embedding)\n\n"

            for i, doc in enumerate(final_docs):
                score = doc.get("rerank_score", doc["score"])
                output += f"--- Result {i+1} (Score: {score:.4f}) ---\n"
                output += f"{doc['content']}\n\n"

            return output

        except Exception as e:
            logger.exception("Pipeline failed")
            return f"Pipeline Error: {str(e)}"


# --------------------------------------------------
# MCP Tool
# --------------------------------------------------
@mcp.tool()
async def universal_rag_executor(
    query: str,
    db_type: str,
    connection_config: Dict[str, Any],
    collection_name: str,
    retrieval_config: Dict[str, Any],
    rerank_config: Dict[str, Any],
    query_vector: Optional[List[float]] = Field(default=None),
) -> str:
    
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        rag_engine = None
        
    if not rag_engine:
        return "RAG engine not initialized"

    return await rag_engine.execute_pipeline(
        query,
        db_type,
        connection_config,
        collection_name,
        retrieval_config,
        rerank_config,
        query_vector,
    )


# --------------------------------------------------
# IMPORTANT
# --------------------------------------------------
# ❌ DO NOT call mcp.run() when using MCP CLI
# The CLI already manages the asyncio event loop.
#
# If you run this file directly (python file.py),
# then uncomment below.
#
# if __name__ == "__main__":
#     mcp.run(transport="sse", host="0.0.0.0", port=8000)
