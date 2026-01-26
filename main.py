import logging
from typing import Dict, Any, Optional, List
from typing_extensions import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from rag.engine import RAGEngine


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
# DB-SPECIFIC CONNECTION PROFILES (SERVER ONLY)
# --------------------------------------------------
CONNECTION_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "qdrant": {
        "default": {
            "host": "localhost",
            "port": 6333,
            "api_key": None,
        },
        "prod": {
            "host": "qdrant.prod.internal",
            "port": 6333,
            "api_key": None,
        },
    },

    "milvus": {
        "default": {
            "host": "localhost",
            "port": 19530,
            "user": None,
            "password": None,
        },
    },

    "pinecone": {
        "default": {
            "api_key": "ENV:PINECONE_API_KEY",
            "environment": "us-east-1",
            "index_name": "docs-index",
        },
    },
}


# --------------------------------------------------
# CONFIG MODELS (STRICT + ENUM-SAFE)
# --------------------------------------------------
class RetrievalConfig(BaseModel):
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of documents to retrieve before reranking."
    )


class RerankConfig(BaseModel):
    method: Literal["none", "cross_encoder"] = Field(
        default="cross_encoder",
        description="Reranking method to apply."
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of documents after reranking."
    )


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def resolve_connection(db_type: str, profile: str) -> Dict[str, Any]:
    """
    Resolve DB-specific connection config safely.
    """
    try:
        return CONNECTION_PROFILES[db_type][profile]
    except KeyError:
        raise ValueError(
            f"Invalid connection profile '{profile}' for db '{db_type}'"
        )


# --------------------------------------------------
# MCP TOOL
# --------------------------------------------------
@mcp.tool(
    name="rag_retrieve_and_rerank",
    description=(
        "Retrieve documents from a knowledge base and optionally rerank them. "
        "Use this tool when the question requires external knowledge or factual grounding."
    )
)
def rag_retrieve_and_rerank(
    query: str = Field(
        description="The user's question or search query."
    ),

    db_type: Literal["qdrant", "milvus", "pinecone"] = Field(
        default="qdrant",
        description="Vector database to use."
    ),

    connection_profile: Literal["default", "prod"] = Field(
        default="default",
        description="Predefined database connection profile."
    ),

    collection_name: str = Field(
        description="Name of the collection / index to search."
    ),

    retrieval_config: RetrievalConfig = RetrievalConfig(),

    rerank_config: RerankConfig = RerankConfig(),
) -> str:
    """
    MCP tool that executes retrieval + optional reranking.
    Returns CONTEXT, not a final answer.
    """

    # -----------------------------
    # Resolve connection safely
    # -----------------------------
    try:
        connection_config = resolve_connection(db_type, connection_profile)
    except ValueError as e:
        logger.error(str(e))
        return str(e)

    # -----------------------------
    # Initialize RAG engine
    # -----------------------------
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        logger.exception("Failed to initialize RAG engine")
        return "RAG engine initialization failed."

    # -----------------------------
    # Execute pipeline
    # -----------------------------
    try:
        return rag_engine.retrieveAndRerankDocuments(
            query=query,
            db_type=db_type,
            connection_config=connection_config,
            collection_name=collection_name,
            retrieval_config=retrieval_config.model_dump(),
            rerank_config=rerank_config.model_dump(),
        )
    except Exception as e:
        logger.exception("RAG pipeline execution failed")
        return f"Pipeline execution error: {str(e)}"


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    # mcp.run(
    #     transport="sse",
    #     host="0.0.0.0",
    #     port=8000,
    # )
    mcp.run()
