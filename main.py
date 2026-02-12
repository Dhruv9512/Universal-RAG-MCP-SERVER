from typing_extensions import Literal
from pydantic import Field

from mcp.server.fastmcp import FastMCP

import logging

from pydamic import RetrievalConfig, RerankConfig



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
    from rag.engine import RAGEngine
    from utility import resolve_connection
    # -----------------------------
    # Initialize RAG engine
    # -----------------------------
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        logger.exception("Failed to initialize RAG engine")
    # -----------------------------
    # Resolve connection safely
    # -----------------------------
    try:
        connection_config = resolve_connection(db_type, connection_profile)
    except ValueError as e:
        logger.error(str(e))
        return str(e)
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


if __name__ == "__main__":
    
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
    # mcp.run()
    
