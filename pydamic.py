
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field
from enum import Enum
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