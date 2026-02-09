import os
import logging
from typing import Dict, Any, Optional, List
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from rag.strategy.base import Strategy




# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# --------------------------------------------------
# RAG Engine
# --------------------------------------------------
class RAGEngine:
    reranker=None
    # -----------------------------
    # retrieveAndRerankDocuments pipeline
    # -----------------------------
    def retrieveAndRerankDocuments(
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

            # 1. Find which client and vecterstore need to get for retrieval
            raw_docs = Strategy().retriver(
                query,
                connection_config,
                collection_name,
                retrieval_config,
                db_type,
            )
            
            if not raw_docs:
                return "No documents found."

            # 3. Get Re-rank Model
            self._getEmbedderAndReranker()

            # 4. Reranking
            rerank_method = rerank_config.get("method", "none")
            top_n = rerank_config.get("top_n", 3)

            if rerank_method == "cross_encoder":
                pairs = [[query, d["content"]] for d in raw_docs]

                scores =  self.reranker.text_classification(pairs)

                for i, d in enumerate(raw_docs):
                    d["rerank_score"] = float(scores[i])

                raw_docs.sort(
                    key=lambda x: x["rerank_score"],
                    reverse=True,
                )

            final_docs = raw_docs[:top_n]

            # 5. Output formatting
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
    
    # -----------------------------Helpers----------------------
    def _getEmbedderAndReranker(self):
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set")

        # if self.embedder is not None:        
        #     try:
        #         self.embedder = HuggingFaceEndpointEmbeddings(
        #             model="sentence-transformers/all-MiniLM-L6-v2",
        #             huggingfacehub_api_token=api_token,
        #         )
        #     except Exception as e:
        #         logger.error(f"Failed to initialize embedder: {e}")

        if self.reranker is None:
            try:
                # HF Inference API cross-encoder (REMOTE, blocking HTTP)
                self.reranker = InferenceClient(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    token=api_token,
                )
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
