"""Cross-encoder reranking backends for evidence re-scoring."""

from cam_rag.reranking.ollama import OllamaRerankerBackend
from cam_rag.reranking.openrouter import OpenRouterRerankerBackend
from cam_rag.reranking.prompt import build_rerank_prompt, parse_rerank_scores

__all__ = [
    "LocalCrossEncoderBackend",
    "OllamaRerankerBackend",
    "OpenRouterRerankerBackend",
    "build_rerank_prompt",
    "parse_rerank_scores",
]


def __getattr__(name: str):
    if name == "LocalCrossEncoderBackend":
        from cam_rag.reranking.cross_encoder import LocalCrossEncoderBackend
        return LocalCrossEncoderBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
