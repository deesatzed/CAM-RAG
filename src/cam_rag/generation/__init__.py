"""LLM generation backends for RAG answer synthesis."""

from cam_rag.generation.ollama import OllamaGenerationBackend
from cam_rag.generation.openrouter import OpenRouterGenerationBackend
from cam_rag.generation.prompt import build_rag_prompt

__all__ = [
    "OllamaGenerationBackend",
    "OpenRouterGenerationBackend",
    "build_rag_prompt",
]
