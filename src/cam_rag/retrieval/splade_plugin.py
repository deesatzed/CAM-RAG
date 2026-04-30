"""SPLADE retriever plugin for learned sparse retrieval.

SPLADE (SParse Lexical AnD Expansion) learns sparse, high-dimensional
representations where each dimension corresponds to a vocabulary token.
The model produces token-level importance weights via:

    ReLU(log(1 + ReLU(MLM_logits)))

followed by max-pooling over the sequence length, yielding a sparse
vector in ``(vocab_size,)`` space.  Document and query vectors are
compared by dot-product, giving an efficient and interpretable learned
sparse retrieval method.

This plugin implements the ``RetrieverPlugin`` protocol from
``cam_rag.retrieval.plugin`` so it can participate in N-way RRF fusion
alongside BM25 and dense retrievers.

References
----------
- Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for
  Information Retrieval", 2022.
- Default model: ``naver/splade-cocondenser-ensembledistil``

Example usage::

    from cam_rag.retrieval.splade_plugin import SPLADEPlugin

    plugin = SPLADEPlugin()  # loads lazily on first use
    plugin.index(documents)
    results = plugin.retrieve("search query", k=10)
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
_DEFAULT_BATCH_SIZE = 32


def _auto_device() -> str:
    """Select the best available PyTorch device for inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class SPLADEPlugin:
    """Learned sparse retriever using SPLADE models from HuggingFace.

    Implements the ``RetrieverPlugin`` protocol so it can be registered
    with the pipeline and participate in N-way RRF fusion.

    The underlying transformers model and tokenizer are loaded lazily
    on the first call to ``index()`` or ``retrieve()``.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.  Defaults to
        ``naver/splade-cocondenser-ensembledistil``.
    device : str or None
        PyTorch device string.  Auto-detects CUDA/MPS/CPU when ``None``.
    batch_size : int
        Number of documents to process per forward pass during indexing.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name_or_path
        self._device_requested = device
        self._batch_size = batch_size

        # Lazily initialised
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = ""

        # Index state
        self._documents: list[RetrievalDocument] = []
        self._doc_vectors: Any = None  # torch.Tensor (num_docs, vocab_size)

    @property
    def name(self) -> str:
        return "splade"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the SPLADE model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        if importlib.util.find_spec("torch") is None:
            raise ImportError(
                "torch is required for SPLADEPlugin. "
                "Install with: pip install torch"
            )
        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                "transformers is required for SPLADEPlugin. "
                "Install with: pip install transformers"
            )

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._device = self._device_requested or _auto_device()

        logger.info(
            "loading SPLADE model %s on %s",
            self._model_name,
            self._device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            "SPLADE model loaded: vocab_size=%d",
            self._model.config.vocab_size,
        )

    # ------------------------------------------------------------------
    # SPLADE sparse encoding
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: list[str]) -> Any:
        """Encode a list of texts into SPLADE sparse vectors.

        Returns a ``torch.Tensor`` of shape ``(len(texts), vocab_size)``
        with non-negative sparse weights.
        """
        import torch

        self._ensure_loaded()

        all_vectors: list[Any] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]

            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}

            with torch.no_grad():
                output = self._model(**tokens)

            # SPLADE sparse vector:  max(log(1 + ReLU(logits)), dim=seq_len)
            # output.logits shape: (batch, seq_len, vocab_size)
            sparse_vecs = torch.max(
                torch.log1p(torch.relu(output.logits)),
                dim=1,
            ).values  # (batch, vocab_size)

            all_vectors.append(sparse_vecs.cpu())

        return torch.cat(all_vectors, dim=0)  # (num_texts, vocab_size)

    # ------------------------------------------------------------------
    # RetrieverPlugin protocol
    # ------------------------------------------------------------------

    def index(self, documents: list[RetrievalDocument]) -> None:
        """Build the in-memory sparse index from documents.

        Encodes all document texts through the SPLADE model in batches
        and stores the resulting sparse vectors.
        """
        self._documents = list(documents)
        if not documents:
            self._doc_vectors = None
            return

        texts = [doc.text for doc in documents]
        self._doc_vectors = self._encode_texts(texts)

        logger.info(
            "SPLADE index built: %d documents, vector shape %s",
            len(documents),
            tuple(self._doc_vectors.shape),
        )

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k documents by SPLADE dot-product similarity.

        Encodes the query through the SPLADE model, computes dot-product
        scores against all indexed document vectors, and returns the
        top-k results sorted by descending score.
        """
        if not query or not query.strip():
            return []

        if self._doc_vectors is None or len(self._documents) == 0:
            return []

        import torch

        # Encode query
        query_vec = self._encode_texts([query])  # (1, vocab_size)

        # Dot-product scores: (1, vocab_size) @ (vocab_size, num_docs) -> (1, num_docs)
        scores = torch.matmul(query_vec, self._doc_vectors.t()).squeeze(0)  # (num_docs,)

        # Get top-k indices
        actual_k = min(k, len(self._documents))
        top_scores, top_indices = torch.topk(scores, actual_k)

        results: list[RetrievalResult] = []
        for rank, (idx, score_val) in enumerate(
            zip(top_indices.tolist(), top_scores.tolist(), strict=True), start=1
        ):
            doc = self._documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=score_val,
                    rank=rank,
                    metadata=dict(doc.metadata),
                )
            )

        return results
