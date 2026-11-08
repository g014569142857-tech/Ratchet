"""Hybrid retrieval over the experience store: BM25 (lexical) + vector (semantic).

Why hybrid: BM25 nails exact terminology ("LoRA" must match "LoRA") where vector
search can drift, while vectors catch paraphrases that share no keywords. In
practice the fusion of both beats either alone, which is what
``test_hybrid_beats_lexical_only_on_paraphrase`` guards.

Zero third-party dependencies: BM25 is implemented here (~40 lines) and the
default embedder is a deterministic hashed n-gram vector. Swap in a real
embedding model by implementing the ``Embedder`` protocol - nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ratchet.memory.provenance import Provenance, rule_for
from ratchet.memory.store import TRUSTED, ExperienceStore

_TOKEN_RE = re.compile(r"[a-z0-9_]+|\S")

# Okapi BM25 defaults from the original paper.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase tokeniser that keeps CJK characters as single tokens.

    ``\\S`` as a fallback alternative matters here: CJK text has no spaces, so
    character-level tokens are the only way BM25 can match Chinese experience
    entries at all.
    """
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Okapi BM25 over an in-memory corpus."""

    def __init__(self, *, k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._b = b
        self._docs: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._avg_length = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._docs[doc_id] = counts
        self._doc_lengths[doc_id] = len(tokens)
        for term in counts:
            self._doc_freqs[term] += 1
        total = sum(self._doc_lengths.values())
        self._avg_length = total / len(self._docs) if self._docs else 0.0

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score)]`` sorted by descending BM25 score."""
        if not self._docs:
            return []
        scores: dict[str, float] = defaultdict(float)
        n_docs = len(self._docs)
        for term in set(tokenize(query)):
            doc_freq = self._doc_freqs.get(term, 0)
            if doc_freq == 0:
                continue
            # Standard BM25 IDF; clamped at 0 so a term appearing in every
            # document cannot produce a negative contribution.
            idf = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id, counts in self._docs.items():
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                length = self._doc_lengths[doc_id]
                norm = 1 - self._b + self._b * (length / self._avg_length)
                scores[doc_id] += idf * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(doc_id, max(0.0, score)) for doc_id, score in ranked[:top_k]]


class Embedder(Protocol):
    """Any callable text-to-vector with a similarity helper."""

    def embed(self, text: str) -> Sequence[float]: ...

    def similarity(self, a: str, b: str) -> float: ...


class HashedEmbedder:
    """Deterministic hashed character-n-gram embedder.

    No network, no model download, no API key - so CI and offline users get the
    same behaviour. Quality is well below a real embedding model; that is why
    the class exists behind a protocol rather than being baked into the retriever.
    """

    def __init__(self, *, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        normalised = " ".join((text or "").lower().split())
        for token in tokenize(normalised):
            padded = f"${token}$"
            for i in range(max(1, len(padded) - self.ngram + 1)):
                gram = padded[i : i + self.ngram]
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed(a), self.embed(b)
        # strict=True: a length mismatch means the embedder is broken. Silently
        # truncating would produce a plausible-looking but wrong similarity score.
        return sum(x * y for x, y in zip(va, vb, strict=True))


@dataclass(frozen=True)
class RetrievalHit:
    """One recalled experience, carrying provenance so downstream code can judge it."""

    id: str
    content: str
    task_type: str | None
    provenance: Provenance
    kind: str
    score: float
    lexical_score: float
    semantic_score: float
    trust_level: str = TRUSTED


class HybridRetriever:
    """Fuse BM25 and vector similarity, then apply provenance and budget limits."""

    def __init__(
        self,
        *,
        store: ExperienceStore,
        embedder: Embedder | None = None,
        lexical_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be within [0, 1]")
        self.store = store
        self.embedder = embedder or HashedEmbedder()
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_chars: int | None = None,
    ) -> list[RetrievalHit]:
        """Recall the most relevant active experiences for ``query``.

        Only active rows are considered: a soft-deleted or auto-deactivated
        (poisoned) entry must never be recalled again.
        """
        experiences = self.store.list_active()
        if not experiences:
            return []

        index = BM25Index()
        for exp in experiences:
            index.add(exp.id, exp.content)

        # Over-fetch lexically so the fusion step has candidates to re-rank.
        candidate_ids = {doc_id for doc_id, _ in index.search(query, top_k=top_k * 5)}
        candidate_ids |= {exp.id for exp in experiences}

        query_vector = self.embedder.embed(query)
        lexical_by_id = dict(index.search(query, top_k=top_k * 5))
        max_lexical = max(lexical_by_id.values(), default=0.0) or 1.0

        by_id = {exp.id: exp for exp in experiences}
        hits: list[RetrievalHit] = []
        for doc_id in candidate_ids:
            candidate = by_id.get(doc_id)
            if candidate is None:
                continue
            exp = candidate
            lexical = lexical_by_id.get(doc_id, 0.0) / max_lexical
            doc_vector = self.embedder.embed(exp.content)
            semantic = max(
                0.0, sum(a * b for a, b in zip(query_vector, doc_vector, strict=True))
            )
            fused = self.lexical_weight * lexical + (1 - self.lexical_weight) * semantic
            # Provenance priority is added as a small deterministic tie-break so
            # human-authored rules win ties without drowning out relevance.
            tie_break = rule_for(exp.provenance).priority / 1000.0
            hits.append(
                RetrievalHit(
                    id=exp.id,
                    content=exp.content,
                    task_type=exp.task_type,
                    provenance=exp.provenance,
                    kind=exp.kind,
                    score=fused + tie_break,
                    lexical_score=lexical,
                    semantic_score=semantic,
                    trust_level=exp.trust_level,
                )
            )

        hits.sort(key=lambda h: (-h.score, h.id))
        ranked = hits[:top_k]

        if max_chars is None:
            return ranked

        # Enforce the context budget: this is what stops the experience panel
        # from crowding out the user's actual request.
        budgeted: list[RetrievalHit] = []
        used = 0
        for hit in ranked:
            if used + len(hit.content) > max_chars:
                continue
            budgeted.append(hit)
            used += len(hit.content)
        return budgeted
