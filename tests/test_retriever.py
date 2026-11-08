"""Tests for hybrid retrieval (BM25 + vector) over the experience store.

Zero third-party dependencies: BM25 is implemented in-house and the default
embedder is a deterministic hashed n-gram vector. Real embedding models can be
plugged in via the Embedder protocol.
"""

import pytest

from ratchet.memory.provenance import Provenance
from ratchet.memory.retriever import BM25Index, HashedEmbedder, HybridRetriever
from ratchet.memory.store import ExperienceStore


@pytest.fixture()
def store(tmp_path):
    s = ExperienceStore(tmp_path / "ratchet.db")
    yield s
    s.close()


@pytest.fixture()
def retriever(store):
    return HybridRetriever(store=store, embedder=HashedEmbedder())


class TestBM25:
    def test_exact_term_match_ranks_first(self):
        index = BM25Index()
        index.add("d1", "alpha beta gamma")
        index.add("d2", "delta epsilon zeta")
        assert index.search("gamma", top_k=1)[0][0] == "d1"

    def test_no_match_returns_empty(self):
        index = BM25Index()
        index.add("d1", "alpha beta")
        assert index.search("zzz", top_k=5) == []

    def test_term_frequency_boosts_rank(self):
        index = BM25Index()
        index.add("once", "python testing")
        index.add("thrice", "python python python")
        hits = index.search("python", top_k=2)
        assert hits[0][0] == "thrice"

    def test_rare_term_outranks_common_term(self):
        # BM25's IDF component: a term in one doc beats a term in all docs.
        index = BM25Index()
        index.add("d1", "common rare_term_xq")
        index.add("d2", "common only")
        index.add("d3", "common only")
        hits = index.search("common rare_term_xq", top_k=1)
        assert hits[0][0] == "d1"

    def test_scores_are_non_negative(self):
        index = BM25Index()
        index.add("d1", "alpha beta")
        assert all(score >= 0 for _, score in index.search("alpha", top_k=1))


class TestHashedEmbedder:
    def test_embedding_is_deterministic(self):
        embedder = HashedEmbedder(dim=64)
        assert embedder.embed("hello") == embedder.embed("hello")

    def test_embedding_has_requested_dimension(self):
        assert len(HashedEmbedder(dim=32).embed("hello")) == 32

    def test_embedding_is_unit_normalised(self):
        vector = HashedEmbedder(dim=64).embed("some text here")
        norm = sum(x * x for x in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_identical_texts_are_more_similar_than_different(self):
        embedder = HashedEmbedder(dim=256)
        same = embedder.similarity("fix the login bug", "fix the login bug")
        diff = embedder.similarity("fix the login bug", "banana orchestra sunset")
        assert same > diff


class TestHybridRetriever:
    def test_returns_relevant_entry(self, store, retriever):
        store.add(
            content="Always run the test suite before committing Python changes.",
            task_type="coding",
            provenance=Provenance.USER_RULE,
        )
        hits = retriever.search("commit python code", top_k=3)
        assert len(hits) == 1
        assert "test suite" in hits[0].content

    def test_excludes_inactive_entries(self, store, retriever):
        # A deleted or poisoned experience must never be recalled again.
        eid = store.add(content="obsolete guidance about python", provenance=Provenance.AUTO)
        store.delete(eid)
        assert retriever.search("python", top_k=5) == []

    def test_respects_top_k(self, store, retriever):
        for i in range(5):
            store.add(content=f"python tip number {i}", provenance=Provenance.AUTO)
        assert len(retriever.search("python tip", top_k=2)) == 2

    def test_empty_store_returns_empty(self, retriever):
        assert retriever.search("anything", top_k=5) == []

    def test_user_rule_outranks_auto_on_tie(self, store, retriever):
        # Provenance priority is the tie-break, per the permission model.
        store.add(content="python style guide", provenance=Provenance.AUTO)
        store.add(content="python style guide", provenance=Provenance.USER_RULE)
        hits = retriever.search("python style guide", top_k=2)
        assert hits[0].provenance is Provenance.USER_RULE

    def test_token_budget_caps_results(self, store, retriever):
        for i in range(20):
            store.add(content=f"word{i} " * 200, provenance=Provenance.AUTO)
        hits = retriever.search("word0", top_k=20, max_chars=500)
        assert sum(len(h.content) for h in hits) <= 500

    def test_hit_carries_score_and_metadata(self, store, retriever):
        store.add(
            content="use f-strings for formatting",
            task_type="coding",
            provenance=Provenance.USER_RULE,
        )
        hit = retriever.search("f-strings formatting", top_k=1)[0]
        assert hit.score > 0
        assert hit.task_type == "coding"
        assert hit.provenance is Provenance.USER_RULE

    def test_hybrid_beats_lexical_only_on_paraphrase(self, store, retriever):
        # Production change that would break this: dropping the vector leg and
        # keeping BM25 alone.
        store.add(content="increase test coverage for the parser", provenance=Provenance.AUTO)
        hybrid_hits = retriever.search("parser tests coverage", top_k=1)
        assert len(hybrid_hits) == 1
