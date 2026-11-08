"""End-to-end tests: the full memory-reflection-selection loop.

These run against a scripted provider, so CI needs no API key, no network and no
GPU. The point is to prove the closed loop actually closes: an interaction is
scored, the score updates the policy, low scores produce reflections, and the
next interaction recalls what was learned.
"""

import pytest

from ratchet.core.policy import ThompsonPolicy
from ratchet.core.prompt import PromptAssembler
from ratchet.core.provider import EchoProvider, Provider, ProviderError
from ratchet.core.session import Session, SessionConfig, SessionResult
from ratchet.improve.reflector import Reflector
from ratchet.improve.verifier import ObjectiveVerifier, RewardChain, RewardSource
from ratchet.memory.provenance import Provenance
from ratchet.memory.retriever import HashedEmbedder, HybridRetriever
from ratchet.memory.store import ExperienceStore


class _ScriptedProvider(Provider):
    """Returns queued outputs and records every prompt it received."""

    name = "scripted"

    def __init__(self, *outputs: str) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        if not self._outputs:
            raise ProviderError("scripted provider exhausted")
        return self._outputs.pop(0)


class _FixedVerifier(ObjectiveVerifier):
    name = "fixed"

    def __init__(self, score: float) -> None:
        self._score = score

    def verify(self, task: str, output: str) -> float | None:
        return self._score


@pytest.fixture()
def store(tmp_path):
    s = ExperienceStore(tmp_path / "ratchet.db")
    yield s
    s.close()


@pytest.fixture()
def session(store):
    provider = _ScriptedProvider("first answer", "second answer", "third answer")
    return Session(
        store=store,
        provider=provider,
        retriever=HybridRetriever(store=store, embedder=HashedEmbedder()),
        policy=ThompsonPolicy(seed=3),
        verifier=RewardChain(objective=[_FixedVerifier(1.0)]),
        assembler=PromptAssembler(base_system="You are helpful."),
        reflector=Reflector(store=store, model=provider, threshold=0.4),
        config=SessionConfig(seed=3),
    ), provider


class TestSingleInteraction:
    def test_returns_the_model_output(self, session):
        s, _ = session
        result = s.run(task="write a poem", user_input="about the sea")
        assert result.output == "first answer"

    def test_result_is_a_session_result(self, session):
        s, _ = session
        assert isinstance(s.run(task="t", user_input="u"), SessionResult)

    def test_result_carries_reward(self, session):
        s, _ = session
        result = s.run(task="t", user_input="u")
        assert result.reward.value == pytest.approx(1.0)
        assert result.reward.source is RewardSource.OBJECTIVE

    def test_interaction_is_persisted_as_an_event(self, store, session):
        s, _ = session
        s.run(task="t", user_input="u")
        assert len(store.list_events()) == 1

    def test_event_records_input_and_output(self, store, session):
        s, _ = session
        s.run(task="write a poem", user_input="about the sea")
        event = store.list_events()[0]
        assert event.task == "write a poem"
        assert event.user_input == "about the sea"
        assert event.output == "first answer"

    def test_event_records_reward(self, store, session):
        s, _ = session
        s.run(task="t", user_input="u")
        assert store.list_events()[0].reward == pytest.approx(1.0)

    def test_selected_variant_is_recorded(self, store, session):
        s, _ = session
        result = s.run(task="t", user_input="u")
        assert store.list_events()[0].variant_id == result.variant_id

    def test_recalled_experience_ids_are_recorded(self, store, session):
        s, _ = session
        store.add(content="a useful rule", provenance=Provenance.USER_RULE)
        result = s.run(task="useful rule task", user_input="u")
        assert store.list_events()[0].retrieved_ids == tuple(result.retrieved_ids)

    def test_user_input_reaches_the_provider(self, session):
        s, provider = session
        s.run(task="t", user_input="my exact request")
        assert "my exact request" in provider.prompts[0]


class TestRewardUpdatesPolicy:
    def test_selected_variant_gets_reward_update(self, session):
        s, _ = session
        result = s.run(task="t", user_input="u")
        assert s.policy.arms[result.variant_id].pulls == 1

    def test_good_reward_raises_variant_alpha(self, session):
        s, _ = session
        result = s.run(task="t", user_input="u")
        assert s.policy.arms[result.variant_id].alpha == pytest.approx(2.0)


class TestReflectionOnFailure:
    def test_low_reward_produces_a_stored_rule(self, store):
        # Production change that would break this: skipping the reflect call in
        # Session.run() when reward is below threshold.
        provider = _ScriptedProvider("bad output", "Prefer small functions.")
        session = Session(
            store=store,
            provider=provider,
            retriever=HybridRetriever(store=store, embedder=HashedEmbedder()),
            policy=ThompsonPolicy(seed=3),
            verifier=RewardChain(objective=[_FixedVerifier(0.0)]),
            assembler=PromptAssembler(base_system="You are helpful."),
            reflector=Reflector(store=store, model=provider, threshold=0.4),
            config=SessionConfig(seed=3),
        )
        session.run(task="t", user_input="u")
        rules = [e for e in store.list_active() if e.kind == "rule"]
        assert len(rules) == 1
        assert rules[0].provenance is Provenance.AUTO

    def test_high_reward_produces_no_rule(self, store, session):
        s, _ = session
        s.run(task="t", user_input="u")
        assert [e for e in store.list_active() if e.kind == "rule"] == []


class TestCompoundingMemory:
    def test_second_interaction_recalls_first_lesson(self, store):
        # This is the compounding claim the whole project rests on: interaction
        # N can retrieve what interaction N-1 learned.
        provider = _ScriptedProvider(
            "bad output",                   # 1st call: first answer (fails)
            "Validate JSON before parse.",  # 2nd call: reflection on failure #1
            "good output",                  # 3rd call: second answer
            "Reuse the validated parser.",  # 4th call: reflection on failure #2
        )
        session = Session(
            store=store,
            provider=provider,
            retriever=HybridRetriever(store=store, embedder=HashedEmbedder()),
            policy=ThompsonPolicy(seed=3),
            verifier=RewardChain(objective=[_FixedVerifier(0.0)]),
            assembler=PromptAssembler(base_system="You are helpful."),
            reflector=Reflector(store=store, model=provider, threshold=0.4),
            config=SessionConfig(seed=3),
        )
        session.run(task="parse the JSON config file", user_input="do it")
        result = session.run(task="parse the JSON config file", user_input="do it again")
        assert len(result.retrieved_ids) == 1
        # The learned rule must be visible in the prompt assembled for THIS
        # interaction. result.prompt is that prompt; provider.prompts[-1] would
        # be the *next* reflection call, which is not what we are asserting.
        assert "Validate JSON before parse." in result.prompt


class TestExplicitUserFeedback:
    def test_thumbs_down_overrides_objective_reward(self, store):
        provider = _ScriptedProvider("answer", "Be more concise.")
        session = Session(
            store=store,
            provider=provider,
            retriever=HybridRetriever(store=store, embedder=HashedEmbedder()),
            policy=ThompsonPolicy(seed=3),
            verifier=RewardChain(objective=[_FixedVerifier(1.0)]),
            assembler=PromptAssembler(base_system="You are helpful."),
            reflector=Reflector(store=store, model=provider, threshold=0.4),
            config=SessionConfig(seed=3),
        )
        result = session.run(task="t", user_input="u", user_feedback=-1)
        assert result.reward.value < 0.5
        assert result.reward.source is RewardSource.USER

    def test_thumbs_up_yields_high_reward(self, session):
        s, _ = session
        result = s.run(task="t", user_input="u", user_feedback=1)
        assert result.reward.value == pytest.approx(1.0)


class TestEchoProvider:
    def test_echo_provider_needs_no_network(self):
        provider = EchoProvider()
        assert "hello" in provider.complete("hello")

    def test_echo_provider_is_a_provider(self):
        assert isinstance(EchoProvider(), Provider)

    def test_provider_is_abstract(self):
        with pytest.raises(TypeError):
            Provider()  # type: ignore[abstract]


class TestCostTracking:
    def test_event_records_token_counts(self, store, session):
        s, _ = session
        s.run(task="t", user_input="u")
        event = store.list_events()[0]
        assert event.tokens_in > 0
        assert event.tokens_out > 0

    def test_latency_is_recorded(self, store, session):
        s, _ = session
        s.run(task="t", user_input="u")
        assert store.list_events()[0].latency_ms >= 0

    def test_session_reports_running_totals(self, session):
        s, _ = session
        s.run(task="t", user_input="u")
        s.run(task="t", user_input="u")
        stats = s.stats()
        assert stats["interactions"] == 2
        assert stats["tokens_in"] > 0


class TestTokenEstimate:
    def test_chinese_is_not_undercounted_like_latin_text(self):
        from ratchet.core.session import _approx_tokens

        zh = "请用中文解释一下低秩适配器的原理"  # 16 CJK characters
        # A flat len // 4 would say 4 tokens; CJK is closer to 1-2 per character.
        assert _approx_tokens(zh) >= len(zh)

    def test_latin_text_keeps_the_four_chars_per_token_rule(self):
        from ratchet.core.session import _approx_tokens

        assert _approx_tokens("a" * 40) == 10

    def test_empty_is_zero_and_tiny_is_at_least_one(self):
        from ratchet.core.session import _approx_tokens

        assert _approx_tokens("") == 0
        assert _approx_tokens("a") == 1
