"""End-to-end test: does the loop actually improve anything?

This file exists because every other test checks a component in isolation, and a
system whose parts all work can still close no loop at all. The original skeleton
had exactly that failure: the bandit selected an arm, the reward updated the arm's
posterior, and the arm influenced nothing - so the learning was real arithmetic
about an imaginary difference.

The test below makes the outcome depend on the arm, so the only way to pass is
for selection -> prompt -> output -> reward -> posterior to form a genuine circuit.
"""

from __future__ import annotations

import pytest

from ratchet.core.prompt import VARIANT_STRATEGIES
from ratchet.core.provider import Provider
from ratchet.engine import Engine
from ratchet.improve.distiller import pinned_principles
from ratchet.improve.verifier import ObjectiveVerifier, RewardChain

GOOD = "CORRECT"
BAD = "WRONG"

# A distinctive fragment of the ``concise`` strategy. Only the concise arm's
# prompt can contain it, so the provider's answer really is a function of the
# action the bandit chose.
_CONCISE_MARKER = VARIANT_STRATEGIES["concise"][:40]


class OnlyConciseIsRight(Provider):
    """Rewards exactly one strategy, and declines to reflect."""

    name = "scripted"

    def complete(self, prompt: str, **kwargs: object) -> str:
        if "failed interaction" in prompt:
            return "NONE"
        return GOOD if _CONCISE_MARKER in prompt else BAD


class ExactMatch(ObjectiveVerifier):
    name = "exact_match"

    def verify(self, task: str, output: str) -> float | None:
        return 1.0 if output.strip() == GOOD else 0.0


@pytest.fixture
def engine(tmp_path):
    e = Engine(
        db_path=tmp_path / "e2e.db",
        provider=OnlyConciseIsRight(),
        reward_chain=RewardChain(objective=[ExactMatch()]),
        seed=7,
    )
    yield e
    e.close()


# ------------------------------------------------------- the loop closes


def test_the_bandit_learns_to_prefer_the_strategy_that_works(engine):
    for i in range(40):
        engine.run(task=f"task {i}", user_input="give the short answer")

    arms = engine.policy.arms
    assert arms["concise"].pulls > 0
    # The arm that produced the right answer must end up looking best.
    assert arms["concise"].mean() > arms["default"].mean()
    assert arms["concise"].mean() > arms["chain_of_thought"].mean()


def test_learned_preference_changes_what_the_model_is_asked_to_do(engine):
    # Warm up, then confirm the better strategy is the one actually being used.
    for i in range(40):
        engine.run(task=f"task {i}", user_input="give the short answer")

    used = [
        engine.run(task=f"probe {i}", user_input="give the short answer").variant_id
        for i in range(15)
    ]
    assert used.count("concise") > used.count("default")
    assert used.count("concise") > used.count("chain_of_thought")


def test_the_rewarded_arm_produces_high_rewards_consistently(engine):
    for i in range(40):
        engine.run(task=f"task {i}", user_input="give the short answer")

    focused = [
        engine.run(task=f"check {i}", user_input="give the short answer")
        for i in range(15)
    ]
    mean = sum(r.reward.value for r in focused if r.reward.value is not None) / len(focused)
    assert mean > 0.6


def test_no_learning_happens_without_a_reward(engine):
    # An unmeasured interaction must not be recorded as a failure, or the bandit
    # would learn to avoid strategies that were simply never evaluated. The
    # verifier lives on the session, so that is what has to be swapped.
    engine.session.verifier = RewardChain()
    before = {n: (a.alpha, a.beta) for n, a in engine.policy.arms.items()}
    engine.run(task="unmeasured", user_input="hello")
    after = {n: (a.alpha, a.beta) for n, a in engine.policy.arms.items()}
    assert before == after


# -------------------------------------------------- interaction N feeds N+1


def test_a_lesson_learned_is_recalled_on_the_next_similar_task(engine):
    lesson = "When explaining LoRA, mention low-rank adaptation explicitly."
    engine.add_rule(lesson)
    result = engine.run(task="explain LoRA", user_input="briefly")
    assert any(lesson == hit.content for hit in result.hits)


def test_deleting_a_lesson_stops_it_being_recalled(engine):
    lesson = "Always mention low-rank adaptation when discussing LoRA."
    exp_id = engine.add_rule(lesson)
    assert any(lesson == h.content for h in engine.run(task="explain LoRA", user_input="x").hits)

    engine.forget(exp_id)
    assert not any(
        lesson == h.content for h in engine.run(task="explain LoRA", user_input="x").hits
    )


# ------------------------------------------------------ autonomy is opt-in


def test_evolution_is_off_by_default(tmp_path):
    e = Engine(db_path=tmp_path / "off.db", provider=OnlyConciseIsRight(), seed=1)
    try:
        assert e.evolver is None
        for _ in range(25):
            e.run(task="t", user_input="u")
        assert e.stats()["evolved_variants"] == 0
    finally:
        e.close()


def test_evolution_adds_an_arm_when_enabled(tmp_path):
    class Suggests(Provider):
        name = "suggests"

        def complete(self, prompt: str, **kwargs: object) -> str:
            if "failed interaction" in prompt:
                return "NONE"
            if "REVISED INSTRUCTION" in prompt:
                return "Lead with the single most relevant fact."
            return GOOD

    e = Engine(
        db_path=tmp_path / "on.db",
        provider=Suggests(),
        reward_chain=RewardChain(objective=[ExactMatch()]),
        evolve=True,
        evolve_every=5,
        seed=3,
    )
    try:
        for i in range(30):
            e.run(task=f"t{i}", user_input="u")
        assert e.evolver is not None
        assert any(v.origin == "mutation" for v in e.evolver.pool.values())
    finally:
        e.close()


# ------------------------------------------------------------- distillation


def test_distillation_pins_a_persona_block_when_enabled(tmp_path):
    class Compresses(Provider):
        name = "compresses"

        def complete(self, prompt: str, **kwargs: object) -> str:
            if "failed interaction" in prompt:
                return "NONE"
            if "PRINCIPLES" in prompt:
                return "- be concise\n- verify claims"
            return GOOD

    e = Engine(
        db_path=tmp_path / "d.db",
        provider=Compresses(),
        reward_chain=RewardChain(objective=[ExactMatch()]),
        distill_every=5,
        seed=5,
    )
    try:
        # Earn some rules first: a rule needs score and uses before it is worth
        # compressing.
        for i in range(4):
            exp_id = e.store.add(content=f"rule {i}", provenance="auto", kind="rule")
            e.store.record_reward(exp_id, 1.5)
        for i in range(6):
            e.run(task=f"t{i}", user_input="u")

        blocks = pinned_principles(e.store)
        assert len(blocks) >= 1
        assert blocks[0].kind == "distilled"

        # And the pinned block must reach the prompt.
        result = e.run(task="anything", user_input="u")
        assert "distilled principles" in result.prompt
    finally:
        e.close()


def test_deleting_the_persona_stops_it_being_pinned(tmp_path):
    class Compresses(Provider):
        name = "compresses"

        def complete(self, prompt: str, **kwargs: object) -> str:
            if "PRINCIPLES" in prompt:
                return "- be concise"
            if "failed interaction" in prompt:
                return "NONE"
            return GOOD

    e = Engine(
        db_path=tmp_path / "d2.db",
        provider=Compresses(),
        reward_chain=RewardChain(objective=[ExactMatch()]),
        distill_every=3,
        seed=5,
    )
    try:
        for i in range(4):
            exp_id = e.store.add(content=f"rule {i}", provenance="auto", kind="rule")
            e.store.record_reward(exp_id, 1.5)
        for i in range(4):
            e.run(task=f"t{i}", user_input="u")

        block = pinned_principles(e.store)[0]
        e.forget(block.id)
        assert pinned_principles(e.store) == []
        assert "distilled principles" not in e.run(task="x", user_input="u").prompt
    finally:
        e.close()
