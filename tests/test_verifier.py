"""Tests for the verifier / reward chain.

This is the highest-stakes module in Ratchet: the reward function decides what
the system optimises for. Two failure modes are explicitly guarded against here:

  * reward hacking  - the judge preferring verbose or keyword-stuffed answers
  * judge dominance - a subjective LLM opinion overriding objective evidence

Each test names the production change that would make it fail.
"""

from dataclasses import FrozenInstanceError

import pytest

from ratchet.improve.verifier import (
    LengthPenalty,
    LLMJudgeVerifier,
    ObjectiveVerifier,
    Reward,
    RewardChain,
    RewardSource,
)


class _AlwaysPass(ObjectiveVerifier):
    name = "always_pass"

    def verify(self, task: str, output: str) -> float | None:
        return 1.0


class _AlwaysFail(ObjectiveVerifier):
    name = "always_fail"

    def verify(self, task: str, output: str) -> float | None:
        return 0.0


class _NotApplicable(ObjectiveVerifier):
    name = "not_applicable"

    def verify(self, task: str, output: str) -> float | None:
        return None


class TestRewardSources:
    def test_three_sources_exist(self):
        # Fails if the priority chain loses a tier.
        assert {s.value for s in RewardSource} == {"objective", "user", "judge"}

    def test_objective_outranks_user_outranks_judge(self):
        assert RewardSource.OBJECTIVE.weight > RewardSource.USER.weight
        assert RewardSource.USER.weight > RewardSource.JUDGE.weight


class TestObjectiveVerifier:
    def test_applicable_verifier_contributes_full_reward(self):
        chain = RewardChain(objective=[_AlwaysPass()])
        reward = chain.score(task="t", output="o")
        assert reward.value == pytest.approx(1.0)
        assert reward.source is RewardSource.OBJECTIVE

    def test_failing_verifier_yields_zero(self):
        chain = RewardChain(objective=[_AlwaysFail()])
        assert chain.score(task="t", output="o").value == pytest.approx(0.0)

    def test_not_applicable_verifier_is_skipped(self):
        # Returning None means "I have no opinion", not "this is bad".
        chain = RewardChain(objective=[_NotApplicable()])
        reward = chain.score(task="t", output="o")
        assert len(reward.contributions) == 0
        assert reward.value is None

    def test_multiple_objective_verifiers_are_averaged(self):
        chain = RewardChain(objective=[_AlwaysPass(), _AlwaysFail()])
        assert chain.score(task="t", output="o").value == pytest.approx(0.5)

    def test_not_applicable_does_not_drag_down_the_average(self):
        chain = RewardChain(objective=[_AlwaysPass(), _NotApplicable()])
        assert chain.score(task="t", output="o").value == pytest.approx(1.0)


class TestUserFeedback:
    def test_thumbs_up_is_full_reward(self):
        chain = RewardChain()
        reward = chain.score(task="t", output="o", user_feedback=1)
        assert reward.value == pytest.approx(1.0)
        assert reward.source is RewardSource.USER

    def test_thumbs_down_is_zero_reward(self):
        chain = RewardChain()
        reward = chain.score(task="t", output="o", user_feedback=-1)
        assert reward.value == pytest.approx(0.0)
        assert reward.source is RewardSource.USER

    def test_absent_feedback_contributes_nothing(self):
        chain = RewardChain()
        reward = chain.score(task="t", output="o")
        assert reward.value is None

    def test_feedback_is_validated(self):
        chain = RewardChain()
        with pytest.raises(ValueError, match="feedback"):
            chain.score(task="t", output="o", user_feedback=7)

    def test_explicit_feedback_overrides_judge(self):
        # Production change that would break this: letting the judge outrank an
        # explicit human thumbs-down.
        judge = LLMJudgeVerifier(lambda task, output: 1.0)
        chain = RewardChain(judge=judge)
        reward = chain.score(task="t", output="o", user_feedback=-1)
        assert reward.value < 0.5
        assert reward.source is RewardSource.USER


class TestJudgeWeightCap:
    def test_judge_alone_gets_full_weight_when_configured(self):
        judge = LLMJudgeVerifier(lambda task, output: 0.8)
        chain = RewardChain(judge=judge, judge_weight_cap=1.0)
        reward = chain.score(task="t", output="o")
        assert reward.value == pytest.approx(0.8)

    def test_judge_weight_is_capped_by_default(self):
        # Default cap must stay below objective/user influence.
        assert RewardChain().judge_weight_cap == pytest.approx(0.3)

    def test_judge_cannot_outvote_objective_evidence(self):
        # Production change that would break this: weighting the judge at 1.0
        # regardless of whether objective evidence exists.
        judge = LLMJudgeVerifier(lambda task, output: 1.0)
        chain = RewardChain(objective=[_AlwaysFail()], judge=judge, judge_weight_cap=0.3)
        reward = chain.score(task="t", output="o")
        # objective 0.0 at weight 1.0, judge 1.0 at weight 0.3 -> 0.3/1.3
        assert reward.value == pytest.approx(0.3 / 1.3, abs=1e-6)
        assert reward.source is RewardSource.OBJECTIVE

    def test_explicit_cap_above_one_is_rejected(self):
        with pytest.raises(ValueError, match="cap"):
            RewardChain(judge_weight_cap=1.5)

    def test_judge_records_its_own_contribution(self):
        judge = LLMJudgeVerifier(lambda task, output: 0.5)
        chain = RewardChain(judge=judge)
        reward = chain.score(task="t", output="o")
        assert reward.contributions[0].source is RewardSource.JUDGE
        assert reward.contributions[0].value == pytest.approx(0.5)


class TestLengthPenalty:
    def test_no_penalty_under_budget(self):
        penalty = LengthPenalty(soft_limit_chars=100, hard_limit_chars=400)
        assert penalty.apply("x" * 90, score=1.0) == pytest.approx(1.0)

    def test_full_penalty_at_hard_limit(self):
        penalty = LengthPenalty(soft_limit_chars=100, hard_limit_chars=400)
        assert penalty.apply("x" * 400, score=1.0) == pytest.approx(0.0)

    def test_penalty_is_linear_between_limits(self):
        penalty = LengthPenalty(soft_limit_chars=100, hard_limit_chars=400)
        # midpoint 250 -> half the score retained
        assert penalty.apply("x" * 250, score=1.0) == pytest.approx(0.5)

    def test_penalty_never_goes_negative(self):
        penalty = LengthPenalty(soft_limit_chars=100, hard_limit_chars=400)
        assert penalty.apply("x" * 10_000, score=1.0) == pytest.approx(0.0)

    def test_judge_applies_length_penalty(self):
        # Production change that would break this: removing the penalty call in
        # LLMJudgeVerifier.score(), which would let verbosity buy reward.
        judge = LLMJudgeVerifier(lambda task, output: 1.0)
        chain = RewardChain(
            judge=judge,
            judge_weight_cap=1.0,
            length_penalty=LengthPenalty(soft_limit_chars=100, hard_limit_chars=200),
        )
        short = chain.score(task="t", output="x" * 50).value
        long = chain.score(task="t", output="x" * 500).value
        assert short == pytest.approx(1.0)
        assert long == pytest.approx(0.0)


class TestRewardIntegrity:
    def test_no_sources_yields_no_reward(self):
        reward = RewardChain().score(task="t", output="o")
        assert reward.value is None
        assert len(reward.contributions) == 0

    def test_reward_value_is_always_in_unit_interval(self):
        judge = LLMJudgeVerifier(lambda task, output: 1.0)
        chain = RewardChain(objective=[_AlwaysPass()], judge=judge)
        reward = chain.score(task="t", output="o")
        assert 0.0 <= reward.value <= 1.0

    def test_reward_exposes_confidence(self):
        # Confidence should reflect how much independent evidence backed the score.
        judge = LLMJudgeVerifier(lambda task, output: 1.0)
        rich = RewardChain(objective=[_AlwaysPass()], judge=judge).score(
            task="t", output="o"
        )
        thin = RewardChain(judge=judge).score(task="t", output="o")
        assert rich.confidence > thin.confidence

    def test_reward_is_immutable(self):
        # Frozen dataclass: mutation must raise FrozenInstanceError specifically.
        # Asserting a bare Exception would pass for almost any error and prove
        # nothing about immutability.
        reward = Reward(value=0.5, source=RewardSource.OBJECTIVE, contributions=(), confidence=1.0)
        with pytest.raises(FrozenInstanceError):
            reward.value = 0.9  # type: ignore[misc]
