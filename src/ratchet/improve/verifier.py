"""Verifier: the reward chain that decides what Ratchet optimises for.

Why this module gets more engineering attention than prompt evolution
---------------------------------------------------------------------
In any reinforcement-learning system the reward function determines what is
actually being optimised. If the verifier cannot tell good from bad, the system
optimises confidently towards garbage. Goodhart's law applies with full force:
once a metric becomes the target, it stops being a good metric.

Two failure modes are designed against here:

1. **Reward hacking.** An LLM judge is known to prefer longer answers, answers
   in the first position, and answers stylistically similar to its own. So the
   judge is (a) never the top-priority source, (b) weight-capped, and (c) run
   through a length penalty so verbosity cannot buy reward.

2. **Judge dominance.** Evidence is strictly ordered. An objective, mechanically
   checkable signal (test passed, format matched, answer key correct) always
   outranks a subjective opinion. This follows RLVR: prefer verifiable rewards.

Priority chain::

    user     (overrides)  explicit human thumbs up/down - highest *information*
                          per event; when present it REPLACES the blend rather
                          than joining it, because only the owner can judge
                          whether the output actually satisfied them
      > objective (weight 1.0)  mechanical, reproducible, hardest to game;
                                but it only checks the property it encodes
        > judge     (weight cap 0.3)  cheap, dense, but biased - fallback only

Returning ``None`` from a verifier means "I have no opinion about this task",
which is different from "this output scored zero". Verifiers with no opinion are
excluded from the average entirely, so a not-applicable check cannot drag the
score down.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class RewardSource(str, Enum):
    """Where a reward signal came from, ordered by trustworthiness."""

    OBJECTIVE = "objective"
    USER = "user"
    JUDGE = "judge"

    @property
    def weight(self) -> float:
        """Canonical trust weight used when blending contributions."""
        return _SOURCE_WEIGHTS[self]


_SOURCE_WEIGHTS: dict[RewardSource, float] = {
    RewardSource.OBJECTIVE: 1.0,
    RewardSource.USER: 0.8,
    RewardSource.JUDGE: 0.3,
}

# Total weight when every source agrees; used to normalise confidence to [0, 1].
_MAX_EXPECTED_WEIGHT = 2.0


class ObjectiveVerifier(ABC):
    """A mechanically checkable reward source.

    Subclass this for anything that can be decided without asking another model:
    running unit tests, matching a regex/schema, comparing against an answer key,
    checking that code compiles, validating JSON parses.
    """

    name: str = "objective"

    @abstractmethod
    def verify(self, task: str, output: str) -> float | None:
        """Return a score in [0, 1], or ``None`` when not applicable."""


@dataclass(frozen=True)
class Contribution:
    """One source's input to the final reward."""

    source: RewardSource
    name: str
    value: float
    weight: float


@dataclass(frozen=True)
class Reward:
    """Blended reward with a full audit trail.

    ``value`` is ``None`` when no source had an opinion - callers must not
    record that as a zero, or unmeasurable interactions would be logged as
    failures and poison the policy.
    """

    value: float | None
    source: RewardSource | None
    contributions: tuple[Contribution, ...] = field(default_factory=tuple)
    confidence: float = 0.0


@dataclass(frozen=True)
class LengthPenalty:
    """Linear decay between a soft and a hard character budget.

    Applied only to judge scores. Objective verification must not be penalised
    for length: a passing test suite passes regardless of how much was written.
    """

    soft_limit_chars: int
    hard_limit_chars: int

    def __post_init__(self) -> None:
        if self.hard_limit_chars <= self.soft_limit_chars:
            raise ValueError("hard_limit_chars must exceed soft_limit_chars")

    def apply(self, text: str, *, score: float) -> float:
        length = len(text or "")
        if length <= self.soft_limit_chars:
            return score
        if length >= self.hard_limit_chars:
            return 0.0
        span = self.hard_limit_chars - self.soft_limit_chars
        overshoot = (length - self.soft_limit_chars) / span
        return max(0.0, score * (1.0 - overshoot))


class UserFeedbackVerifier:
    """Converts an explicit human thumbs up/down into a reward.

    This is the highest-information signal available - the premise RLHF is built
    on - which is why human correction stays editable in the provenance model.
    """

    name = "user_feedback"

    _MAP: ClassVar[dict[int, float]] = {1: 1.0, -1: 0.0}

    def verify(self, feedback: int | None) -> float | None:
        if feedback is None:
            return None
        if feedback not in self._MAP and feedback != 0:
            raise ValueError("user_feedback must be -1, 0 or 1")
        # 0 means "neutral / no opinion", contributing nothing.
        return self._MAP.get(feedback)


class LLMJudgeVerifier:
    """Last-resort reward source: ask a model to rate the output.

    Cheap and dense, but biased. Keep the callable simple and deterministic in
    tests; in production, randomise option order and supply a rubric.
    """

    name = "llm_judge"

    def __init__(self, rate: Callable[[str, str], float]) -> None:
        self._rate = rate

    def score(self, task: str, output: str) -> float:
        raw = float(self._rate(task, output))
        return min(1.0, max(0.0, raw))


class RewardChain:
    """Blend all available evidence into a single auditable reward."""

    def __init__(
        self,
        *,
        objective: Sequence[ObjectiveVerifier] = (),
        judge: LLMJudgeVerifier | None = None,
        judge_weight_cap: float = RewardSource.JUDGE.weight,
        length_penalty: LengthPenalty | None = None,
        user_verifier: UserFeedbackVerifier | None = None,
    ) -> None:
        if not 0.0 <= judge_weight_cap <= 1.0:
            raise ValueError("judge_weight_cap must be within [0, 1]")
        self.objective = list(objective)
        self.judge = judge
        self.judge_weight_cap = judge_weight_cap
        self.length_penalty = length_penalty
        self.user_verifier = user_verifier or UserFeedbackVerifier()

    def score(
        self,
        *,
        task: str,
        output: str,
        user_feedback: int | None = None,
    ) -> Reward:
        """Collect every opinion, then blend by trust weight."""
        contributions: list[Contribution] = []

        for verifier in self.objective:
            value = verifier.verify(task, output)
            if value is None:
                continue
            contributions.append(
                Contribution(
                    source=RewardSource.OBJECTIVE,
                    name=verifier.name,
                    value=min(1.0, max(0.0, float(value))),
                    weight=RewardSource.OBJECTIVE.weight,
                )
            )

        user_value = self.user_verifier.verify(user_feedback)
        if user_value is not None:
            contributions.append(
                Contribution(
                    source=RewardSource.USER,
                    name=self.user_verifier.name,
                    value=user_value,
                    weight=RewardSource.USER.weight,
                )
            )

        if self.judge is not None:
            judge_value = self.judge.score(task, output)
            if self.length_penalty is not None:
                judge_value = self.length_penalty.apply(output, score=judge_value)
            contributions.append(
                Contribution(
                    source=RewardSource.JUDGE,
                    name=self.judge.name,
                    value=judge_value,
                    weight=self.judge_weight_cap,
                )
            )

        if not contributions:
            return Reward(value=None, source=None, contributions=(), confidence=0.0)

        # Explicit human feedback OVERRIDES the weighted blend instead of joining
        # it. Rationale: an objective verifier only checks one mechanical property
        # (does it compile, is it valid JSON, do the tests pass), whereas a human
        # thumb judges whether the output actually satisfied them. When those
        # disagree, the verifier is measuring the wrong thing - the human is the
        # arbiter of their own satisfaction.
        #
        # Blending instead of overriding is actively harmful: a thumbs-down
        # against a passing objective check would average out to ~0.55, which is
        # ABOVE the 0.5 midpoint, so the bandit would *increase* the selection
        # probability of a variant the user just rejected.
        user_contribution = next(
            (c for c in contributions if c.source is RewardSource.USER), None
        )
        if user_contribution is not None:
            return Reward(
                value=user_contribution.value,
                source=RewardSource.USER,
                contributions=tuple(contributions),
                # A direct human judgement on this exact output is maximal evidence.
                confidence=1.0,
            )

        total_weight = sum(c.weight for c in contributions)
        value = sum(c.value * c.weight for c in contributions) / total_weight
        value = min(1.0, max(0.0, value))
        # The most trusted source that actually contributed names the reward,
        # so downstream code can tell objective evidence from a guess.
        source = max(contributions, key=lambda c: c.weight).source
        confidence = min(1.0, total_weight / _MAX_EXPECTED_WEIGHT)
        return Reward(
            value=value,
            source=source,
            contributions=tuple(contributions),
            confidence=confidence,
        )
