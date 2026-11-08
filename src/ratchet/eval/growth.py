"""Growth measurement: does the system actually get better?

A self-improvement system that cannot demonstrate improvement is a belief, not a
tool. This module is the cheapest honest answer to "is this working?", built on
data the loop already records rather than on a new benchmark the owner has to
trust.

The core measure is a **cohort curve**: interactions are split into equal slices
in the order they happened, and the mean reward of each slice is compared. Rising
cohorts mean the system is doing better on later work than on earlier work.

What this does and does not establish
-------------------------------------
A rising curve is evidence, not proof. Reward depends on what the owner happened
to ask, so a run of easy tasks at the end looks identical to genuine improvement.
Two things make the number mean something:

* **Unmeasured interactions are excluded, never counted as zero.** An interaction
  with no reward is not a failure; scoring it as one would make the curve a
  measure of how often the verifier had an opinion.
* **Small samples are refused.** Below ``min_per_bucket`` the module returns an
  empty curve rather than a dramatic-looking number built on three data points.
  A growth curve that cannot be trusted is worse than no growth curve.

For a real answer, use ``EvalSuite``: re-run a fixed task list against the engine
at two points in time and compare. That holds the tasks constant, which is the
only way to attribute a difference to the system rather than to the workload.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ratchet.memory.store import Event

MIN_PER_BUCKET = 3
"""Below this a bucket is too thin to report."""


@dataclass(frozen=True)
class Bucket:
    """One slice of the interaction history."""

    index: int
    n: int
    mean_reward: float
    measured: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "n": self.n,
            "mean_reward": round(self.mean_reward, 4),
            "measured": self.measured,
        }


@dataclass(frozen=True)
class GrowthCurve:
    """The result of one growth measurement."""

    buckets: tuple[Bucket, ...] = ()
    first_mean: float = 0.0
    last_mean: float = 0.0
    delta: float = 0.0
    measured_total: int = 0
    unmeasured_total: int = 0
    verdict: str = "insufficient data"

    def as_dict(self) -> dict[str, object]:
        return {
            "buckets": [b.as_dict() for b in self.buckets],
            "first_mean": round(self.first_mean, 4),
            "last_mean": round(self.last_mean, 4),
            "delta": round(self.delta, 4),
            "measured_total": self.measured_total,
            "unmeasured_total": self.unmeasured_total,
            "verdict": self.verdict,
        }


def _verdict(delta: float, measured: int, *, min_sample: int = MIN_PER_BUCKET * 2) -> str:
    """Classify a delta, refusing to classify at all below ``min_sample``.

    The threshold is a parameter because the two callers count different things.
    A growth curve slices dozens of interactions and needs a handful per slice
    before a mean means anything; a suite comparison counts fixed tasks the owner
    wrote by hand, where two graded tasks is a legitimate sample. Sharing one
    hardcoded bar would leave every small suite permanently "insufficient".
    """
    if measured < min_sample:
        return "insufficient data"
    if delta > 0.1:
        return "improving"
    if delta < -0.1:
        return "regressing"
    return "flat"


def growth_curve(
    events: Sequence[Event], *, buckets: int = 4, min_per_bucket: int = MIN_PER_BUCKET
) -> GrowthCurve:
    """Mean reward per chronological slice of ``events``.

    Events are assumed to be in the order they were recorded, which is how
    ``store.list_events`` returns them. Only events carrying a real reward
    contribute to a mean; the rest are counted and reported separately so the
    owner can see how much of the history was actually measured.
    """
    if buckets < 2:
        raise ValueError("buckets must be at least 2")

    ordered = sorted(events or [], key=lambda e: e.created_at)
    measured = [e for e in ordered if e.reward is not None]
    unmeasured = len(ordered) - len(measured)

    if not measured:
        return GrowthCurve(unmeasured_total=unmeasured)

    size = max(1, len(measured) // buckets)
    slices: list[list[Event]] = []
    for start in range(0, len(measured), size):
        slices.append(measured[start : start + size])

    # An off-by-one remainder can leave a final slice of one event, which would
    # be reported as confidently as a slice of fifty. It is folded into the
    # previous slice instead.
    if len(slices) > 1 and len(slices[-1]) < min_per_bucket:
        tail = slices.pop()
        slices[-1] = slices[-1] + tail

    out: list[Bucket] = []
    for index, group in enumerate(slices):
        if len(group) < min_per_bucket:
            continue
        rewards = [e.reward for e in group if e.reward is not None]
        out.append(
            Bucket(
                index=index,
                n=len(group),
                mean_reward=sum(rewards) / len(rewards),
                measured=len(rewards),
            )
        )

    if len(out) < 2:
        return GrowthCurve(
            buckets=tuple(out),
            measured_total=len(measured),
            unmeasured_total=unmeasured,
        )

    first, last = out[0].mean_reward, out[-1].mean_reward
    return GrowthCurve(
        buckets=tuple(out),
        first_mean=first,
        last_mean=last,
        delta=last - first,
        measured_total=len(measured),
        unmeasured_total=unmeasured,
        verdict=_verdict(last - first, len(measured)),
    )


@dataclass(frozen=True)
class EvalTask:
    """One fixed item in a repeatable suite."""

    name: str
    task: str
    user_input: str
    check: Callable[[str], float | None]
    """Returns a reward in [0, 1], or ``None`` if not applicable."""


@dataclass(frozen=True)
class EvalSuite:
    """A fixed task list, so two runs are comparable."""

    name: str = "suite"
    tasks: tuple[EvalTask, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tasks": [
                {"name": t.name, "task": t.task, "user_input": t.user_input}
                for t in self.tasks
            ],
        }


@dataclass(frozen=True)
class EvalRun:
    """One pass over a suite."""

    suite: str
    scores: dict[str, float | None] = field(default_factory=dict)
    mean: float = 0.0
    graded: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "scores": {k: v for k, v in self.scores.items()},
            "mean": round(self.mean, 4),
            "graded": self.graded,
        }


def run_suite(
    suite: EvalSuite, complete: Callable[[str, str], str]
) -> EvalRun:
    """Run ``suite`` by calling ``complete(task, user_input) -> output``.

    The suite never runs a Ratchet cycle itself - it takes a completion function
    so that the same tasks can be run against a bare provider and against
    Ratchet, which is the comparison that isolates Ratchet's contribution.
    """
    scores: dict[str, float | None] = {}
    for item in suite.tasks:
        output = complete(item.task, item.user_input)
        scores[item.name] = item.check(output)

    graded = [s for s in scores.values() if s is not None]
    return EvalRun(
        suite=suite.name,
        scores=scores,
        mean=(sum(graded) / len(graded)) if graded else 0.0,
        graded=len(graded),
    )


def compare(before: EvalRun, after: EvalRun) -> dict[str, object]:
    """Compare two runs of the same suite."""
    shared = sorted(set(before.scores) & set(after.scores))
    deltas = {
        name: (after.scores[name] or 0.0) - (before.scores[name] or 0.0)
        for name in shared
        if before.scores[name] is not None and after.scores[name] is not None
    }
    return {
        "before_mean": round(before.mean, 4),
        "after_mean": round(after.mean, 4),
        "delta": round(after.mean - before.mean, 4),
        "graded": len(shared),
        "per_task_delta": {k: round(v, 4) for k, v in deltas.items()},
        "verdict": _verdict(after.mean - before.mean, len(shared), min_sample=2),
    }
