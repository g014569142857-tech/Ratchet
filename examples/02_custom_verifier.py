"""Example 2 — teach Ratchet what "good" means for your task.

The reward function decides what the system optimises for. Anything you can check
mechanically is a better reward than asking a model to rate the answer: it is
cheaper, reproducible, and cannot be charmed into a higher score.

This example uses a scripted provider so it runs offline, and two objective
verifiers so you can watch the bandit converge on the strategy that satisfies them.

    python examples/02_custom_verifier.py
"""

import os

import ratchet
from ratchet.core.provider import Provider
from ratchet.improve.verifier import ObjectiveVerifier, RewardChain

# ---------------------------------------------------------------- the reward


class MentionsTerm(ObjectiveVerifier):
    """Reward an answer for containing the term the task is about."""

    name = "mentions_term"

    def __init__(self, term: str) -> None:
        self.term = term.lower()

    def verify(self, task: str, output: str) -> float | None:
        # Returning None means "not applicable", which is excluded from the
        # average. Never score an inapplicable check as zero.
        return 1.0 if self.term in output.lower() else 0.0


class UnderLength(ObjectiveVerifier):
    """Reward brevity, because the human rule asked for it."""

    name = "under_length"

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def verify(self, task: str, output: str) -> float | None:
        length = len(output)
        if length <= self.limit:
            return 1.0
        # Linear falloff rather than a cliff, so the bandit gets a gradient.
        return max(0.0, 1.0 - (length - self.limit) / self.limit)


# --------------------------------------------------------------- the provider


class ScriptedProvider(Provider):
    """A model that only answers well under the `concise` strategy.

    Stands in for a real one so this example costs nothing to run. The point is
    that the correct answer is a function of the strategy the bandit picked —
    which is what makes the reward informative.
    """

    name = "scripted"

    def __init__(self) -> None:
        self.marker = ratchet.VARIANT_STRATEGIES["concise"][:40]

    def complete(self, prompt: str, **kwargs: object) -> str:
        if "failed interaction" in prompt:
            return "NONE"  # decline to reflect
        if self.marker in prompt:
            return "LoRA: low-rank adaptation. It trains a small side matrix."
        return (
            "Low-rank adaptation is a technique that, broadly speaking and with "
            "many caveats, reparameterises weight updates in a manner which one "
            "might describe as efficient, though the details are contested."
        )


def main() -> None:
    # Start from an empty store so the numbers below are reproducible: otherwise
    # a second run would keep accumulating on the first run's posterior.
    if os.path.exists("example_verifier.db"):
        os.remove("example_verifier.db")

    engine = ratchet.Engine(
        db_path="example_verifier.db",
        provider=ScriptedProvider(),
        reward_chain=RewardChain(
            objective=[MentionsTerm("low-rank"), UnderLength(120)]
        ),
        seed=42,
    )

    for _i in range(30):
        engine.run(task="explain LoRA", user_input="briefly")

    print("posterior means after 30 interactions:")
    for name in sorted(engine.policy.list_arms()):
        arm = engine.policy.arms[name]
        print(f"  {name:<18} mean={arm.mean():.3f}  pulls={arm.pulls}")

    events = engine.store.list_events()
    measured = [e.reward for e in events if e.reward is not None]
    print(f"\nmean reward: {sum(measured) / len(measured):.3f} over {len(measured)} events")

    print("\ngrowth curve:")
    print(ratchet.growth_curve(events, buckets=3).verdict)
    engine.close()


if __name__ == "__main__":
    main()
