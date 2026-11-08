"""Thompson Sampling policy: the reinforcement-learning core of Ratchet.

Mapping onto the standard RL framework:
  * arm       -> a prompt variant / reasoning strategy (the Action)
  * Beta(a,b) -> the Policy's belief about that arm's success rate
  * reward    -> the Verifier's score, clamped to [0, 1]
  * update()  -> the Policy Update step

Thompson Sampling samples one draw from each arm's Beta posterior and plays the
highest. Rarely-tried arms keep wide posteriors, so exploration happens
automatically without any epsilon hyper-parameter to tune.

Uses ``random.betavariate`` from the standard library, so there is no
third-party dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Arm:
    """Beta-distribution posterior for one action."""

    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)


@dataclass
class ThompsonPolicy:
    """Multi-armed bandit over named strategies."""

    arms: dict[str, Arm] = field(default_factory=dict)
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------ lifecycle

    def register(self, name: str) -> None:
        """Add an arm with a uniform Beta(1,1) prior. Idempotent."""
        self.arms.setdefault(name, Arm())

    def list_arms(self) -> list[str]:
        return list(self.arms)

    # ------------------------------------------------------------- decision

    def select(self) -> str:
        """Sample every arm and return the name with the highest draw."""
        if not self.arms:
            raise ValueError("no arms registered")
        return max(self.arms, key=lambda name: self.arms[name].sample(self._rng))

    # --------------------------------------------------------------- update

    def update(self, name: str, reward: float) -> None:
        """Fold a reward in [0, 1] into the arm's posterior.

        Out-of-range rewards are clamped, so a mis-scaled verifier can never
        corrupt the posterior. An unknown arm is registered on the fly.
        """
        if name not in self.arms:
            self.register(name)
        arm = self.arms[name]
        clamped = min(1.0, max(0.0, float(reward)))
        arm.alpha += clamped
        arm.beta += 1.0 - clamped
        arm.pulls += 1

    def mean(self, name: str) -> float:
        """Posterior mean success rate for one arm."""
        if name not in self.arms:
            raise KeyError(f"unknown arm: {name}")
        return self.arms[name].mean()

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "arms": {
                name: {"alpha": a.alpha, "beta": a.beta, "pulls": a.pulls}
                for name, a in self.arms.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, seed: int | None = None
    ) -> ThompsonPolicy:
        policy = cls(seed=data.get("seed") if seed is None else seed)
        for name, arm in data.get("arms", {}).items():
            policy.arms[name] = Arm(
                alpha=float(arm["alpha"]),
                beta=float(arm["beta"]),
                pulls=int(arm["pulls"]),
            )
        return policy
