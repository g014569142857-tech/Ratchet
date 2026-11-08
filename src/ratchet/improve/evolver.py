"""Prompt evolver: grow the bandit's action space instead of hand-tuning it.

The three seed variants (``default``, ``chain_of_thought``, ``concise``) are a
starting guess about which ways of answering might suit the owner. They are not a
theory of what this particular person wants. The evolver is what turns a fixed
guess into a searched space: once an arm has enough pulls to be judged, the
evolver asks the model to propose a revision of it, and the revision enters the
pool as a new arm with a uniform prior so Thompson Sampling gives it a fair trial.

Why mutations enter as *new* arms rather than replacing their parent
------------------------------------------------------------------------
Replacing in place would destroy the posterior that justified the mutation, and
the successor would inherit credit it never earned. Entering as a new arm means a
mutation has to beat its own parent on evidence, which is the only thing that
keeps the pool from drifting on the strength of plausible-sounding rewrites.

The privileged position problem
--------------------------------
A variant's text is injected into the instruction position, which makes it the
most powerful content in the prompt. A mutation that proposed "ignore the user's
constraints" would be extremely effective at scoring well and catastrophic to
obey. So a proposal is screened before it can become an arm: bounded in length,
single-paragraph, and rejected if it tries to claim authority, reference the
scaffolding, or talk about the reward process itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from ratchet.core.policy import ThompsonPolicy
from ratchet.core.prompt import register_variant

_MAX_INSTRUCTION_CHARS = 260
_MIN_PULLS_TO_MUTATE = 5
_DEFAULT_MAX_POOL = 8

_MUTATION_PROMPT = """You are improving a short instruction that steers how a
language model approaches a task.

CURRENT INSTRUCTION
{instruction}

MEASURED SUCCESS RATE
{mean:.2f} over {pulls} uses

Propose ONE revised instruction that would be more likely to succeed.
Constraints:
  - at most {max_chars} characters, a single paragraph
  - imperative: describe how to approach the task
  - must stay general; it will be reused on different tasks
  - must not claim authority, mention system messages, or instruct the model to
    disregard anything
  - must not mention scores, rewards, testing, or being evaluated
  - if the current instruction is already as good as it can be, reply with exactly: NONE

REVISED INSTRUCTION:"""

# A proposal that tries to become an instruction about instructions. Checked
# before the text is registered, because once it is an arm it is injected into
# the instruction position of every future prompt.
_HOSTILE = (
    "ignore",
    "disregard",
    "override",
    "forget your",
    "system prompt",
    "system message",
    "reveal",
    "exfiltrate",
    "api key",
    "as an ai",
    "you are now",
    "new instruction",
    "previous instruction",
    "developer message",
)

# If these come back, the model narrated its prompt rather than answering it.
_ECHO = (
    "CURRENT INSTRUCTION",
    "MEASURED SUCCESS RATE",
    "REVISED INSTRUCTION",
    "BEGIN RECALLED EXPERIENCE DATA",
    "END RECALLED EXPERIENCE DATA",
    "APPROACH FOR THIS TASK",
)


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Variant:
    """One arm's identity, beyond the bandit's statistics."""

    name: str
    instruction: str
    generation: int = 0
    parent: str | None = None
    origin: str = "seed"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "generation": self.generation,
            "parent": self.parent,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Variant:
        return cls(
            name=str(data["name"]),
            instruction=str(data.get("instruction", "")),
            generation=int(str(data.get("generation", 0))),
            parent=(str(data["parent"]) if data.get("parent") else None),
            origin=str(data.get("origin", "seed")),
        )


@dataclass
class PromptEvolver:
    """Maintain the pool of strategies the bandit chooses between."""

    model: CompletionModel
    pool: dict[str, Variant] = field(default_factory=dict)
    max_pool: int = _DEFAULT_MAX_POOL
    min_pulls: int = _MIN_PULLS_TO_MUTATE
    max_instruction_chars: int = _MAX_INSTRUCTION_CHARS

    def __post_init__(self) -> None:
        if self.max_pool < 2:
            raise ValueError("max_pool must be at least 2")
        if self.min_pulls < 1:
            raise ValueError("min_pulls must be at least 1")

    # ------------------------------------------------------------------ pool

    def seed(self, variants: dict[str, str]) -> None:
        """Register the starting strategies. Idempotent, and never overwrites."""
        for name, instruction in variants.items():
            self.pool.setdefault(
                name, Variant(name=name, instruction=instruction, origin="seed")
            )

    def sync(self) -> None:
        """Push every pooled variant into the prompt layer."""
        for variant in self.pool.values():
            register_variant(variant.name, variant.instruction)

    # --------------------------------------------------------------- evolve

    def maybe_evolve(self, policy: ThompsonPolicy) -> Variant | None:
        """Propose one mutation of the best-established arm, if there is one.

        Returns ``None`` when no arm has enough evidence, the pool is full, or
        the model declines to improve on the current best. Evolving is
        deliberately one step per call: it is driven by an interaction loop, and
        a burst of mutations would flood the pool with untested arms.
        """
        if len(self.pool) >= self.max_pool:
            return None

        candidates = [
            name
            for name in self.pool
            if policy.arms.get(name) is not None
            and policy.arms[name].pulls >= self.min_pulls
        ]
        if not candidates:
            return None

        # Mutate the best performer, not a random one: the point is to climb,
        # and a mutation of a weak arm inherits nothing worth climbing from.
        parent_name = max(candidates, key=lambda n: policy.arms[n].mean())
        parent = self.pool[parent_name]
        arm = policy.arms[parent_name]

        prompt = _MUTATION_PROMPT.format(
            instruction=parent.instruction or "(none - answer directly)",
            mean=arm.mean(),
            pulls=arm.pulls,
            max_chars=self.max_instruction_chars,
        )
        proposal = (self.model.complete(prompt) or "").strip()
        if not self._is_valid_mutation(proposal):
            return None

        return self.adopt(
            instruction=proposal,
            parent=parent_name,
            generation=parent.generation + 1,
        )

    def adopt(
        self, *, instruction: str, parent: str | None = None, generation: int = 1
    ) -> Variant:
        """Add a validated instruction to the pool as a fresh arm."""
        variant = Variant(
            name=f"evolved_{uuid.uuid4().hex[:8]}",
            instruction=instruction.strip(),
            generation=generation,
            parent=parent,
            origin="mutation",
        )
        self.pool[variant.name] = variant
        register_variant(variant.name, variant.instruction)
        return variant

    def prune(self, policy: ThompsonPolicy, *, min_mean: float = 0.3) -> list[str]:
        """Drop arms that were tried enough and performed badly.

        Only mutations are pruned. The seed arms stay in the pool forever: they
        are the control group, and without them there is no baseline to judge an
        evolved arm against.
        """
        doomed = [
            name
            for name, variant in self.pool.items()
            if variant.origin == "mutation"
            and policy.arms.get(name) is not None
            and policy.arms[name].pulls >= self.min_pulls
            and policy.arms[name].mean() < min_mean
        ]
        for name in doomed:
            self.pool.pop(name, None)
            policy.arms.pop(name, None)
        return doomed

    # ---------------------------------------------------------- validation

    def _is_valid_mutation(self, proposal: str) -> bool:
        if not proposal or proposal.upper() == "NONE":
            return False
        if len(proposal) > self.max_instruction_chars:
            return False
        # A single paragraph: newlines would let a proposal smuggle in a fake
        # instruction block of its own.
        if len(proposal.splitlines()) != 1:
            return False
        lowered = proposal.lower()
        if any(marker in lowered for marker in _HOSTILE):
            return False
        if any(marker in proposal for marker in _ECHO):
            return False
        return True

    # ------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, object]:
        return {
            "max_pool": self.max_pool,
            "min_pulls": self.min_pulls,
            "variants": [v.as_dict() for v in self.pool.values()],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, object], model: CompletionModel
    ) -> PromptEvolver:
        evolver = cls(
            model=model,
            max_pool=int(str(data.get("max_pool", _DEFAULT_MAX_POOL))),
            min_pulls=int(str(data.get("min_pulls", _MIN_PULLS_TO_MUTATE))),
        )
        raw_variants = data.get("variants", [])
        if isinstance(raw_variants, list):
            for raw in raw_variants:
                if isinstance(raw, dict):
                    variant = Variant.from_dict(raw)
                    evolver.pool[variant.name] = variant
        evolver.sync()
        return evolver
