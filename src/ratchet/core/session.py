"""Session: the closed loop that ties every subsystem together.

One ``run()`` call performs the full cycle::

    retrieve  -> recall relevant experience for this task
    select    -> Thompson Sampling picks a prompt variant (the Action)
    assemble  -> build the prompt with experience as data, never instructions
    complete  -> call the model provider
    verify    -> RewardChain scores the output (a human verdict overrides;
                 otherwise objective checks outrank the LLM judge)
    record    -> append to the audit trail; update the policy posterior
    reflect   -> on low reward, distil a rule for next time

The last two steps are what make the loop compound: interaction N writes
something interaction N+1 can retrieve. ``test_second_interaction_recalls_first_
lesson`` is the assertion that this actually happens.

A missing reward is recorded as ``None``, never as zero. Logging an unmeasured
interaction as a failure would teach the policy to avoid strategies that were
simply never evaluated.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ratchet.core.policy import ThompsonPolicy
from ratchet.core.prompt import PromptAssembler, variant_strategy
from ratchet.core.provider import Provider
from ratchet.improve.distiller import pinned_principles
from ratchet.improve.reflector import Reflector
from ratchet.improve.verifier import Reward, RewardChain
from ratchet.memory.retriever import HybridRetriever, RetrievalHit
from ratchet.memory.store import TRUSTED, ExperienceStore


@dataclass(frozen=True)
class SessionConfig:
    """Tunables for one session."""

    seed: int | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    top_k: int = 5
    max_experience_chars: int = 2000
    reflect_on_low_reward: bool = True
    pin_distilled: bool = True
    """Always inject the distilled persona, independent of retrieval."""
    variants: tuple[str, ...] = (
        "default",
        "chain_of_thought",
        "concise",
    )


@dataclass(frozen=True)
class SessionResult:
    """Everything a caller needs to inspect or act on one interaction."""

    output: str
    reward: Reward
    variant_id: str
    retrieved_ids: tuple[str, ...]
    strategy: str = ""
    """The instruction text the chosen arm injected, for auditability."""

    hits: tuple[RetrievalHit, ...] = ()
    prompt: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    reflections: tuple[str, ...] = ()


def _is_cjk(ch: str) -> bool:
    """True for Chinese / Japanese / Korean characters, which tokenise densely."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK unified ideographs
        or 0x3400 <= code <= 0x4DBF  # extension A
        or 0x3040 <= code <= 0x30FF  # hiragana / katakana
        or 0xAC00 <= code <= 0xD7AF  # hangul syllables
        or 0x3000 <= code <= 0x303F  # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def _approx_tokens(text: str) -> int:
    """Rough token estimate. Always at least 1 for non-empty text.

    Latin text averages ~4 characters per token, but CJK text is closer to
    1-2 tokens per character, so a flat ``len // 4`` under-counts Chinese input
    by several times and makes the cost report look far cheaper than it is.
    CJK characters are therefore counted at 1.5 tokens each.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, round(cjk * 1.5 + other / 4))


class Session:
    """Orchestrates the memory-reflection-selection loop over any provider."""

    def __init__(
        self,
        *,
        store: ExperienceStore,
        provider: Provider,
        retriever: HybridRetriever,
        policy: ThompsonPolicy,
        verifier: RewardChain,
        assembler: PromptAssembler,
        reflector: Reflector | None = None,
        config: SessionConfig | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.retriever = retriever
        self.policy = policy
        self.verifier = verifier
        self.assembler = assembler
        self.reflector = reflector
        self.config = config or SessionConfig()
        for variant in self.config.variants:
            self.policy.register(variant)
        self._interactions = 0
        self._tokens_in = 0
        self._tokens_out = 0

    def run(
        self,
        *,
        task: str,
        user_input: str,
        user_feedback: int | None = None,
        trust_level: str = TRUSTED,
        profile_summary: str | None = None,
    ) -> SessionResult:
        """Execute one full self-improvement cycle."""
        hits = self.retriever.search(
            f"{task} {user_input}",
            top_k=self.config.top_k,
            max_chars=self.config.max_experience_chars,
        )
        usable_hits = [h for h in hits if h.trust_level != "untrusted"]

        pinned = pinned_principles(self.store) if self.config.pin_distilled else []
        variant_id = self.policy.select()
        prompt = self.assembler.assemble(
            user_input=f"{task}\n\n{user_input}",
            hits=usable_hits,
            profile_summary=profile_summary,
            # The arm the bandit chose has to reach the model, or the reward it
            # earns carries no information about it.
            variant_id=variant_id,
            pinned=pinned,
        )

        started = time.perf_counter()
        output = self.provider.complete(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0

        reward = self.verifier.score(
            task=task, output=output, user_feedback=user_feedback
        )

        tokens_in = _approx_tokens(prompt)
        tokens_out = _approx_tokens(output)
        retrieved_ids = tuple(h.id for h in usable_hits)

        # Only fold a real measurement into the posterior.
        if reward.value is not None:
            self.policy.update(variant_id, reward.value)
            for experience_id in retrieved_ids:
                self.store.record_reward(experience_id, reward.value - 0.5)

        self.store.record_event(
            session_id=self.config.session_id,
            task=task,
            user_input=user_input,
            output=output,
            variant_id=variant_id,
            retrieved_ids=retrieved_ids,
            reward=reward.value,
            reward_source=reward.source.value if reward.source else None,
            confidence=reward.confidence,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            trust_level=trust_level,
        )

        self._interactions += 1
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out

        reflections: tuple[str, ...] = ()
        if (
            self.reflector is not None
            and self.config.reflect_on_low_reward
            and reward.value is not None
        ):
            created = self.reflector.reflect_on(
                task=task,
                output=output,
                reward=reward.value,
                trust_level=trust_level,
            )
            reflections = tuple(r.content for r in created)

        return SessionResult(
            output=output,
            reward=reward,
            variant_id=variant_id,
            strategy=variant_strategy(variant_id),
            retrieved_ids=retrieved_ids,
            hits=tuple(usable_hits),
            prompt=prompt,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            reflections=reflections,
        )

    def stats(self) -> dict[str, float | int | str]:
        """Running totals, so cost growth is visible rather than surprising."""
        return {
            "session_id": self.config.session_id,
            "interactions": self._interactions,
            "tokens_in": self._tokens_in,
            "tokens_out": self._tokens_out,
            "active_experiences": len(self.store.list_active()),
        }
