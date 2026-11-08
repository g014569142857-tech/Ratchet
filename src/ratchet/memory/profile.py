"""Machine-inferred user profile: the part that learns you without being told.

Ratchet's whole premise is that the owner's job is *judgement*, not
prompt-writing. The profile is where the system converts a stream of judgements
into a standing understanding of how this person wants to be answered - the thing
that makes the difference between a generic assistant and one that has been
yours for a while.

Design constraints, in the order they bite
------------------------------------------

1. **Every belief is traceable.** Each row keeps the ids of the events that
   suggested it. A preference with no addressable evidence is indistinguishable
   from a hallucination, and the panel has to be able to show its work.

2. **Confidence decays.** The failure mode this prevents is the locked-in
   mistake: the system concludes "this owner wants terse answers", answers
   tersely, receives no correction because the answers are adequate, and treats
   the silence as confirmation. Time decay means an un-reconfirmed belief slowly
   loses authority instead of compounding.

3. **Contradiction can flip a belief.** Support and contradiction are counted
   separately. Enough counter-evidence flips the value rather than being averaged
   into it, so the system cannot entrench an early guess.

4. **The owner outranks the inference.** A human override is marked as such and
   is never auto-flipped; only another human action changes it. This is the same
   asymmetry as the provenance model - the machine may propose, the owner disposes.

Inference is deliberately **evidence-based rather than model-based**. Asking a
model "what does this user like?" would make the profile unauditable and
dependent on a network call; the signals below are all observable facts about the
interaction (a correction was issued, the input was mostly Chinese, an accepted
answer was short). That keeps the whole module deterministic and testable, which
matters more than its coverage - a profile you can inspect beats a profile you
must trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ratchet.governance.correction import PROFILE_IMPACT, Correction
from ratchet.memory.store import ExperienceStore, ProfileEntry

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# A belief reaches ~0.5 confidence on the first observation, ~0.88 by the third.
# Saturation is intentional: the tenth confirmation should not be worth as much
# as the first, or one early fluke could never be overtaken.
_CONFIDENCE_SATURATION = 0.5

# Confidence halves every ``half_life_days`` without new evidence.
_DEFAULT_HALF_LIFE_DAYS = 14.0

# Below this a belief is treated as a rumour and kept out of the prompt.
_DEFAULT_MIN_CONFIDENCE = 0.35

# Accepted-output length thresholds used for the weak verbosity signal.
_SHORT_OUTPUT_CHARS = 280
_LONG_OUTPUT_CHARS = 1200

# Minimum input length before language inference will hazard a guess.
_MIN_LANGUAGE_SAMPLE = 8

_HUMAN = "human_override"
_INFERRED = "inferred"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


def detect_language(text: str) -> str | None:
    """Guess ``zh`` / ``en`` from CJK share. ``None`` when the sample is too small.

    Refusing to guess on short input matters: "ok" and "好" are both common, and
    a wrong language belief is one of the most annoying ones to be stuck with.
    """
    sample = (text or "").strip()
    if len(sample) < _MIN_LANGUAGE_SAMPLE:
        return None
    cjk = len(_CJK.findall(sample))
    ratio = cjk / len(sample)
    if ratio > 0.25:
        return "zh"
    if ratio == 0:
        return "en"
    return None


class Profile:
    """Infer, decay and render the owner's preferences."""

    def __init__(
        self,
        store: ExperienceStore,
        *,
        half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        self.store = store
        self.half_life_days = half_life_days
        self.min_confidence = min_confidence

    # ------------------------------------------------------------ observation

    def observe(
        self,
        key: str,
        value: str,
        *,
        evidence_id: str | None = None,
        now: datetime | None = None,
    ) -> ProfileEntry:
        """Record one piece of evidence for or against ``key=value``.

        Confirming evidence raises ``support``; contradicting evidence raises
        ``contradict``, and once contradiction exceeds support the belief flips
        to the new value. A human-authored row is never flipped - only another
        human action may move it.
        """
        moment = now or _now()
        existing = self.store.get_profile_entry(key)

        support, contradict, value_now = 1, 0, value
        evidence: list[str] = []
        source = _INFERRED

        if existing is not None:
            source = existing.source
            if existing.source == _HUMAN:
                # The owner spoke; inference may not overrule them.
                return existing
            if existing.value == value:
                support = existing.support + 1
                contradict = existing.contradict
                value_now = existing.value
            else:
                contradict = existing.contradict + 1
                support = existing.support
                if contradict >= existing.support:
                    # Enough counter-evidence: adopt the new value and start over
                    # rather than carrying the old tally across. The comparison
                    # is inclusive so that a tie goes to the newer signal -
                    # requiring contradiction to strictly exceed support would
                    # let an early lucky guess hold out indefinitely against
                    # equal evidence to the contrary.
                    value_now, support, contradict = value, 1, 0
                    evidence = []
                else:
                    value_now = existing.value
            if existing.evidence:
                evidence = list(existing.evidence)

        if evidence_id and evidence_id not in evidence:
            evidence.append(evidence_id)

        entry = ProfileEntry(
            key=key,
            value=value_now,
            confidence=self._confidence(support, moment),
            support=support,
            contradict=contradict,
            evidence=tuple(evidence),
            source=source,
            created_at=existing.created_at if existing else moment.isoformat(),
            updated_at=moment.isoformat(),
        )
        self.store.save_profile_entry(entry)
        return entry

    def observe_correction(
        self, correction: Correction, *, now: datetime | None = None
    ) -> list[ProfileEntry]:
        """Fold one structured correction into the profile."""
        impact = PROFILE_IMPACT.get(correction.kind)
        if impact is None:
            return []
        key, value = impact
        # A task-scoped correction is still evidence about the owner, just weaker;
        # it is recorded once and allowed to decay rather than being discarded.
        entry = self.observe(key, value, evidence_id=correction.event_id, now=now)
        return [entry]

    def observe_interaction(
        self,
        *,
        user_input: str,
        output: str,
        reward: float | None,
        event_id: str | None = None,
        now: datetime | None = None,
    ) -> list[ProfileEntry]:
        """Infer from an ordinary interaction, without any explicit feedback.

        Language is a strong signal and is safe to infer from the owner's own
        words. Verbosity is inferred only from *accepted* output - using rejected
        output would teach the system that the length it was just told off about
        is what the owner wants.
        """
        entries: list[ProfileEntry] = []

        language = detect_language(user_input)
        if language:
            entries.append(
                self.observe("language", language, evidence_id=event_id, now=now)
            )

        if reward is not None and reward >= 0.7:
            length = len(output or "")
            if length <= _SHORT_OUTPUT_CHARS:
                value = "concise"
            elif length >= _LONG_OUTPUT_CHARS:
                value = "detailed"
            else:
                value = ""
            if value:
                entries.append(
                    self.observe("verbosity", value, evidence_id=event_id, now=now)
                )

        return entries

    # ----------------------------------------------------------------- output

    def effective_confidence(self, entry: ProfileEntry, *, now: datetime | None = None) -> float:
        """Confidence after time decay, which is what the prompt should use.

        The stored value is a snapshot; the number that matters is the one
        computed now, so a belief stops counting purely because it went stale.
        """
        if entry.source == _HUMAN:
            return 1.0
        moment = now or _now()
        updated = _parse(entry.updated_at)
        if updated is None:
            return 0.0
        days = max(0.0, (moment - updated).total_seconds() / 86400.0)
        decayed = self._confidence(entry.support, moment) * (
            0.5 ** (days / self.half_life_days)
        )
        return max(0.0, min(1.0, decayed))

    def entries(self, *, now: datetime | None = None) -> list[ProfileEntry]:
        """All rows with their confidence recomputed as of ``now``."""
        moment = now or _now()
        out: list[ProfileEntry] = []
        for entry in self.store.list_profile_entries():
            out.append(
                ProfileEntry(
                    key=entry.key,
                    value=entry.value,
                    confidence=self.effective_confidence(entry, now=moment),
                    support=entry.support,
                    contradict=entry.contradict,
                    evidence=entry.evidence,
                    source=entry.source,
                    created_at=entry.created_at,
                    updated_at=entry.updated_at,
                )
            )
        return out

    def active(self, *, now: datetime | None = None) -> list[ProfileEntry]:
        """Rows confident enough to mention to the model."""
        return [
            e
            for e in self.entries(now=now)
            if e.confidence >= self.min_confidence
        ]

    def summary(self, *, now: datetime | None = None) -> str:
        """Render the profile for injection into the prompt.

        Deliberately hedged. The model is told these are fallible machine
        inferences so that a stale belief is treated as a hint rather than a
        constraint - which is also what stops the profile from silently
        overriding what the owner literally asked for in this turn.
        """
        rows = self.active(now=now)
        if not rows:
            return ""
        lines = [
            "- " + self._render_line(e) for e in sorted(rows, key=lambda e: -e.confidence)
        ]
        return "\n".join(lines)

    def _render_line(self, entry: ProfileEntry) -> str:
        basis = f"{entry.support} observation(s)" if entry.support else "no observations"
        origin = "set by owner" if entry.source == _HUMAN else "inferred"
        return (
            f"{entry.key} = {entry.value} ({origin}, from {basis}, "
            f"confidence {entry.confidence:.2f})"
        )

    # ------------------------------------------------------------ owner power

    def override(self, key: str, value: str) -> ProfileEntry:
        """The owner states a preference outright. Inference may not undo this."""
        moment = _now()
        existing = self.store.get_profile_entry(key)
        entry = ProfileEntry(
            key=key,
            value=value,
            confidence=1.0,
            support=(existing.support if existing else 0) + 1,
            contradict=0,
            evidence=existing.evidence if existing else (),
            source=_HUMAN,
            created_at=existing.created_at if existing else moment.isoformat(),
            updated_at=moment.isoformat(),
        )
        self.store.save_profile_entry(entry)
        return entry

    def forget(self, key: str) -> bool:
        """Drop one belief, including human overrides."""
        return self.store.delete_profile_entry(key)

    def reset(self) -> int:
        """Drop every belief. The panic button behind ``ratchet profile --reset``."""
        return self.store.clear_profile()

    def prune(self, *, now: datetime | None = None) -> int:
        """Delete rows whose confidence has decayed to nothing.

        Keeps the table from accumulating rumours that will never be revived,
        and makes the audit view honest about what the system currently believes.
        """
        removed = 0
        for entry in self.store.list_profile_entries():
            if entry.source == _HUMAN:
                continue
            if self.effective_confidence(entry, now=now) <= 0.01:
                removed += 1 if self.store.delete_profile_entry(entry.key) else 0
        return removed

    # ---------------------------------------------------------------- helpers

    def _confidence(self, support: int, _now: datetime) -> float:
        """Saturating confidence from a count of confirming observations."""
        if support <= 0:
            return 0.0
        return 1.0 - (_CONFIDENCE_SATURATION ** support)


@dataclass(frozen=True)
class ProfileSnapshot:
    """Profile state at one point in time, for the panel and tests."""

    entries: tuple[ProfileEntry, ...]

    def as_dict(self) -> dict[str, str]:
        return {e.key: e.value for e in self.entries}


def confidence_curve(
    support: int,
    *,
    elapsed_days: float = 0.0,
    half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Confidence for ``support`` observations after ``elapsed_days`` of silence.

    Exposed separately because the decay behaviour is the thing most likely to
    be tuned, and it is easier to reason about (and test) as a pure function.
    """
    if support <= 0:
        return 0.0
    base = 1.0 - (_CONFIDENCE_SATURATION ** support)
    return max(0.0, min(1.0, base * (0.5 ** (max(0.0, elapsed_days) / half_life_days))))
