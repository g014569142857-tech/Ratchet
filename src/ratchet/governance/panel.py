"""Governance panel: the surface through which the owner stays in charge.

This module is the executable form of one rule: **the machine may add, the owner
may subtract.** Reflection, distillation and evolution write freely; the human
side of the API offers view, delete, restore and structured correction - and
nothing that lets new prose reach the store except through the bounded
correction envelope.

The asymmetry is not a UX preference, it is the security argument. Deletion can
only ever remove capability, so it is safe to grant without review. Insertion
changes what the model will be told on every future task, so it is not.

Everything both the CLI and the web console do goes through here, so there is
exactly one place where the permission rules are enforced and one place to audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ratchet.governance.correction import PROFILE_IMPACT, Correction, CorrectionKind
from ratchet.memory.profile import Profile
from ratchet.memory.provenance import Provenance, rule_for
from ratchet.memory.store import Experience, ExperienceStore


@dataclass(frozen=True)
class ExperienceView:
    """One row as the owner should see it, including what they may do to it."""

    id: str
    content: str
    provenance: str
    kind: str
    trust_level: str
    score: float
    uses: int
    active: bool
    created_at: str
    updated_at: str
    editable: bool
    deletable: bool
    can_restore: bool
    priority: int
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "provenance": self.provenance,
            "kind": self.kind,
            "trust_level": self.trust_level,
            "score": self.score,
            "uses": self.uses,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "editable": self.editable,
            "deletable": self.deletable,
            "can_restore": self.can_restore,
            "priority": self.priority,
            "note": self.note,
        }


@dataclass(frozen=True)
class PanelSnapshot:
    """Everything the panel shows in one read."""

    experiences: tuple[ExperienceView, ...]
    profile: tuple[dict[str, object], ...]
    counts: dict[str, int] = field(default_factory=dict)


class GovernancePanel:
    """Read, retract and correct - the owner's whole API surface."""

    def __init__(self, store: ExperienceStore, profile: Profile | None = None) -> None:
        self.store = store
        self.profile = profile or Profile(store)

    # ------------------------------------------------------------------- view

    def _view(self, exp: Experience) -> ExperienceView:
        policy = rule_for(exp.provenance)
        return ExperienceView(
            id=exp.id,
            content=exp.content,
            provenance=exp.provenance.value,
            kind=exp.kind,
            trust_level=exp.trust_level,
            score=exp.score,
            uses=exp.uses,
            active=exp.active,
            created_at=exp.created_at,
            updated_at=exp.updated_at,
            editable=policy.human_editable,
            deletable=policy.human_deletable,
            can_restore=not exp.active,
            priority=policy.priority,
        )

    def view(
        self,
        *,
        include_inactive: bool = False,
        provenance: Provenance | str | None = None,
        kind: str | None = None,
    ) -> tuple[ExperienceView, ...]:
        """List what the system currently believes, most trusted first."""
        entries = (
            self.store.list_all() if include_inactive else self.store.list_active()
        )
        if provenance is not None:
            wanted = Provenance(provenance)
            entries = [e for e in entries if e.provenance is wanted]
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        views = [self._view(e) for e in entries]
        # Human rules first, then untrusted, then by score: the ordering the
        # owner needs in order to spot a bad machine inference quickly.
        views.sort(key=lambda v: (-v.priority, -v.score, v.id))
        return tuple(views)

    def snapshot(self, *, include_inactive: bool = False) -> PanelSnapshot:
        experiences = self.view(include_inactive=include_inactive)
        profile = tuple(
            {
                "key": e.key,
                "value": e.value,
                "confidence": round(e.confidence, 3),
                "support": e.support,
                "contradict": e.contradict,
                "evidence": list(e.evidence),
                "source": e.source,
                "updated_at": e.updated_at,
            }
            for e in self.profile.entries()
        )
        counts = {
            "total": len(experiences),
            "active": sum(1 for v in experiences if v.active),
            "machine_written": sum(
                1
                for v in experiences
                if v.provenance in (Provenance.AUTO.value, Provenance.DERIVED.value)
            ),
            "human_written": sum(
                1 for v in experiences if v.provenance == Provenance.USER_RULE.value
            ),
        }
        return PanelSnapshot(experiences=experiences, profile=profile, counts=counts)

    # ------------------------------------------------------------- retracting

    def delete(self, prefix: str) -> str:
        """Soft-delete by id or unique prefix. Always permitted."""
        resolved = self.store.resolve_id(prefix)
        if resolved is None:
            raise KeyError(f"experience not found: {prefix!r}")
        self.store.delete(resolved)
        return resolved

    def restore(self, prefix: str) -> str:
        """Undo a deletion. Text that was already reviewed comes back as it was."""
        resolved = self.store.resolve_id(prefix)
        if resolved is None:
            raise KeyError(f"experience not found: {prefix!r}")
        if not self.store.restore(resolved):
            raise KeyError(f"experience not found: {prefix!r}")
        return resolved

    def purge(self, prefix: str) -> str:
        """Physically erase one entry. Irreversible; requires an explicit call."""
        resolved = self.store.resolve_id(prefix)
        if resolved is None:
            raise KeyError(f"experience not found: {prefix!r}")
        self.store.purge(resolved)
        return resolved

    # ------------------------------------------------------------- correcting

    def correct(self, correction: Correction) -> tuple[str, tuple[str, ...]]:
        """Submit a structured correction.

        Returns ``(experience_id, profile_keys_touched)``. The correction is
        stored as a ``derived`` entry - machine-authored, so the owner cannot
        later edit its text, only delete it - and its note has already been
        escaped and framed by ``Correction`` itself.
        """
        content = correction.render()
        experience_id = self.store.add(
            content=content,
            provenance=Provenance.DERIVED,
            task_type=correction.task_type,
            kind="correction",
        )
        touched: tuple[str, ...] = ()
        if correction.kind in PROFILE_IMPACT:
            entries = self.profile.observe_correction(correction)
            touched = tuple(e.key for e in entries)
        return experience_id, touched

    def override_profile(self, key: str, value: str) -> None:
        """The owner states a preference outright, outranking inference."""
        self.profile.override(key, value)

    def forget_profile(self, key: str) -> bool:
        return self.profile.forget(key)

    def reset_profile(self) -> int:
        return self.profile.reset()

    # ----------------------------------------------------------------- audit

    def explain(self, prefix: str) -> dict[str, object]:
        """Show why one entry exists, as far as the audit trail allows."""
        resolved = self.store.resolve_id(prefix)
        if resolved is None:
            raise KeyError(f"experience not found: {prefix!r}")
        entry = self.store.get(resolved, include_inactive=True)
        assert entry is not None
        events = [
            e
            for e in self.store.list_events()
            if resolved in e.retrieved_ids
        ]
        return {
            "experience": self._view(entry).as_dict(),
            "policy": {
                "human_editable": rule_for(entry.provenance).human_editable,
                "human_deletable": rule_for(entry.provenance).human_deletable,
                "max_chars": rule_for(entry.provenance).max_chars,
                "priority": rule_for(entry.provenance).priority,
            },
            "times_recalled": len(events),
            "mean_reward_when_recalled": (
                sum(e.reward for e in events if e.reward is not None)
                / len([e for e in events if e.reward is not None])
                if any(e.reward is not None for e in events)
                else None
            ),
        }


CORRECTION_KINDS: tuple[str, ...] = tuple(k.value for k in CorrectionKind)
