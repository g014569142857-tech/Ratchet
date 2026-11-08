"""Turn accumulated experience into training data (Route A).

Route B - the context-level loop - is the default because it works on any model.
But everything it accumulates is *also* the training signal Route A needs: a log
of real tasks, real outputs and measured rewards for the same owner's actual work.
That is a far better dataset than synthetic prompts, and it is the payoff for
having run Route B first.

Three shapes are produced, matching the three ways people actually train:

* **SFT** - keep the good completions and imitate them. Simplest, and the right
  first step when you have little data.
* **DPO** - pair a good completion against a bad one for the *same* task. This is
  the one Ratchet is built for, because the loop naturally produces both sides.
* **GRPO** - group every completion of one task together and let the trainer work
  out advantages within the group. Needs several samples per task.

A note on honesty about reward provenance
-----------------------------------------
Each record carries the ``reward_source`` it was scored under. A pair built from
two *judge* scores is much weaker evidence than a pair where one side passed an
objective check, and anyone training on this data deserves to be able to filter
on that. Silently mixing the two is how people end up training on a judge's
length bias.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ratchet.memory.store import Event

DEFAULT_MIN_REWARD = 0.7
"""Above this a completion is treated as a good example for SFT."""

DEFAULT_MAX_REWARD = 0.35
"""Below this a completion is a candidate rejection in a preference pair."""

DEFAULT_MIN_GAP = 0.3
"""Minimum reward distance before two completions form a pair."""


@dataclass(frozen=True)
class DatasetStats:
    """What a build produced, so a short run does not look like a broken one."""

    considered: int = 0
    written: int = 0
    skipped_unmeasured: int = 0
    skipped_singleton: int = 0
    skipped_small_gap: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "considered": self.considered,
            "written": self.written,
            "skipped_unmeasured": self.skipped_unmeasured,
            "skipped_singleton": self.skipped_singleton,
            "skipped_small_gap": self.skipped_small_gap,
        }


def _prompt_of(event: Event) -> str:
    """Reconstruct the task text a completion answered.

    The stored ``user_input`` is the owner's raw message; the ``task`` is the
    framing. Recording only one of them would lose either the instruction or the
    context, and a model trained on half of it learns the wrong mapping.
    """
    parts = [p for p in (event.task, event.user_input) if p]
    return "\n\n".join(parts).strip()


def build_sft_records(
    events: list[Event], *, min_reward: float = DEFAULT_MIN_REWARD
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """High-reward completions as ``prompt``/``completion`` pairs."""
    records: list[dict[str, Any]] = []
    unmeasured = 0
    for event in events:
        if event.reward is None:
            unmeasured += 1
            continue
        if event.reward < min_reward:
            continue
        if not (event.output or "").strip():
            continue
        records.append(
            {
                "prompt": _prompt_of(event),
                "completion": event.output,
                "reward": event.reward,
                "reward_source": event.reward_source,
                "variant_id": event.variant_id,
                "event_id": event.id,
            }
        )
    stats = DatasetStats(
        considered=len(events), written=len(records), skipped_unmeasured=unmeasured
    )
    return records, stats


def build_dpo_pairs(
    events: list[Event],
    *,
    min_reward: float = DEFAULT_MIN_REWARD,
    max_reward: float = DEFAULT_MAX_REWARD,
    min_gap: float = DEFAULT_MIN_GAP,
    require_different_sources: bool = False,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Pair a good completion against a bad one for the same task.

    Grouping key is the *task text*, not the raw input: the loop records one
    event per interaction, and it is the same task attempted more than once that
    yields the contrast a preference pair needs.

    ``require_different_sources`` keeps a pair only when the two sides were
    scored differently - the strongest evidence, since one side was verified
    objectively and the other was not.
    """
    groups: dict[str, list[Event]] = defaultdict(list)
    unmeasured = 0
    for event in events:
        if event.reward is None:
            unmeasured += 1
            continue
        if not (event.output or "").strip():
            continue
        groups[_prompt_of(event)].append(event)

    records: list[dict[str, Any]] = []
    singletons = 0
    small_gap = 0
    for prompt, group in groups.items():
        if len(group) < 2:
            singletons += 1
            continue
        best = max(group, key=lambda e: e.reward)  # type: ignore[arg-type,return-value]
        worst = min(group, key=lambda e: e.reward)  # type: ignore[arg-type,return-value]
        if best is worst or best.reward is None or worst.reward is None:
            singletons += 1
            continue
        if best.reward < min_reward or worst.reward > max_reward:
            small_gap += 1
            continue
        if best.reward - worst.reward < min_gap:
            small_gap += 1
            continue
        if require_different_sources and best.reward_source == worst.reward_source:
            small_gap += 1
            continue
        records.append(
            {
                "prompt": prompt,
                "chosen": best.output,
                "rejected": worst.output,
                "chosen_reward": best.reward,
                "rejected_reward": worst.reward,
                "chosen_source": best.reward_source,
                "rejected_source": worst.reward_source,
                "chosen_variant": best.variant_id,
                "rejected_variant": worst.variant_id,
            }
        )

    stats = DatasetStats(
        considered=len(events),
        written=len(records),
        skipped_unmeasured=unmeasured,
        skipped_singleton=singletons,
        skipped_small_gap=small_gap,
    )
    return records, stats


def build_grpo_groups(
    events: list[Event], *, min_group: int = 2
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Every completion of one task, with its reward, for group-relative training."""
    groups: dict[str, list[Event]] = defaultdict(list)
    unmeasured = 0
    for event in events:
        if event.reward is None:
            unmeasured += 1
            continue
        if not (event.output or "").strip():
            continue
        groups[_prompt_of(event)].append(event)

    records: list[dict[str, Any]] = []
    singletons = 0
    for prompt, group in groups.items():
        if len(group) < min_group:
            singletons += 1
            continue
        records.append(
            {
                "prompt": prompt,
                "completions": [
                    {
                        "completion": e.output,
                        "reward": e.reward,
                        "reward_source": e.reward_source,
                        "variant_id": e.variant_id,
                    }
                    for e in group
                ],
            }
        )

    stats = DatasetStats(
        considered=len(events),
        written=len(records),
        skipped_unmeasured=unmeasured,
        skipped_singleton=singletons,
    )
    return records, stats


def build(
    events: list[Event], fmt: str = "dpo", **kwargs: Any
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Dispatch on format name. Raises ``ValueError`` for an unknown format."""
    if fmt == "sft":
        return build_sft_records(events, **kwargs)
    if fmt == "dpo":
        return build_dpo_pairs(events, **kwargs)
    if fmt == "grpo":
        return build_grpo_groups(events, **kwargs)
    raise ValueError(f"unknown dataset format: {fmt!r}")


def write_jsonl(records: list[dict[str, Any]], path: str) -> int:
    """Write records as one JSON object per line. Returns the count written."""
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
