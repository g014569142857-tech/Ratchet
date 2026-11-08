"""Tests for evaluation and dataset export.

Both modules exist to answer "is this actually working?" with a number rather
than a feeling, so the tests concentrate on the cases where a number would be
misleading: too little data, unmeasured interactions, and tasks seen only once.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ratchet.eval.growth import EvalSuite, EvalTask, compare, growth_curve, run_suite
from ratchet.memory.store import ExperienceStore
from ratchet.train.dataset import build, build_dpo_pairs, build_grpo_groups, build_sft_records

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    s = ExperienceStore(tmp_path / "e.db")
    yield s
    s.close()


def _record(store, task, reward, output, day=0, variant="default", source="objective"):
    return store.record_event(
        session_id="s",
        task=task,
        user_input=f"input for {task}",
        output=output,
        variant_id=variant,
        reward=reward,
        reward_source=source,
    )


# --------------------------------------------------------------- growth curve


def test_no_curve_without_measured_interactions(store):
    _record(store, "t", None, "x", day=0)
    curve = growth_curve(store.list_events())
    assert curve.buckets == ()
    assert curve.verdict == "insufficient data"
    assert curve.unmeasured_total == 1


def test_unmeasured_interactions_are_not_counted_as_failures(store):
    # Logging an unmeasured interaction as zero would make the curve a measure
    # of how often the verifier had an opinion, not of performance.
    for i in range(8):
        _record(store, f"good {i}", 1.0, "x", day=i)
    for i in range(4):
        _record(store, f"unmeasured {i}", None, "x", day=i + 8)
    curve = growth_curve(store.list_events(), buckets=2)
    assert curve.measured_total == 8
    assert curve.unmeasured_total == 4
    assert curve.first_mean == 1.0
    assert curve.last_mean == 1.0


def test_rising_rewards_are_reported_as_improving(store):
    for i in range(12):
        _record(store, f"t{i}", 0.2 if i < 6 else 0.9, "x", day=i)
    curve = growth_curve(store.list_events(), buckets=4)
    assert curve.verdict == "improving"
    assert curve.delta > 0


def test_falling_rewards_are_reported_as_regressing(store):
    for i in range(12):
        _record(store, f"t{i}", 0.9 if i < 6 else 0.2, "x", day=i)
    curve = growth_curve(store.list_events(), buckets=4)
    assert curve.verdict == "regressing"


def test_a_flat_history_is_not_dressed_up_as_improvement(store):
    for i in range(12):
        _record(store, f"t{i}", 0.5, "x", day=i)
    curve = growth_curve(store.list_events(), buckets=4)
    assert curve.verdict == "flat"


def test_thin_buckets_are_refused(store):
    for i in range(3):
        _record(store, f"t{i}", 1.0, "x", day=i)
    assert growth_curve(store.list_events(), buckets=4).buckets == ()


def test_buckets_must_be_at_least_two(store):
    with pytest.raises(ValueError):
        growth_curve(store.list_events(), buckets=1)


def test_curve_serialises(store):
    for i in range(8):
        _record(store, f"t{i}", 0.5, "x", day=i)
    payload = growth_curve(store.list_events()).as_dict()
    assert json.dumps(payload)


# ------------------------------------------------------------------- suites


def _suite():
    return EvalSuite(
        name="demo",
        tasks=(
            EvalTask(
                name="mentions-lora",
                task="explain LoRA",
                user_input="briefly",
                check=lambda out: 1.0 if "lora" in out.lower() else 0.0,
            ),
            EvalTask(
                name="length",
                task="summarise",
                user_input="x",
                check=lambda out: 1.0 if len(out) < 80 else 0.0,
            ),
        ),
    )


def test_suite_scores_are_comparable_across_runs():
    suite = _suite()
    before = run_suite(suite, lambda t, u: "I cannot help with that at all, sorry.")
    after = run_suite(suite, lambda t, u: "LoRA: low-rank adaptation.")
    report = compare(before, after)
    assert report["delta"] > 0
    assert report["graded"] == 2
    assert report["verdict"] == "improving"


def test_compare_reports_per_task_deltas():
    suite = _suite()
    before = run_suite(suite, lambda t, u: "nope")
    after = run_suite(suite, lambda t, u: "LoRA")
    report = compare(before, after)
    assert "mentions-lora" in report["per_task_delta"]


def test_suite_handles_ungradable_tasks():
    suite = EvalSuite(
        name="na",
        tasks=(
            EvalTask(name="n", task="t", user_input="u", check=lambda out: None),
        ),
    )
    run = run_suite(suite, lambda t, u: "anything")
    assert run.graded == 0
    assert run.mean == 0.0


# ------------------------------------------------------------------ datasets


def test_sft_keeps_only_high_reward_completions(store):
    _record(store, "good", 0.9, "good output")
    _record(store, "bad", 0.1, "bad output")
    _record(store, "none", None, "unmeasured output")
    events = store.list_events()
    records, stats = build_sft_records(events)
    assert [r["completion"] for r in records] == ["good output"]
    assert stats.skipped_unmeasured == 1


def test_dpo_needs_two_attempts_of_the_same_task(store):
    _record(store, "same task", 0.9, "chosen output")
    _record(store, "same task", 0.1, "rejected output")
    events = store.list_events()
    records, stats = build_dpo_pairs(events)
    assert len(records) == 1
    assert records[0]["chosen"] == "chosen output"
    assert records[0]["rejected"] == "rejected output"
    assert stats.written == 1


def test_tasks_seen_once_yield_no_preference_pair(store):
    _record(store, "task a", 0.9, "a")
    _record(store, "task b", 0.1, "b")
    records, stats = build_dpo_pairs(store.list_events())
    assert records == []
    assert stats.skipped_singleton == 2


def test_a_small_reward_gap_is_not_treated_as_a_preference(store):
    _record(store, "same", 0.72, "slightly better")
    _record(store, "same", 0.68, "slightly worse")
    records, stats = build_dpo_pairs(store.list_events(), min_gap=0.3)
    assert records == []
    assert stats.skipped_small_gap == 1


def test_dpo_records_the_reward_provenance_of_each_side(store):
    _record(store, "same", 0.9, "verified", source="objective")
    _record(store, "same", 0.1, "unverified", source="judge")
    records, _ = build_dpo_pairs(store.list_events())
    assert records[0]["chosen_source"] == "objective"
    assert records[0]["rejected_source"] == "judge"


def test_grpo_groups_every_attempt_of_a_task(store):
    for reward in (0.9, 0.4, 0.7):
        _record(store, "same", reward, f"out-{reward}")
    records, stats = build_grpo_groups(store.list_events())
    assert len(records) == 1
    assert len(records[0]["completions"]) == 3


def test_build_dispatches_and_rejects_unknown_formats(store):
    _record(store, "t", 0.9, "out")
    events = store.list_events()
    assert build(events, "sft")[0]
    # A task attempted once cannot produce a preference pair, so dpo exists but
    # is empty here - the point of this test is dispatch, not content.
    assert build(events, "dpo")[0] == []
    assert build(events, "grpo")[0] == []
    with pytest.raises(ValueError):
        build(events, "mystery")


def test_exported_jsonl_is_one_object_per_line(store, tmp_path):
    from ratchet.train.dataset import write_jsonl

    _record(store, "good", 0.9, "out")
    records, _ = build_sft_records(store.list_events())
    path = tmp_path / "out.jsonl"
    assert write_jsonl(records, str(path)) == 1
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1
    assert json.loads(lines[0])["completion"] == "out"
