"""Tests for the inferred user profile.

The profile is the module most likely to be quietly wrong, so these tests pin the
three behaviours that make it safe rather than merely clever: confidence decays,
contradiction can flip a belief, and a human override is never auto-reversed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ratchet.governance.correction import Correction, CorrectionKind, CorrectionScope
from ratchet.memory.profile import Profile, confidence_curve, detect_language
from ratchet.memory.store import ExperienceStore

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def profile(tmp_path):
    store = ExperienceStore(tmp_path / "p.db")
    yield Profile(store)
    store.close()


# ------------------------------------------------------------- observation


def test_first_observation_is_uncertain(profile):
    entry = profile.observe("verbosity", "concise", evidence_id="e1", now=BASE)
    assert entry.support == 1
    assert 0.0 < entry.confidence < 1.0


def test_confidence_saturates_rather_than_reaching_one(profile):
    for i in range(6):
        entry = profile.observe("verbosity", "concise", evidence_id=f"e{i}", now=BASE)
    assert entry.support == 6
    assert entry.confidence < 1.0
    assert entry.confidence > 0.9


def test_evidence_ids_accumulate_and_deduplicate(profile):
    profile.observe("verbosity", "concise", evidence_id="e1", now=BASE)
    profile.observe("verbosity", "concise", evidence_id="e1", now=BASE)
    entry = profile.observe("verbosity", "concise", evidence_id="e2", now=BASE)
    assert entry.evidence == ("e1", "e2")


# ------------------------------------------------------------- contradiction


def test_contradiction_flips_the_belief_once_it_dominates(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    profile.observe("verbosity", "concise", evidence_id="b", now=BASE)
    # Two against one: the new value wins and the tally restarts.
    profile.observe("verbosity", "detailed", evidence_id="c", now=BASE)
    entry = profile.observe("verbosity", "detailed", evidence_id="d", now=BASE)
    assert entry.value == "detailed"
    assert entry.support == 1
    assert entry.contradict == 0


def test_a_single_contradiction_does_not_flip(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    profile.observe("verbosity", "concise", evidence_id="b", now=BASE)
    entry = profile.observe("verbosity", "detailed", evidence_id="c", now=BASE)
    assert entry.value == "concise"
    assert entry.contradict == 1


# -------------------------------------------------------------------- decay


def test_confidence_decays_with_silence(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    entry = profile.store.get_profile_entry("verbosity")
    assert entry is not None
    fresh = profile.effective_confidence(entry, now=BASE)
    stale = profile.effective_confidence(entry, now=BASE + timedelta(days=14))
    assert stale < fresh


def test_decay_destroys_the_rumour_eventually(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    entry = profile.store.get_profile_entry("verbosity")
    assert entry is not None
    long_after = BASE + timedelta(days=200)
    assert profile.effective_confidence(entry, now=long_after) < 0.01


def test_prune_drops_decayed_beliefs(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    assert profile.prune(now=BASE + timedelta(days=200)) == 1
    assert profile.store.list_profile_entries() == []


def test_prune_leaves_human_overrides_alone(profile):
    profile.override("verbosity", "detailed")
    assert profile.prune(now=BASE + timedelta(days=500)) == 0
    assert profile.store.get_profile_entry("verbosity") is not None


def test_new_evidence_revives_a_decayed_belief(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    entry = profile.store.get_profile_entry("verbosity")
    decayed = profile.effective_confidence(entry, now=BASE + timedelta(days=60))  # type: ignore[arg-type]
    assert decayed < 0.25
    later = BASE + timedelta(days=60)
    revived = profile.observe("verbosity", "concise", evidence_id="b", now=later)
    assert revived.confidence > decayed


def test_confidence_curve_is_a_pure_function():
    assert confidence_curve(0) == 0.0
    assert confidence_curve(1) > confidence_curve(1, elapsed_days=30)
    assert confidence_curve(3, elapsed_days=0) > confidence_curve(1, elapsed_days=0)
    assert 0.0 <= confidence_curve(10, elapsed_days=1) <= 1.0


# ------------------------------------------------------------------ override


def test_human_override_is_never_overturned_by_inference(profile):
    profile.override("verbosity", "detailed")
    entry = profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    assert entry.value == "detailed"
    assert entry.source == "human_override"


def test_human_override_reports_full_confidence(profile):
    profile.override("tone", "blunt")
    entries = profile.entries()
    assert entries[0].confidence == 1.0


def test_reset_clears_everything_including_overrides(profile):
    profile.override("tone", "blunt")
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    assert profile.reset() == 2
    assert profile.store.list_profile_entries() == []


# ---------------------------------------------------------------- inference


def test_correction_feeds_the_matching_dimension(profile):
    correction = Correction.build(
        kind=CorrectionKind.TOO_LONG, scope=CorrectionScope.GLOBAL, note="too long"
    )
    entries = profile.observe_correction(correction, now=BASE)
    assert [e.key for e in entries] == ["verbosity"]
    assert entries[0].value == "concise"


def test_language_is_inferred_from_the_owners_own_words(profile):
    profile.observe_interaction(
        user_input="请用中文解释一下这个概念",
        output="...",
        reward=None,
        event_id="e1",
        now=BASE,
    )
    assert profile.store.get_profile_entry("language").value == "zh"  # type: ignore[union-attr]


def test_language_is_not_guessed_from_a_two_character_input(profile):
    assert detect_language("好") is None
    profile.observe_interaction(
        user_input="好", output="ok", reward=1.0, event_id="e1", now=BASE
    )
    assert profile.store.get_profile_entry("language") is None


def test_verbosity_is_only_inferred_from_accepted_output(profile):
    # A thumbs-down must not teach the system that the length it was just
    # criticised for is what the owner wants.
    profile.observe_interaction(
        user_input="x" * 40, output="short", reward=0.0, event_id="e1", now=BASE
    )
    assert profile.store.get_profile_entry("verbosity") is None

    profile.observe_interaction(
        user_input="x" * 40, output="short", reward=1.0, event_id="e2", now=BASE
    )
    assert profile.store.get_profile_entry("verbosity").value == "concise"  # type: ignore[union-attr]


def test_purging_an_experience_removes_its_evidence_citation(profile, tmp_path):

    store = profile.store
    exp_id = store.add(content="a rule", provenance="auto", kind="rule")
    profile.observe("tone", "blunt", evidence_id=exp_id, now=BASE)
    assert profile.store.get_profile_entry("tone").evidence == (exp_id,)  # type: ignore[union-attr]
    store.purge(exp_id)
    assert profile.store.get_profile_entry("tone").evidence == ()  # type: ignore[union-attr]


# ------------------------------------------------------------------ summary


def test_summary_omits_beliefs_below_the_threshold(profile):
    profile.observe("tone", "blunt", evidence_id="a", now=BASE)
    stale = BASE + timedelta(days=120)
    assert profile.summary(now=stale) == ""
    assert profile.summary(now=BASE) != ""


def test_summary_hedges_that_the_beliefs_are_inferred(profile):
    profile.observe("verbosity", "concise", evidence_id="a", now=BASE)
    text = profile.summary(now=BASE)
    assert "verbosity" in text
    assert "inferred" in text
