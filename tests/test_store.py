"""Tests for the SQLite experience store.

Focus areas: CRUD, soft-delete (audit trail), reward accounting with
automatic deactivation of poisoned entries, and the permission rules
enforced at the storage layer.
"""

import sqlite3

import pytest

from ratchet.memory.provenance import Provenance
from ratchet.memory.store import ExperienceStore


@pytest.fixture()
def store(tmp_path):
    return ExperienceStore(tmp_path / "ratchet.db")


class TestCreateAndRead:
    def test_add_returns_uuid(self, store):
        eid = store.add(
            content="Prefer pytest over unittest for this project.",
            task_type="coding",
            provenance=Provenance.USER_RULE,
        )
        assert isinstance(eid, str) and len(eid) == 36

    def test_get_roundtrips_all_fields(self, store):
        eid = store.add(
            content="Keep answers under three sentences.",
            task_type="writing",
            provenance=Provenance.USER_RULE,
            kind="rule",
        )
        row = store.get(eid)
        assert row is not None
        assert row.content == "Keep answers under three sentences."
        assert row.task_type == "writing"
        assert row.provenance is Provenance.USER_RULE
        assert row.kind == "rule"
        assert row.active is True
        assert row.uses == 0
        assert row.score == pytest.approx(0.0)

    def test_get_unknown_id_returns_none(self, store):
        assert store.get("no-such-id") is None

    def test_list_excludes_soft_deleted_by_default(self, store):
        eid = store.add(content="temp", provenance=Provenance.AUTO)
        store.delete(eid)
        assert all(r.id != eid for r in store.list_active())

    def test_list_all_includes_soft_deleted(self, store):
        eid = store.add(content="temp", provenance=Provenance.AUTO)
        store.delete(eid)
        assert any(r.id == eid for r in store.list_all())


class TestDeleteIsSoftAndAuditable:
    def test_delete_sets_inactive_but_keeps_row(self, store):
        eid = store.add(content="bad rule", provenance=Provenance.AUTO)
        assert store.delete(eid) is True
        row = store.get(eid, include_inactive=True)
        assert row is not None
        assert row.active is False

    def test_delete_unknown_id_returns_false(self, store):
        assert store.delete("nope") is False


class TestPermissionEnforcement:
    def test_edit_allowed_for_user_rule(self, store):
        eid = store.add(content="old text", provenance=Provenance.USER_RULE)
        store.edit(eid, "new text")
        assert store.get(eid).content == "new text"

    def test_edit_forbidden_for_auto(self, store):
        # Fails if a human-edit path is ever opened for machine entries.
        eid = store.add(content="machine reflection", provenance=Provenance.AUTO)
        with pytest.raises(PermissionError, match="delete"):
            store.edit(eid, "tampered")

    def test_edit_forbidden_for_derived(self, store):
        eid = store.add(content="derived", provenance=Provenance.DERIVED)
        with pytest.raises(PermissionError, match="delete"):
            store.edit(eid, "tampered")

    def test_edit_rejects_overlong_user_rule(self, store):
        eid = store.add(content="ok", provenance=Provenance.USER_RULE)
        with pytest.raises(ValueError, match="200"):
            store.edit(eid, "x" * 201)

    def test_delete_allowed_for_every_provenance(self, store):
        ids = [
            store.add(content="a", provenance=Provenance.AUTO),
            store.add(content="b", provenance=Provenance.USER_RULE),
            store.add(content="c", provenance=Provenance.DERIVED),
        ]
        assert all(store.delete(i) for i in ids)

    def test_provenance_cannot_be_changed_by_edit(self, store):
        # Editing must not be a back door for privilege escalation.
        eid = store.add(content="user rule", provenance=Provenance.USER_RULE)
        with pytest.raises(TypeError):
            store.edit(eid, "text", provenance=Provenance.AUTO)


class TestRewardAccounting:
    def test_record_reward_updates_score_and_uses(self, store):
        eid = store.add(content="strategy", provenance=Provenance.AUTO)
        store.record_reward(eid, 1.0)
        store.record_reward(eid, 0.5)
        row = store.get(eid)
        assert row.score == pytest.approx(1.5)
        assert row.uses == 2

    def test_negative_rewards_accumulate(self, store):
        eid = store.add(content="strategy", provenance=Provenance.AUTO)
        store.record_reward(eid, -1.0)
        assert store.get(eid).score == pytest.approx(-1.0)

    def test_poisoned_entry_auto_deactivates_below_threshold(self, store):
        # This is the main defence against experience poisoning.
        eid = store.add(content="harmful", provenance=Provenance.AUTO)
        for _ in range(3):
            store.record_reward(eid, -1.0)
        assert store.get(eid, include_inactive=True).active is False
        assert eid not in [r.id for r in store.list_active()]

    def test_threshold_defaults_to_negative_three(self, store):
        assert store.deactivation_threshold == pytest.approx(-3.0)

    def test_record_reward_on_inactive_entry_is_noop(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO)
        store.delete(eid)
        store.record_reward(eid, 5.0)
        assert store.get(eid, include_inactive=True).score == pytest.approx(0.0)


class TestTrustLevel:
    """Provenance says WHO wrote an entry; trust_level says whether the
    underlying content came from a trusted source. Both are needed to stop
    third-party content being laundered into persistent instructions."""

    def test_default_trust_level_is_trusted(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO)
        assert store.get(eid).trust_level == "trusted"

    def test_trust_level_is_persisted(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO, trust_level="untrusted")
        assert store.get(eid).trust_level == "untrusted"

    def test_unknown_trust_level_is_rejected(self, store):
        with pytest.raises(ValueError, match="trust_level"):
            store.add(content="x", provenance=Provenance.AUTO, trust_level="maybe")

    def test_trust_level_cannot_be_changed_by_edit(self, store):
        # Editing must not be a back door for laundering untrusted content.
        eid = store.add(
            content="x", provenance=Provenance.USER_RULE, trust_level="untrusted"
        )
        store.edit(eid, "y")
        assert store.get(eid).trust_level == "untrusted"

    def test_untrusted_entries_can_be_filtered_out(self, store):
        store.add(content="safe", provenance=Provenance.AUTO, trust_level="trusted")
        store.add(content="tainted", provenance=Provenance.AUTO, trust_level="untrusted")
        assert [r.content for r in store.list_active(trust_level="trusted")] == ["safe"]


class TestStatePersistence:
    """Without this the bandit restarts from scratch on every CLI invocation,
    which would silently break learning across separate processes."""

    def test_missing_key_returns_none(self, store):
        assert store.load_state("nope") is None

    def test_state_roundtrips(self, store):
        store.save_state("policy", {"arms": {"a": {"alpha": 2.0}}})
        assert store.load_state("policy") == {"arms": {"a": {"alpha": 2.0}}}

    def test_state_overwrites_previous_value(self, store):
        store.save_state("policy", {"v": 1})
        store.save_state("policy", {"v": 2})
        assert store.load_state("policy") == {"v": 2}

    def test_state_survives_reopen(self, tmp_path):
        path = tmp_path / "ratchet.db"
        first = ExperienceStore(path)
        first.save_state("policy", {"arms": {}})
        first.close()
        second = ExperienceStore(path)
        assert second.load_state("policy") == {"arms": {}}
        second.close()


class TestResolveExperienceId:
    """Shared by the CLI `rm` and the MCP `forget` tool, so owners never have to
    copy a full uuid. Ambiguity must fail closed."""

    def test_full_id_resolves(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO)
        assert store.resolve_id(eid) == eid

    def test_unique_prefix_resolves(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO)
        assert store.resolve_id(eid[:8]) == eid

    def test_unknown_id_returns_none(self, store):
        assert store.resolve_id("no-such-id") is None

    def test_empty_prefix_is_ambiguous_when_multiple_exist(self, store):
        store.add(content="a", provenance=Provenance.AUTO)
        store.add(content="b", provenance=Provenance.AUTO)
        with pytest.raises(ValueError, match="ambiguous"):
            store.resolve_id("")

    def test_empty_prefix_resolves_when_exactly_one_exists(self, store):
        eid = store.add(content="only one", provenance=Provenance.AUTO)
        assert store.resolve_id("") == eid

    def test_error_reports_the_match_count(self, store):
        for _ in range(3):
            store.add(content="x", provenance=Provenance.AUTO)
        with pytest.raises(ValueError, match="3"):
            store.resolve_id("")

    def test_resolves_inactive_entries_too(self, store):
        # Owners must be able to delete something the system auto-deactivated.
        eid = store.add(content="x", provenance=Provenance.AUTO)
        store.delete(eid)
        assert store.resolve_id(eid[:8]) == eid

    def test_prefix_matching_is_case_insensitive(self, store):
        # Ids are lowercase uuid4 read off a terminal and typed back by hand.
        # Treating a capitalised prefix as a different id looks like data loss.
        eid = store.add(content="x", provenance=Provenance.AUTO)
        assert store.resolve_id(eid[:8].upper()) == eid
        assert store.resolve_id(eid.upper()) == eid

    def test_surrounding_whitespace_is_ignored(self, store):
        eid = store.add(content="x", provenance=Provenance.AUTO)
        assert store.resolve_id(f"  {eid[:8]}  ") == eid


class TestUpdateEventReward:
    """An explicit human verdict arrives after the interaction is already
    recorded, so the audit trail must be correctable - otherwise a later
    thumbs-down would leave a stale reward that contradicts reality."""

    def test_updates_reward_and_source(self, store):
        from ratchet.improve.verifier import (
            Contribution,
            Reward,
            RewardSource,
        )

        event_id = store.record_event(task="t", output="o", reward=0.2)
        reward = Reward(
            value=0.0,
            source=RewardSource.USER,
            contributions=(
                Contribution(
                    source=RewardSource.USER,
                    name="user_feedback",
                    value=0.0,
                    weight=1.0,
                ),
            ),
            confidence=1.0,
        )
        store.update_event_reward(event_id, reward)
        event = store.list_events()[0]
        assert event.reward == pytest.approx(0.0)
        assert event.reward_source == "user"
        assert event.confidence == pytest.approx(1.0)

    def test_unknown_event_id_is_a_noop(self, store):
        from ratchet.improve.verifier import Reward

        store.record_event(task="t", output="o", reward=1.0)
        store.update_event_reward("nope", Reward(value=0.0, source=None, confidence=0.0))
        assert store.list_events()[0].reward == pytest.approx(1.0)

    def test_none_reward_is_preserved_as_none(self, store):
        from ratchet.improve.verifier import Reward

        event_id = store.record_event(task="t", output="o", reward=None)
        store.update_event_reward(event_id, Reward(value=None, source=None, confidence=0.0))
        assert store.list_events()[0].reward is None


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        path = tmp_path / "ratchet.db"
        first = ExperienceStore(path)
        eid = first.add(content="persisted", provenance=Provenance.USER_RULE)
        first.close()

        second = ExperienceStore(path)
        assert second.get(eid).content == "persisted"
        second.close()

    def test_storage_is_a_single_sqlite_file(self, tmp_path):
        path = tmp_path / "ratchet.db"
        store = ExperienceStore(path)
        store.add(content="x", provenance=Provenance.AUTO)
        store.close()
        # Must be a real SQLite database, not a pile of loose files.
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM experience").fetchone()[0] == 1
