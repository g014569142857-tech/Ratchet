"""Tests for the three-tier provenance / permission model.

These are the safety-critical rules the user asked for:
  * machine-written entries: humans may DELETE, never EDIT
  * user-written rules: humans may delete and edit, hard length cap
  * derived entries: humans may DELETE, never EDIT

Each test below names the production change that would make it fail.
"""

import pytest

from ratchet.memory.provenance import PROVENANCE_RULES, Provenance


class TestProvenanceLevels:
    def test_three_levels_exist(self):
        # Fails if any of the three tiers is renamed or removed.
        assert {p.value for p in Provenance} == {"auto", "user_rule", "derived"}

    def test_auto_entries_are_not_human_editable(self):
        # Fails if auto entries are ever made editable.
        assert PROVENANCE_RULES[Provenance.AUTO].human_editable is False

    def test_derived_entries_are_not_human_editable(self):
        assert PROVENANCE_RULES[Provenance.DERIVED].human_editable is False

    def test_user_rule_entries_are_human_editable(self):
        assert PROVENANCE_RULES[Provenance.USER_RULE].human_editable is True

    def test_every_level_is_human_deletable(self):
        # Deletion must be allowed for all tiers - that is the escape hatch.
        for level in Provenance:
            assert PROVENANCE_RULES[level].human_deletable is True

    def test_auto_entries_cannot_be_promoted_to_rule(self):
        # An auto entry must never be able to claim user_rule authority.
        assert PROVENANCE_RULES[Provenance.AUTO].promotable_to_user_rule is False

    def test_derived_entries_cannot_be_promoted_to_rule(self):
        assert PROVENANCE_RULES[Provenance.DERIVED].promotable_to_user_rule is False


class TestUserRuleLengthCap:
    def test_cap_is_two_hundred_characters(self):
        # Fails if the hard cap drifts from 200.
        assert PROVENANCE_RULES[Provenance.USER_RULE].max_chars == 200

    def test_cap_rejects_overlong_text(self):
        text = "x" * 201
        with pytest.raises(ValueError, match="200"):
            PROVENANCE_RULES[Provenance.USER_RULE].validate_content(text)

    def test_cap_accepts_text_at_boundary(self):
        text = "x" * 200
        assert PROVENANCE_RULES[Provenance.USER_RULE].validate_content(text) == text

    def test_cap_rejects_empty_text(self):
        with pytest.raises(ValueError, match="empty"):
            PROVENANCE_RULES[Provenance.USER_RULE].validate_content("   ")

    def test_cap_rejects_conversational_structure(self):
        # A 200-char cap alone is not enough: block dialogue-shaped text so the
        # experience panel cannot be abused as a chat window.
        with pytest.raises(ValueError, match="conversation"):
            PROVENANCE_RULES[Provenance.USER_RULE].validate_content(
                "user: hello\nassistant: hi there"
            )

    def test_cap_rejects_cjk_conversational_structure(self):
        # The fullwidth colon "：" is deliberate production behaviour, so it needs
        # coverage: without it the 200-char cap would still admit a short Chinese
        # chat log as a "rule". Production change that would break this: removing
        # "：" from the conversation-marker character class.
        with pytest.raises(ValueError, match="conversation"):
            PROVENANCE_RULES[Provenance.USER_RULE].validate_content(
                "用户：你好\n助手：有什么可以帮你"
            )

    def test_cap_accepts_cjk_rule_with_ascii_colon_inside(self):
        # A colon mid-sentence is legitimate punctuation, not a transcript marker,
        # so it must not be rejected - only a leading speaker label counts.
        text = "回答时使用中文：简洁、准确。"
        assert PROVENANCE_RULES[Provenance.USER_RULE].validate_content(text) == text

    def test_cap_accepts_plain_atomic_rule(self):
        assert (
            PROVENANCE_RULES[Provenance.USER_RULE].validate_content(
                "Always answer in Chinese, keep it under three sentences."
            )
            == "Always answer in Chinese, keep it under three sentences."
        )


class TestRetrievalPriority:
    def test_user_rule_outranks_derived_and_auto(self):
        # Retrieval tie-breaking must prefer human-authored rules.
        priorities = {p: PROVENANCE_RULES[p].priority for p in Provenance}
        assert priorities[Provenance.USER_RULE] > priorities[Provenance.DERIVED]
        assert priorities[Provenance.DERIVED] > priorities[Provenance.AUTO]
