"""Three-tier provenance model: the safety rules for who may change an experience.

Design rationale (see docs/architecture.md):
  * Machine-written entries are DELETE-only for humans. Deletion can only ever
    remove capability, never inject it, so it is the one safe escape hatch.
  * Human-written rules are the highest-confidence signal in the system - the
    entire RLHF literature rests on explicit human correction - so they stay
    editable, but under a hard character cap that makes it impossible to use
    the experience panel as a chat window.
  * Derived entries (machine inferences from user feedback) inherit the
    delete-only rule, because letting a human edit a machine inference would
    blur the audit trail of "what did the system conclude on its own".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    """Who authored an experience entry."""

    AUTO = "auto"
    """Machine reflection / distillation output."""

    USER_RULE = "user_rule"
    """Written explicitly by the human owner."""

    DERIVED = "derived"
    """Machine inference drawn from user feedback signals."""


# Matches a leading speaker label, in English or Chinese, followed by either
# colon form: "user:", "assistant:", "用户：", "助手：".
#
# Both halves are load-bearing. Listing only English labels would make the
# fullwidth colon branch dead code - a Chinese transcript like "用户：你好" would
# sail through, and the 200-character cap alone would not stop a short chat log
# from being stored as a "rule".
#
# The labels are matched only at line start, so a colon used as ordinary
# punctuation mid-sentence ("回答时使用中文：简洁。") is still accepted.
_SPEAKER_LABELS = r"""
    user | assistant | human | system | ai | bot | you | me
    | 用户 | 使用者 | 助手 | 人工智能 | 系统 | 你 | 我 | 人类
"""

_CONVERSATION_MARKER = re.compile(
    rf"(?imx)^\s*(?:{_SPEAKER_LABELS})\s*[:：]"
)

_MAX_AUTO_CHARS = 4000
_MAX_DERIVED_CHARS = 2000
_MAX_USER_RULE_CHARS = 200


@dataclass(frozen=True)
class ProvenanceRule:
    """Permission and validation policy attached to one provenance level."""

    level: Provenance
    human_editable: bool
    human_deletable: bool
    promotable_to_user_rule: bool
    max_chars: int | None
    priority: int

    def validate_content(self, text: str) -> str:
        """Normalise and validate entry text, returning the cleaned content.

        Raises ``ValueError`` when the text is empty, over the length cap for
        this level, or shaped like a conversation transcript.
        """
        content = (text or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        if self.max_chars is not None and len(content) > self.max_chars:
            raise ValueError(
                f"{self.level.value} content must not exceed {self.max_chars} characters"
            )
        if _CONVERSATION_MARKER.search(content):
            raise ValueError(
                "content must be a single atomic rule, not a conversation transcript"
            )
        return content


PROVENANCE_RULES: dict[Provenance, ProvenanceRule] = {
    Provenance.USER_RULE: ProvenanceRule(
        level=Provenance.USER_RULE,
        human_editable=True,
        human_deletable=True,
        promotable_to_user_rule=False,
        max_chars=_MAX_USER_RULE_CHARS,
        priority=30,
    ),
    Provenance.DERIVED: ProvenanceRule(
        level=Provenance.DERIVED,
        human_editable=False,
        human_deletable=True,
        promotable_to_user_rule=False,
        max_chars=_MAX_DERIVED_CHARS,
        priority=20,
    ),
    Provenance.AUTO: ProvenanceRule(
        level=Provenance.AUTO,
        human_editable=False,
        human_deletable=True,
        promotable_to_user_rule=False,
        max_chars=_MAX_AUTO_CHARS,
        priority=10,
    ),
}


def rule_for(provenance: Provenance | str) -> ProvenanceRule:
    """Look up the policy for a provenance level."""
    return PROVENANCE_RULES[Provenance(provenance)]
