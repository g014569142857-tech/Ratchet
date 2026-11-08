"""Redaction of secrets and personal identifiers before they reach the store.

The experience store accumulates the owner's real work: code, draft prose, error
logs, contact details pasted in for context. It is a single file on disk, and it
is the one artefact a user is most likely to back up, sync, or attach to a bug
report. A secret that lands in it tends to stay in it.

So redaction runs **by default on machine-written entries**. That includes
reflections, distilled rules and anything derived from feedback - the paths where
content enters without the owner having typed it into a box they can see.

It is applied at the boundary rather than at render time because the threat is
the stored copy, not the displayed one: redacting only what gets sent to a model
would leave the plaintext sitting in the database.

Deliberately conservative: a missed secret is worse than a mangled log line, so
the patterns over-match slightly and the replacement keeps the shape of the
original (``sk-...`` stays recognisably a key) so that a redacted entry is still
readable as a lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: ``(name, pattern, replacement)`` applied in order.
DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # Provider API keys: OpenAI-style sk-, Anthropic sk-ant-, and generic long
    # high-entropy tokens that follow an assignment.
    (
        "api_key",
        re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,})\b"),
        "[REDACTED_KEY]",
    ),
    (
        "bearer",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
        "Bearer [REDACTED_TOKEN]",
    ),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|secret|token|password|passwd|pwd)"
            r"(\s*[:=]\s*)(\"?)([A-Za-z0-9._\-]{8,})(\"?)"
        ),
        r"\1\2\3[REDACTED]\5",
    ),
    # Private keys - the BEGIN/END armour is unmistakable, so match the block.
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "[REDACTED_EMAIL]"),
    # Mainland mobile numbers. Anchored on the leading 1 to avoid eating long
    # numeric ids that merely happen to be 11 digits.
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
    # IPv4 with a port or in a URL context; bare version numbers are left alone
    # by requiring four dotted octets in the 0-255 range.
    (
        "ipv4",
        re.compile(
            r"(?<![\d.])((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])"
        ),
        "[REDACTED_IP]",
    ),
    # Long hex runs are usually hashes or keys, not meaningful content.
    ("hex_secret", re.compile(r"\b[0-9a-fA-F]{32,}\b"), "[REDACTED_HEX]"),
)


@dataclass(frozen=True)
class Redactor:
    """Apply a pattern set to text, reporting what it touched."""

    patterns: tuple[tuple[str, re.Pattern[str], str], ...] = DEFAULT_PATTERNS
    enabled: bool = True
    #: Keys to leave alone. ``hex_secret`` is the usual candidate: a codebase full
    #: of git SHAs produces a lot of noise for little real protection.
    disabled: frozenset[str] = field(default_factory=frozenset)

    def scrub(self, text: str) -> tuple[str, tuple[str, ...]]:
        """Redact ``text``, returning ``(cleaned, names_of_patterns_that_fired)``.

        Returning the names matters for the audit view: the owner should be able
        to see *that* something was stripped from an entry, without the log
        itself revealing what it was.
        """
        if not self.enabled or not text:
            return text, ()
        cleaned = text
        fired: list[str] = []
        for name, pattern, replacement in self.patterns:
            if name in self.disabled:
                continue
            cleaned, count = pattern.subn(replacement, cleaned)
            if count:
                fired.append(name)
        return cleaned, tuple(fired)

    def __call__(self, text: str) -> str:
        return self.scrub(text)[0]


def redact(text: str) -> str:
    """Convenience wrapper using the defaults."""
    return Redactor()(text)
