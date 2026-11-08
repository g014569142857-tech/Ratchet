"""Governance: the owner's side of the system.

Only the correction data model is re-exported here. ``panel`` imports the
profile module, and the profile module imports ``correction`` - keeping the
package's eager exports limited to the leaf module keeps that import graph
acyclic no matter which entry point a caller reaches first.
"""

from ratchet.governance.correction import (
    MAX_NOTE_CHARS,
    PROFILE_IMPACT,
    Correction,
    CorrectionKind,
    CorrectionRecord,
    CorrectionScope,
    escape_note,
)

__all__ = [
    "MAX_NOTE_CHARS",
    "PROFILE_IMPACT",
    "Correction",
    "CorrectionKind",
    "CorrectionRecord",
    "CorrectionScope",
    "escape_note",
]
