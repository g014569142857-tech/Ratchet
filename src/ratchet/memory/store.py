"""SQLite-backed experience store.

Single-file, zero-server, no third-party dependencies. Every human mutation
goes through the provenance policy, so the permission rules cannot be bypassed
by calling the storage layer directly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ratchet.memory.provenance import Provenance, rule_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ratchet.improve.verifier import Reward

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experience (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    task_type    TEXT,
    provenance   TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'episode',
    trust_level  TEXT NOT NULL DEFAULT 'trusted',
    score        REAL NOT NULL DEFAULT 0.0,
    uses         INTEGER NOT NULL DEFAULT 0,
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experience_active ON experience (active);
CREATE INDEX IF NOT EXISTS idx_experience_task_type ON experience (task_type);
CREATE INDEX IF NOT EXISTS idx_experience_trust ON experience (trust_level);

CREATE TABLE IF NOT EXISTS event (
    id            TEXT PRIMARY KEY,
    session_id    TEXT,
    created_at    TEXT NOT NULL,
    task          TEXT,
    user_input    TEXT,
    output        TEXT,
    variant_id    TEXT,
    retrieved_ids TEXT NOT NULL DEFAULT '[]',
    reward        REAL,
    reward_source TEXT,
    confidence    REAL NOT NULL DEFAULT 0.0,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    latency_ms    REAL NOT NULL DEFAULT 0.0,
    trust_level   TEXT NOT NULL DEFAULT 'trusted'
);
CREATE INDEX IF NOT EXISTS idx_event_session ON event (session_id);
CREATE INDEX IF NOT EXISTS idx_event_created ON event (created_at);

-- Machine-inferred user profile. Every row carries the event ids that support
-- it, because a preference the owner cannot trace back to evidence is exactly
-- the kind of quiet wrong assumption that self-reinforces. See
-- docs/architecture.md, "profile drift".
CREATE TABLE IF NOT EXISTS profile (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.0,
    support     INTEGER NOT NULL DEFAULT 0,
    contradict  INTEGER NOT NULL DEFAULT 0,
    evidence    TEXT NOT NULL DEFAULT '[]',
    source      TEXT NOT NULL DEFAULT 'inferred',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_DEFAULT_DEACTIVATION_THRESHOLD = -3.0

TRUSTED = "trusted"
UNTRUSTED = "untrusted"
VALID_TRUST_LEVELS = frozenset({TRUSTED, UNTRUSTED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Rows:
    """Materialised query result, supporting the cursor methods this store uses.

    Rows are pulled eagerly so that nothing touches the connection once the lock
    has been released. Returning the raw cursor would let a caller iterate it
    after the lock was dropped, which is exactly the interleaving the lock
    exists to prevent.
    """

    def __init__(self, rows: list[Any], rowcount: int) -> None:
        self._rows = rows
        self._index = 0
        self.rowcount = rowcount

    def fetchone(self) -> Any | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)


class _LockedConnection:
    """Serialise every use of one SQLite connection.

    The store is opened once and shared, but adapters are not single-threaded -
    the web console serves each request from a pool. Without this, a request
    handled off the creating thread fails outright ("SQLite objects created in a
    thread can only be used in that same thread"), which takes the whole
    governance panel down.

    ``check_same_thread=False`` lets the connection cross threads; the lock makes
    it safe to do so. Both are needed: relaxing the check without serialising
    would trade a loud failure for silent interleaving.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    @staticmethod
    def open(path: Path) -> _LockedConnection:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return _LockedConnection(conn)

    def execute(self, query: str, params: Sequence[object] = ()) -> _Rows:
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
            return _Rows(rows, cursor.rowcount)

    def executescript(self, script: str) -> None:
        with self._lock:
            self._conn.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass(frozen=True)
class Experience:
    """One row of the experience table."""

    id: str
    content: str
    task_type: str | None
    provenance: Provenance
    kind: str
    score: float
    uses: int
    active: bool
    created_at: str
    updated_at: str
    trust_level: str = TRUSTED


@dataclass(frozen=True)
class ProfileEntry:
    """One inferred preference, always traceable to the events that suggested it.

    ``support`` / ``contradict`` are kept separately from ``confidence``: the
    confidence is the *present* strength after time decay, while the counters are
    the raw tally. Decay is recomputed from the counters on every read, so a
    confidence that falls only ever reflects the passage of time, never a loss of
    history - and a single new confirming observation restores it.
    """

    key: str
    value: str
    confidence: float = 0.0
    support: int = 0
    contradict: int = 0
    evidence: tuple[str, ...] = ()
    source: str = "inferred"
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class Event:
    """One row of the event table: a full audit trail of one interaction."""

    id: str
    session_id: str | None
    created_at: str
    task: str | None
    user_input: str | None
    output: str | None
    variant_id: str | None
    retrieved_ids: tuple[str, ...]
    reward: float | None
    reward_source: str | None
    confidence: float
    tokens_in: int
    tokens_out: int
    latency_ms: float
    trust_level: str = TRUSTED


class ExperienceStore:
    """CRUD over experiences with soft delete, reward accounting and permissions."""

    def __init__(
        self,
        path: str | Path,
        *,
        deactivation_threshold: float = _DEFAULT_DEACTIVATION_THRESHOLD,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.deactivation_threshold = deactivation_threshold
        self._conn = _LockedConnection.open(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ read

    def get(self, experience_id: str, *, include_inactive: bool = False) -> Experience | None:
        query = "SELECT * FROM experience WHERE id = ?"
        params: list[object] = [experience_id]
        if not include_inactive:
            query += " AND active = 1"
        row = self._conn.execute(query, params).fetchone()
        return self._to_experience(row) if row else None

    def list_active(
        self,
        *,
        task_type: str | None = None,
        trust_level: str | None = None,
    ) -> list[Experience]:
        query = "SELECT * FROM experience WHERE active = 1"
        params: list[object] = []
        if task_type is not None:
            query += " AND task_type = ?"
            params.append(task_type)
        if trust_level is not None:
            if trust_level not in VALID_TRUST_LEVELS:
                raise ValueError(
                    f"trust_level must be one of {sorted(VALID_TRUST_LEVELS)}"
                )
            query += " AND trust_level = ?"
            params.append(trust_level)
        query += " ORDER BY created_at DESC"
        return [self._to_experience(r) for r in self._conn.execute(query, params)]

    def list_all(self) -> list[Experience]:
        rows = self._conn.execute("SELECT * FROM experience ORDER BY created_at DESC")
        return [self._to_experience(r) for r in rows]

    def resolve_id(self, prefix: str) -> str | None:
        """Resolve a full id or a prefix matching exactly one entry.

        Returns ``None`` when nothing matches and raises ``ValueError`` when the
        prefix is ambiguous. Ambiguity fails closed: picking one at random and
        deleting it would be far worse than refusing.

        Inactive entries are included on purpose - an owner must be able to
        delete something the reward system auto-deactivated.

        Matching is case-insensitive. Ids are lowercase uuid4, but they are read
        off a terminal and typed back by hand, so a capitalised prefix is a typo
        rather than a different id; failing on it looks like data loss.
        """
        needle = prefix.strip().lower()
        entries = self.list_all()
        exact = [e for e in entries if e.id == needle]
        if exact:
            return exact[0].id
        matches = [e for e in entries if e.id.startswith(needle)]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous prefix {prefix!r} matches {len(matches)} entries"
            )
        return None

    # ----------------------------------------------------------------- write

    def add(
        self,
        content: str,
        *,
        provenance: Provenance | str,
        task_type: str | None = None,
        kind: str = "episode",
        trust_level: str = TRUSTED,
    ) -> str:
        """Insert a new experience and return its id.

        Content is validated against the provenance policy before insertion.
        ``trust_level`` records whether the underlying content came from a
        trusted source, which is separate from ``provenance`` (who wrote the
        entry). An auto-reflection over an untrusted web page is authored by the
        machine but still tainted, so both labels are needed.
        """
        policy = rule_for(provenance)
        cleaned = policy.validate_content(content)
        if trust_level not in VALID_TRUST_LEVELS:
            raise ValueError(
                f"trust_level must be one of {sorted(VALID_TRUST_LEVELS)}"
            )
        experience_id = str(uuid.uuid4())
        timestamp = _now()
        self._conn.execute(
            """
            INSERT INTO experience
                (id, content, task_type, provenance, kind, trust_level, score,
                 uses, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0.0, 0, 1, ?, ?)
            """,
            (
                experience_id,
                cleaned,
                task_type,
                policy.level.value,
                kind,
                trust_level,
                timestamp,
                timestamp,
            ),
        )
        self._conn.commit()
        return experience_id

    def edit(self, experience_id: str, content: str) -> None:
        """Edit content in place. Only permitted for human-editable provenance.

        ``provenance`` is deliberately not a parameter: re-labelling an entry
        would be a privilege-escalation back door, so the call raises TypeError.
        """
        existing = self.get(experience_id, include_inactive=True)
        if existing is None:
            raise KeyError(f"unknown experience id: {experience_id}")

        policy = rule_for(existing.provenance)
        if not policy.human_editable:
            raise PermissionError(
                f"{existing.provenance.value} entries are not editable; "
                "you may only delete them"
            )
        cleaned = policy.validate_content(content)
        self._conn.execute(
            "UPDATE experience SET content = ?, updated_at = ? WHERE id = ?",
            (cleaned, _now(), experience_id),
        )
        self._conn.commit()

    def delete(self, experience_id: str) -> bool:
        """Soft delete. Deletion is allowed for every provenance level."""
        existing = self.get(experience_id, include_inactive=True)
        if existing is None:
            return False
        self._conn.execute(
            "UPDATE experience SET active = 0, updated_at = ? WHERE id = ?",
            (_now(), experience_id),
        )
        self._conn.commit()
        return True

    def restore(self, experience_id: str) -> bool:
        """Re-activate a soft-deleted entry.

        Restoring is a closed operation - it re-admits text that was already
        reviewed and stored once, so unlike editing it cannot introduce new
        content. That is why the owner is allowed to undo a deletion freely.
        """
        existing = self.get(experience_id, include_inactive=True)
        if existing is None:
            return False
        self._conn.execute(
            "UPDATE experience SET active = 1, updated_at = ? WHERE id = ?",
            (_now(), experience_id),
        )
        self._conn.commit()
        return True

    def purge(self, experience_id: str) -> bool:
        """Hard-delete one entry and the profile evidence that cites it.

        This is the right-to-erasure path. It is deliberately NOT the default:
        soft delete keeps the audit trail that answers "why did the model start
        behaving differently on Tuesday". Purging is irreversible, so it is only
        exposed through an explicit command.
        """
        existing = self.get(experience_id, include_inactive=True)
        if existing is None:
            return False
        self._conn.execute("DELETE FROM experience WHERE id = ?", (experience_id,))
        self._prune_evidence(experience_id)
        self._conn.commit()
        return True

    def _prune_evidence(self, experience_id: str) -> None:
        """Drop a purged experience id from every profile entry citing it."""
        rows = self._conn.execute("SELECT key, evidence FROM profile").fetchall()
        for row in rows:
            evidence = [e for e in json.loads(row["evidence"]) if e != experience_id]
            if len(evidence) != len(json.loads(row["evidence"])):
                self._conn.execute(
                    "UPDATE profile SET evidence = ?, updated_at = ? WHERE key = ?",
                    (json.dumps(evidence), _now(), row["key"]),
                )

    # --------------------------------------------------------------- profile

    def save_profile_entry(self, entry: ProfileEntry) -> None:
        """Insert or replace one profile row (upsert on ``key``).

        Timestamps come from the entry rather than from the wall clock. They are
        load-bearing: confidence decay is computed from ``updated_at``, so
        stamping the real "now" here instead of the caller's time would make a
        back-dated observation indistinguishable from a fresh one and silently
        disable decay.
        """
        created = entry.created_at or _now()
        updated = entry.updated_at or _now()
        self._conn.execute(
            """
            INSERT INTO profile
                (key, value, confidence, support, contradict, evidence, source,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                support = excluded.support,
                contradict = excluded.contradict,
                evidence = excluded.evidence,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                entry.key,
                entry.value,
                float(entry.confidence),
                int(entry.support),
                int(entry.contradict),
                json.dumps(list(entry.evidence)),
                entry.source,
                created,
                updated,
            ),
        )
        self._conn.commit()

    def get_profile_entry(self, key: str) -> ProfileEntry | None:
        row = self._conn.execute(
            "SELECT * FROM profile WHERE key = ?", (key,)
        ).fetchone()
        return self._to_profile_entry(row) if row else None

    def list_profile_entries(self) -> list[ProfileEntry]:
        rows = self._conn.execute("SELECT * FROM profile ORDER BY key ASC")
        return [self._to_profile_entry(r) for r in rows]

    def delete_profile_entry(self, key: str) -> bool:
        cursor = self._conn.execute("DELETE FROM profile WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def clear_profile(self) -> int:
        """Drop every inferred row, keeping nothing - the panic button."""
        cursor = self._conn.execute("DELETE FROM profile")
        self._conn.commit()
        return cursor.rowcount or 0

    # ---------------------------------------------------------------- reward

    def record_reward(self, experience_id: str, reward: float) -> None:
        """Accumulate a reward signal and deactivate poisoned entries.

        No-op for unknown or already-inactive entries, so late-arriving rewards
        cannot resurrect a deactivated experience.
        """
        existing = self.get(experience_id)
        if existing is None:
            return
        new_score = existing.score + float(reward)
        new_uses = existing.uses + 1
        active = 1 if new_score > self.deactivation_threshold else 0
        self._conn.execute(
            "UPDATE experience SET score = ?, uses = ?, active = ?, updated_at = ? WHERE id = ?",
            (new_score, new_uses, active, _now(), experience_id),
        )
        self._conn.commit()

    # ----------------------------------------------------------------- event

    def record_event(
        self,
        *,
        session_id: str | None = None,
        task: str | None = None,
        user_input: str | None = None,
        output: str | None = None,
        variant_id: str | None = None,
        retrieved_ids: Sequence[str] = (),
        reward: float | None = None,
        reward_source: str | None = None,
        confidence: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        trust_level: str = TRUSTED,
    ) -> str:
        """Append one interaction to the immutable audit trail."""
        if trust_level not in VALID_TRUST_LEVELS:
            raise ValueError(f"trust_level must be one of {sorted(VALID_TRUST_LEVELS)}")
        event_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO event
                (id, session_id, created_at, task, user_input, output, variant_id,
                 retrieved_ids, reward, reward_source, confidence, tokens_in,
                 tokens_out, latency_ms, trust_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                _now(),
                task,
                user_input,
                output,
                variant_id,
                json.dumps(list(retrieved_ids)),
                reward,
                reward_source,
                confidence,
                tokens_in,
                tokens_out,
                latency_ms,
                trust_level,
            ),
        )
        self._conn.commit()
        return event_id

    def list_events(
        self, *, session_id: str | None = None, limit: int | None = None
    ) -> list[Event]:
        query = "SELECT * FROM event"
        params: list[object] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return [self._to_event(r) for r in self._conn.execute(query, params)]

    def update_event_reward(self, event_id: str, reward: Reward) -> None:
        """Correct the reward recorded on an existing event.

        An explicit human verdict usually arrives *after* the interaction has
        been recorded, so the audit trail has to be correctable. Without this,
        a later thumbs-down would leave a stale reward that contradicts the
        owner's actual judgement, and any growth-curve metric computed from the
        event table would be wrong.

        No-op for unknown ids: a late verdict must not create phantom events.
        """
        row = self._conn.execute(
            "SELECT id FROM event WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return
        self._conn.execute(
            """
            UPDATE event
               SET reward = ?, reward_source = ?, confidence = ?
             WHERE id = ?
            """,
            (
                reward.value,
                reward.source.value if reward.source else None,
                reward.confidence,
                event_id,
            ),
        )
        self._conn.commit()

    # ----------------------------------------------------------------- state

    def save_state(self, key: str, value: object) -> None:
        """Persist an arbitrary JSON-serialisable value under ``key``.

        Used for the bandit policy, so separate CLI invocations keep learning
        instead of restarting from a uniform prior every time.
        """
        self._conn.execute(
            """
            INSERT INTO state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), _now()),
        )
        self._conn.commit()

    def load_state(self, key: str) -> object | None:
        """Return the value stored under ``key``, or ``None`` if absent."""
        row = self._conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    # ------------------------------------------------------------------ misc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExperienceStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _to_experience(row: sqlite3.Row) -> Experience:
        return Experience(
            id=row["id"],
            content=row["content"],
            task_type=row["task_type"],
            provenance=Provenance(row["provenance"]),
            kind=row["kind"],
            score=row["score"],
            uses=row["uses"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            trust_level=row["trust_level"],
        )

    @staticmethod
    def _to_profile_entry(row: sqlite3.Row) -> ProfileEntry:
        return ProfileEntry(
            key=row["key"],
            value=row["value"],
            confidence=float(row["confidence"]),
            support=int(row["support"]),
            contradict=int(row["contradict"]),
            evidence=tuple(json.loads(row["evidence"])),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            task=row["task"],
            user_input=row["user_input"],
            output=row["output"],
            variant_id=row["variant_id"],
            retrieved_ids=tuple(json.loads(row["retrieved_ids"])),
            reward=row["reward"],
            reward_source=row["reward_source"],
            confidence=row["confidence"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            latency_ms=row["latency_ms"],
            trust_level=row["trust_level"],
        )
