"""Tests for the CLI and the public top-level API.

The CLI is the human-facing surface for the permission model, so `audit` and
`rm` get the most attention here: the owner must be able to see everything the
machine learned and delete any of it.
"""

import pytest

from ratchet.cli.main import main
from ratchet.memory.provenance import Provenance
from ratchet.memory.store import ExperienceStore


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "ratchet.db")


def run(*args):
    return main(list(args))


class TestInit:
    def test_init_creates_database(self, db, tmp_path):
        assert run("init", "--db", db) == 0
        assert (tmp_path / "ratchet.db").exists()

    def test_init_on_existing_db_is_idempotent(self, db):
        assert run("init", "--db", db) == 0
        assert run("init", "--db", db) == 0

    def test_init_prints_confirmation(self, db, capsys):
        run("init", "--db", db)
        assert "initialized" in capsys.readouterr().out.lower()


class TestAddRule:
    def test_add_creates_user_rule(self, db):
        run("init", "--db", db)
        assert run("add", "--db", db, "Always answer in Chinese.") == 0
        with ExperienceStore(db) as store:
            entries = store.list_active()
            assert len(entries) == 1
            assert entries[0].provenance is Provenance.USER_RULE

    def test_add_rejects_overlong_rule(self, db, capsys):
        run("init", "--db", db)
        assert run("add", "--db", db, "x" * 201) == 1
        assert "200" in capsys.readouterr().err

    def test_add_rejects_conversation(self, db, capsys):
        run("init", "--db", db)
        assert run("add", "--db", db, "user: hi\nassistant: hello") == 1
        assert "conversation" in capsys.readouterr().err.lower()

    def test_add_rejects_empty(self, db, capsys):
        run("init", "--db", db)
        assert run("add", "--db", db, "   ") == 1
        assert "empty" in capsys.readouterr().err.lower()


class TestAudit:
    def test_audit_lists_active_entries(self, db, capsys):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        run("audit", "--db", db)
        assert "Prefer pytest." in capsys.readouterr().out

    def test_audit_shows_provenance_and_id(self, db, capsys):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        out = capsys.readouterr().out
        run("audit", "--db", db)
        out = capsys.readouterr().out
        assert "user_rule" in out

    def test_audit_includes_inactive_with_flag(self, db, capsys):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        with ExperienceStore(db) as store:
            store.delete(store.list_active()[0].id)
        assert "Prefer pytest." not in _capture(run, "audit", "--db", db)
        assert "Prefer pytest." in _capture(run, "audit", "--db", db, "--all")

    def test_audit_on_empty_store_reports_cleanly(self, db, capsys):
        run("init", "--db", db)
        assert run("audit", "--db", db) == 0
        assert "no active" in capsys.readouterr().out.lower()


class TestRemove:
    def test_remove_deletes_entry(self, db):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        with ExperienceStore(db) as store:
            entry_id = store.list_active()[0].id
        assert run("rm", "--db", db, entry_id) == 0
        with ExperienceStore(db) as store:
            assert store.list_active() == []

    def test_remove_unknown_id_fails(self, db, capsys):
        run("init", "--db", db)
        assert run("rm", "--db", db, "no-such-id") == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_remove_by_prefix(self, db):
        # Owners should not have to copy a full uuid.
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        with ExperienceStore(db) as store:
            entry_id = store.list_active()[0].id
        assert run("rm", "--db", db, entry_id[:8]) == 0

    def test_ambiguous_prefix_fails_safely(self, db, capsys):
        # An empty prefix necessarily matches every entry, so with two entries
        # stored the ambiguity is deterministic. (A literal prefix like "a" would
        # be flaky, since uuid4 hex digits are random.)
        run("init", "--db", db)
        run("add", "--db", db, "Rule one.")
        run("add", "--db", db, "Rule two.")
        assert run("rm", "--db", db, "") == 1
        assert "ambiguous" in capsys.readouterr().err.lower()

    def test_ambiguous_prefix_deletes_nothing(self, db):
        # Ambiguity must fail closed, never pick one and delete it.
        run("init", "--db", db)
        run("add", "--db", db, "Rule one.")
        run("add", "--db", db, "Rule two.")
        run("rm", "--db", db, "")
        with ExperienceStore(db) as store:
            assert len(store.list_active()) == 2


class TestStatus:
    def test_status_reports_counts(self, db, capsys):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        assert run("status", "--db", db) == 0
        out = capsys.readouterr().out
        assert "experiences" in out.lower()
        assert "events" in out.lower()

    def test_status_shows_policy_arms(self, db, capsys):
        run("init", "--db", db)
        assert run("status", "--db", db) == 0
        assert "variant" in capsys.readouterr().out.lower()


class TestExport:
    def test_export_writes_dpo_pairs_file(self, db, tmp_path):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        target = tmp_path / "out.jsonl"
        assert run("export", "--db", db, "--format", "dpo", "--out", str(target)) == 0
        assert target.exists()

    def test_export_writes_jsonl_lines(self, db, tmp_path):
        run("init", "--db", db)
        run("add", "--db", db, "Prefer pytest.")
        target = tmp_path / "out.jsonl"
        run("export", "--db", db, "--format", "experiences", "--out", str(target))
        assert target.read_text(encoding="utf-8").strip().startswith("{")

    def test_export_unknown_format_fails(self, db, tmp_path, capsys):
        run("init", "--db", db)
        # --format uses argparse choices, so an invalid value is a usage error
        # and exits 2 (the POSIX convention) rather than returning 1.
        with pytest.raises(SystemExit) as excinfo:
            run("export", "--db", db, "--format", "nope", "--out", "x")
        assert excinfo.value.code == 2
        assert "format" in capsys.readouterr().err.lower()


class TestChat:
    def test_chat_runs_offline_with_echo_provider(self, db, capsys):
        # Production change that would break this: making chat require an API key.
        run("init", "--db", db)
        assert run("chat", "--db", db, "--provider", "echo", "--task", "t", "hello") == 0
        out = capsys.readouterr().out
        assert "echo" in out.lower()

    def test_chat_records_an_event(self, db):
        run("init", "--db", db)
        run("chat", "--db", db, "--provider", "echo", "--task", "t", "hello")
        with ExperienceStore(db) as store:
            assert len(store.list_events()) == 1

    def test_chat_accepts_explicit_feedback(self, db):
        run("init", "--db", db)
        assert (
            run("chat", "--db", db, "--provider", "echo", "--task", "t",
                "--feedback", "down", "hello")
            == 0
        )
        with ExperienceStore(db) as store:
            assert store.list_events()[0].reward == pytest.approx(0.0)

    def test_chat_rejects_invalid_feedback(self, db, capsys):
        run("init", "--db", db)
        # choices-constrained, so an invalid value is a usage error -> exit 2.
        with pytest.raises(SystemExit) as excinfo:
            run("chat", "--db", db, "--provider", "echo", "--task", "t",
                "--feedback", "sideways", "hello")
        assert excinfo.value.code == 2

    def test_chat_requires_api_key_for_openai(self, db, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert (
            run("chat", "--db", db, "--provider", "openai", "--task", "t", "hello") == 1
        )
        assert "api key" in capsys.readouterr().err.lower()


class TestPublicApi:
    def test_top_level_exports_are_importable(self):
        import ratchet

        for name in (
            "Engine",
            "ExperienceStore",
            "Provenance",
            "ThompsonPolicy",
            "RewardChain",
            "HybridRetriever",
            "Reflector",
            "PromptAssembler",
            "EchoProvider",
            "OpenAICompatProvider",
            "__version__",
        ):
            assert hasattr(ratchet, name), f"missing public export: {name}"

    def test_engine_runs_the_loop_in_one_call(self, tmp_path):
        import ratchet

        engine = ratchet.Engine(db_path=tmp_path / "r.db", provider=ratchet.EchoProvider())
        result = engine.run(task="greet", user_input="hi")
        assert "echo" in result.output
        assert engine.stats()["interactions"] == 1

    def test_engine_add_rule_respects_permissions(self, tmp_path):
        import ratchet

        engine = ratchet.Engine(db_path=tmp_path / "r.db", provider=ratchet.EchoProvider())
        engine.add_rule("Keep answers short.")
        assert len(engine.list_experiences()) == 1
        with pytest.raises(ValueError):
            engine.add_rule("x" * 201)

    def test_engine_forgets_only_allowed_operations(self, tmp_path):
        import ratchet

        engine = ratchet.Engine(db_path=tmp_path / "r.db", provider=ratchet.EchoProvider())
        engine.add_rule("Keep answers short.")
        entry = engine.list_experiences()[0]
        engine.edit_rule(entry.id, "Keep answers very short.")
        assert engine.list_experiences()[0].content == "Keep answers very short."
        assert engine.forget(entry.id) is True
        assert engine.list_experiences() == []

    def test_engine_refuses_to_edit_machine_entries(self, tmp_path):
        # The escape hatch is deletion only - enforced through the public API too.
        import ratchet

        engine = ratchet.Engine(db_path=tmp_path / "r.db", provider=ratchet.EchoProvider())
        eid = engine.store.add(content="machine rule", provenance=Provenance.AUTO)
        with pytest.raises(PermissionError):
            engine.edit_rule(eid, "tampered")


def _capture(fn, *args):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args)
    return buffer.getvalue()
