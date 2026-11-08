"""Tests for the local governance console.

The console is the owner's window into the system, so the tests assert the two
things that make it trustworthy: it shows everything, and it can only be changed
through the same permission rules the library enforces - including when the
request did not come from the page itself.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from ratchet.adapters.web import build_server
from ratchet.memory.provenance import Provenance


@pytest.fixture
def server(tmp_path):
    srv = build_server(db_path=tmp_path / "w.db", host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    srv.ratchet_engine.close()  # type: ignore[attr-defined]
    thread.join(timeout=5)


def _url(server: HTTPServer, path: str) -> str:
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}{path}"


def _get(server: HTTPServer, path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(_url(server, path), timeout=10) as response:
            raw = response.read().decode("utf-8")
            ctype = response.headers.get("Content-Type", "")
            body = json.loads(raw) if "json" in ctype else raw
            return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(server: HTTPServer, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        _url(server, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _token(server: HTTPServer) -> str:
    _, page = _get(server, "/")
    assert isinstance(page, str)
    marker = "const CSRF = \""
    start = page.index(marker) + len(marker)
    return page[start : page.index('"', start)]


# ------------------------------------------------------------------ reading


def test_index_serves_the_console(server):
    status, page = _get(server, "/")
    assert status == 200
    assert isinstance(page, str)
    assert "治理面板" in page


def test_index_carries_a_csrf_token(server):
    assert len(_token(server)) > 20


def test_snapshot_starts_empty(server):
    status, payload = _get(server, "/api/snapshot")
    assert status == 200
    assert payload["experiences"] == []
    assert payload["counts"]["total"] == 0


def test_snapshot_shows_machine_written_entries(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    engine.store.add(content="a machine lesson", provenance=Provenance.AUTO, kind="rule")
    _, payload = _get(server, "/api/snapshot")
    assert payload["counts"]["machine_written"] == 1
    assert payload["experiences"][0]["editable"] is False
    assert payload["experiences"][0]["deletable"] is True


def test_correction_kinds_are_exposed(server):
    status, payload = _get(server, "/api/correction-kinds")
    assert status == 200
    assert "too_long" in payload["kinds"]


def test_unknown_routes_404(server):
    assert _get(server, "/api/nope")[0] == 404


# ---------------------------------------------------------------- mutations


def test_delete_requires_the_csrf_token(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    exp_id = engine.store.add(content="x", provenance=Provenance.AUTO, kind="rule")
    status, _ = _post(server, "/api/delete", {"id": exp_id})
    assert status == 403
    assert engine.store.get(exp_id) is not None


def test_delete_works_with_the_token(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    exp_id = engine.store.add(content="x", provenance=Provenance.AUTO, kind="rule")
    status, payload = _post(server, "/api/delete", {"id": exp_id, "csrf": _token(server)})
    assert status == 200
    assert payload["deleted"] == exp_id
    assert engine.store.get(exp_id) is None


def test_restore_works_with_the_token(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    exp_id = engine.store.add(content="x", provenance=Provenance.AUTO, kind="rule")
    token = _token(server)
    _post(server, "/api/delete", {"id": exp_id, "csrf": token})
    status, _ = _post(server, "/api/restore", {"id": exp_id, "csrf": token})
    assert status == 200
    assert engine.store.get(exp_id) is not None


def test_deleting_an_unknown_id_is_a_404(server):
    status, _ = _post(server, "/api/delete", {"id": "nope", "csrf": _token(server)})
    assert status == 404


def test_correction_is_recorded_through_the_console(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    status, payload = _post(
        server,
        "/api/correct",
        {"kind": "too_long", "scope": "global", "note": "太长了", "csrf": _token(server)},
    )
    assert status == 200
    entry = engine.store.get(payload["experience_id"])
    assert entry is not None
    assert entry.provenance is Provenance.DERIVED


def test_correction_rejects_an_unknown_kind(server):
    status, _ = _post(
        server, "/api/correct", {"kind": "nonsense", "csrf": _token(server)}
    )
    assert status == 400


def test_profile_can_be_forgotten(server):
    engine = server.ratchet_engine  # type: ignore[attr-defined]
    engine.profile.override("verbosity", "concise")
    status, _ = _post(
        server, "/api/profile/forget", {"key": "verbosity", "csrf": _token(server)}
    )
    assert status == 200
    assert engine.profile.store.get_profile_entry("verbosity") is None


def test_profile_override_requires_both_fields(server):
    token = _token(server)
    assert _post(server, "/api/profile/override", {"key": "tone", "csrf": token})[0] == 400


def test_rule_add_is_refused_when_editing_is_disabled(server):
    status, _ = _post(
        server, "/api/rule/add", {"content": "be terse", "csrf": _token(server)}
    )
    assert status == 403


def test_rule_add_works_when_explicitly_enabled(tmp_path):
    srv = build_server(db_path=tmp_path / "w2.db", host="127.0.0.1", port=0, allow_edit=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _post(srv, "/api/rule/add", {"content": "be terse", "csrf": _token(srv)})
        assert status == 200
        entry = srv.ratchet_engine.store.get(payload["id"])  # type: ignore[attr-defined]
        assert entry is not None
        assert entry.provenance is Provenance.USER_RULE
    finally:
        srv.shutdown()
        srv.server_close()
        srv.ratchet_engine.close()  # type: ignore[attr-defined]
        thread.join(timeout=5)


def test_the_console_refuses_to_bind_outside_loopback(tmp_path):
    with pytest.raises(ValueError):
        build_server(db_path=tmp_path / "w3.db", host="0.0.0.0", port=0)
