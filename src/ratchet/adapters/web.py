"""Local web console: the owner's window into what the machine has learned.

The governance panel is the project's central promise, and a CLI listing is a
poor place to keep a promise. Learning is cumulative and quiet - the interesting
question is never "what happened just now" but "what does it currently believe
about me", and that is a thing you look at, not a thing you grep.

Two decisions shape this module.

**One HTML file, zero dependencies.** The console is served from the standard
library and ships its own inline CSS. A dashboard that pulled in a framework
would be a supply-chain decision, a build step and a bundle to audit, in order to
render one table. Refusing that keeps ``pip install ratchet-ai`` small and keeps
the console working on a machine with no package index.

**The console cannot be a chat box.** There is no free-text field except the
bounded correction note, and by default no editing at all: machine-written
entries are delete-only, and the human-authored rule box is capped and validated.
The whole point of the asymmetric permission model is that adding prose and
removing prose are not equivalent operations, and a textarea would quietly make
them equivalent.

Binding and request safety
--------------------------
The server binds ``127.0.0.1`` only. That is necessary but not sufficient: a page
in any browser tab can attempt a POST to localhost regardless of origin, so
mutations require a CSRF token minted at startup and handed to the page itself.
Deletion is the one action a hostile page could usefully perform, and it is
exactly the action this protects.
"""

from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ratchet.core.provider import Provider
from ratchet.engine import Engine
from ratchet.governance.correction import (
    MAX_NOTE_CHARS,
    Correction,
    CorrectionKind,
    CorrectionScope,
)
from ratchet.governance.panel import CORRECTION_KINDS

_TOKEN_PLACEHOLDER = "__RATCHET_CSRF__"

_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ratchet · 治理面板</title>
<style>
  :root {
    --accent: #0078d4;
    --accent-soft: #e8f3fd;
    --danger: #c50f1f;
    --ok: #107c10;
    --text: #1b1b1f;
    --muted: #616166;
    --line: #e3e3e8;
    --surface: #ffffff;
    --canvas: #f3f4f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI Variable", "Segoe UI", -apple-system, "Microsoft YaHei", sans-serif;
    background: var(--canvas);
    color: var(--text);
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--line);
    padding: 18px 28px;
    display: flex; align-items: baseline; gap: 14px;
    position: sticky; top: 0; z-index: 5;
  }
  header h1 { margin: 0; font-size: 19px; font-weight: 600; letter-spacing: .2px; }
  header .sub { color: var(--muted); font-size: 12.5px; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px 28px 64px; }
  h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: .8px;
    color: var(--muted); margin: 30px 0 12px; font-weight: 600;
  }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
    padding: 14px 16px;
  }
  .card .n { font-size: 22px; font-weight: 600; }
  .card .l { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .row { padding: 14px 18px; border-bottom: 1px solid var(--line); display: flex; gap: 14px; align-items: flex-start; }
  .row:last-child { border-bottom: none; }
  .row.inactive { opacity: .55; background: #fafafb; }
  .body { flex: 1; min-width: 0; }
  .content { white-space: pre-wrap; word-break: break-word; line-height: 1.55; }
  .meta { margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 11.5px; color: var(--muted); }
  .badge {
    border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 600;
    background: #f0f0f3; color: #3b3b41; border: 1px solid var(--line);
  }
  .badge.user_rule { background: var(--accent-soft); color: #005a9e; border-color: #cfe4f7; }
  .badge.derived { background: #fff4e5; color: #8a5300; border-color: #f5dcb8; }
  .badge.auto { background: #f3f0fb; color: #5c3fa8; border-color: #e0d8f5; }
  .badge.untrusted { background: #fdeaea; color: var(--danger); border-color: #f7cfcf; }
  .actions { display: flex; gap: 6px; flex-shrink: 0; }
  button {
    font: inherit; font-size: 12.5px; padding: 5px 11px; border-radius: 5px;
    border: 1px solid var(--line); background: #fbfbfc; color: var(--text); cursor: pointer;
  }
  button:hover { background: #f2f2f5; }
  button.danger { color: var(--danger); border-color: #f0cdd0; }
  button.danger:hover { background: #fdf1f1; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover { background: #106ebe; }
  form.box { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; }
  label { display: block; font-size: 12.5px; color: var(--muted); margin-bottom: 5px; }
  select, input, textarea {
    font: inherit; font-size: 13px; padding: 7px 9px; border: 1px solid var(--line);
    border-radius: 5px; background: #fff; color: var(--text); width: 100%;
  }
  textarea { resize: vertical; min-height: 62px; font-family: inherit; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .note { font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.5; }
  .empty { padding: 26px; text-align: center; color: var(--muted); font-size: 13px; }
  .toast {
    position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%);
    background: #1b1b1f; color: #fff; padding: 10px 18px; border-radius: 6px;
    font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none;
  }
  .toast.show { opacity: 1; }
  code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Ratchet</h1>
  <span class="sub">本地自强化引擎 · 治理面板 · 仅绑定 127.0.0.1</span>
</header>
<main>
  <h2>概览</h2>
  <div class="cards" id="cards"></div>

  <h2>经验库 <span class="sub">机器可增 · 人可删</span></h2>
  <div class="panel" id="experiences"></div>

  <h2>推断中的用户画像</h2>
  <div class="panel" id="profile"></div>

  <h2>结构化纠正</h2>
  <form class="box" id="correct-form">
    <div class="grid">
      <div>
        <label for="kind">不满意的方面</label>
        <select id="kind" name="kind"></select>
      </div>
      <div>
        <label for="scope">适用范围</label>
        <select id="scope" name="scope">
          <option value="this_task_only">仅当前这类任务</option>
          <option value="global">所有交互</option>
        </select>
      </div>
    </div>
    <div style="margin-top:12px">
      <label for="note">补充说明（可选，上限 __MAXNOTE__ 字符）</label>
      <textarea id="note" name="note" maxlength="__MAXNOTE__" placeholder="这段只是你反应的记录，不会作为指令执行"></textarea>
    </div>
    <div style="margin-top:12px"><button class="primary" type="submit">提交纠正</button></div>
    <div class="note">纠正以枚举字段承载主要信息，自由文本只作为附注、入库前转义，并以「用户原话」身份渲染。</div>
  </form>
</main>
<div class="toast" id="toast"></div>
<script>
const CSRF = "__RATCHET_CSRF__";
let snapshot = { experiences: [], profile: [], counts: {} };

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

async function api(path, options) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, options || {}));
  const data = await res.json();
  if (!res.ok) { toast(data.error || '请求失败'); throw new Error(data.error || 'error'); }
  return data;
}

function render() {
  const c = snapshot.counts || {};
  document.getElementById('cards').innerHTML = [
    ['活跃经验', c.active], ['机器写入', c.machine_written],
    ['人写规则', c.human_written], ['画像条目', (snapshot.profile || []).length]
  ].map(([l, n]) => '<div class="card"><div class="n">' + (n ?? 0) + '</div><div class="l">' + esc(l) + '</div></div>').join('');

  const ex = snapshot.experiences || [];
  document.getElementById('experiences').innerHTML = ex.length ? ex.map(e => {
    const acts = e.active
      ? '<button class="danger" onclick="del(\\'' + e.id + '\\')">删除</button>'
      : '<button onclick="restore(\\'' + e.id + '\\')">恢复</button>';
    const editable = e.editable ? '<span class="badge">可编辑</span>' : '<span class="badge">仅可删除</span>';
    const trust = e.trust_level === 'untrusted' ? '<span class="badge untrusted">untrusted</span>' : '';
    return '<div class="row' + (e.active ? '' : ' inactive') + '">'
      + '<div class="body"><div class="content">' + esc(e.content) + '</div>'
      + '<div class="meta"><span class="badge ' + esc(e.provenance) + '">' + esc(e.provenance) + '</span>'
      + '<span class="badge">' + esc(e.kind) + '</span>' + trust + editable
      + '<span>score ' + e.score.toFixed(1) + ' · uses ' + e.uses + '</span>'
      + '<span><code>' + esc(e.id.slice(0, 8)) + '</code></span></div></div>'
      + '<div class="actions">' + acts + '</div></div>';
  }).join('') : '<div class="empty">还没有任何经验条目。</div>';

  const pf = snapshot.profile || [];
  document.getElementById('profile').innerHTML = pf.length ? pf.map(p =>
    '<div class="row"><div class="body"><div class="content"><strong>' + esc(p.key) + '</strong> = ' + esc(p.value)
    + '</div><div class="meta"><span class="badge">' + esc(p.source) + '</span>'
    + '<span>置信度 ' + Number(p.confidence).toFixed(2) + '</span>'
    + '<span>支持 ' + p.support + ' · 反驳 ' + p.contradict + ' · 证据 ' + (p.evidence || []).length + ' 条</span></div></div>'
    + '<div class="actions"><button class="danger" onclick="forget(\\'' + esc(p.key) + '\\')">忘记</button></div></div>'
  ).join('') : '<div class="empty">系统还没有形成任何偏好判断。</div>';
}

async function load() { snapshot = await api('/api/snapshot'); render(); }
async function del(id) { await api('/api/delete', { method: 'POST', body: JSON.stringify({ id, csrf: CSRF }) }); toast('已删除'); load(); }
async function restore(id) { await api('/api/restore', { method: 'POST', body: JSON.stringify({ id, csrf: CSRF }) }); toast('已恢复'); load(); }
async function forget(key) { await api('/api/profile/forget', { method: 'POST', body: JSON.stringify({ key, csrf: CSRF }) }); toast('已忘记'); load(); }

document.getElementById('correct-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const payload = {
    kind: document.getElementById('kind').value,
    scope: document.getElementById('scope').value,
    note: document.getElementById('note').value,
    csrf: CSRF
  };
  await api('/api/correct', { method: 'POST', body: JSON.stringify(payload) });
  document.getElementById('note').value = '';
  toast('纠正已记录'); load();
});

(async function init() {
  const kinds = await api('/api/correction-kinds');
  document.getElementById('kind').innerHTML = kinds.kinds.map(k =>
    '<option value="' + esc(k) + '">' + esc(k) + '</option>').join('');
  load();
})();
</script>
</body>
</html>
"""


class _PanelHandler(BaseHTTPRequestHandler):
    """Request handler bound to one engine instance."""

    engine: Engine
    csrf_token: str
    allow_edit: bool

    # Silence the default per-request logging: it writes to stderr and would
    # spam the terminal the console was launched from.
    def log_message(self, format: str, *args: Any) -> None:
        return

    # ------------------------------------------------------------- helpers

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # No CORS: the console is a local tool and nothing else should be able
        # to read from it, including a page in another tab.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"
        )

    def _html(self, body: str) -> None:
        self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _guard(self, data: dict[str, Any]) -> bool:
        """Reject a mutation without a valid CSRF token."""
        if not secrets.compare_digest(str(data.get("csrf") or ""), self.csrf_token):
            self._json(403, {"error": "invalid csrf token"})
            return False
        return True

    # --------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - http.server's fixed API
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                page = _HTML.replace(_TOKEN_PLACEHOLDER, self.csrf_token)
                page = page.replace("__MAXNOTE__", str(MAX_NOTE_CHARS))
                self._html(page)
            elif path == "/api/snapshot":
                snap = self.engine.panel.snapshot(include_inactive=True)
                self._json(
                    200,
                    {
                        "experiences": [v.as_dict() for v in snap.experiences],
                        "profile": list(snap.profile),
                        "counts": snap.counts,
                    },
                )
            elif path == "/api/status":
                self._json(200, {"status": self.engine.stats()})
            elif path == "/api/correction-kinds":
                self._json(200, {"kinds": list(CORRECTION_KINDS)})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover - defensive surfacing
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - http.server's fixed API
        path = urlparse(self.path).path
        data = self._body()
        try:
            if not self._guard(data):
                return

            if path == "/api/delete":
                resolved = self.engine.panel.delete(str(data.get("id") or ""))
                self._json(200, {"deleted": resolved})

            elif path == "/api/restore":
                resolved = self.engine.panel.restore(str(data.get("id") or ""))
                self._json(200, {"restored": resolved})

            elif path == "/api/correct":
                correction = Correction.build(
                    kind=str(data.get("kind") or CorrectionKind.OTHER),
                    scope=str(data.get("scope") or CorrectionScope.THIS_TASK_ONLY),
                    note=data.get("note"),
                    task_type=data.get("task_type"),
                )
                experience_id, touched = self.engine.panel.correct(correction)
                self._json(
                    200, {"experience_id": experience_id, "profile_keys": list(touched)}
                )

            elif path == "/api/profile/forget":
                key = str(data.get("key") or "")
                self._json(200, {"forgotten": self.engine.panel.forget_profile(key)})

            elif path == "/api/profile/override":
                key, value = str(data.get("key") or ""), str(data.get("value") or "")
                if not key or not value:
                    self._json(400, {"error": "key and value are required"})
                    return
                self.engine.panel.override_profile(key, value)
                self._json(200, {"overridden": key})

            elif path == "/api/rule/add":
                # Human-authored rules are the one place the owner writes prose.
                # It is capped and validated server-side by the provenance
                # policy, and disabled entirely when the panel is opened with
                # human editing turned off.
                if not self.allow_edit:
                    self._json(403, {"error": "human editing is disabled"})
                    return
                content = str(data.get("content") or "")
                if not content:
                    self._json(400, {"error": "content is required"})
                    return
                identifier = self.engine.add_rule(
                    content, task_type=data.get("task_type")
                )
                self._json(200, {"id": identifier})

            else:
                self._json(404, {"error": "not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive surfacing
            self._json(500, {"error": str(exc)})


def build_server(
    *,
    db_path: str | Path = "ratchet.db",
    host: str = "127.0.0.1",
    port: int = 8787,
    provider: Provider | None = None,
    allow_edit: bool = False,
) -> ThreadingHTTPServer:
    """Create the console server without starting it.

    ``allow_edit`` is False by default: the owner can view and delete, while
    every insertion is the machine's. Turning it on exposes the capped,
    validated rule box for owners who want to write their own rules directly.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            "the console only binds to the loopback interface; "
            "exposing it on a network would make the governance panel remote"
        )
    engine = Engine(db_path=db_path, provider=provider)
    token = secrets.token_urlsafe(32)

    handler = type(
        "BoundPanelHandler",
        (_PanelHandler,),
        {"engine": engine, "csrf_token": token, "allow_edit": allow_edit},
    )
    server = ThreadingHTTPServer((host, port), handler)
    # Keep a reference so the engine is not garbage-collected mid-session; its
    # SQLite connection must outlive individual requests.
    server.ratchet_engine = engine  # type: ignore[attr-defined]
    return server


def serve(
    *,
    db_path: str | Path = "ratchet.db",
    host: str = "127.0.0.1",
    port: int = 8787,
    provider: Provider | None = None,
    allow_edit: bool = False,
    open_browser: bool = True,
) -> None:
    """Run the console in the foreground until interrupted.

    ``open_browser`` is a convenience for ``ratchet panel``. It is best-effort:
    a headless machine must not fail to serve just because it has no browser.
    """
    server = build_server(
        db_path=db_path, host=host, port=port, provider=provider, allow_edit=allow_edit
    )
    url = f"http://{host}:{port}/"
    print(f"Ratchet governance panel: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        server.ratchet_engine.close()  # type: ignore[attr-defined]
