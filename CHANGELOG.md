# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **`ratchet rate up|down`.** Judges the most recent answer after you have read it.
  Until now the only CLI feedback was `chat --feedback`, which has to be chosen
  *before* the model answers, so from a terminal the loop could never receive a
  reward and never learned anything. Backed by `Engine.rate_last()`, which also
  refuses to rate an event that already has a reward (it would be double-counted).
- `chat` now says so on stderr when an answer produced no reward signal.

### Fixed

- **Token estimate under-counted Chinese.** `len // 4` is right for English but
  CJK text is ~1-2 tokens per character, so cost reports for Chinese input were
  several times too low. CJK characters now count as 1.5 tokens.
- Line endings: the repository was stored with CRLF. Added `.gitattributes` and
  normalised to LF so diffs and Linux checkouts are clean.
- README install instructions no longer promise a PyPI package that does not exist
  yet; they install from a clone.

### Documented

- New entries under Limitations: no reward means no learning, the bandit is
  context-free, the posterior never forgets, credit assignment is coarse, and the
  loop is a bandit plus in-context memory rather than full RL.

## [0.2.0]

The release that makes the loop actually close, and gives the owner a real window
into it.

### Fixed

- **The bandit was inert.** `Session` selected a strategy, updated that strategy's
  posterior, and never passed the selection to the prompt builder — so the three
  arms produced byte-identical prompts and the reward was fitting noise. The chosen
  variant now renders an `APPROACH FOR THIS TASK` block, which is what makes a
  reward earned under an arm evidence about that arm.
- **Delimiter escape was possible.** A stored entry containing a block marker
  verbatim could close the data block early or open a convincing fake one. Markers
  are now stripped at render time and matched as lines, not as the multi-line
  header, because forging one line is enough.
- **The web console could not serve a request.** The SQLite connection was created
  on the main thread while `ThreadingHTTPServer` handled requests on worker
  threads. The store now serialises access across threads.
- **`status` reported zero tokens.** It read the current process's session counter
  rather than the cumulative event table, so every invocation showed zero.
- **Profile decay never happened.** `save_profile_entry` stamped the wall clock
  instead of the caller's timestamp, so a back-dated observation was
  indistinguishable from a fresh one.
- **Contradiction could never flip a belief.** The flip required contradiction to
  *exceed* support; it now requires it to reach support, so a tie goes to the newer
  signal.
- **Id prefixes resolved case-sensitively.** Ids are lowercase uuid4 but are read
  off a terminal and typed back by hand, so a capitalised prefix failed with
  "experience not found" — indistinguishable from the entry being gone. Prefix
  resolution is now case-insensitive and ignores surrounding whitespace.
- **The MCP adapter broke on `mcp` 2.x.** The SDK renamed `FastMCP` to `MCPServer`
  and left behind a `fastmcp` shim whose only job is to raise
  `ModuleNotFoundError`. The server class is now resolved dynamically, so 1.x and
  2.x both work.

### Added

- **Inferred user profile** (`memory/profile.py`) — preferences derived from
  observable evidence, with time decay, contradiction flipping, evidence
  traceability, and human override that inference may never overturn.
- **Prompt evolution** (`improve/evolver.py`) — the bandit's action space grows
  instead of being hand-tuned. Mutations enter as new arms with uniform priors so
  they must beat their parent on evidence.
- **Distillation** (`improve/distiller.py`) — validated rules are periodically
  compressed into a compact persona block that is pinned into every prompt.
  Deletable like any other entry, and never the only copy of the knowledge.
- **Governance panel** (`governance/`) — view, delete, restore, purge, explain, and
  submit bounded structured corrections. One place where the permission rules are
  enforced, used by both the CLI and the console.
- **Web console** (`adapters/web.py`, `ratchet panel`) — a local, dependency-free
  dashboard on `127.0.0.1` with CSRF-protected mutations and no free-text editing
  box.
- **Redaction** (`memory/redact.py`) — secrets and identifiers are scrubbed from
  machine-written entries before storage, on by default.
- **Route A export** (`train/dataset.py`) — SFT, DPO and GRPO datasets, each record
  carrying the reward provenance it was scored under.
- **Evaluation** (`eval/growth.py`) — cohort growth curves and fixed suites, with
  unmeasured interactions excluded and thin samples refused rather than plotted.
- CLI commands: `restore`, `purge`, `explain`, `correct`, `profile`, `train`,
  `eval`, `panel`.
- **Acceptance script** (`scripts/verify.py`) — nine falsifiable checks, each
  guarding a defect that was real at some point, runnable in seconds without
  installing the package.
- Docs: `README.md`, `README_zh.md`, and `docs/` covering architecture, quickstart,
  concepts and governance, plus `docs/how_to_verify.md`, which explains how to
  reproduce every claim and how to break each fix to confirm the guard goes red.

### Changed

- `SessionResult` carries the `strategy` text the chosen arm injected, for
  auditability.
- Evolution and distillation are **off by default**: both cost model calls, and the
  cost of autonomy should be visible at the call site rather than hidden in config.
- `examples/02_custom_verifier.py` starts from an empty store, so the posteriors it
  reports are reproducible instead of accumulating across runs.
- pytest `addopts` no longer passes `-q`. On pytest 9 that flag suppresses the
  pass/fail summary line entirely, which made a green run indistinguishable from a
  truncated one.

## [0.1.0]

The skeleton: provenance-tagged SQLite store, BM25 + hashed-vector hybrid
retrieval, Thompson Sampling policy, reward chain with source prioritisation,
Reflexion-style reflector, injection-aware prompt assembly, CLI, and an MCP server.
