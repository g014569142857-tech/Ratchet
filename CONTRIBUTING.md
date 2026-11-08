# Contributing

Thanks for considering it. The loop is the interesting part, and most useful
contributions will touch it.

## Setup

```bash
git clone https://github.com/g014569142857-tech/Ratchet.git
cd ratchet
pip install -e ".[dev]"
pytest
```

The suite runs offline with no API key and no GPU — every test uses a scripted or
echo provider. This is deliberate and should stay true: a test that needs
credentials is a test nobody runs.

## Ground rules

These are the things a review will push back on, so they are worth knowing up front.

### 1. Core stays dependency-free

`src/ratchet/` outside `train/` must import nothing outside the standard library.
No `requests`, no `numpy`, no `torch`. If you need one, it belongs behind an
optional extra and a lazy import.

### 2. Tests keep running offline and deterministically

Seed anything random. Use a scripted provider. Do not hit the network.

### 3. The permission model is not negotiable without discussion

Machine-written entries are delete-only for humans. Human insertion is bounded and
structured. If a change makes it easier for arbitrary text to reach the store, it
needs a very good argument in the PR description — this is the project's central
safety claim, not a style preference.

### 4. Anything injected into a prompt gets screened

New text that reaches the instruction position (evolved strategies, distilled
blocks, reflected rules) needs validation, hostile-marker checks, and rejection of
echoed scaffolding. "The model would never say that" is not a defence — an offline
provider echoes its prompt verbatim, and that case is tested.

### 5. Reward changes need arithmetic, not adjectives

If you touch `verifier.py`, explain what the new blend does to a thumbs-down
against a passing objective check. The current override behaviour exists because
averaging them lands at ~0.55, above the midpoint, which raises the probability of
a variant the human just rejected.

### 6. Honest docs

If a feature has a boundary, write it down. Overclaiming is how a project like
this loses trust, and the Limitations section in the README is deliberate.

## What is most wanted

- **Objective verifiers.** Anything mechanically checkable: does it compile, does
  the test pass, does the JSON parse, does it match the answer key. RLVR says these
  beat a judge, and there can never be too many.
- **Embedders.** Real embedding models behind the `Embedder` protocol. The default
  hashed n-gram vector is deliberately mediocre and offline.
- **Providers.** Anthropic, Gemini, local runtimes. One method: `complete(prompt)`.
- **Eval suites.** Fixed task lists with mechanical checks, so "is it working?" has
  a repeatable answer.

## Pull requests

- one logical change per PR
- tests for new behaviour, and a test that fails without the fix
- run `pytest` and `ruff check src tests` before pushing
- update the README if you change the CLI surface

## Reporting a security issue

Do not open a public issue. See [SECURITY.md](./SECURITY.md).
