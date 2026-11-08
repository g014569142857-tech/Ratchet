## What this changes

<!-- One or two sentences. What is different after this PR? -->

## Why

<!-- What problem does it solve? If it fixes an issue, link it. -->

Fixes #

## How it was verified

<!-- Which tests cover it? Ideally one that fails without the change. -->

- [ ] `pytest` passes
- [ ] new behaviour has a test
- [ ] tests still run offline, with no API key and no network

## Checklist for reviewers

- [ ] **Core stays dependency-free** — nothing outside `train/` imports a
      third-party package
- [ ] **The permission model is intact** — machine-written entries are still
      delete-only for humans; no new path lets arbitrary prose reach the store
- [ ] **Anything injected into a prompt is screened** — length-bounded, checked
      for hostile content, and rejected if it echoes the scaffolding (an offline
      provider echoes its prompt verbatim, and that case is tested)
- [ ] **Reward changes are justified with arithmetic**, not adjectives — state
      what happens to a thumbs-down against a passing objective check
- [ ] **Randomness is seeded** so tests are deterministic
- [ ] Docs updated if the CLI or public API surface changed

## Anything else

<!-- Trade-offs you considered, or things you deliberately did not do. -->
