# Security Policy

## Scope

Ratchet stores the owner's real work and injects accumulated experience into every
prompt. Two properties make security load-bearing here:

1. **The store is a privileged position.** Whatever is in it shapes what the model
   is told on every future task.
2. **The data is personal.** It contains the owner's actual work, code and
   preferences.

## Reporting

Please do not open a public issue for a vulnerability.

Email the maintainers privately with:

- what the issue is and how to reproduce it
- which version and platform
- the impact you believe it has

You should get an acknowledgement within a few days. We will agree a disclosure
timeline with you before publishing anything.

## The threat model, stated

### What is defended against

**Persistent prompt injection.** Experience is replayed into every future prompt,
so a poisoned entry is worse than a poisoned message. Defences:

- recalled content is rendered inside a delimited block that states it is data
- block markers are stripped from rendered content, matched as lines so forging
  one line is not enough
- provenance and trust level are labelled at the injection site
- entries flagged `untrusted` never reach the prompt
- reflections derived from untrusted material are never stored
- evolved strategies and distilled blocks are screened for hostile content and
  echoed scaffolding
- secrets are redacted before storage, not at render time

**Reward hacking.** The LLM judge is weight-capped, length-penalised and always
outranked by mechanically verifiable signals. Explicit human feedback overrides the
blend rather than joining it.

**Profile drift.** Inferred preferences decay with time, can be flipped by
contradiction, and are always outranked by an explicit human override.

**Network exposure of the console.** The governance panel binds `127.0.0.1` only
and refuses any other interface; mutations require a CSRF token minted at startup.

### What is explicitly not defended against

**The owner editing their own local experience panel.** This is not a security
boundary. It affects only their own model, there is no privilege escalation, and
the worst outcome is self-inflicted and reversible by deletion. Building a
permission system to stop someone from editing their own file would add complexity
for no safety gain.

**A malicious model provider.** If the endpoint you call returns hostile
completions, Ratchet's screening reduces the damage but cannot guarantee safety.
Treat provider output as untrusted input.

**A compromised host.** Ratchet is a library on your machine. If your machine is
compromised, so is `ratchet.db`.

## Data handling

- Everything stays local. The only network traffic is the model call itself.
- `*.db` is in `.gitignore` by default so the database cannot be committed by
  accident.
- `ratchet purge <id>` physically erases an entry and strips it from any profile
  belief citing it as evidence.

## Known limitations

- Redaction is pattern-based and will not catch every secret shape. It is a
  safety net, not a guarantee — review before sharing a database.
- The web console has no authentication. It relies entirely on loopback binding and
  the CSRF token; do not expose it through a proxy or tunnel.
- Screening of generated text is heuristic. A determined model could attempt to
  phrase a hostile instruction in ways the marker checks do not catch.
