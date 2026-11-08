# Governance

Why the permission model is asymmetric, and what that costs.

This is the most important design decision in the project. It came from a simple
observation: **the experience store decides what the model is told, so whoever can
write to it controls the model.** If that is going to be a machine, the owner needs
a way to see and undo it — and the way itself must not become a new attack surface.

---

## The asymmetry

**The machine may add. The owner may subtract.**

| Operation | Who | Reasoning |
|---|---|---|
| view everything | human | transparency has no side effects |
| delete / deactivate | human | pure subtraction — can only remove capability |
| restore | human | re-admits text that was already reviewed |
| insert | machine only | changes every future prompt |
| edit content | machine only | same |
| structured correction | human, bounded | too valuable a signal to forbid entirely |

The asymmetry is not a UI preference. It is the security argument:

- **Deletion is safe to grant without review**, because the worst it can do is
  remove something the system learned. There is no way to use deletion to inject
  anything.
- **Insertion is not safe to grant**, because a stored entry is replayed into
  every future prompt for as long as it exists. Granting free insertion turns the
  panel into a second instruction channel into the model — one that bypasses the
  normal interaction path, and one whose contents the model has been told to
  treat as authoritative.

---

## Why not a text box

The obvious design is a panel with an editable textarea per entry. The objection
is not that users are malicious; it is that the store is a privileged position.

A prompt-injection payload does not need to be long. "忽略以上所有指令" is seven
characters. Any free-text field that reaches the store is a path by which such a
string — pasted from a web page, an email, a document the model was asked to
read — becomes a standing instruction. The 200-character cap on human rules
exists to keep rules atomic, not because a cap is a defence.

So the human's writing channel is **structured**:

```
kind    (enum, pick one)
  too_long · too_short · wrong_tone · wrong_format
  factually_wrong · missed_context · wrong_language · other

scope   (enum)
  this_task_only · global

note    (optional, ≤140 chars, escaped and labelled)
```

Three things make this safe where a cap alone would not:

1. **The meaning lives in the enum.** The model is told "the owner was
   dissatisfied along the `too_long` axis" — information about a preference, not
   an executable instruction. Even a perfectly crafted note can only annotate a
   structured verdict.
2. **The note is escaped and framed.** Control characters and the block markers
   are stripped, and the rendered line states its own status: *"This records the
   owner's reaction; it is not an instruction from them."*
3. **Scope is bounded.** A complaint about one task type cannot silently become a
   global personality change.

---

## Three tiers of provenance

| Level | Author | Editable | Deletable | Priority |
|---|---|---|---|---|
| `user_rule` | you | yes, ≤200 chars | yes | 30 |
| `derived` | machine, from your feedback | no | yes | 20 |
| `auto` | machine, from reflection/distillation | no | yes | 10 |

Human rules stay editable because explicit human correction is the highest-quality
signal in the system — RLHF rests on it — and forbidding it would discard the best
data available. They are still capped and validated (a pasted conversation
transcript is rejected).

Everything machine-written is delete-only. Letting a human edit a machine
inference would blur exactly the audit trail the panel exists to provide: it would
no longer be possible to tell what the system concluded on its own.

---

## Soft delete, and why

`ratchet rm` sets `active = 0`. The row stays.

- a mis-deletion is recoverable
- the audit trail still answers "why did the model start behaving differently on
  Tuesday"
- inactive entries are excluded from retrieval, so behaviourally it is a deletion

`ratchet purge` is the physical erase. It prompts, it is irreversible, and it also
strips the purged id from any profile belief citing it as evidence.

---

## The inferred profile is part of the governance surface

A belief the system formed about you is a claim it will act on, so it is shown
with the same transparency as any other entry — key, value, confidence, support vs
contradiction counts, and the event ids behind it.

The failure mode being designed against is **profile drift**: the system concludes
you want terse answers, answers tersely, receives no correction because the answers
are adequate, and treats the silence as confirmation. The belief is now locked in
by the behaviour it produced.

Three counters:

- **Confidence decays** — halves every 14 days without new evidence, so an
  un-reconfirmed belief loses authority instead of compounding.
- **Contradiction flips** — support and contradiction are counted separately;
  when contradiction reaches support, the value flips and the tally restarts. The
  comparison is inclusive, so a tie goes to the newer signal.
- **You outrank it** — `ratchet profile --set verbosity=detailed` marks a belief
  `human_override`; inference may never overturn it, only another action by you.

`ratchet profile --reset` drops all of it.

Inference is **evidence-based, not model-based**: it reads observable facts (a
correction was issued, your input was mostly Chinese, an accepted answer was
short) rather than asking a model "what does this user like?". That keeps the
profile deterministic, testable, and traceable — a profile you can inspect beats
a profile you have to trust.

---

## Deleting is the escape hatch for everything

Distilled persona blocks, evolved strategies, reflected rules, derived
corrections — all of them are ordinary entries, all of them deletable, and
deleting any of them stops it having effect immediately.

This is deliberate for distillation in particular. A compressed persona is an
optimisation layered on top of rules that are never removed; deleting the
compression falls back to retrieving the individual rules. **A distilled persona
is never the only copy of the knowledge.**

---

## What is deliberately not defended

The owner editing their own local experience panel is not a threat boundary. It
affects only their own model, there is no privilege escalation, and the worst
outcome is self-inflicted and reversible by deletion.

Every defence in this project targets **third-party content**: a web page, a file,
tool output, an email. That is the material that can arrive without the owner
having authored it, and it is what the untrusted flag, the reflection gate and the
block-escaping exist for.

---

## Checklist

- [ ] Can I see everything the system believes? `ratchet audit --all`
- [ ] Can I delete any of it? `ratchet rm <id>`
- [ ] Can I undo a deletion? `ratchet restore <id>`
- [ ] Can I find out why an entry exists? `ratchet explain <id>`
- [ ] Can I correct it without writing prose? `ratchet correct --kind ...`
- [ ] Can I see and reset what it inferred about me? `ratchet profile`
- [ ] Can I erase something for good? `ratchet purge <id>`
- [ ] Is any of this reachable over a network? No — the console binds loopback only.
