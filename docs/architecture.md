# Architecture

The reasoning behind each part, and the trade-offs that were decided rather than
deferred. Read [concepts.md](./concepts.md) first if the vocabulary is new.

---

## Layered view

```
┌──────────────────────────────────────────────────────────────┐
│  Adapters      CLI  ·  Python SDK  ·  MCP server  ·  web console
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│  Core           Session (one cycle)  ·  Policy (Thompson bandit)
│                 Provider (any backend)  ·  PromptAssembler
└──────────┬──────────────────────────────────┬────────────────┘
           │                                  │
┌──────────▼─────────────────┐  ┌─────────────▼────────────────┐
│  Memory                    │  │  Improve                     │
│  store (SQLite)            │◄─┤  verifier (reward chain)     │
│  retriever (BM25 + vector) │  │  reflector (Reflexion)       │
│  profile (inferred prefs)  │  │  evolver (strategy search)   │
│  provenance (permissions)  │  │  distiller (compression)     │
│  redact (secret scrubbing) │  │                              │
└──────────┬─────────────────┘  └──────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│  Governance    panel (view/delete/restore) · correction
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Train (optional)   dataset → SFT / DPO / GRPO  ·  LoRA      │
│  Eval               growth curve  ·  fixed suites            │
└──────────────────────────────────────────────────────────────┘
```

`governance` sits beside rather than under `memory` on purpose: it is the layer
that constrains what may be written, so it has to be able to see every write path,
and every adapter has to reach it.

---

## One cycle, in order

```
1  retrieve    HybridRetriever searches BM25 + hashed n-gram vectors over active
               entries only. Untrusted entries are dropped before assembly.
2  select      ThompsonPolicy samples each arm's Beta posterior, picks the max.
3  assemble    PromptAssembler renders, in fixed order:
                 base system
                 APPROACH FOR THIS TASK   <- the chosen arm's text
                 USER PROFILE             <- inferred preferences, hedged
                 BEGIN RECALLED EXPERIENCE DATA ... END   <- pinned + retrieved
                 the user's request       <- last, so it cannot be crowded out
4  complete    Provider.complete(prompt). One method; that is the whole seam.
5  verify      RewardChain blends every source that had an opinion.
6  record      Event row written; the arm's posterior updated; retrieved entries
               receive a share of the credit.
7  reflect     Below the reward threshold, the Reflector asks for one atomic rule
               and stores it — screened, redacted, and delete-only.
8  learn       Profile folds the interaction into inferred preferences.
9  automate    Every N interactions: the evolver may propose a strategy, the
               distiller may re-compress validated rules.
```

Steps 7–9 are why the loop compounds, and why they are individually switchable:
each one costs a model call.

---

## Why the strategy block is load-bearing

The bandit's action has to change the text the model sees, or the reward carries
no information about it. The original skeleton selected an arm, updated that
arm's posterior — and never passed the arm to the prompt builder. The arithmetic
was correct and the loop was inert: three arms producing byte-identical prompts.

`VARIANT_STRATEGIES` maps an arm name to the instruction it injects. Two details
matter:

- **`default` is the empty string.** It is the control arm and must stay in the
  pool, so an evolved strategy is always measured against doing nothing.
- **Mutations enter as new arms**, never replacing their parent. Replacing in
  place would destroy the posterior that justified the mutation and let the
  successor inherit credit it never earned.

---

## Reward: why the chain is ordered, not averaged

Sources, in the order they are trusted:

| Source | Weight | Notes |
|---|---|---|
| your verdict | — | **Overrides** the blend rather than joining it |
| objective | 1.0 | mechanically checkable; only checks the property it encodes |
| LLM judge | 0.3 (capped) | dense and cheap, but biased; length-penalised |

The override is the important part. Blending a thumbs-down against a passing
objective check gives ~0.55, which is above the 0.5 midpoint — the bandit would
*increase* the selection probability of a variant you just rejected. Human
satisfaction and mechanical correctness disagreeing means the verifier is
measuring the wrong thing, and the human is the arbiter of their own
satisfaction.

A verifier returning `None` means "no opinion", and is excluded from the average
entirely. Recording it as zero would teach the policy to avoid strategies that
were merely never evaluated.

---

## Memory: why SQLite, and why hashed vectors

**One file, no server.** The store is the owner's private work history; it should
be trivial to back up, easy to `.gitignore`, and impossible to accidentally
expose. A database server is a second thing to run and a second thing to leak.

**Hashed n-gram embeddings by default.** Real embedding models are better, but
they are a download, an API key, or a GPU. The default is deterministic and
offline so that CI and a first-time user get identical behaviour. The `Embedder`
protocol exists so this is a swap, not a rewrite.

**BM25 + vectors, fused.** BM25 nails exact terminology ("LoRA" must match
"LoRA"); vectors catch paraphrases that share no keywords. The tokeniser falls
back to single non-space characters, which is what lets Chinese entries — with no
spaces — match at all.

**The store is thread-safe.** Adapters are not single-threaded (the web console
serves from a pool), and a SQLite connection is bound to its creating thread.
`_LockedConnection` relaxes `check_same_thread` *and* serialises access, because
relaxing it alone would trade a loud failure for silent interleaving.

---

## Security: persistent injection

The threat that shapes most of this code is not a poisoned message — it is a
poisoned **memory**. Experience is replayed into every future prompt, so an entry
that survives the session is materially worse than one that does not.

Defences, in order of importance:

1. **Data, never instructions.** Recalled content lands inside a delimited block
   that states it is data. It never occupies the instruction position.
2. **Delimiter escape is neutralised.** An entry containing the block markers
   verbatim could close the block early or open a convincing fake one, so those
   strings are stripped at render time. The markers are matched as *lines*, not
   as the multi-line header, because forging one line is enough.
3. **Provenance and trust are labelled at the injection site**, so both the model
   and a human reading a transcript can weigh each line.
4. **Untrusted entries never reach the prompt.** Dropped silently, or raised in
   strict mode.
5. **Reflections over untrusted material are never stored.** Deleting is safe;
   storing is not.
6. **Evolved strategies and distilled blocks are screened** for hostile content
   and for echoed scaffolding, because both are injected into privileged
   positions.
7. **Secrets are redacted before storage**, not at render time — the threat is
   the stored copy, and machine-written entries are never reviewed by a human.

Note what is deliberately *not* treated as a threat: the owner editing their own
local experience panel. There is no privilege escalation — it only affects their
own model, and the worst outcome is self-inflicted and reversible by deletion.
The defences target third-party content.

---

## Governance: the asymmetry

| Operation | Who | Why |
|---|---|---|
| view everything | human | transparency has no side effects |
| delete / deactivate | human | pure subtraction |
| restore | human | re-admits already-reviewed text |
| insert | machine only | changes every future prompt |
| edit content | machine only | same |
| structured correction | human, bounded | too valuable a signal to forbid |

`user_rule` entries are human-authored and stay editable under a 200-character
cap, with conversation transcripts rejected. Auto and derived entries are
delete-only. See [governance.md](./governance.md) for the full argument.

---

## Profile drift

The failure mode this design is built against: the system concludes "this owner
wants terse answers", answers tersely, receives no correction because the answers
are adequate, and treats the silence as confirmation. The belief is now locked in
by the very behaviour it produced.

Three counters:

- **Time decay.** Confidence halves every 14 days without new evidence, so an
  un-reconfirmed belief loses authority instead of compounding.
- **Contradiction flips.** Support and contradiction are counted separately;
  when contradiction reaches support, the value flips and the tally restarts.
  The comparison is inclusive so a tie goes to the newer signal.
- **Human override.** Marked as such, never auto-reversed.

Every belief carries the event ids that produced it, so the panel can show its
work and `explain` can answer "why does this exist?".

---

## Distillation: the growing-library problem

Retrieval gets weaker as the store grows: the budget does not grow with it, and
the best lessons compete with near-duplicates for a handful of slots. The
distiller periodically compresses validated rules into a compact block that is
**pinned** into every prompt.

It is stored as an ordinary `auto` entry with `kind=distilled`, which means the
owner can see it and delete it — and deleting it stops it being pinned, falling
back to the individual rules, which are never removed. A distilled persona is an
optimisation, never the only copy of the knowledge.

---

## Route A: why it is separate

Everything Route B accumulates is a preference dataset: real tasks, real outputs,
real measured rewards, from the owner's own work. That is a better training set
than synthetic prompts, and it is the payoff for having run Route B first.

It lives in `train/` behind optional dependencies because:

- it is impossible for closed APIs — you cannot update weights you cannot see
- it needs a GPU
- a first install should stay small, CPU-only and network-free

Records carry the `reward_source` they were scored under, because a pair built
from two judge scores is much weaker evidence than one where a side was
objectively verified, and silently mixing them is how people end up training on a
judge's length bias.

---

## Evaluation

`eval/` answers "is this working?" two ways:

- **Growth curve** — mean reward per chronological slice of recorded
  interactions. Unmeasured interactions are excluded rather than counted as
  failures; slices too thin to trust are refused rather than plotted.
- **Fixed suites** — re-run the same tasks at two points in time, holding the
  workload constant. This is the only version that attributes a difference to
  the system rather than to what you happened to ask that week.

Neither is proof. A rising curve can be a run of easy tasks. They exist so the
question has an answer that is a number.

---

## Extending

| I want to... | Implement |
|---|---|
| use a new model backend | `Provider.complete(prompt) -> str` |
| add a mechanical check | `ObjectiveVerifier.verify(task, output) -> float \| None` |
| use real embeddings | `Embedder.embed` and `Embedder.similarity` |
| add a strategy to the pool | `register_variant(name, instruction)` |
| plug into another agent | the MCP server, or `Engine` directly |
