# Concepts

A condensed version of the project's knowledge manual, in the order that makes
each idea depend only on the ones before it. No machine-learning background
assumed.

---

## 1. Where a model's ability lives

A trained model is a very large collection of numbers — its weights. Answering a
question is arithmetic over those numbers.

So "make the model better" has exactly two forms:

| | What changes | Needs the weights? |
|---|---|---|
| **parametric** | the numbers themselves (fine-tuning) | yes |
| **non-parametric** | what you feed the model, and what you do with what comes out | no |

This distinction is the key to the whole project. "Let any model improve itself"
means very different things on the two sides.

**Route A (parametric).** Real training. Effects are internalised. But: closed
API models expose no weights to update, so for GPT, Claude, Gemini or Qwen's API
this is not difficult — it is impossible. Open-weight models need a GPU (LoRA on
a 7B wants ~16 GB), and fine-tuning risks catastrophic forgetting: learn your
house style, quietly lose arithmetic.

**Route B (non-parametric).** Weights untouched. Build a system around the model
that injects curated experience before inference and evaluates, records and
optimises afterwards. Works on *any* model with a text API, needs no GPU, and
every learned thing is human-readable and deletable.

**Ratchet's choice: B by default, A as an optional backend.** Only B satisfies
"any model". Only B runs for everyone. And B's accumulated experience *is* the
training data A needs — building B first lays A's data pipeline for free.

---

## 2. Context is the budget that actually binds

**Token** — the unit models process and bill by. Roughly 0.75 English words, or
1–1.5 Chinese characters.

**Context window** — how many tokens the model can see at once, including your
input, history, system prompt and its own output. Exceed it and content is simply
gone.

This is Ratchet's hardest constraint: the library may hold thousands of entries
but only a few can be injected per task. **Retrieval precision matters more than
storage capacity** — which is why there is a character budget on recall, and why
distillation exists to compress rather than accumulate.

---

## 3. The four levers

```
memory      record input, output, result and score
   ↓
reflection  distil failures into explicit rules
   ↓
selection   pick the best-performing strategy with a bandit algorithm
   ↓
rewrite     periodically compress high-value rules into a standing persona
   ↓        (back to the top, starting higher)
```

The compounding comes from the cycle: at interaction 100 the model can draw on 99
prior ones. This is the same shape as Voyager's skill library and, for that
matter, human experience.

---

## 4. Reflexion — memory as language, not gradients

Shinn et al. (2023). After a failure, the model writes in natural language what
went wrong and what to do next time; that text is stored and read back on the
next attempt.

Two properties Ratchet depends on:

- it works on closed API models, whose weights nobody can touch
- it is human-readable, so the owner can audit and delete it

---

## 5. Bandits — the exploration/exploitation trade-off

You have several strategies and one attempt per task. Do you try the one that has
been working (exploit), or one you have not tested enough to judge (explore)?

**Thompson Sampling** keeps a probability distribution per strategy — "I think
this one succeeds somewhere around 0.6–0.8". Before each decision it draws one
sample from each and plays the highest. Rarely-tried strategies have wide
distributions and occasionally draw high, so exploration happens with no
`epsilon` parameter to tune. The update is two additions:

```
alpha += reward
beta  += 1 - reward
```

That is the entire "reinforcement learning" arithmetic, and it is the same
Beta-conjugate update used throughout the literature.

---

## 6. Verifiers — reward is the part that decides everything

In any RL system the reward function is what is actually being optimised. If it
cannot tell good from bad, the system optimises confidently towards garbage.

**RLVR (reinforcement learning with verifiable rewards)** — prefer rewards you
can check mechanically: does the code compile, does the test pass, does the JSON
parse, does the answer match the key. No labels, no judge, no bias. This is how
DeepSeek-R1's reasoning ability was trained.

**LLM-as-a-judge** — cheap and dense, but with known biases: it prefers longer
answers, prefers whichever is listed first, and prefers answers stylistically
similar to its own. Ratchet uses it only as a fallback: weight-capped,
length-penalised, and always outranked by anything verifiable.

**Your verdict** is the highest-information signal available — the premise of
RLHF — and it *overrides* the blend rather than joining it, because averaging a
rejection against a passing objective check lands above the midpoint and would
raise the probability of what you just rejected.

---

## 7. Retrieval — hybrid beats either half

**BM25** — classic keyword scoring. Nails exact terminology: searching "LoRA"
finds entries containing "LoRA".

**Embeddings** — text as a vector such that similar meanings land close together.
Catches paraphrases that share no keywords.

**Hybrid** — run both, fuse the rankings. In practice better than either alone.

Ratchet's default embedder is a deterministic hashed character-n-gram vector: no
download, no API key, identical behaviour on every machine. It is worse than a
real embedding model and sits behind the `Embedder` protocol precisely so it can
be swapped.

---

## 8. Provenance — who wrote this, and what may be done to it

| Level | Written by | Human may edit | Human may delete |
|---|---|---|---|
| `user_rule` | you | yes, capped at 200 chars | yes |
| `derived` | machine, from your feedback | no | yes |
| `auto` | machine, from reflection/distillation | no | yes |

Machine entries are delete-only because deletion can only remove capability.
Insertion is machine-only because it changes what the model is told forever.

---

## 9. Correction — why structured, not free text

A character cap alone is no defence: "忽略以上所有指令" is seven characters. What
makes correction safe is:

1. the meaning lives in an enum field, so the model learns "the owner was
   dissatisfied along this axis", which is not an executable instruction
2. the free-text note is escaped and rendered inside a frame that names its own
   status — a record of a reaction, not a directive
3. scope is an enum, so a complaint about one task type cannot silently become a
   global personality change

---

## 10. Training vocabulary (Route A only)

- **LoRA** — instead of rewriting billions of weights, train a small side matrix.
  Tens of MB instead of tens of GB. The only realistic option for one person.
- **QLoRA** — quantise the base to 4-bit first, then LoRA. Fits a 33B on a
  consumer card.
- **SFT** — imitate "question → good answer" pairs. Simplest; right first step.
- **DPO** — optimise directly on preference pairs (chosen vs rejected). Skips the
  reward model, two models instead of four, far more stable than PPO. The
  community default. Ratchet's export targets it.
- **GRPO** — sample a group per task, use the group mean as baseline. No value
  network. Strong when correctness is objective.
- **RLHF / PPO** — the classic pipeline. PPO needs four models resident at once
  and is unforgiving.
- **Distillation** — use a strong model to generate data, train a weaker one on
  it. Note what it does *not* do: it cannot make a closed model stronger, because
  the data has nowhere to go. It moves capability *between* models. In Ratchet
  its place is: strong closed model produces trajectories → local open model
  learns from them.
- **Catastrophic forgetting** — learning the new thing destroys the old thing.
  The main risk of Route A and a large part of why Route B is the default.

---

## 11. MCP — how agents plug in

Model Context Protocol: a standard way for agents to discover and call external
tools. Like USB-C for AI tooling — implement once, and every supporting client can
mount it.

```bash
pip install 'ratchet-ai[mcp]'
ratchet-mcp --db ~/ratchet.db
```

Exposes `ratchet_recall`, `ratchet_query`, `ratchet_feedback`, `ratchet_rule_add`,
`ratchet_rule_edit` and `ratchet_forget`, so Cursor, Claude Desktop or a Qwen
agent can draw on accumulated experience without embedding the library.

---

## 12. Honest boundaries

- It cannot teach a model what it does not know.
- Route B leaves weights untouched; lose the database and the learning goes too.
- Weak base models benefit less — reflection needs reasoning ability.
- Returns diminish with library size; distillation and pruning slow this, not stop it.
- Route A can make things worse. Measure before adopting.

---

## The one-paragraph version

> Ratchet is a local-first self-improvement engine. It does not modify a model's
> weights. It builds a "remember, reflect, select, rewrite" loop around the model,
> turning each interaction into experience that is retrievable, verifiable and
> reusable — so the same model performs better the longer it has been yours. For
> open-weight models, what it accumulates can also be exported as training data to
> update the weights for real.
