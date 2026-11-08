# Ratchet

**A local-first self-improvement engine that makes any LLM better the more you use it.**

[中文文档](./README_zh.md) · [Architecture](./docs/architecture.md) · [Quickstart](./docs/quickstart.md) · [Concepts](./docs/concepts.md) · [Governance](./docs/governance.md) · [How to verify](./docs/how_to_verify.md) · [Upload to GitHub](./docs/upload_to_github.md)

Ratchet does not fine-tune your model. It wraps it in a closed loop — **remember, reflect, select, rewrite** — so that each interaction leaves behind something the next one can use. It works on any model you can call, including closed APIs whose weights you will never see, on a laptop, with no GPU.

> **棘轮 (ratchet)**: a gear that only turns one way. Capability clicks up and does not click back down.

---

## The one rule

**The machine may add. The owner may subtract.**

Everything Ratchet learns is stored as plain text in a local SQLite file, and you can see all of it at any time. Machine-written entries are **delete-only** — you cannot rewrite them, only remove them. Deleting can only ever remove capability, so it is safe to grant without review; inserting changes what the model is told on every future task, so it is not.

```
ratchet audit     # SEE everything it concluded on its own
ratchet rm af3c   # and retract any of it
ratchet panel     # or do both in a browser
```

The only way a human writes into the store is a **bounded, structured correction**: you pick an axis (`too_long`, `wrong_tone`, `factually_wrong`, ...) and an optional note capped at 140 characters that is escaped and labelled as a quotation of your reaction. There is no free-text field, because the store is injected into every future prompt and a textarea would turn it into a second, unaudited instruction channel into the model.

---

## Quickstart

```bash
git clone https://github.com/g014569142857-tech/Ratchet.git
cd ratchet
pip install -e .

ratchet init
ratchet add "Answer in Chinese, three sentences at most."
ratchet chat "explain LoRA" --task "explain a concept"
ratchet rate up      # judge the answer you just read: no reward signal, nothing is learned
ratchet audit
```

> Not on PyPI yet, so install from a clone. Once it is published, `pip install ratchet-ai` will work.

No API key needed — the default provider is an offline echo, so the whole loop runs and you can inspect it before spending anything. To use a real model:

```bash
export OPENAI_API_KEY=sk-...
ratchet chat "explain LoRA" --provider openai --model gpt-4o-mini
```

Any OpenAI-compatible endpoint works, including Ollama, vLLM, LM Studio and Qwen/DashScope:

```bash
ratchet chat "你好" --provider openai --base-url http://localhost:11434/v1 --model qwen2.5
```

As a library:

```python
import ratchet
from ratchet.core.provider import OpenAICompatProvider

engine = ratchet.Engine(
    db_path="ratchet.db",
    provider=OpenAICompatProvider(model="gpt-4o-mini", api_key="..."),
)

result = engine.run(task="summarise", user_input="this quarter's report")
print(result.output)

engine.panel.view()          # what has it learned?
engine.panel.delete("af3c")  # retract anything wrong
```

---

## How it works

One `run()` performs a full cycle:

```
retrieve   hybrid BM25 + vector search recalls relevant experience
select     Thompson Sampling picks a strategy — and it is injected into the prompt
assemble   system + strategy + profile + recalled experience + your request, in that order
complete   call the model
verify     score the output: your verdict > objective check > LLM judge
record     append to the audit trail; update the arm's posterior
reflect    on a low score, distil a rule for next time
learn      fold the interaction into an inferred picture of how you like to be answered
automate   periodically propose a better strategy and re-compress what has been learned
```

The last three steps are why it compounds: interaction *N* writes something interaction *N+1* retrieves.

### This is reinforcement learning, minus the weights

| RL concept | In Ratchet |
|---|---|
| State | the task, your inferred profile, and the experience recalled for it |
| Action | which strategy to answer with (`concise`, `chain_of_thought`, ... and ones it evolves) |
| Reward | the verifier's score, blended by trust: your verdict > objective check > LLM judge |
| Policy | a Beta posterior per strategy, maintained by Thompson Sampling |
| Policy update | `alpha += reward`, `beta += 1 - reward` |

The policy is a probability distribution plus a set of readable rules instead of a neural network, but the update is the same arithmetic. This is the **in-context RL** family (Reflexion, Voyager, and the "LLM as optimizer" line of work) — a real, published idea, not a metaphor.

Two things keep the reward honest, because a mis-specified reward is worse than no reward:

- **Your thumbs-down overrides the blend** instead of joining it. Averaging a rejection against a passing objective check lands at ~0.55, above the midpoint, which would *raise* the probability of a variant you just rejected.
- **The LLM judge is a last resort**: weight-capped, length-penalised, and always outranked by anything mechanically checkable.

### Two routes, one dataset

- **Route B (default)** — no weight changes. Works on every model, including closed APIs. This is what the loop above does.
- **Route A (optional)** — real training. Everything Route B accumulates is a preference dataset built from your own work; export it and run LoRA + DPO on an open-weight model.

```bash
ratchet export --format dpo --out pairs.jsonl
ratchet train --dataset pairs.jsonl --base-model Qwen/Qwen2.5-7B --out-dir ./adapter
```

Route A is impossible for closed models by construction — you cannot update weights you cannot see — and it needs a GPU, so it sits behind `pip install 'ratchet-ai[train]'` and is never imported by the core.

---

## Governance panel

```bash
ratchet panel
```

Opens a local console on `127.0.0.1` showing every experience entry, where it came from, and what it has contributed — plus your inferred profile with the evidence behind each belief. Delete and restore are one click each. There is no free-text editing box.

```
ratchet audit --provenance auto    # only machine-written entries
ratchet explain af3c               # why does this entry exist?
ratchet restore af3c               # undo a deletion
ratchet purge af3c                 # physically erase it (asks first)
ratchet profile                    # what does it think it knows about you?
ratchet profile --reset            # and forget all of it
ratchet correct --kind too_long --scope global --note "three points max"
```

---

## The inferred profile

Ratchet works out how you like to be answered without being told. The signals are observable facts — you flagged an answer as too long, your input was mostly Chinese, an answer you accepted was short — not a model's speculation about you, so every belief can be traced to the interactions that produced it.

Three properties keep it from going wrong quietly:

- **Confidence decays.** An un-reconfirmed belief slowly loses authority. Without this, "this user wants terse answers" becomes self-confirming: answer tersely, receive no correction, treat the silence as agreement.
- **Contradiction can flip it.** Support and contradiction are counted separately; enough counter-evidence flips the value instead of being averaged into it.
- **You outrank it.** `ratchet profile --set verbosity=detailed` sets a belief the machine may never overturn.

---

## Limitations

Stated plainly, because overclaiming is how a project like this loses trust.

- **It cannot teach a model something it does not know.** A model that cannot write Rust still cannot. What improves is the *use* of existing capability, and the avoidance of repeated mistakes.
- **Route B does not touch weights.** Nothing learned survives the loss of the database — which is also what makes it reversible.
- **Weak models benefit less.** Reflection requires the model to be able to reason about its own failures.
- **Returns diminish.** As the library grows, retrieval precision drops. Distillation and automatic deactivation of low-scoring entries slow this, they do not eliminate it.
- **Route A is not guaranteed to help.** Fine-tuning can be neutral or harmful, and can cost you capabilities you never meant to trade away. Train into a separate adapter, keep the base model, and measure with `ratchet eval` before adopting anything.
- **Nothing is learned without a reward signal.** With no verifier configured, an unjudged answer is recorded as *unmeasured*, the bandit posterior does not move and no reflection fires. Use `ratchet rate up|down` after reading an answer, or configure an objective verifier (tests, schema, answer key) in the library API.
- **The bandit is context-free.** One global posterior covers every kind of task, so a strategy that is best for summaries and worst for code is averaged into a single number. Keying arms by task type is the obvious next step.
- **The posterior never forgets.** `alpha` and `beta` only grow, so after a few hundred interactions a single new rating barely moves the policy even if the model or your taste has changed. A discount factor would fix this.
- **Credit assignment is coarse.** The reward is attributed to the chosen strategy *and* to every recalled experience at once, with no control group, so "this experience helped" is a correlation, not a measurement.
- **"Reinforcement learning" is shorthand for a multi-armed bandit plus in-context memory.** There are no state transitions and no weight updates, so it belongs with in-context learning, not with RL in the full sense.
- **"Any LLM" means any model with a text API.** It does not mean closed products you cannot call programmatically.

---

## Install and develop

```bash
pip install -e ".[dev]"
pytest
```

Core has **zero third-party runtime dependencies** and the entire test suite runs offline with no API key, so CI is fast and free.

```bash
pip install -e '.[train]'   # Route A: trl / peft / datasets
pip install -e '.[mcp]'     # MCP server for Cursor, Claude Desktop, etc.
```

---

## Roadmap

| Version | Theme | Status |
|---|---|---|
| v0.1 | memory loop: store, retrieval, verifier, reflector, bandit | done |
| v0.2 | selection actually steers the prompt; inferred profile; evolution; distillation; governance console; eval; Route A export | **current** |
| v0.3 | richer objective verifiers (tests, schemas, compilers) shipped in-tree | planned |
| v0.4 | real embedding backends behind the `Embedder` protocol; more agent adapters | planned |

---

## License

MIT. See [LICENSE](./LICENSE).
