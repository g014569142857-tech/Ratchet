# Quickstart

Five minutes from an empty directory to a loop that is visibly learning, without
an API key.

## 1. Install

```bash
git clone https://github.com/g014569142857-tech/Ratchet.git
cd ratchet
pip install -e .
```

(Not on PyPI yet. After it is published, `pip install ratchet-ai` will do.)

Core has zero third-party runtime dependencies. Nothing downloads, no GPU.

## 2. Create a database and a rule

```bash
ratchet init
ratchet add "Answer in Chinese, three sentences at most."
```

You just wrote the only kind of entry a human may author: a rule, capped at 200
characters and validated to be a single atomic sentence rather than a pasted
conversation.

## 3. Run a cycle

```bash
ratchet chat "explain LoRA" --task "explain a concept"
```

The default provider is an offline echo, so this costs nothing and works on a
plane. You will see something like:

```
--- reward: unmeasured  variant: chain_of_thought
```

`unmeasured` because there is nothing to grade an echo against yet.

## 4. Make it measurable

The quickest way to see the loop close is to attach a real model and a real
check. In Python:

```python
import ratchet
from ratchet.core.provider import OpenAICompatProvider
from ratchet.improve.verifier import ObjectiveVerifier, RewardChain


class MentionsTerm(ObjectiveVerifier):
    """Reward: did the answer actually contain the key term?"""

    name = "mentions_term"

    def __init__(self, term: str):
        self.term = term.lower()

    def verify(self, task: str, output: str) -> float | None:
        return 1.0 if self.term in output.lower() else 0.0


engine = ratchet.Engine(
    db_path="ratchet.db",
    provider=OpenAICompatProvider(model="gpt-4o-mini", api_key="..."),
    reward_chain=RewardChain(objective=[MentionsTerm("low-rank")]),
)

for i in range(20):
    result = engine.run(task="explain LoRA", user_input="briefly")
    print(result.variant_id, result.reward.value)

print(engine.stats())
```

Watch `variant_id`: as one strategy starts scoring better, Thompson Sampling
selects it more often. That is the whole reinforcement loop, visible.

## 5. See what it learned

```bash
ratchet audit
```

```
af3c1d...  provenance=user_rule  kind=rule  score=+2.0  uses=4
    Answer in Chinese, three sentences at most.
7b21e9...  provenance=auto       kind=rule  score=-1.0  uses=2
    Mention low-rank adaptation explicitly when explaining LoRA.
```

The second line was written by the machine after a low-scoring interaction. You
cannot edit it — only delete it:

```bash
ratchet explain 7b21e9    # why does this exist? how has it done?
ratchet rm 7b21e9         # retract it
ratchet restore 7b21e9    # changed your mind
```

Or do it visually:

```bash
ratchet panel
```

## 6. Correct it structurally

Free-text correction is deliberately unavailable. Instead name the axis:

```bash
ratchet correct --kind too_long --scope global --note "three points max"
```

This is stored as a `derived` entry (delete-only, like everything the machine
writes) and also feeds the inferred profile:

```bash
ratchet profile
#   verbosity = concise   conf=0.50  support=1  contradict=0  [inferred]
```

## 7. Check whether it is working

```bash
ratchet eval
```

```
  period 1: 0.410  ##########
  period 2: 0.680  ################
first 0.410 -> last 0.680  (+0.270)
verdict: improving
```

Refuses to draw a curve from fewer than a handful of measured interactions
rather than showing you a dramatic number built on three data points.

## 8. Optional: export training data

```bash
ratchet export --format dpo --out pairs.jsonl
```

Then, on a machine with a GPU and an open-weight model:

```bash
pip install 'ratchet-ai[train]'
ratchet train --dataset pairs.jsonl --base-model Qwen/Qwen2.5-7B --out-dir ./adapter
```

Add `--dry-run` to see the exact settings first. This never modifies the base
model — only writes an adapter.

## Where the data lives

One file, `ratchet.db`, next to where you ran the command. It holds your real
work, so:

- it is already in `.gitignore` (`*.db`)
- `ratchet purge <id>` physically erases an entry, if soft deletion is not enough
- everything stays local; the only network traffic is the model call itself

## Next

- [concepts.md](./concepts.md) — the vocabulary, in dependency order
- [architecture.md](./architecture.md) — why each part is built the way it is
- [governance.md](./governance.md) — the permission model in full
