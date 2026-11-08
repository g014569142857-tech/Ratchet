# Example 5 — Route A: turn experience into weight updates

**Read the warning at the bottom before running any of this.**

Route A is impossible for closed API models by construction — you cannot update
weights you cannot see — and it needs a GPU. It is an optional extra, not the
default, and the core package never imports `torch`, `trl` or `peft`.

What Ratchet contributes here is the **dataset**: real tasks, real outputs and
real measured rewards drawn from your own work. That is a better training set than
synthetic prompts, and it is the payoff for having run Route B first.

## 1. Export

```bash
# Preference pairs: a good completion against a bad one for the same task.
ratchet export --format dpo --out pairs.jsonl

# Or: keep only the good completions.
ratchet export --format sft --out sft.jsonl --min-reward 0.8

# Or: every attempt of a task, for group-relative training.
ratchet export --format grpo --out groups.jsonl
```

The command reports what it skipped and why:

```
exported 41 dpo records to pairs.jsonl
  considered 210, skipped: 62 unmeasured, 88 single-attempt, 19 small-gap
```

Most skips are expected. A task you attempted once cannot produce a preference
pair, and a 0.04 reward gap is noise, not a preference.

## 2. Filter on reward provenance

Each record carries `chosen_source` and `rejected_source`. A pair where one side
was objectively verified is strong evidence; a pair built from two judge scores is
much weaker — the judge is known to prefer longer answers, and training on that
teaches your model to be verbose.

```python
import json

strong = [
    r for r in map(json.loads, open("pairs.jsonl", encoding="utf-8"))
    if "objective" in (r["chosen_source"], r["rejected_source"])
]
```

## 3. Inspect before training

```bash
ratchet train --dataset pairs.jsonl --base-model Qwen/Qwen2.5-7B --dry-run
```

Prints the exact settings without touching a GPU.

## 4. Train

```bash
pip install 'ratchet-ai[train]'
ratchet train --dataset pairs.jsonl --base-model Qwen/Qwen2.5-7B --out-dir ./adapter
```

This writes a LoRA adapter only. **The base model is never modified**, so a bad run
costs you time and a directory you can delete.

## 5. Evaluate before adopting

```bash
ratchet eval
```

Or better, hold the workload constant and compare — see `EvalSuite` in
`ratchet.eval.growth`. A rising growth curve can just be a run of easy tasks;
re-running the same fixed task list is the only version that attributes a
difference to the system.

---

## The warning

Fine-tuning can make a model **worse**. It can also cost you capabilities you
never meant to trade away — the classic catastrophic forgetting failure is a model
that learns your house style and quietly loses its arithmetic. Nothing in this
project can detect that for you.

- train into a separate adapter, keep the base model untouched
- use a low learning rate and few epochs first
- evaluate on tasks that have nothing to do with what you trained on
- if the base scores better, throw the adapter away

Route B, by contrast, is reversible by deleting a row. That asymmetry is the main
reason it is the default.
