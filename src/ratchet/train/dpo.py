"""Optional LoRA / DPO training backend for open-weight models (Route A).

This module is deliberately thin and deliberately optional.

It is optional because Route A is impossible for closed APIs by construction -
you cannot update weights you cannot see - and because it needs a GPU. The core
package therefore never imports ``trl``, ``peft`` or ``torch``; they are imported
lazily inside the functions here so that ``pip install ratchet-ai`` stays a
small, CPU-only, network-free install.

It is thin because a training script is not where this project's value is. What
Ratchet contributes to Route A is the *dataset*: real tasks, real outputs and
real measured rewards drawn from the owner's own work. Handing that to ``trl`` is
a few dozen lines, and pretending to reimplement DPO here would only add surface
area to maintain.

An honest warning, kept next to the code rather than buried in the docs
-----------------------------------------------------------------------
Fine-tuning can make a model worse. It can also destroy capabilities the owner
never asked to trade away - the classic catastrophic forgetting failure is a
model that learns your house style and quietly loses its arithmetic. Nothing here
can detect that for you. Train into a separate adapter directory, keep the base
model untouched, and evaluate before you adopt the result. ``ratchet eval``
exists precisely so that "did this actually help?" is a measurement rather than
a feeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TrainingUnavailable(RuntimeError):
    """Raised when the optional training stack is not installed."""


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter hyperparameters, with defaults that fit a single consumer GPU."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True)
class TrainingConfig:
    """One training run's settings."""

    base_model: str
    output_dir: str
    dataset_path: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: float = 1.0
    batch_size: int = 1
    max_length: int = 1024
    lora: LoRAConfig = LoRAConfig()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "dataset_path": self.dataset_path,
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        data.update(self.lora.as_dict())
        return data


def check_available() -> None:
    """Raise ``TrainingUnavailable`` with an actionable message if deps are missing."""
    missing = []
    for module in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise TrainingUnavailable(
            "Route A training needs the optional dependencies. Install them with:\n"
            "    pip install 'ratchet-ai[train]'\n"
            f"missing: {', '.join(missing)}"
        )


def train_dpo(config: TrainingConfig) -> str:
    """Run one DPO + LoRA pass over a dataset built by ``train.dataset``.

    Returns the output directory. The base model is never modified: only the
    adapter is written, so a bad run costs nothing but the time it took and a
    directory the owner can delete.
    """
    check_available()

    import json

    from datasets import Dataset  # type: ignore[import-not-found]
    from peft import LoraConfig  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found]

    with open(config.dataset_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("dataset is empty; nothing to train on")

    dataset = Dataset.from_list(
        [
            {
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        # Without a pad token, batching silently produces garbage rather than
        # failing, so it is set explicitly.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.base_model)

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=config.output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        max_length=config.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    return config.output_dir


def plan(config: TrainingConfig) -> dict[str, Any]:
    """Describe what a run *would* do, without touching a GPU.

    Exposed so ``ratchet train --dry-run`` can show the owner the exact settings
    before they spend an hour of GPU time on them.
    """
    return {"action": "dpo_lora", **config.as_dict()}
