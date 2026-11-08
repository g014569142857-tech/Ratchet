"""Ratchet: local-first self-improvement engine for any LLM.

    import ratchet
    engine = ratchet.Engine(db_path="ratchet.db")
    engine.add_rule("Always answer in Chinese.")
    result = engine.run(task="write a haiku", user_input="about rain")
    print(result.output)
    engine.panel.view()           # SEE what it concluded on its own
    engine.panel.delete("af3c")   # and retract it

Core has zero third-party runtime dependencies: no GPU, no API key and no
database server required. See README.md for the architecture overview and
Limitations for an honest statement of what this does and does not do.

The permission model in one line: **the machine may add, the owner may subtract.**
Reflection, distillation and evolution write freely; the owner sees everything,
can delete anything, and reaches the store only through a bounded, structured
correction. Deletion can only remove capability, so it is safe to grant without
review; insertion changes what the model is told on every future task, so it is
not.
"""

from ratchet.core.policy import Arm, ThompsonPolicy
from ratchet.core.prompt import (
    VARIANT_STRATEGIES,
    PromptAssembler,
    UntrustedContentError,
    register_variant,
    variant_strategy,
)
from ratchet.core.provider import EchoProvider, OpenAICompatProvider, Provider, ProviderError
from ratchet.core.session import Session, SessionConfig, SessionResult
from ratchet.engine import Engine, openai_compatible
from ratchet.eval.growth import (
    Bucket,
    EvalRun,
    EvalSuite,
    EvalTask,
    GrowthCurve,
    compare,
    growth_curve,
    run_suite,
)
from ratchet.governance.correction import (
    MAX_NOTE_CHARS,
    PROFILE_IMPACT,
    Correction,
    CorrectionKind,
    CorrectionRecord,
    CorrectionScope,
    escape_note,
)
from ratchet.governance.panel import ExperienceView, GovernancePanel, PanelSnapshot
from ratchet.improve.distiller import Distillation, Distiller, pinned_principles
from ratchet.improve.evolver import PromptEvolver, Variant
from ratchet.improve.reflector import Reflection, Reflector
from ratchet.improve.verifier import (
    Contribution,
    LengthPenalty,
    LLMJudgeVerifier,
    ObjectiveVerifier,
    Reward,
    RewardChain,
    RewardSource,
    UserFeedbackVerifier,
)
from ratchet.memory.profile import Profile, ProfileSnapshot, detect_language
from ratchet.memory.provenance import PROVENANCE_RULES, Provenance, ProvenanceRule, rule_for
from ratchet.memory.redact import Redactor, redact
from ratchet.memory.retriever import (
    BM25Index,
    Embedder,
    HashedEmbedder,
    HybridRetriever,
    RetrievalHit,
)
from ratchet.memory.store import Event, Experience, ExperienceStore, ProfileEntry
from ratchet.train.dataset import build_dpo_pairs, build_grpo_groups, build_sft_records

__version__ = "0.2.0"

# Kept in isort order so RUF022 stays stable across ruff releases.
__all__ = [
    "MAX_NOTE_CHARS",
    "PROFILE_IMPACT",
    "PROVENANCE_RULES",
    "Arm",
    "BM25Index",
    "Bucket",
    "Contribution",
    "Correction",
    "CorrectionKind",
    "CorrectionRecord",
    "CorrectionScope",
    "Distillation",
    "Distiller",
    "EchoProvider",
    "Embedder",
    "Engine",
    "EvalRun",
    "EvalSuite",
    "EvalTask",
    "Event",
    "Experience",
    "ExperienceStore",
    "ExperienceView",
    "GovernancePanel",
    "GrowthCurve",
    "HashedEmbedder",
    "HybridRetriever",
    "LLMJudgeVerifier",
    "LengthPenalty",
    "ObjectiveVerifier",
    "OpenAICompatProvider",
    "PanelSnapshot",
    "Profile",
    "ProfileEntry",
    "ProfileSnapshot",
    "PromptAssembler",
    "PromptEvolver",
    "Provenance",
    "ProvenanceRule",
    "Provider",
    "ProviderError",
    "Redactor",
    "Reflection",
    "Reflector",
    "RetrievalHit",
    "Reward",
    "RewardChain",
    "RewardSource",
    "Session",
    "SessionConfig",
    "SessionResult",
    "ThompsonPolicy",
    "UntrustedContentError",
    "UserFeedbackVerifier",
    "VARIANT_STRATEGIES",
    "Variant",
    "__version__",
    "build_dpo_pairs",
    "build_grpo_groups",
    "build_sft_records",
    "compare",
    "detect_language",
    "escape_note",
    "growth_curve",
    "openai_compatible",
    "pinned_principles",
    "redact",
    "register_variant",
    "run_suite",
    "rule_for",
    "variant_strategy",
]
