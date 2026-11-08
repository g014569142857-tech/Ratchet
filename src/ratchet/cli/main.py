"""Command-line interface for Ratchet.

Subcommands map directly onto the permission model:

  init      create the local database
  add       write a human rule  (capped, validated, highest priority)
  audit     SEE everything the machine learned  <- the review entry point
  rm        DELETE anything (the universal escape hatch)
  restore   undo a deletion
  explain   show why one entry exists and what it has done
  purge     physically erase one entry (irreversible)
  correct   submit a structured, bounded correction
  profile   view / set / forget / reset inferred preferences
  status    counts, cost, and the bandit's current beliefs
  export    dump experiences, SFT / DPO / GRPO datasets
  train     Route A: LoRA + DPO on exported data (optional deps + GPU)
  eval      growth curve: is this actually working?
  chat      run one interaction offline or against a real endpoint
  panel     open the local governance console

``audit`` + ``rm`` + ``correct`` exist because the owner must always be able to
see what the system concluded on its own and retract it. Machine entries are
delete-only by design: deletion can only remove capability, never inject it.

Exit codes: 0 success, 1 handled error (message on stderr), 2 usage error
(argparse default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ratchet import __version__
from ratchet.core.provider import EchoProvider, OpenAICompatProvider, ProviderError
from ratchet.engine import Engine
from ratchet.eval.growth import growth_curve
from ratchet.governance.correction import MAX_NOTE_CHARS, CorrectionKind, CorrectionScope
from ratchet.governance.panel import CORRECTION_KINDS
from ratchet.memory.store import Experience, ExperienceStore
from ratchet.train import dataset as train_dataset

_FEEDBACK_MAP = {"up": 1, "down": -1, "none": None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ratchet",
        description="Local-first self-improvement engine for any LLM.",
    )
    parser.add_argument("--version", action="version", version=f"ratchet {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_db(target: argparse.ArgumentParser) -> None:
        target.add_argument("--db", default="ratchet.db", help="path to the SQLite database")

    init_p = subparsers.add_parser("init", help="create the local database")
    add_db(init_p)

    add_p = subparsers.add_parser("add", help="add a human-authored rule")
    add_db(add_p)
    add_p.add_argument("content", help="one atomic rule, at most 200 characters")
    add_p.add_argument("--task-type", default=None)

    audit_p = subparsers.add_parser("audit", help="review what the system learned")
    add_db(audit_p)
    audit_p.add_argument("--all", action="store_true", help="include deleted/deactivated entries")
    audit_p.add_argument("--provenance", choices=("auto", "user_rule", "derived"), default=None)
    audit_p.add_argument("--kind", default=None, help="rule / episode / distilled / correction")

    rm_p = subparsers.add_parser("rm", help="delete an experience by id or unique prefix")
    add_db(rm_p)
    rm_p.add_argument("prefix", help="full id, or a prefix that matches exactly one entry")

    restore_p = subparsers.add_parser("restore", help="re-activate a deleted entry")
    add_db(restore_p)
    restore_p.add_argument("prefix")

    purge_p = subparsers.add_parser("purge", help="physically erase an entry (irreversible)")
    add_db(purge_p)
    purge_p.add_argument("prefix")
    purge_p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    explain_p = subparsers.add_parser("explain", help="show why one entry exists")
    add_db(explain_p)
    explain_p.add_argument("prefix")

    correct_p = subparsers.add_parser("correct", help="submit a structured correction")
    add_db(correct_p)
    correct_p.add_argument("--kind", required=True, choices=list(CORRECTION_KINDS))
    correct_p.add_argument(
        "--scope", default="this_task_only", choices=("this_task_only", "global")
    )
    correct_p.add_argument("--note", default=None, help=f"optional, at most {MAX_NOTE_CHARS} chars")
    correct_p.add_argument("--task-type", default=None)

    profile_p = subparsers.add_parser("profile", help="inferred preferences")
    add_db(profile_p)
    profile_p.add_argument("--reset", action="store_true", help="drop every belief")
    profile_p.add_argument("--set", dest="set_kv", default=None, metavar="KEY=VALUE")
    profile_p.add_argument("--forget", default=None, metavar="KEY")

    status_p = subparsers.add_parser("status", help="show counts, cost and policy beliefs")
    add_db(status_p)

    export_p = subparsers.add_parser("export", help="export data for inspection or training")
    add_db(export_p)
    export_p.add_argument(
        "--format", choices=("experiences", "sft", "dpo", "grpo"), default="experiences",
        help="experiences = full dump; sft/dpo/grpo = Route A training datasets",
    )
    export_p.add_argument("--out", required=True, help="output JSONL path")
    export_p.add_argument("--min-reward", type=float, default=train_dataset.DEFAULT_MIN_REWARD)

    train_p = subparsers.add_parser("train", help="Route A: LoRA + DPO (optional deps)")
    add_db(train_p)
    train_p.add_argument("--dataset", required=True, help="JSONL produced by 'export --format dpo'")
    train_p.add_argument("--base-model", required=True)
    train_p.add_argument("--out-dir", default="ratchet-adapter")
    train_p.add_argument("--epochs", type=float, default=1.0)
    train_p.add_argument("--dry-run", action="store_true", help="print the plan, train nothing")

    eval_p = subparsers.add_parser("eval", help="growth curve over recorded interactions")
    add_db(eval_p)
    eval_p.add_argument("--buckets", type=int, default=4)
    eval_p.add_argument("--json", action="store_true")

    panel_p = subparsers.add_parser("panel", help="open the local governance console")
    add_db(panel_p)
    panel_p.add_argument("--host", default="127.0.0.1")
    panel_p.add_argument("--port", type=int, default=8787)
    panel_p.add_argument("--no-browser", action="store_true")
    panel_p.add_argument(
        "--allow-edit", action="store_true",
        help="expose the capped rule box (off by default: humans delete, machines add)",
    )

    chat_p = subparsers.add_parser("chat", help="run one interaction")
    add_db(chat_p)
    chat_p.add_argument("user_input")
    chat_p.add_argument("--task", default=None)
    chat_p.add_argument("--provider", choices=("echo", "openai"), default="echo")
    chat_p.add_argument("--base-url", default="https://api.openai.com/v1")
    chat_p.add_argument("--model", default="gpt-4o-mini")
    chat_p.add_argument(
        "--feedback", choices=tuple(_FEEDBACK_MAP), default="none",
        help="explicit judgement of the output; overrides blended reward",
    )
    chat_p.add_argument(
        "--api-key-env", default="OPENAI_API_KEY",
        help="environment variable holding the API key",
    )
    return parser


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _make_provider(args: argparse.Namespace) -> tuple[object, str | None]:
    """Build the requested provider, returning (provider, error_message)."""
    if args.provider == "echo":
        return EchoProvider(), None
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        return None, (
            f"no API key found in {args.api_key_env}. Export it, or use "
            f"--provider echo to run offline."
        )
    return OpenAICompatProvider(
        base_url=args.base_url, model=args.model, api_key=api_key
    ), None


# --------------------------------------------------------------------- render

def _render_audit(entries: Sequence[Experience]) -> None:
    for entry in entries:
        flag = "" if entry.active else " [inactive]"
        print(
            f"{entry.id}  provenance={entry.provenance.value}  "
            f"kind={entry.kind}  trust={entry.trust_level}  "
            f"score={entry.score:+.1f}  uses={entry.uses}{flag}"
        )
        print(f"    {entry.content}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "init":
        with ExperienceStore(args.db):
            pass
        print(f"initialized ratchet database at {args.db}")
        return 0

    if args.command == "add":
        with Engine(db_path=args.db) as engine:
            try:
                entry_id = engine.add_rule(args.content, task_type=args.task_type)
            except ValueError as exc:
                return _fail(str(exc))
        print(f"added user_rule {entry_id}")
        return 0

    if args.command == "audit":
        with Engine(db_path=args.db) as engine:
            entries = engine.panel.view(
                include_inactive=args.all,
                provenance=args.provenance,
                kind=args.kind,
            )
        if not entries:
            print("no active experiences yet")
            return 0
        with ExperienceStore(args.db) as store:
            lookup = {e.id: e for e in store.list_all()}
        _render_audit([lookup[v.id] for v in entries if v.id in lookup])
        return 0

    if args.command == "rm":
        with Engine(db_path=args.db) as engine:
            try:
                resolved = engine.panel.delete(args.prefix)
            except ValueError as exc:
                return _fail(str(exc))
            except KeyError as exc:
                return _fail(str(exc))
        print(f"deleted {resolved}")
        return 0

    if args.command == "restore":
        with Engine(db_path=args.db) as engine:
            try:
                resolved = engine.panel.restore(args.prefix)
            except (ValueError, KeyError) as exc:
                return _fail(str(exc))
        print(f"restored {resolved}")
        return 0

    if args.command == "purge":
        with Engine(db_path=args.db) as engine:
            try:
                target = engine.store.resolve_id(args.prefix)
            except ValueError as exc:
                return _fail(str(exc))
            if target is None:
                return _fail(f"experience not found: {args.prefix!r}")
            entry = engine.store.get(target, include_inactive=True)
            if not args.yes:
                print(f"about to permanently erase {target}")
                if entry:
                    print(f"    {entry.content}")
                answer = input("type 'purge' to confirm: ").strip().lower()
                if answer != "purge":
                    print("aborted")
                    return 1
            engine.panel.purge(target)
        print(f"purged {target}")
        return 0

    if args.command == "explain":
        with Engine(db_path=args.db) as engine:
            try:
                info = engine.panel.explain(args.prefix)
            except (ValueError, KeyError) as exc:
                return _fail(str(exc))
        exp = cast(dict[str, Any], info["experience"])
        policy = cast(dict[str, Any], info["policy"])
        mean = cast("float | None", info["mean_reward_when_recalled"])
        print(f"{exp['id']}  {exp['provenance']}/{exp['kind']}")
        print(f"    {exp['content']}")
        print(f"  recalled {info['times_recalled']} time(s)")
        print(f"  mean reward when recalled: {mean if mean is None else round(mean, 3)}")
        print(
            f"  human_editable={policy['human_editable']}  "
            f"human_deletable={policy['human_deletable']}  "
            f"max_chars={policy['max_chars']}"
        )
        return 0

    if args.command == "correct":
        from ratchet.governance.correction import Correction

        with Engine(db_path=args.db) as engine:
            try:
                correction = Correction.build(
                    kind=CorrectionKind(args.kind),
                    scope=CorrectionScope(args.scope),
                    note=args.note,
                    task_type=args.task_type,
                )
            except ValueError as exc:
                return _fail(str(exc))
            experience_id, touched = engine.panel.correct(correction)
        print(f"recorded correction {experience_id}")
        if touched:
            print(f"  profile updated: {', '.join(touched)}")
        return 0

    if args.command == "profile":
        with Engine(db_path=args.db) as engine:
            if args.reset:
                removed = engine.panel.reset_profile()
                print(f"dropped {removed} belief(s)")
                return 0
            if args.forget:
                print("forgotten" if engine.panel.forget_profile(args.forget) else "no such key")
                return 0
            if args.set_kv:
                if "=" not in args.set_kv:
                    return _fail("expected KEY=VALUE")
                key, _, value = args.set_kv.partition("=")
                engine.panel.override_profile(key.strip(), value.strip())
                print(f"set {key.strip()} = {value.strip()} (owner override)")
                return 0
            rows = engine.profile.entries()
        if not rows:
            print("no beliefs inferred yet")
            return 0
        for row in rows:
            origin = "owner" if row.source == "human_override" else "inferred"
            print(
                f"  {row.key:<18} = {row.value:<26} "
                f"conf={row.confidence:.2f}  support={row.support}  "
                f"contradict={row.contradict}  [{origin}]"
            )
        return 0

    if args.command == "status":
        with Engine(db_path=args.db) as engine:
            stats = engine.stats()
            events = engine.store.list_events()
        measured = [e.reward for e in events if e.reward is not None]
        mean_reward = sum(measured) / len(measured) if measured else 0.0
        # Token totals are summed from the event table, not from the Session's
        # counter: every CLI invocation is a new process, so a per-session
        # counter would always read zero and tell the owner nothing about cost.
        tokens_in = sum(e.tokens_in for e in events)
        tokens_out = sum(e.tokens_out for e in events)
        print(f"database      : {args.db}")
        print(f"experiences   : {stats['active_experiences']} active")
        print(f"events        : {len(events)} recorded")
        print(f"tokens        : {tokens_in} in / {tokens_out} out")
        print(f"mean reward   : {mean_reward:.3f}")
        print(f"interactions  : {stats['interactions']}")
        print(f"profile       : {stats['profile_beliefs']} active belief(s)")
        print(f"distilled     : {stats['distilled_blocks']} persona block(s)")
        print("policy beliefs (variant -> posterior mean, pulls):")
        with Engine(db_path=args.db) as engine:
            for name in sorted(engine.policy.list_arms()):
                arm = engine.policy.arms[name]
                print(f"  {name:<16} mean={arm.mean():.3f}  pulls={arm.pulls}")
        return 0

    if args.command == "export":
        records: list[dict[str, Any]]
        dataset_stats: train_dataset.DatasetStats | None = None
        with ExperienceStore(args.db) as store:
            if args.format == "experiences":
                records = [
                    {
                        "id": e.id,
                        "content": e.content,
                        "provenance": e.provenance.value,
                        "kind": e.kind,
                        "trust_level": e.trust_level,
                        "score": e.score,
                        "uses": e.uses,
                    }
                    for e in store.list_active()
                ]
            else:
                events = store.list_events()
                records, dataset_stats = train_dataset.build(
                    events, args.format, min_reward=args.min_reward
                )
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = train_dataset.write_jsonl(records, str(path))
        print(f"exported {written} {args.format} records to {path}")
        if dataset_stats is not None:
            detail = dataset_stats.as_dict()
            print(
                "  considered {considered}, skipped: {skipped_unmeasured} unmeasured, "
                "{skipped_singleton} single-attempt, {skipped_small_gap} small-gap".format(
                    **detail
                )
            )
        return 0

    if args.command == "train":
        from ratchet.train.dpo import TrainingConfig, plan, train_dpo

        config = TrainingConfig(
            base_model=args.base_model,
            output_dir=args.out_dir,
            dataset_path=args.dataset,
            epochs=args.epochs,
        )
        if args.dry_run:
            print(json.dumps(plan(config), indent=2, ensure_ascii=False))
            return 0
        try:
            out = train_dpo(config)
        except RuntimeError as exc:
            return _fail(str(exc))
        print(f"adapter written to {out}")
        print("Evaluate it with 'ratchet eval' before adopting it.")
        return 0

    if args.command == "eval":
        with ExperienceStore(args.db) as store:
            curve = growth_curve(store.list_events(), buckets=args.buckets)
        if args.json:
            print(json.dumps(curve.as_dict(), indent=2, ensure_ascii=False))
            return 0
        if not curve.buckets:
            print("not enough measured interactions to draw a curve")
            print(f"  measured={curve.measured_total} unmeasured={curve.unmeasured_total}")
            return 0
        for bucket in curve.buckets:
            bar = "#" * max(1, int(round(bucket.mean_reward * 24)))
            print(f"  period {bucket.index + 1}: {bucket.mean_reward:.3f}  {bar}")
        print(f"first {curve.first_mean:.3f} -> last {curve.last_mean:.3f}  ({curve.delta:+.3f})")
        print(f"verdict: {curve.verdict}")
        return 0

    if args.command == "panel":
        from ratchet.adapters.web import serve

        serve(
            db_path=args.db,
            host=args.host,
            port=args.port,
            allow_edit=args.allow_edit,
            open_browser=not args.no_browser,
        )
        return 0

    if args.command == "chat":
        provider, error = _make_provider(args)
        if error:
            return _fail(error)
        with Engine(db_path=args.db, provider=provider) as engine:  # type: ignore[arg-type]
            try:
                result = engine.run(
                    task=args.task or args.user_input,
                    user_input=args.user_input,
                    user_feedback=_FEEDBACK_MAP[args.feedback],
                )
            except ProviderError as exc:
                return _fail(str(exc))
        print(result.output)
        if result.reward.value is None:
            reward_text = "unmeasured"
        else:
            # A measured reward always names a source, but the type cannot say so.
            source = result.reward.source.value if result.reward.source else "unknown"
            reward_text = f"{result.reward.value:.3f} ({source})"
        print(f"\n--- reward: {reward_text}  variant: {result.variant_id}", file=sys.stderr)
        for reflection in result.reflections:
            print(f"--- learned: {reflection}", file=sys.stderr)
        return 0

    return _fail(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
