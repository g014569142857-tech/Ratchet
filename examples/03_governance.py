"""Example 3 — the owner's side of the system.

Everything here is about seeing what the machine concluded and being able to
retract it, plus the one bounded way a human writes into the store.

    python examples/03_governance.py
"""

import ratchet
from ratchet.governance.correction import Correction, CorrectionKind, CorrectionScope

DB = "example_governance.db"


def main() -> None:
    engine = ratchet.Engine(db_path=DB)

    # Simulate what reflection would have written, so there is something to
    # govern. In normal use these arrive on their own after low-scoring runs.
    machine_rule = engine.store.add(
        content="When explaining LoRA, mention low-rank adaptation explicitly.",
        provenance=ratchet.Provenance.AUTO,
        kind="rule",
    )
    engine.add_rule("Answer in Chinese.")

    # ---- 1. See everything -------------------------------------------------
    print("library:")
    for view in engine.panel.view():
        mode = "editable" if view.editable else "delete-only"
        print(f"  [{view.provenance:<9} {mode:<12}] {view.content}")

    # ---- 2. Ask why an entry exists ---------------------------------------
    print("\nexplain:")
    info = engine.panel.explain(machine_rule[:8])
    print(f"  recalled {info['times_recalled']} time(s)")
    print(f"  human_editable={info['policy']['human_editable']}")

    # ---- 3. Machine entries refuse to be edited ----------------------------
    print("\nediting a machine entry:")
    try:
        engine.store.edit(machine_rule, "rewritten by a human")
    except PermissionError as exc:
        print(f"  refused: {exc}")

    # ---- 4. But they can always be deleted ---------------------------------
    print("\ndeleting it:")
    engine.panel.delete(machine_rule[:8])
    print("  deleted:", engine.store.get(machine_rule) is None)
    print("  restored:", engine.panel.restore(machine_rule[:8]))

    # ---- 5. Structured correction ------------------------------------------
    print("\nstructured correction:")
    correction = Correction.build(
        kind=CorrectionKind.TOO_LONG,
        scope=CorrectionScope.GLOBAL,
        note="keep it to three points",
    )
    experience_id, touched = engine.panel.correct(correction)
    print(f"  stored as {experience_id[:8]}  profile touched: {touched}")

    # The note arrives escaped and framed — never as an instruction.
    print("  rendered:", correction.render())

    # ---- 6. The inferred profile, with its evidence ------------------------
    print("\nprofile:")
    for entry in engine.profile.entries():
        origin = "owner" if entry.source == "human_override" else "inferred"
        print(
            f"  {entry.key:<12} = {entry.value:<24} "
            f"conf={entry.confidence:.2f}  evidence={len(entry.evidence)}  [{origin}]"
        )

    # ---- 7. The owner outranks inference -----------------------------------
    engine.panel.override_profile("verbosity", "detailed")
    print("\nafter override:")
    for entry in engine.profile.entries():
        if entry.key == "verbosity":
            print(f"  verbosity = {entry.value} (confidence {entry.confidence:.2f})")

    # A later contradicting correction must not move it.
    engine.panel.correct(
        Correction.build(kind=CorrectionKind.TOO_LONG, scope=CorrectionScope.GLOBAL)
    )
    print(
        "  still",
        engine.store.get_profile_entry("verbosity").value,
        "- inference may not overturn the owner",
    )

    # ---- 8. And everything can be forgotten --------------------------------
    print("\nreset:", engine.panel.reset_profile(), "belief(s) dropped")
    engine.close()


if __name__ == "__main__":
    main()
