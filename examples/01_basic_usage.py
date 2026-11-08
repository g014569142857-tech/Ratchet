"""Example 1 — the shortest path through the loop.

Runs offline: the default provider is an echo, so no API key, no network, no cost.

    python examples/01_basic_usage.py
"""

import ratchet

DB = "example.db"


def main() -> None:
    engine = ratchet.Engine(db_path=DB)

    # Human-authored rules are the only prose you write. They are capped at 200
    # characters and validated to be one atomic sentence.
    engine.add_rule("Answer in Chinese, three sentences at most.")

    result = engine.run(task="explain a concept", user_input="what is LoRA?")

    print("output      :", result.output[:80])
    print("strategy    :", result.variant_id)
    print("reward      :", result.reward.value)
    print("recalled    :", len(result.retrieved_ids), "experience entries")

    # See everything the system currently believes, machine-written included.
    print("\nexperience library:")
    for view in engine.panel.view():
        editable = "editable" if view.editable else "delete-only"
        print(f"  [{view.provenance}/{view.kind} {editable}] {view.content}")

    # Anything wrong can be retracted. Deletion is the universal escape hatch.
    machine_written = [v for v in engine.panel.view() if not v.editable]
    if machine_written:
        engine.panel.delete(machine_written[0].id)
        print(f"\ndeleted machine-written entry {machine_written[0].id[:8]}")

    print("\nstats:", engine.stats())
    engine.close()


if __name__ == "__main__":
    main()
