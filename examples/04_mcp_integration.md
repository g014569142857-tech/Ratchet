# Example 4 — mount Ratchet in an agent over MCP

MCP (Model Context Protocol) is the standard way agents discover and call external
tools. Implement once, and any supporting client — Cursor, Claude Desktop, a Qwen
agent — can draw on your accumulated experience without embedding the library.

## Install and run

```bash
pip install 'ratchet-ai[mcp]'
ratchet-mcp --db ~/ratchet.db
```

## Client configuration

Add the server to your client's MCP config, for example:

```json
{
  "mcpServers": {
    "ratchet": {
      "command": "ratchet-mcp",
      "args": ["--db", "C:\\Users\\you\\ratchet.db"]
    }
  }
}
```

## Tools exposed

| Tool | Purpose |
|---|---|
| `ratchet_recall` | search the experience library without running a cycle |
| `ratchet_query` | run one full cycle: retrieve → select → answer → score |
| `ratchet_feedback` | report a verdict (`up` / `down`) on the last answer |
| `ratchet_rule_add` | add a human-authored rule (capped, validated) |
| `ratchet_rule_edit` | edit a human-authored rule only |
| `ratchet_forget` | delete any entry — the universal escape hatch |

## What this buys the agent

Without Ratchet, an agent starts every session from zero. With it:

- a question it has answered badly before retrieves the lesson it wrote at the time
- the strategy it selects is the one that has actually scored well on your work
- your corrections persist across sessions and across clients

## A note on `ratchet_feedback`

Human feedback overrides the blended reward rather than joining it, so a
thumbs-down genuinely lowers the probability of the strategy that produced the
answer — it is not diluted by an objective check that happened to pass.

## Security note

The MCP server holds the same permission model as the library: machine-written
entries are delete-only, and `ratchet_rule_add` is capped and validated. The
client can ask Ratchet to forget things; it cannot use it to inject arbitrary
prose into the store.
