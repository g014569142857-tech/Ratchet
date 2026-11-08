"""Adapters for plugging Ratchet into external hosts.

``mcp_server`` exposes Ratchet as an MCP (Model Context Protocol) server so any
supporting agent - Cursor, Claude Desktop, Qwen and similar - can recall
experience, run a cycle and report feedback without embedding this library.

The tool logic lives in plain functions so it is fully testable without the
optional ``mcp`` dependency installed; only ``serve()`` imports it, lazily.
"""
