"""Optional training backend (Route A): experience data into weight updates.

Impossible for closed API models by construction - you cannot update weights you
cannot see - so this package is an optional extra behind ``pip install
'ratchet-ai[train]'`` and is never imported by the core loop. See
``docs/architecture.md`` for why Route B is the default.
"""
