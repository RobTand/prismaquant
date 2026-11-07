"""Tessera fleet workflows using native Flux IDs and external .flux-state.

Payloads run from committed detached shared sources. The current baseline
qualifies CPU fixtures only; numerical GPU/container work fails closed.
Historical PB workspaces remain untouched.

Run drivers as modules, for example
``python3 -m tools.tessera_fleet.dispatch_model --help``.
"""
