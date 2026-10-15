#!/usr/bin/env python3
"""Verify global bindings for the campaign operations moved into domain owners."""
from __future__ import annotations

import dis
from pathlib import Path
from types import CodeType, FunctionType


OWNED_OPERATIONS = ('resolve_campaign_scope',)


def unbound_global_reads(operation: FunctionType) -> tuple[tuple[str, str], ...]:
    """Missing real global/builtin bindings, including deferred nested code."""
    namespace = operation.__globals__
    builtin_namespace = operation.__builtins__
    pending = [operation.__code__]
    missing = set()
    while pending:
        code = pending.pop()
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
        for instruction in dis.get_instructions(code):
            if instruction.opname == 'LOAD_GLOBAL':
                name = instruction.argval
                if name not in namespace and name not in builtin_namespace:
                    missing.add((code.co_qualname, name))
    return tuple(sorted(missing))


def main() -> int:
    from prismaquant import tessera_campaign as campaign

    root = Path(__file__).resolve().parents[1]
    if Path(campaign.__file__).resolve() != root/'prismaquant/tessera_campaign.py':
        raise RuntimeError('campaign binding verification requires this checkout owner')
    for name in OWNED_OPERATIONS:
        missing = unbound_global_reads(getattr(campaign, name))
        print(f'GLOBAL-BINDINGS {name}: unresolved {missing}')
        if missing:
            raise RuntimeError(f'{name} has unbound global reads: {missing}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
