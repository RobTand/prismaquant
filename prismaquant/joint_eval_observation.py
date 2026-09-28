"""What a diagnostic joint-evaluation panel observed, per unit.

Neutral home for the two facts the cost stage and the currency gate read off a
pilot panel's rows (decoupling step 6, PQ #1550): the panel status every
diagnostic row carries, and whether an expert was observed at all. The panel
itself (``tessera_joint_eval_panel``) selects rows and re-exports both.
"""
from __future__ import annotations

#: The status a diagnostic pilot panel stamps on its provenance and rows.
STATUS = 'diagnostic_pilot'


def observation_status(count):
    """An invocation with zero token rows does not observe that expert."""
    if (not isinstance(count, dict) or not {'tokens', 'calls'} <= set(count)
            or any(type(count[key]) is not int or count[key] < 0 for key in ('tokens', 'calls'))
            or (count['tokens'] > 0 and count['calls'] == 0)):
        raise ValueError('joint evaluation observation counts are invalid')
    return 'observed' if count['tokens'] > 0 else 'unknown_unobserved'
