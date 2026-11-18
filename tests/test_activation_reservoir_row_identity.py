"""The activation reservoir must sample the same rows however much a run stores.

One ``torch.Generator`` feeds every hooked Linear's priority reservoir, so the
slice of the random stream a given Linear receives is a function of how many
rows every *earlier* hook consumed. If priorities are drawn only for the
Linears a run happens to STORE, then a run that stores a subset keeps
different rows than a full run -- and the rendered bytes follow the rows.

``fill_production_weight_cache`` hooks the full ``eligible_qnames``
enumeration but stores only ``qnames_needing_activation``, which is
resume-dependent. So this invariant is what makes a ``--resume`` build
reproduce a fresh build, and it is a precondition for any per-unit split of a
render. This docstring described the intended behaviour before the code had it:
the hook set was the render-narrowed ``qname_set`` until #130/#135 restored the
full enumeration -- see ``tests/test_striped_render_row_identity.py``.

Salvaged from PR #104 (``tests/test_unit_sharding.py``
``test_stored_subset_keeps_the_full_runs_rows``); the sharding machinery that
PR carried is superseded by ``production_cache_stripes`` +
``union_production_cache``, but this invariant is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismaquant.production_weight_cache import (  # noqa: E402
    _LinearActivationCollector,
    update_priority_reservoir,
)


class _ResAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _ResModel(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([_ResAttn(hidden) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _collect(model, names, store):
    collector = _LinearActivationCollector(
        model, set(names), max_rows=16, store_qnames=store
    )
    collector.install()
    try:
        with torch.no_grad():
            for _ in range(3):
                model(torch.randn(1, 40, 64))
    finally:
        collector.remove()
    return collector.collected()


def test_stored_subset_keeps_the_full_runs_rows():
    """A run that STORES a subset must sample the same rows as the full run."""
    model = _ResModel().eval()
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    assert len(names) == 8
    subset = set(names[4:])

    torch.manual_seed(0)
    full = _collect(model, names, set(names))
    torch.manual_seed(0)
    partial = _collect(model, names, subset)

    assert set(partial) == subset
    for name in sorted(subset):
        assert torch.equal(full[name], partial[name]), (
            f"{name} sampled different rows when only a subset was stored"
        )


def test_empty_activation_rows_keep_the_existing_reservoir_and_random_stream():
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    priorities = torch.tensor([0.2, 0.8])
    incoming = torch.empty((0, 2))
    generator = torch.Generator().manual_seed(17)
    before = generator.get_state()

    result_rows, result_priorities = update_priority_reservoir(
        rows, priorities, incoming, max_rows=2, generator=generator
    )
    assert result_rows is rows
    assert result_priorities is priorities
    assert rows.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert torch.equal(priorities, torch.tensor([0.2, 0.8]))
    assert incoming.shape == (0, 2)
    assert torch.equal(generator.get_state(), before)
