"""Bounded GLM routing identity tap recovered from the retained diagnostic.

Source: evidence/pq-1962-claude-7a182085, glm_aside_replay.py RoutingTap.
"""
import torch
import torch.nn.functional as F


class Refused(RuntimeError):
    pass


class RoutingTap:
    """Forward pre-hook on the routed experts: keeps this forward's routing and resolves the
    (token, slot) of each row a dispatch hands an expert.  grouped_mm sorts the flattened
    (token, slot) pairs by expert (``torch.sort`` of ``top_k_index.reshape(-1)``); the eager
    loop gathers ``torch.where(one_hot(top_k_index).permute(2, 1, 0)[e])``."""

    def __init__(self, experts):
        self.num_experts = int(experts.num_experts)
        self.handle = experts.register_forward_pre_hook(self._pre)
        self.hidden = self.index = self.weights = self.perm = None
        self.impl, self.starts, self.counts = None, None, None

    def _pre(self, module, args):
        hidden, index, weights = args[0], args[1], args[2]
        impl = getattr(module.config, "_experts_implementation", None) or "eager"
        if impl not in ("grouped_mm", "eager"):
            raise Refused(f"routing tap supports grouped_mm and eager dispatch, not {impl!r}")
        self.impl = impl
        self.hidden, self.index, self.weights = hidden.detach(), index.detach(), weights.detach()
        ids = self.index.reshape(-1)
        counts = torch.bincount(ids, minlength=self.num_experts)
        self.counts = counts.tolist()
        self.starts = (torch.cumsum(counts, 0) - counts).tolist()
        self.perm = torch.sort(ids)[1] if impl == "grouped_mm" else None

    def rows(self, expert, row_slice, n_rows):
        """(token, slot) of each row ``expert`` was handed, in the dispatch's row order."""
        if n_rows != self.counts[expert]:
            raise Refused(f"expert {expert}: {n_rows} rows observed, {self.counts[expert]} routed")
        if self.impl == "grouped_mm":
            if row_slice is None or int(row_slice.start) != self.starts[expert] \
                    or int(row_slice.stop) - int(row_slice.start) != n_rows:
                raise Refused(f"expert {expert}: grouped rows {row_slice} differ from the "
                              f"routing offsets ({self.starts[expert]}, {n_rows})")
            token = self.perm[int(row_slice.start):int(row_slice.stop)] // self.index.shape[-1]
        else:
            if row_slice is not None:
                raise Refused("eager dispatch observed a row slice")
            mask = F.one_hot(self.index, num_classes=self.num_experts).permute(2, 1, 0)
            token = torch.where(mask[expert])[1]
        match = self.index[token] == expert
        if not bool((match.sum(-1) == 1).all()):
            raise Refused(f"expert {expert}: a row's token does not route to it exactly once")
        return token, match.int().argmax(-1)

    def remove(self):
        self.handle.remove()

