"""Same-pass disjoint FIT/HELDOUT calibration row moments (Stage-1 research).

``DisjointRowMoments`` is the external ``row_consumer`` for the default-off
seam in :func:`prismaquant.tessera_campaign._collect_activations`.  The seam
calls :meth:`consume` ONCE per canonical shared input group -- dense pre-hook
or routed derivation, both feed the single ``accumulate`` path -- with the
uncapped flat rows, before any scoring-prefix cap and independent of
``want_hessian``.  Both split moments accumulate in the SAME forward pass, so
the research caller never computes a full-draw H (``want_hessian=False``,
``max_rows=0``).

Fixed source-calibration coordinates: sample indices ``0..fit_stop-1`` are
FIT, ``fit_stop..total_samples-1`` are HELDOUT; the parent's forward wrapper
feeds exactly one sample per forward.  A batch larger than one sample
(``tokens_per_sample`` rows) is refused, never guessed into a role; routed
per-expert derivations legally produce any non-empty subset of one sample.

Moment residency: the ``XᵀX`` gram is accumulated ON the rows' device into ONE
owned float32 buffer per canonical ``(role, group)``, mutated in place per
batch; nothing but the bounded owned prefix ever leaves the capture device
before :meth:`finish`.  ``resource_check`` runs BEFORE every allocation
(moment growth, prefix growth) and BEFORE every device-to-CPU transfer, and
its refusals propagate.  Aliases of one proven shared input map to the same
state: exactly one H per group, materialized once and handed to every alias
unit at :meth:`finish` under explicit read-only publication semantics (a
record's ``hessian`` may alias its group siblings' record until the parent
writes it out; the parent owns per-unit copies if its writer requires them).

This class does not fit, encode or publish; it only accumulates disjoint
moments.  :meth:`finish` returns, per role and exact unit name, the full
unnormalized float32 ``XᵀX``, the bounded float32 prefix, the routed-row
count, the max|x| and one int64 sample index PER retained prefix row (so
``len(prefix_sample_ids) == inputs.shape[0]`` always).  The parent verifies
per-unit fit+heldout counts against this forward's actual routed counts before
writing each unit+role ".pt".
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

FIT = "fit"
HELDOUT = "heldout"


class DisjointRowMoments:
    """Per-``(role, canonical group)`` disjoint split moments, one pass."""

    def __init__(self, *, fit_stop: int = 384, total_samples: int = 512,
                 tokens_per_sample: int = 512, max_prefix_rows: int = 512,
                 prefix_device: str = "cpu",
                 resource_check: Optional[Callable[[str], None]] = None):
        if fit_stop <= 0 or total_samples <= fit_stop:
            raise ValueError(
                f"split needs 0 < fit_stop < total_samples, got "
                f"{fit_stop}/{total_samples}")
        if tokens_per_sample <= 0 or max_prefix_rows < 0:
            raise ValueError(
                f"split needs tokens_per_sample > 0 and max_prefix_rows >= 0, "
                f"got {tokens_per_sample}/{max_prefix_rows}")
        if prefix_device not in ("cpu", "device"):
            raise ValueError(
                f"prefix_device must be 'cpu' or 'device', got {prefix_device!r}")
        self.fit_stop = int(fit_stop)
        self.total_samples = int(total_samples)
        self.tokens_per_sample = int(tokens_per_sample)
        self.max_prefix_rows = int(max_prefix_rows)
        self.prefix_device = prefix_device
        self.resource_check = resource_check
        self._sample_index: Optional[int] = None
        # ONE state per canonical (role, group); every alias unit in the group
        # maps to it.  Keyed by the exact unit-name tuple from the seam.  The
        # role is part of the key: the two splits never share a buffer.
        self._groups: dict[tuple[str, tuple[str, ...]], dict] = {}
        self._published: Optional[dict] = None
        self.closed = False

    # -- parent forward wrapper -------------------------------------------

    def set_sample(self, index: int) -> None:
        """Set the current global calibration sample index (0..total-1)."""
        if not isinstance(index, int) or isinstance(index, bool) or \
                not 0 <= index < self.total_samples:
            raise ValueError(
                f"sample index must be an int in [0, {self.total_samples}), "
                f"got {index!r}")
        self._sample_index = index

    # -- row_consumer seam -------------------------------------------------

    def consume(self, names, flat) -> None:
        """Accumulate one uncapped within-one-sample batch of flat rows.

        ``names`` is the tuple of exact unit names sharing this input (the
        canonical shared group; a singleton for dense and unshared packed
        units).  ``flat`` is the detached ``[rows, columns]`` tensor from the
        seam -- the full one-sample batch for dense units, any non-empty
        routed subset of one sample for packed experts.  The gram accumulates
        on ``flat``'s device; only the bounded owned prefix copy is retained.
        """
        if self.closed:
            raise RuntimeError("DisjointRowMoments used after close()")
        if not isinstance(names, tuple) or not names or \
                not all(isinstance(name, str) and name for name in names):
            raise TypeError(f"names must be a non-empty tuple of unit names: {names!r}")
        if not isinstance(flat, torch.Tensor) or flat.ndim != 2:
            raise TypeError("flat must be a 2-D [rows, columns] tensor")
        if flat.shape[0] <= 0 or flat.shape[0] > self.tokens_per_sample:
            # A batch spanning more than one sample (e.g. a dense batch over
            # two samples) cannot be assigned to one split role without
            # guessing.  Routed per-expert derivations legally produce any
            # non-empty subset of one sample's rows; anything larger than one
            # sample, or empty, is refused instead.
            raise ValueError(
                f"split consumer requires a non-empty subset of one sample "
                f"(1..{self.tokens_per_sample} rows), got {tuple(flat.shape)}")
        index = self._sample_index
        if index is None:
            raise RuntimeError("set_sample() was not called before this forward")
        role = FIT if index < self.fit_stop else HELDOUT
        group = (role, names)
        if group not in self._groups:
            if self.resource_check is not None:
                self.resource_check(f"split_moments:before_moment_growth:{role}:{names[0]}")
            state = self._groups[group] = {
                "units": names, "hessian": None, "count": 0,
                "max_abs": None,  # owned 0-dim f32 tensor on the moment device
                "inputs": [], "prefix_rows": 0, "prefix_sample_ids": [],
                "moment_device": flat.device,
            }
        else:
            state = self._groups[group]
            if self.resource_check is not None:
                self.resource_check(f"split_moments:before_moment_growth:{role}:{names[0]}")

        # Owned f32 moment on the rows' device; no device-to-host sync here.
        f32 = flat.detach().to(dtype=torch.float32)
        gram = f32.t() @ f32
        f32 = None
        if state["hessian"] is None:
            state["hessian"] = gram
        else:
            state["hessian"] += gram  # in place, one owned buffer per group
        batch_max = flat.detach().abs().amax()
        if state["max_abs"] is None:
            state["max_abs"] = batch_max
        else:
            state["max_abs"] = torch.fmax(state["max_abs"], batch_max)
        state["count"] += int(flat.shape[0])
        del gram

        # Bounded owned prefix; never a view of the source plane.
        if self.max_prefix_rows and state["prefix_rows"] < self.max_prefix_rows:
            if self.resource_check is not None:
                self.resource_check(f"split_moments:before_prefix_growth:{role}:{names[0]}")
            take = min(self.max_prefix_rows - state["prefix_rows"],
                       int(flat.shape[0]))
            if self.prefix_device == "device":
                chunk = flat.detach().to(dtype=torch.float32)[:take].clone()
            else:
                chunk = (flat.detach().to(dtype=torch.float32, device="cpu")
                         [:take].clone())
            state["inputs"].append(chunk)
            # One int64 sample index per retained row.
            state["prefix_sample_ids"].extend([index] * take)
            state["prefix_rows"] += take

    # -- reads -------------------------------------------------------------

    def finish(self) -> dict:
        """Publish-ready disjoint moments: ``{'fit': {...}, 'heldout': {...}}``.

        Materializes each canonical group once: the device H moves to CPU
        (``resource_check`` before each transfer), max|x| is materialized, and
        the SAME read-only CPU hessian object is handed to every alias unit of
        the group.  Each role maps every observed exact unit name to
        ``{'name', 'role', 'hessian', 'inputs', 'count', 'max_abs',
        'prefix_sample_ids'}``; ``prefix_sample_ids`` holds one int64 index per
        retained prefix row, so its length is exactly
        ``inputs.shape[0]``.
        """
        if self.closed:
            raise RuntimeError("DisjointRowMoments used after close()")
        if self._published is not None:
            # Publication is a materialization snapshot; repeating it returns
            # the same read-only records instead of re-draining states.
            return self._published
        out: dict[str, dict] = {FIT: {}, HELDOUT: {}}
        for (role, names), state in sorted(self._groups.items(),
                                           key=lambda item: (item[0][0],
                                                             item[0][1])):
            if self.resource_check is not None:
                self.resource_check(f"split_moments:before_hessian_transfer:{role}:{names[0]}")
            h = state["hessian"]
            released_cuda = bool(h is not None and h.is_cuda)
            cpu_h = None if h is None else h.to(device="cpu", dtype=torch.float32)
            if cpu_h is h:
                # Already-CPU moments must still publish read-only.
                cpu_h = cpu_h.clone()
            h = None
            state["hessian"] = None
            max_abs = state["max_abs"]
            max_abs = 0.0 if max_abs is None else float(
                max_abs.to(device="cpu", dtype=torch.float32))
            if self.resource_check is not None:
                self.resource_check(f"split_moments:before_prefix_transfer:{role}:{names[0]}")
            inputs = (torch.cat(state["inputs"], dim=0).to(device="cpu")
                      if state["inputs"] else None)
            ids = torch.tensor(state["prefix_sample_ids"], dtype=torch.int64)
            state["inputs"].clear()
            state["prefix_sample_ids"].clear()
            state["max_abs"] = None
            if released_cuda:
                torch.cuda.empty_cache()
            if inputs is not None:
                assert ids.shape[0] == inputs.shape[0]
            for name in names:
                out[role][name] = {
                    "name": name,
                    "role": role,
                    "hessian": cpu_h,
                    "inputs": inputs,
                    "count": state["count"],
                    "max_abs": max_abs,
                    "prefix_sample_ids": ids,
                }
        self._published = out
        return out

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Release every owned buffer.  Idempotent; on failure the caller
        closes before propagating so no partial capture is retained."""
        self._groups.clear()
        self._published = None
        self.closed = True


__all__ = ["DisjointRowMoments", "FIT", "HELDOUT"]
