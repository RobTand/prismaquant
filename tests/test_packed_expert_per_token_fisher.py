"""Pinning tests for the 2026-07-02 numerical-audit probe fixes.

FIX 1 (audit M3): packed-MoE expert Fisher uses the per-token-summed
    estimator Σ_t ‖∇_t‖² captured at the expert `F.linear(x, packed[e])`
    boundary — not the sum-then-square ‖Σ_t ∇_t‖² of the token-summed
    weight gradient. Micro-repro: matches a brute-force per-token
    autograd Fisher to <1%, is invariant (in per-token-mean units) to
    doubling T, and the old estimator provably is not. Non-interceptable
    packed compute (e.g. bmm) fail-fasts unless
    PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1.

FIX 2 (audit M4, denominator superseded by PR #14):
    `FisherAccumulator.finalize` applies a SINGLE normalization — no
    second ÷route_prob (route_prob is metadata only) — so the
    `FisherAccumulator` backend agrees with the incremental (production)
    backend's convention. M4 originally made that single division
    per-ROUTED-token; PR #14 deliberately reversed the denominator to
    the GLOBAL calib token count for every row (tokens never routed to
    an expert contribute zero gradient, so per-routed-token inflated
    sparse-expert rows by global/routed). The no-double-division
    property M4 pinned still holds and is still pinned here.

FIX 3 (audit M9): every h-detail writer goes through
    `probe_statistics.h_detail_blob` (per-token units + explicit
    ``units: "per_token"`` marker); `HDetailIndex.h_diag_from_blob`
    refuses legacy raw token-summed ``H`` blobs. Since PR #14 the blob
    denominator is the GLOBAL calib token count (v4), matching the
    scalar `h_trace`.
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.sensitivity_probe import FisherAccumulator, install_packed_expert_hooks
from prismaquant.probe_statistics import h_detail_blob
from prismaquant.measure_quant_cost import HDetailIndex

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ToyPackedExperts(nn.Module):
    """Mirrors the transformers packed-experts pattern (Qwen3MoeExperts /
    Qwen3NextExperts / Lfm2MoeExperts): 3D packed params consumed via
    per-expert `nn.functional.linear(x_routed, packed[e])`."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, inter))
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index,
                             num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            cur = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                cur, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = nn.functional.linear(h, self.down_proj[expert_idx])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h)
        return final


class _PackedWrap(nn.Module):
    def __init__(self, num_experts=4, hidden=5, inter=6):
        super().__init__()
        self.experts = ToyPackedExperts(num_experts, hidden, inter)


def _routed_batch(T, E, K, hidden, device, correlated=True, seed=0):
    """Correlated tokens (strong shared component) + distinct top-K
    routing, as real routers produce (`torch.topk` never repeats an
    expert within a token)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(1, hidden, generator=g)
    x = (base + 0.1 * torch.randn(T, hidden, generator=g)).to(device)
    scores = torch.randn(T, E, generator=g).to(device)
    w, idx = torch.topk(F.softmax(scores, dim=-1), K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    v = torch.randn(T, hidden, generator=g).to(device)
    return x, idx, w, v


class TestPackedPerTokenFisher(unittest.TestCase):
    """FIX 1 — per-token packed-expert Fisher estimator (audit M3)."""

    def _brute_force(self, model, x, idx, w, v):
        """Per-token autograd Fisher: Σ_t (∂L_t/∂W)² per packed param."""
        bf = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(True)
        for t in range(x.size(0)):
            model.zero_grad(set_to_none=True)
            out = model.experts(x[t:t + 1], idx[t:t + 1], w[t:t + 1])
            (out * v[t:t + 1]).sum().backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    bf[n] += p.grad.detach().pow(2)
        model.zero_grad(set_to_none=True)
        return bf

    def test_matches_bruteforce_per_token_fisher(self):
        torch.manual_seed(0)
        E, hidden, inter, T, K = 4, 5, 6, 16, 2
        model = _PackedWrap(E, hidden, inter).to(DEV)
        x, idx, w, v = _routed_batch(T, E, K, hidden, DEV)

        bf = self._brute_force(model, x, idx, w, v)

        scalar, channel, full = {}, {}, {}
        install_packed_expert_hooks(
            model, accumulator=scalar,
            channel_accumulator=channel, full_accumulator=full)
        xg = x.detach().requires_grad_(True)
        (model.experts(xg, idx, w) * v).sum().backward()

        for name, bf_h in bf.items():
            with self.subTest(param=name):
                ref_trace = float(bf_h.sum())
                got_trace = float(scalar[name])
                # Acceptance criterion (a): <1% of brute force.
                self.assertLess(abs(got_trace - ref_trace),
                                0.01 * abs(ref_trace))
                # Full per-weight diagonal [E, M, N] matches elementwise.
                self.assertLess(
                    float((full[name].to(DEV) - bf_h).abs().max()),
                    5e-3 * float(bf_h.abs().max()))
                # Per-expert per-channel [E, M] matches Σ over N.
                self.assertLess(
                    float((channel[name].to(DEV)
                           - bf_h.sum(dim=-1)).abs().max()),
                    5e-3 * float(bf_h.sum(dim=-1).abs().max()))

        # No [E, M, N] gradient is ever accumulated on the leaves.
        for n, p in model.named_parameters():
            self.assertIsNone(p.grad, n)

        # The interception's input gradient is exact (the reverse sweep
        # depends on it): compare against plain autograd on an unhooked
        # copy of the same weights.
        plain = _PackedWrap(E, hidden, inter).to(DEV)
        plain.load_state_dict(model.state_dict())
        xr = x.detach().requires_grad_(True)
        (plain.experts(xr, idx, w) * v).sum().backward()
        self.assertLess(
            float((xg.grad - xr.grad).abs().max()),
            1e-5 * max(float(xr.grad.abs().max()), 1e-12))

    def test_per_token_mean_invariant_to_doubling_T_but_old_path_is_not(self):
        torch.manual_seed(1)
        E, hidden, inter, T, K = 4, 5, 6, 12, 2
        model = _PackedWrap(E, hidden, inter).to(DEV)
        x, idx, w, v = _routed_batch(T, E, K, hidden, DEV, seed=7)

        def capture(xin, idxin, win, vin):
            scalar = {}
            install_packed_expert_hooks(model, accumulator=scalar,
                                        channel_accumulator=None,
                                        full_accumulator=None)
            xg = xin.detach().requires_grad_(True)
            (model.experts(xg, idxin, win) * vin).sum().backward()
            return {k: float(t) for k, t in scalar.items()}

        one = capture(x, idx, w, v)
        two = capture(torch.cat([x, x]), torch.cat([idx, idx]),
                      torch.cat([w, w]), torch.cat([v, v]))
        # New estimator: duplicating every token exactly doubles the raw
        # sum → the per-token mean is invariant.
        for name in one:
            with self.subTest(param=name):
                self.assertAlmostEqual(two[name] / (2 * T),
                                       one[name] / T,
                                       delta=1e-3 * abs(one[name] / T))

        # Old sum-then-square estimator on the same batches: Σ∇ doubles,
        # its square quadruples → the per-token mean DOUBLES (this is the
        # non-convergence audit M3 documents).
        def old_sumsq(xin, idxin, win, vin):
            plain = _PackedWrap(E, hidden, inter).to(DEV)
            plain.load_state_dict(model.state_dict())
            for p in plain.parameters():
                p.requires_grad_(True)
            (plain.experts(xin, idxin, win) * vin).sum().backward()
            return {n: float(p.grad.pow(2).sum())
                    for n, p in plain.named_parameters()}

        o1 = old_sumsq(x, idx, w, v)
        o2 = old_sumsq(torch.cat([x, x]), torch.cat([idx, idx]),
                       torch.cat([w, w]), torch.cat([v, v]))
        for name in o1:
            with self.subTest(param=name, path="old"):
                ratio = (o2[name] / (2 * T)) / (o1[name] / T)
                self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_rebind_uses_fresh_accumulators(self):
        """The idempotent re-install path (per-shard) must route stats
        into the NEW dicts."""
        torch.manual_seed(2)
        model = _PackedWrap().to(DEV)
        x, idx, w, v = _routed_batch(10, 4, 2, 5, DEV, seed=3)
        first = {}
        install_packed_expert_hooks(model, accumulator=first)
        second = {}
        install_packed_expert_hooks(model, accumulator=second)
        xg = x.detach().requires_grad_(True)
        (model.experts(xg, idx, w) * v).sum().backward()
        self.assertEqual(first, {})
        self.assertTrue(second)
        for val in second.values():
            self.assertGreater(float(val), 0.0)


class _BmmExperts(nn.Module):
    """Packed experts whose compute is NOT an F.linear on a dim-0 slice
    — the interception cannot capture per-token Fisher here."""

    def __init__(self, E=3, M=4, N=5):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, M, N))

    def forward(self, x):  # x: [E, T, N] dense-batched
        return torch.bmm(x, self.gate_up_proj.transpose(1, 2))


class TestPackedSumSqFailFast(unittest.TestCase):
    """FIX 1 fallback: sum-then-square is fail-fast unless the env
    override opts in (mirrors the KV-shared Fisher guard, MINOR-M33)."""

    def setUp(self):
        self._saved = os.environ.pop(
            "PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER"] = self._saved
        else:
            os.environ.pop("PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER", None)

    def _model(self):
        m = nn.Module()
        m.experts = _BmmExperts()
        return m.to(DEV)

    def test_uninterceptable_compute_fails_fast(self):
        torch.manual_seed(0)
        model = self._model()
        acc = {}
        install_packed_expert_hooks(model, accumulator=acc)
        x = torch.randn(3, 8, 5, device=DEV, requires_grad=True)
        out = model.experts(x)
        with self.assertRaisesRegex(RuntimeError,
                                    "PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER"):
            out.sum().backward()

    def test_env_override_reproduces_legacy_sum_then_square(self):
        torch.manual_seed(0)
        os.environ["PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER"] = "1"
        model = self._model()
        acc = {}
        install_packed_expert_hooks(model, accumulator=acc)
        x = torch.randn(3, 8, 5, device=DEV, requires_grad=True)
        model.experts(x).sum().backward()
        # Legacy semantics: ‖Σ_t ∇_t‖² of the token-summed weight grad.
        plain = _BmmExperts().to(DEV)
        plain.load_state_dict(model.experts.state_dict())
        for p in plain.parameters():
            p.requires_grad_(True)
        plain(x.detach()).sum().backward()
        ref = float(plain.gate_up_proj.grad.pow(2).sum())
        self.assertAlmostEqual(float(acc["experts.gate_up_proj"]), ref,
                               delta=1e-3 * abs(ref))


class _RoutedExpert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)


class _UnpackedMoE(nn.Module):
    """Top-1-routed per-expert nn.Linear MoE (MiniMax/DSv4-style layout,
    reduced to one projection)."""

    def __init__(self, E, hidden, inter):
        super().__init__()
        self.router = nn.Linear(hidden, E, bias=False)
        self.experts = nn.ModuleList(
            [_RoutedExpert(hidden, inter) for _ in range(E)])

    def forward(self, x):  # [T, hidden]
        top = self.router(x).argmax(dim=-1)
        out = x.new_zeros(x.size(0), self.experts[0].w1.out_features)
        for e, expert in enumerate(self.experts):
            sel = (top == e).nonzero(as_tuple=True)[0]
            if sel.numel():
                out[sel] = expert.w1(x[sel])
        return out


class _StubTracker:
    """RouterTracker stand-in returning a fixed routing probability."""

    def __init__(self, prob):
        self._prob = prob

    def prob(self, router_path, expert_id):
        return self._prob


class TestBackendNormalizationAgreement(unittest.TestCase):
    """FIX 2 (audit M4) + the PR #14 denominator reversal: the
    FisherAccumulator backend's expert h_trace is the routed-token gradient
    mass divided ONCE, by the GLOBAL calib token count — the same
    single-division convention the incremental (production) backend
    implements — with route_prob kept as metadata only, never applied
    as a second division."""

    def test_expert_h_trace_is_global_token_mean(self):
        torch.manual_seed(0)
        E, hidden, inter, T = 3, 6, 4, 24
        model = _UnpackedMoE(E, hidden, inter).to(DEV)
        x = torch.randn(T, hidden, device=DEV)
        v = torch.randn(T, inter, device=DEV)
        with torch.no_grad():
            top = model.router(x).argmax(dim=-1)

        # Reference: Σ_t ‖∇_t W_e‖² over ROUTED tokens t (tokens never
        # routed to e contribute zero gradient), divided by the GLOBAL
        # calib token count T — finalize_fisher_stats' convention.
        ref = {}
        for e, expert in enumerate(model.experts):
            sel = (top == e).nonzero(as_tuple=True)[0]
            total = 0.0
            for t in sel.tolist():
                g, = torch.autograd.grad(
                    (expert.w1(x[t:t + 1]) * v[t:t + 1]).sum(),
                    expert.w1.weight)
                total += float(g.pow(2).sum())
            if sel.numel():
                ref[f"experts.{e}.w1"] = (total / T, int(sel.numel()))

        tracked = [f"experts.{e}.w1" for e in range(E)]
        expert_info = {f"experts.{e}.w1": ("router", str(e))
                       for e in range(E)}
        for p in model.parameters():
            p.requires_grad_(False)
        acc = FisherAccumulator(model, tracked, expert_info)
        xg = x.detach().requires_grad_(True)
        (model(xg) * v).sum().backward()
        # A route_prob well below 1 — the pre-M4 code would divide by it
        # as a SECOND division; still pinned as metadata-only.
        acc.finalize(_StubTracker(0.25), global_tokens=T)

        for name, (ref_mean, n_routed) in ref.items():
            with self.subTest(linear=name):
                s = acc.stats[name]
                # n_tokens_seen stays raw (routed count, metadata) —
                # the denominator is the global count.
                self.assertEqual(s["n_tokens_seen"], n_routed)
                self.assertEqual(s["h_trace_norm_tokens"], T)
                self.assertEqual(s["route_prob"], 0.25)  # metadata only
                self.assertLess(abs(s["h_trace"] - ref_mean),
                                5e-3 * abs(ref_mean))
                # Pin the reversal: for a sparsely routed expert the old
                # per-routed-token convention was (T/n_routed)× hotter.
                if n_routed < T:
                    self.assertGreater(
                        (s["h_trace_raw"] / n_routed) / s["h_trace"], 1.0)


class TestHDetailUnits(unittest.TestCase):
    """FIX 3 (audit M9): per-token h-detail units with explicit marker,
    consumer refuses legacy raw token-summed blobs."""

    def test_h_detail_blob_normalizes_and_stamps_units(self):
        raw = torch.full((2, 3), 12.0)
        blob = h_detail_blob(raw, 4, "toy.fc")
        self.assertEqual(blob["units"], "per_token")
        # v4: denominator is the GLOBAL calib token count (PR #14), and
        # the blob records it.
        self.assertEqual(blob["h_detail_version"], 4)
        self.assertEqual(blob["norm_tokens"], 4)
        self.assertEqual(blob["kind"], "linear")
        self.assertTrue(torch.equal(blob["h_diag"],
                                    torch.full((2, 3), 3.0)))

    def test_consumer_accepts_marked_blobs_and_legacy_h_diag(self):
        t = torch.ones(2, 2)
        self.assertTrue(torch.equal(
            HDetailIndex.h_diag_from_blob({"h_diag": t}), t))
        self.assertTrue(torch.equal(
            HDetailIndex.h_diag_from_blob(
                {"h_diag": t, "units": "per_token"}), t))
        self.assertTrue(torch.equal(
            HDetailIndex.h_diag_from_blob(
                {"H": t, "units": "per_token"}), t))

    def test_consumer_refuses_legacy_raw_H_and_unknown_units(self):
        t = torch.ones(2, 2)
        with self.assertRaisesRegex(ValueError, "token-summed"):
            HDetailIndex.h_diag_from_blob({"H": t, "name": "old.fc"})
        with self.assertRaisesRegex(ValueError, "unknown units"):
            HDetailIndex.h_diag_from_blob(
                {"h_diag": t, "units": "per_sequence"})
        with self.assertRaises(KeyError):
            HDetailIndex.h_diag_from_blob({"name": "empty"})

    def test_both_writers_land_on_the_same_per_token_scale(self):
        """The sensitivity writer (FisherAccumulator.finalize) and the
        incremental writer funnel (h_detail_blob on the token-summed
        accumulator) must produce blobs the consumer loads at the SAME
        per-token scale."""
        torch.manual_seed(0)
        hidden, out_f, T = 5, 3, 11
        model = nn.Module()
        model.fc = nn.Linear(hidden, out_f, bias=False).to(DEV)
        x = torch.randn(T, hidden, device=DEV)
        v = torch.randn(T, out_f, device=DEV)

        # Token-summed reference accumulator: Σ_t gy_t² ⊗ x_t².
        xg = x.detach().requires_grad_(True)
        y = model.fc(xg)
        gy, = torch.autograd.grad((y * v).sum(), y)
        raw = (gy.pow(2).t() @ x.pow(2)).cpu()

        with tempfile.TemporaryDirectory(
                dir=os.environ.get("TMPDIR") or None) as td:
            h_dir = Path(td) / "h"
            model.fc.weight.requires_grad_(False)
            acc = FisherAccumulator(model, ["fc"], {}, h_detail_dir=h_dir)
            xg2 = x.detach().requires_grad_(True)
            (model.fc(xg2) * v).sum().backward()
            acc.finalize(None, global_tokens=T)

            index = HDetailIndex(h_dir, ["fc"])
            self.assertIn("fc", index)
            blob = index.load_blob("fc")
            self.assertEqual(blob["units"], "per_token")
            sens = index.load("fc")

        incr = HDetailIndex.h_diag_from_blob(
            h_detail_blob(raw, T, "fc", kind="linear"))
        ref = raw / T
        for got, label in ((sens, "sensitivity"), (incr, "incremental")):
            with self.subTest(writer=label):
                self.assertLess(
                    float((got - ref).abs().max()),
                    5e-3 * float(ref.abs().max()))


if __name__ == "__main__":
    unittest.main()
