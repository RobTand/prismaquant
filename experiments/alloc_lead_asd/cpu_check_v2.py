"""CPU check of a_side_diag v2 against the v1 (production-lease) path on a tiny random Qwen3.

Builds a 2-layer random Qwen3 (fp32, sdpa), runs the full v2 ``run`` (with --profile, layer
arms and single units), then recomputes everything with the v1 functions over ALL rows and
probes and compares:
  * pricing components comps[spec, unit, probe, row] (A8, A8 without position 0, W4);
  * every arm's KL_seq and Q_seq (closed form vs two log-softmaxes);
  * every probe arm's s_real[probe, row].
Exit status 0 only if every check holds.  usage: cpu_check_v2.py OUTDIR
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_side_diag as D  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
D.DEVICE = "cpu"
torch.manual_seed(0)
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=128, tie_word_embeddings=False)
cfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM(cfg).eval()
with torch.no_grad():                       # widen the activation range so FP8 error is visible
    for name, p in model.named_parameters():
        if p.dim() == 2:
            p.mul_(3.0)
for p in model.parameters():
    p.requires_grad_(False)
model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

n, T, K = 4, 32, 3
ids = torch.randint(0, cfg.vocab_size, (n, T), generator=torch.Generator().manual_seed(1))
inputs = os.path.join(OUT, "inputs.safetensors")
save_file({"fit_s42": ids.to(torch.int64).contiguous()}, inputs)
stem = os.path.join(OUT, "tiny")
for suffix in (".pricing.partial.pt", ".arms.partial.pt"):
    if os.path.exists(stem + suffix):
        os.remove(stem + suffix)
args = argparse.Namespace(model="tiny-random-qwen3", inputs=inputs, text="fit_s42", n_seqs=n,
                          n_probes=K, seed_base=7000, dtype="float32", n_single=2,
                          layer_arms=True, dz_dtype="float32", pricing_from=None, profile=True,
                          smoke_first=False, output=stem)
D.run(args, model)

out = json.load(open(stem + ".json"))
pricing = torch.load(stem + ".pricing.pt")
saved = torch.load(stem + ".pt")
units = {nm: m for nm, m in model.named_modules()
         if isinstance(m, torch.nn.Linear) and nm.startswith("model.layers.")}
names = list(units)
spec_a8 = D.fr.get_format(D.A8)
specs_obj = {D.A8: spec_a8, D.A8N: D.nopos0_spec(spec_a8), D.W4: D.fr.get_format(D.W4)}
seeds = out["seeds"]
N = n * T
failures = []

v1 = D.price_v1(model, units, names, ids, seeds, specs_obj, N, "all", 1.0)
v2 = pricing["comps"]
for si, spec in enumerate(D.SPECS):
    a, b = v1[si], v2[si]
    rms = float(a.pow(2).mean().sqrt())
    rel = float((a - b).abs().max()) / rms
    print(f"pricing {spec}: rms {rms:.4g} max|v1-v2|/rms {rel:.3g}")
    if not rel < 1e-4:
        failures.append(f"pricing {spec} rel {rel:.3g}")

plan = D.build_plan(names, {s: 0.5 * v2[i].sum(-1).pow(2).mean(-1) for i, s in enumerate(D.SPECS)},
                    {nm: j for j, nm in enumerate(names)}, args.n_single, args.layer_arms)
contexts = D.make_contexts(plan, units, specs_obj)
for label, _, _ in plan:
    with_probes = not label.split(":")[0].endswith("_unit")
    m = D.measure_v1(model, ids, N, seeds, "all", 1.0, contexts[label], with_probes)
    arm = out["arms"][label]
    kl1, kl2 = m["kl"], torch.tensor(arm["KL_seq"], dtype=torch.float64)
    q1, q2 = m["q"], torch.tensor(arm["Q_seq"], dtype=torch.float64)
    kl_rel = float(((kl1 - kl2).abs() / kl1.abs().clamp_min(1e-30)).max())
    q_rel = float(((q1 - q2).abs() / q1.abs().clamp_min(1e-30)).max())
    msg = f"arm {label:28s} KL {float(kl1.sum()):.4g} rel {kl_rel:.2g}  Q rel {q_rel:.2g}"
    if kl_rel > 1e-8 or q_rel > 1e-8:
        failures.append(f"{label} kl_rel {kl_rel:.3g} q_rel {q_rel:.3g}")
    if with_probes:
        s1, s2 = m["s_real"], saved[f"s_real:{label}"].double()
        s_rel = float((s1 - s2).abs().max()) / float(s1.pow(2).mean().sqrt())
        msg += f"  s_real rel {s_rel:.2g}"
        if not s_rel < 1e-4:
            failures.append(f"{label} s_real rel {s_rel:.3g}")
    print(msg)
print("profile:", json.dumps(out["profile"]["component_crosscheck"]),
      "kl_rel", out["profile"]["arm_crosscheck"]["kl_rel_diff"])
print("A_all ratio chain:", {k: out["arms"]["A_all"][k] for k in
                              ("P_add", "P_joint", "S_real", "Q_real", "KL_true", "corr_slin_sreal")})
if failures:
    print("FAIL", failures)
    sys.exit(1)
print("PASS")
