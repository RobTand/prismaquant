"""CPU-only exact token-subsequence audit against the accepted original draw."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class Substrings:
    """Suffix automaton with negative separators between independent windows."""
    def __init__(self, rows):
        self.edges, self.link, self.length, self.end = [{}], [-1], [0], [-1]
        last = 0
        flat = []
        for row_id, row in enumerate(rows):
            for token in [*row, -row_id-1]:
                position = len(flat)
                flat.append(token)
                current = len(self.edges)
                self.edges.append({})
                self.length.append(self.length[last]+1)
                self.link.append(0)
                self.end.append(position)
                parent = last
                while parent >= 0 and token not in self.edges[parent]:
                    self.edges[parent][token] = current
                    parent = self.link[parent]
                if parent >= 0:
                    target = self.edges[parent][token]
                    if self.length[parent]+1 == self.length[target]:
                        self.link[current] = target
                    else:
                        clone = len(self.edges)
                        self.edges.append(self.edges[target].copy())
                        self.length.append(self.length[parent]+1)
                        self.link.append(self.link[target])
                        self.end.append(self.end[target])
                        while parent >= 0 and self.edges[parent].get(token) == target:
                            self.edges[parent][token] = clone
                            parent = self.link[parent]
                        self.link[target] = self.link[current] = clone
                last = current
        self.flat = flat

    def match(self, tokens):
        state = length = 0
        best = {"length": 0, "eval_start": None, "calibration_flat_start": None}
        coverage = {threshold: set() for threshold in (16, 32, 64, 128, 256, 512)}
        for position, token in enumerate(tokens):
            while state and token not in self.edges[state]:
                state = self.link[state]
                length = self.length[state]
            if token in self.edges[state]:
                state = self.edges[state][token]
                length += 1
            else:
                state = length = 0
            if length > best["length"]:
                best = {"length": length, "eval_start": position-length+1,
                        "calibration_flat_start": self.end[state]-length+1}
            for threshold, positions in coverage.items():
                if length >= threshold:
                    positions.update(range(position-length+1, position+1))
        if best["length"]:
            a, b, count = best["eval_start"], best["calibration_flat_start"], best["length"]
            if tokens[a:a+count] != self.flat[b:b+count] or any(x < 0 for x in self.flat[b:b+count]):
                raise RuntimeError("substring witness failed exact token verification")
        return {"tokens": len(tokens), "longest_match": best,
                "covered_eval_positions_by_minimum_contiguous_match": {
                    str(k): len(v) for k, v in coverage.items()}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from prismaquant.calibration_data import load_calibration_input
    from tools.dsv4_wikitext_inputs import load_wikitext_inputs, wikitext_model_identity
    from tools.full_kl_teacher_payload import tokenizer_identity
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("overlap audit requires PB CPU-only execution")
    root = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
    model = Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
    original = root/'exact-calibration-input-01/calibration_tokens.safetensors'
    original_sha = '9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329'
    ids, binding = load_calibration_input(original, expected_sha256=original_sha,
                                         n_samples=512, seqlen=512)
    expected = dict(seed=0, fit_ids_sha256='6b6a0c4283de3aae633fd2bf00f74ac80928a8c389c42aa6d5101b765115532e',
                    text_sha256='aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5')
    if any(binding['provenance'].get(k) != v for k, v in expected.items()):
        raise RuntimeError('original draw does not match accepted calibration')
    payload = load_wikitext_inputs(args.inputs, expected_sha256=args.inputs_sha256,
        expected_tokenizer_identity=tokenizer_identity(model),
        expected_model_identity=wikitext_model_identity(model))
    rows = ids.tolist()
    matcher = Substrings(rows)
    full = [dict(window=i, **matcher.match(row))
            for i, row in enumerate(payload['full_kl']['token_ids'])]
    ppl = matcher.match(payload['ppl']['token_ids'])
    exact = [{"eval_window": i, "calibration_window": j}
             for i, row in enumerate(payload['full_kl']['token_ids'])
             for j, original_row in enumerate(rows) if row == original_row]
    report = dict(schema='prismaquant.glm_wikitext_token_overlap.v1',
        status='COMPLETE_TOKEN_VALUE_AUDIT', fit_overlap_status='unverified',
        heldout_claim=False, original=binding,
        input_file=dict(path=args.inputs, sha256=args.inputs_sha256),
        input_semantic_sha256=payload['semantic_sha256'],
        full_kl_exact_window_matches=exact, full_kl=full, ppl=ppl,
        semantics='Longest exact contiguous token matches against each original 512-token calibration window; negative separators prohibit crossing calibration-window boundaries. Coverage unions matching evaluation positions at each stated minimum length.',
        limitation='Original calibration includes empty text rows and tokenizer defaults; gold filters nonempty rows and disables special tokens. Corpus start indices are incomparable. This audit does not map corpus character positions or prove held-out origin; repeated text can share token sequences.',
        prior_reconstruction=dict(path=str(root/'exact-calibration-input-01/reconstruction.json'),
                                  sha256=sha(root/'exact-calibration-input-01/reconstruction.json')),
        helper_sha256=sha(__file__), cpu_affinity=sorted(os.sched_getaffinity(0)))
    out = Path(args.output)
    if out.exists():
        raise RuntimeError('refusing to overwrite overlap audit')
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(out), sha256=sha(out), exact_windows=len(exact),
        full_kl_longest=[row['longest_match']['length'] for row in full],
        ppl_longest=ppl['longest_match']['length'], heldout_claim=False), sort_keys=True))


if __name__ == '__main__':
    main()
