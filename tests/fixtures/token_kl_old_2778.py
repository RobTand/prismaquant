"""Independent pre-consolidation checked scorer for #2778 (partial #1303).

Frozen from experiments/glm_tr3_full_vocab.py:token_kl at
914bf79807120fa60f433714e95cffd44b917d92, before deleting its duplicate.
Do not import a production KL owner here: this is the old-body CPU oracle,
not evidence of CUDA or served-metric qualification.
"""
import torch


def token_kl(teacher, candidate, *, tile_rows=32, require_cuda=True):
    """Match upstream token_kld_chunk on raw logits without CPU vocabulary work."""
    if (teacher.ndim != 2 or teacher.shape != candidate.shape or teacher.shape[1] <= 1
            or teacher.shape[0] <= 0 or type(tile_rows) is not int or tile_rows <= 0):
        raise ValueError("teacher/candidate logit geometry or tile size mismatch")
    if teacher.device != candidate.device or (require_cuda and teacher.device.type != "cuda"):
        raise ValueError("KL requires co-resident CUDA tensors")
    result = torch.empty(teacher.shape[0], dtype=torch.float64, device=teacher.device)
    for start in range(0, teacher.shape[0], tile_rows):
        end = min(start + tile_rows, teacher.shape[0])
        t, c = teacher[start:end].to(torch.float64), candidate[start:end].to(torch.float64)
        if not bool(torch.isfinite(t).all() & torch.isfinite(c).all()):
            raise ValueError("teacher/candidate logits must be finite")
        t = torch.log_softmax(t, dim=-1)
        c = torch.log_softmax(c, dim=-1)
        result[start:end] = (t.exp() * (t - c)).sum(dim=-1, dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite full-vocabulary KL")
    return result
