def _tensor_bytes(tensor):
    """The tensor's own bytes, contiguous, on host.

    A byte view rather than ``numpy()``, which raises on some of the dtypes
    this sees and which would hash a converted value rather than the stored
    bit pattern.  Bit patterns are what a bit-identity question is about.
    """
    import torch

    flat = tensor.detach().contiguous().cpu().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()
