"""Read-only C10 allocator sizing through the existing locked Torch extension path.

Only compiled code is cached; effective settings are read on every call.
No CUDA kernel, renderer, allocator setter or separate extension cache is used.
Build this through PrismaBuild using the established extension build policy.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _backend():
    from torch.utils.cpp_extension import load
    from .jit_build_lock import jit_build_lock, torch_build_directory

    source = Path(__file__).with_suffix('.cpp')
    header = Path(torch.__file__).parent / 'include/c10/core/AllocatorConfig.h'
    flags = ['-O2']
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()
                            + str((torch.__version__, torch.version.git_version, flags)).encode()).hexdigest()
    name = 'pq_cuda_allocator_state_' + digest[:16]
    with jit_build_lock(torch_build_directory(name)):
        return load(name=name, sources=[str(source)], extra_cflags=flags,
                    with_cuda=False, verbose=True)


def sizing():
    """The live large-segment and nonsplit-rounding byte sizes, not their text."""
    return _backend().sizing()
