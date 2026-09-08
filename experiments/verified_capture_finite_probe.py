"""CPU semantic/allocation qualification of scalar finite reduction, not a speed claim."""
import json
import torch
from prismaquant.perturbed_x_cache import bounded_cpu_float32_isfinite


def scalar_finite(tensor):
    return bounded_cpu_float32_isfinite(tensor, max_scratch_bytes=4*1024**2)


def main():
    torch.set_num_threads(1)
    cases = []
    for size in (0, 1, 33, 4099):
        positions = sorted({0, size//2, size-1}) if size else [None]
        for position in positions:
            for label, value in [('finite', 1.), ('negative', -1.), ('nan', float('nan')),
                                 ('positive_inf', float('inf')), ('negative_inf', -float('inf'))]:
                tensor = torch.zeros(size, dtype=torch.float32)
                if position is not None:
                    tensor[position] = value
                actual = scalar_finite(tensor)
                expected = bool(torch.isfinite(tensor).all())
                assert actual == expected, (size, position, label, actual, expected)
                cases.append(dict(size=size, position=position, value=label, finite=actual))
    allocations = []
    for nbytes in (4*1024**2, 64*1024**2, 600*1024**2):
        tensor = torch.ones(nbytes//4, dtype=torch.float32)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True) as profile:
            assert scalar_finite(tensor)
        positive = [dict(name=e.name, self_cpu_memory_bytes=e.self_cpu_memory_usage,
                         cpu_memory_bytes=e.cpu_memory_usage)
                    for e in profile.events() if e.self_cpu_memory_usage > 0]
        allocated = sum(e['self_cpu_memory_bytes'] for e in positive)
        assert allocated <= 64, positive
        allocations.append(dict(input_bytes=nbytes, positive_self_cpu_allocation_bytes=allocated,
                                allocating_events=positive))
        del tensor
    print(json.dumps(dict(torch=torch.__version__, torch_git=torch.version.git_version,
        cuda=torch.version.cuda, device='cpu', threads=torch.get_num_threads(),
        semantic_cases=cases, allocations=allocations,
        limits='Profiler accounts Torch CPU allocations; non-Torch allocator bookkeeping is runtime overhead. No latency claim.'),
        indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
