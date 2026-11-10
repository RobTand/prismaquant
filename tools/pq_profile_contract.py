"""Pure argv and destination contracts shared by namespace validation and profiling."""
from pathlib import Path

PROFILE_LOCAL_ROOT = Path('/home/rob/tmp/claude-campaign-20260926/tmp/row-startup/profiles')


def profile_row_parts(command):
    """Locate the existing row boundary without changing command operands."""
    boundary = 0
    if 'tools.tessera_campaign_container' in command:
        module = command.index('tools.tessera_campaign_container')
        boundary = command.index('--', module) + 1
    inner = command[boundary:]
    if not inner or not Path(inner[0]).name.startswith('python'):
        raise ValueError('profiling requires the original Python row argv')
    return boundary, inner


def profiled_row_command(command, *, destination, profiler_executable):
    """Keep container ownership and original argv; profile its same-UID child."""
    boundary, inner = profile_row_parts(command)
    return [*command[:boundary], inner[0], '-u', '-m', 'tools.pq_profile_child',
            '--profiler-executable', profiler_executable, '--out', str(destination),
            '--', *inner]


def profiled_workload_parts(command, *, destination, profiler_executable):
    """Accept only the exact existing canonical child wrapper; keep full argv."""
    boundary, inner = profile_row_parts(command)
    if len(inner) < 10 or inner[8] != '--':
        raise RuntimeError('namespace profile requires the canonical child wrapper')
    original = [*command[:boundary], *inner[9:]]
    if command != profiled_row_command(original, destination=destination,
                                       profiler_executable=profiler_executable):
        raise RuntimeError('namespace profile wrapper differs from its declaration')
    return boundary + 9


def profile_local_destination(request_key):
    """Same existing host-local policy; namespace owns one deterministic child."""
    return PROFILE_LOCAL_ROOT / request_key
