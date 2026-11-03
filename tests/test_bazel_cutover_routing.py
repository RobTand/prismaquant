"""Cutover actions use the declared fleet pools and preserve their isolation."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGES = {
    'amd64': '5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50',
    'arm64': '2f48fc6c0c7f0794a76c74d77ff23b1a80944e05acacff22c63da2ef002ebf9b',
}


def platforms():
    found = {}
    def platform(name, **attributes):
        found[name] = attributes
    def ignore(*args, **kwargs):
        pass
    scope = {name: ignore for name in (
        'load', 'package', 'exports_files', 'filegroup', 'py_library')}
    scope.update(platform=platform, glob=lambda *args, **kwargs: [])
    exec((ROOT / 'tools/bazel/BUILD.bazel').read_text(), scope)
    return found


@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
def test_cutover_platforms_route_actions_to_the_matching_self_hosted_pool(arch):
    configuration = platforms()['linux_' + arch]
    properties = configuration['exec_properties']
    assert properties['Pool'] == 'selfhosted-' + arch
    assert properties['use-self-hosted-executors'] == 'true'
    assert properties['OSFamily'] == 'Linux'
    assert properties['dockerNetwork'] == 'off'
    assert properties['dockerUser'] == '1000:1000'
    assert properties['container-image'] == (
        'docker://gcr.io/flame-public/rbe-ubuntu24-04@sha256:' + IMAGES[arch])
    if arch == 'arm64':
        assert properties['Arch'] == 'arm64'
    constraint = 'x86_64' if arch == 'amd64' else 'aarch64'
    assert '@platforms//cpu:' + constraint in configuration['constraint_values']


@pytest.mark.parametrize('config,arch', [('remote', 'amd64'), ('remote-arm64', 'arm64')])
def test_gate_configs_select_the_pool_platform_without_local_fallback(config, arch):
    lines = (ROOT / '.bazelrc').read_text().splitlines()
    for option in ('platforms', 'extra_execution_platforms'):
        assert f'build:{config} --{option}=//tools/bazel:linux_{arch}' in lines
    assert f'build:{config} --remote_local_fallback=false' in lines
