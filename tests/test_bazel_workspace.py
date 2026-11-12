"""Bazel workspace contracts: pinned toolchains and images, BuildBuddy flags, no secrets (#3216)."""
from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / '.bazelrc').read_text().splitlines()
MODULE = (ROOT / 'MODULE.bazel').read_text()
PLATFORMS = (ROOT / 'tools/bazel/BUILD.bazel').read_text()


def test_bzlmod_python_and_the_gate_requirements_drive_the_build():
    for required in ['bazel_dep(name = "rules_python"', 'python_version = "3.12"', 'pip.parse(',
                     '"//tools/bazel:requirements_linux_x86_64.txt": "linux_x86_64"',
                     '"//tools/bazel:requirements_linux_aarch64.txt": "linux_aarch64"',
                     'hub_name = "pq_pip"', 'explicit_init_py(default = True)']:
        assert required in MODULE
    assert re.fullmatch(r'\d+\.\d+\.\d+\n', (ROOT / '.bazelversion').read_text())
    assert (ROOT / 'MODULE.bazel.lock').is_file()


def test_both_architectures_have_a_registered_cc_toolchain():
    # Without one, an arm64 run resolves no test target at all.
    assert 'bazel_dep(name = "toolchains_buildbuddy", version = "0.0.4")' in MODULE
    for toolchain in ['"@buildbuddy_toolchain//:ubuntu_cc_toolchain"',
                      '"@buildbuddy_toolchain//:ubuntu_cc_toolchain_arm64"']:
        assert toolchain in MODULE
    assert PLATFORMS.count('"@bazel_tools//tools/cpp:gcc"') == 2


def test_both_executor_platforms_pin_an_ubuntu24_image_by_digest_without_network():
    images = re.findall(r'"container-image": "docker://gcr.io/flame-public/rbe-ubuntu24-04@sha256:[0-9a-f]{64}"',
                        PLATFORMS)
    assert len(images) == 2
    assert PLATFORMS.count('"dockerNetwork": "off"') == 2
    assert PLATFORMS.count('"dockerUser": "1000:1000"') == 2


def test_buildbuddy_suggested_flags_apply_to_every_command():
    for flag in ['--remote_cache_compression', '--experimental_remote_cache_chunking',
                 '--remote_grpc_log=bazel-remote-grpc.log',
                 '--execution_log_compact_file=execution_log.binpb.zst']:
        assert f'common {flag}' in CONFIG
    for flag in ['--noslim_profile', '--experimental_profile_include_target_label',
                 '--experimental_profile_include_primary_output']:
        assert f'build {flag}' in CONFIG


def test_remote_configs_execute_everything_remotely_and_leave_room_for_other_gates():
    for mode in ['remote', 'remote-arm64']:
        for flag in ['--remote_download_minimal', '--remote_cache=grpcs://remote.buildbuddy.io',
                     '--remote_executor=grpcs://remote.buildbuddy.io',
                     '--bes_backend=grpcs://remote.buildbuddy.io', '--remote_local_fallback=false',
                     '--nobuild_runfile_links']:
            assert f'build:{mode} {flag}' in CONFIG
        # BuildBuddy's free plan is shared with TL3's and PrismaBuild's gates.
        assert f'build:{mode} --jobs=40' in CONFIG
    assert 'build:remote --platforms=//tools/bazel:linux_amd64' in CONFIG
    for flag in ['--platforms', '--extra_execution_platforms', '--host_platform']:
        assert f'build:remote-arm64 {flag}=//tools/bazel:linux_arm64' in CONFIG
    assert not [line for line in CONFIG if line.startswith('build --jobs')]


def test_ci_refuses_a_stale_module_lockfile_and_keeps_receipts():
    assert 'common:ci --lockfile_mode=error' in CONFIG
    assert 'build:ci --remote_download_regex=.*/test\\.outputs/.*' in CONFIG
    assert 'build:ci --build_metadata=REPO_URL=https://github.com/RobTand/prismaquant' in CONFIG


def test_no_key_is_committed_and_outputs_are_ignored():
    assert not [line for line in CONFIG if 'x-buildbuddy-api-key' in line]
    ignored = (ROOT / '.gitignore').read_text().splitlines()
    for pattern in ['/bazel-*', '/*bazel-remote-grpc.log', '/*execution_log.binpb.zst']:
        assert pattern in ignored


def test_tests_run_with_native_threads_bounded_like_the_hosted_gate():
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS']:
        assert f'test --test_env={name}=1' in CONFIG


def test_special_tests_carry_a_reason_and_a_known_lane():
    table = (ROOT / 'tools/bazel/nonhermetic.bzl').read_text()
    entries = re.findall(r'"(//[^"]+)": struct\(\n        tags = \[([^\]]*)\],\n        reason = "([^"]+)",', table)
    assert entries and len(entries) == table.count('struct(')
    for label, tags, reason in entries:
        assert set(re.findall(r'"([^"]+)"', tags)) <= {'gpu', 'runner', 'hosted', 'hosted-arm64', 'requires-network'}, label
        assert len(reason) > 20 and 'TBD' not in reason, label


def test_every_extra_test_input_carries_a_reason():
    table = (ROOT / 'tools/bazel/test_inputs.bzl').read_text()
    assert table.count('struct(') == len(re.findall(r'\n        reason = "[^"]{20,}",', table))


def test_arm64_disk_tests_remain_executable_on_amd64():
    scope = {'struct': SimpleNamespace,
             'native': SimpleNamespace(package_name=lambda: 'tests')}
    exec((ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), scope)
    # Both failed the real arm64 sweep at its disk-space preflight.
    for name in ('test_capture_layer_chain', 'test_tessera_calibration_cache'):
        tags, properties, environment = scope['nonhermetic'](name)
        assert 'hosted-arm64' in tags
        assert not {'manual', 'gate-lane-hosted', 'no-remote-exec'} & set(tags)
        assert environment.get('PQ_BAZEL_LANE') != 'runner'
        assert '0' in scope['NONHERMETIC']['//tests:' + name].reason


def test_torch_comes_from_the_cpu_index_per_platform():
    # pip on the amd64 runner would otherwise install an x86 torch that arm64 executors can't load.
    assert 'experimental_index_url_overrides = {"torch": "https://download.pytorch.org/whl/cpu"}' in MODULE
    for platform in ('x86_64', 'aarch64'):
        export = (ROOT / f'tools/bazel/requirements_linux_{platform}.txt').read_text()
        assert 'torch==' in export and '+cpu' in export.split('torch==', 1)[1].split()[0]
        assert '--extra-index-url' not in export


def test_graph_impact_keeps_the_git_checkout_on_the_runner():
    scope = {'struct': SimpleNamespace,
             'native': SimpleNamespace(package_name=lambda: 'tests')}
    exec((ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), scope)
    tags, _, environment = scope['nonhermetic']('test_graph_impact')
    assert {'runner', 'no-remote-exec', 'no-sandbox', 'no-cache'} <= set(tags)
    assert environment['PQ_BAZEL_LANE'] == 'runner'


@pytest.mark.parametrize('name', [
    'test_no_secret_on_command_line',
    'test_fleet_acceptance_level1',
])
def test_checkout_sweep_targets_keep_the_real_git_checkout_on_runner(name):
    scope = {'struct': SimpleNamespace,
             'native': SimpleNamespace(package_name=lambda: 'tests')}
    exec((ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), scope)
    tags, _, environment = scope['nonhermetic'](name)
    assert {'runner', 'no-remote-exec', 'no-sandbox', 'no-cache'} <= set(tags)
    assert environment['PQ_BAZEL_LANE'] == 'runner'
    assert 'git' in scope['NONHERMETIC']['//tests:' + name].reason.lower()


def test_sdk_owned_readers_keep_the_hosted_interpreters_linux_apis():
    scope = {'struct': SimpleNamespace,
             'native': SimpleNamespace(package_name=lambda: 'tests')}
    exec((ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), scope)
    tags, _, environment = scope['nonhermetic']('test_pilot_verified_result_1293')
    assert {'hosted', 'manual', 'gate-lane-hosted'} <= set(tags)
    assert 'runner' not in tags
    assert environment.get('PQ_BAZEL_LANE') != 'runner'
    reason = scope['NONHERMETIC']['//tests:test_pilot_verified_result_1293'].reason
    assert reason.startswith('needs os.pidfd_open/P_PIDFD/signal.pidfd_send_signal, absent from '
                             'python-build-standalone 3.12 (rules_python 2.4.6) on amd64;')
    assert 'arm64 standalone has pidfd and is excluded only for table simplicity' in reason
    assert 'Hosted setup-python retains amd64 coverage' in reason


@pytest.mark.parametrize('name', [
    'test_original_diagnostic_context_2149',
    'test_original_source_admission',
])
def test_original_modeling_source_paths_use_hosted_site_packages(name):
    scope = {'struct': SimpleNamespace,
             'native': SimpleNamespace(package_name=lambda: 'tests')}
    exec((ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), scope)
    tags, _, environment = scope['nonhermetic'](name)
    assert {'hosted', 'manual', 'gate-lane-hosted'} <= set(tags)
    assert 'runner' not in tags
    assert environment.get('PQ_BAZEL_LANE') != 'runner'
    assert scope['NONHERMETIC']['//tests:' + name].reason == (
        "Bazel canonical repo paths contain '+', which the modeling-source schema "
        '(schemas.py:592) correctly rejects. Hosted setup-python retains amd64 coverage; '
        'arm64 action coverage is deferred for this same layout constraint.')
