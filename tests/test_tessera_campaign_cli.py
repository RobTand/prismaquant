"""The actual campaign parser is a cold authority for CLI syntax and defaults."""
from pathlib import Path
import subprocess
import sys

import pytest

from prismaquant.tessera_campaign_cli import parse_campaign_arguments
from prismaquant.tessera_campaign_requests import (
    CaptureJoinRequest, CapturePrepRequest, capture_chain_bookend_request,
)


BASE = ['--model', 'checkpoint', '--out', 'cost.pkl', '--cache-dir', 'cache']


def test_actual_parser_modes_defaults_and_help_are_cold():
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys
sys.path.insert(0, ROOT)
class NoNumericalRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.startswith('tessera.serving') or name in {'prismaquant.allocator_solver', 'prismaquant.format_registry'} or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('CLI imported numerical runtime: ' + name)
sys.meta_path.insert(0, NoNumericalRuntime())
from prismaquant.tessera_campaign_cli import campaign_argument_parser, parse_campaign_arguments
parser = campaign_argument_parser(description='campaign')
assert '--capture-chain' in parser.format_help()
args = parse_campaign_arguments(['--model', 'm', '--out', 'o', '--cache-dir', 'c'])
assert (args.anchors, args.menu_mode, args.row_head, args.hessian) == (3, None, 'stream', 'require')
assert (args.nsamples, args.seqlen, args.seed, args.pinned_roster_only) == (8, 512, 0, False)
print('COLD-CAMPAIGN-CLI-OK')
'''.replace('ROOT', repr(str(root)))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'COLD-CAMPAIGN-CLI-OK'


@pytest.mark.parametrize('extra', [
    ['--row-head', 'load-all', '--row-head=stream'],
    ['--row-head=stream'],
    ['--row-he', 'stream'],
])
def test_existing_equal_duplicate_and_abbreviated_syntax(extra):
    assert parse_campaign_arguments(BASE + extra).row_head == 'stream'


@pytest.mark.parametrize('extra', [[], ['--menu-mode', 'unknown'], ['--row-head', 'unknown']])
def test_required_paths_and_closed_mode_choices_still_refuse(extra):
    argv = extra if not extra else BASE + extra
    with pytest.raises(SystemExit) as error:
        parse_campaign_arguments(argv)
    assert error.value.code == 2


def test_phase_binding_retains_explicit_zero_and_false():
    args = parse_campaign_arguments(BASE + ['--capture-chain', 'prep',
        '--calibration-census', 'census.json', '--capture-calibration-out', 'capture',
        '--attention-implementation', 'eager', '--seed=0'])
    # Execution normalizes these values only after its existing ordered gates.
    args.calibration_input_receipt = None
    args.capture_chain_ranges = [(0, 1)]
    args.capture_chain_boundary_storage = {'schema': 'owned-boundary-policy'}
    request = capture_chain_bookend_request(args)
    assert isinstance(request, CapturePrepRequest)
    assert request.seed == 0
    assert request.pinned_roster_only is False
    assert request.capture_chain_ranges == [(0, 1)]
    assert request.capture_chain_boundary_storage == {'schema': 'owned-boundary-policy'}
    args.capture_chain = 'join'
    assert capture_chain_bookend_request(args) == CaptureJoinRequest('census.json', 'capture')


def test_bookend_binding_refuses_quantum_instead_of_guessing_a_phase():
    args = parse_campaign_arguments(BASE + ['--capture-chain', 'quantum'])
    with pytest.raises(ValueError, match='prep or join'):
        capture_chain_bookend_request(args)
