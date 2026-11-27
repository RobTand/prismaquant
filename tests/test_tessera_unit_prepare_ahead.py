"""Check the preparation handoff and its owners."""

from collections import Counter, OrderedDict
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tessera.manifest import ScalePlaneKind

from prismaquant import cost_streaming, layer_streaming, memory_management, model_profiles
from prismaquant.io_engine import ENGINE
from prismaquant.tessera_row_stream import RowStream
from prismaquant.tessera_campaign_wire import require_checkpoint_identity_api
from prismaquant import tessera_calibration_cache as capture
from prismaquant import tessera_calibration_forward as forward
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_campaign_selection as selection
from prismaquant import tessera_hessian as hessian
from prismaquant import tessera_render as render


UNITS = ["model.layers.0.a", "model.layers.0.b", "model.layers.0.c", "model.layers.0.d"]
WIDTH, HEIGHT, ROWS = 256, 8, 4
FORMAT = "TESSERA_E4M3_K1_R1024"


@pytest.fixture
def guard(monkeypatch, tmp_path):
    cgroup = tmp_path / 'cgroup'
    cgroup.mkdir()
    (cgroup / 'memory.max').write_text(str(4 << 30))
    (cgroup / 'memory.current').write_text('0')
    (cgroup / 'memory.stat').write_text('anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n')
    membership = tmp_path / 'membership'
    membership.write_text('0::/\n')
    monkeypatch.setattr(memory_management, '_host_memory_info', lambda: (64 << 30, 128 << 30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *_: 0)
    monkeypatch.setenv('CACHE_HEADROOM_GB', '0')
    monkeypatch.setenv('PREFETCH_MIN_AVAILABLE_GB', '0')
    owner = memory_management.CaptureMemoryGuard('cpu', cgroup_root=cgroup, membership=membership)
    monkeypatch.setattr(memory_management, 'CaptureMemoryGuard', lambda _device: owner)
    return owner


@pytest.fixture
def row(monkeypatch, tmp_path, request):
    if not getattr(request, 'param', False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        request.getfixturevalue('guard')
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    profile = model_profiles.DefaultProfile()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(dict(
        model_type="llama", hidden_size=WIDTH, intermediate_size=WIDTH,
        num_hidden_layers=1, num_attention_heads=1)))
    rng = torch.Generator().manual_seed(2955)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    for name in UNITS:
        linear = torch.nn.Linear(WIDTH, HEIGHT, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(HEIGHT, WIDTH, generator=rng))
        setattr(layer, name.rsplit(".", 1)[-1], linear)
    model.lm_head = torch.nn.Linear(WIDTH, 32, bias=False, dtype=torch.bfloat16)
    model.config = SimpleNamespace(_attn_implementation="eager")
    save_file(model.state_dict(), str(source / "model.safetensors"))
    tokens, text = [torch.ones(1, WIDTH, dtype=torch.long)], "one draw"
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _: profile)
    monkeypatch.setattr(campaign, "_calibration_tokens", lambda *_: (tokens, text))
    monkeypatch.setattr(forward, "capture_activation_statistics",
                        lambda *a, **k: pytest.fail("The campaign repeats calibration."))
    menu = [SimpleNamespace(format_name=FORMAT, family="TESSERA_E4M3_K1",
                            body_rate_q256=1024, bpp=4.0)]
    monkeypatch.setattr(campaign, "expand_menus_for_targets",
                        lambda weights, targets, **k: dict.fromkeys(targets, menu))

    def build(*args, source_authentication=None, **kwargs):
        owned = {} if source_authentication is None else {"source_authentication": source_authentication}
        files, keys = layer_streaming._build_weight_map(str(source), **owned)

        def snapshot(names, **_):
            values = layer_streaming._read_layer_to_device(
                "model.layers.0.", files, keys, torch.bfloat16, torch.device("cpu"), **owned)
            return {name: values[name + ".weight"].clone() for name in names}, {
                "schema": "prismaquant.selected_source_weights.v1", "source_forward_count": 0}

        return SimpleNamespace(model=model, snapshot_selected_weights=snapshot, shutdown=lambda: None)

    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
    version = importlib.metadata.version("transformers")
    initialization = dict(schema="prismaquant.streaming_initialization.v1",
        scope="streamed_text_source_forward", status="completed", transformers_version=version,
        model_class="SyntheticSource", dtype="torch.bfloat16", layers_prefix="model.layers.",
        num_layers=1, persistent_tensors=5, derived_buffers=0,
        state_sha256="a" * 64, source_map_sha256="b" * 64)
    identity = hessian.calibration_identity(text, tokens, fit_tokens=ROWS,
        source="wikitext-2-raw-v1/train", split_role="calibration", model=str(source),
        seed=0, nsamples=32, seqlen=512, fit_tokens_min=ROWS)
    groups = campaign.resolve_anchor_groups(UNITS, profile=profile, expert_members={})
    counts = dict.fromkeys(UNITS, ROWS)
    rows = {name: torch.randn(ROWS, WIDTH, generator=rng) for name in UNITS}
    matrices = {name: x.T @ x + torch.eye(WIDTH) for name, x in rows.items()}
    maxima = {name: float(x.abs().max()) for name, x in rows.items()}
    census = campaign.calibration_census(counts, maxima,
        args=SimpleNamespace(model=str(source), nsamples=32, seqlen=512, seed=0, layer_stride=1),
        groups=groups, dense_targets=UNITS, expert_targets=[],
        shapes=dict.fromkeys(UNITS, [HEIGHT, WIDTH]), identity=identity,
        model_load_contract=initialization, attention_implementation="eager",
        capture_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=version))
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    canonical = capture.capture_identity(census_path, calibration=identity, max_act_rows=ROWS,
        model_load_contract=initialization, attention_implementation="eager")
    complete = capture.publish_capture(tmp_path / "capture", census_path=census_path,
        identity=canonical, acts=rows, hessians=matrices, counts=counts, maxima=maxima)
    selected = tmp_path / "units.json"
    selected.write_text(json.dumps(dict(schema=selection.UNITS_SCHEMA, groups=[
        dict(key=key, members=list(members)) for key, members in sorted(groups.items())])))
    size = 1024 ** 2
    return ["--model", str(source), "--out", str(tmp_path / "cost.pkl"),
        "--cache-dir", str(tmp_path / "cache"), "--checkpoint", str(tmp_path / "checkpoint.json"),
        "--hessian", "require", "--menu-mode", "research", "--max-rounds", "1",
        "--attention-implementation", "eager", "--streaming", "--streaming-cache-headroom-gb", "0",
        "--units", str(selected), "--calibration-census", str(census_path),
        "--calibration-cache", complete["path"], "--calibration-cache-sha256", complete["sha256"],
        "--nsamples", "32", "--seqlen", "512", "--layer-stride", "1", "--max-act-rows", str(ROWS),
        "--campaign-identity-threads", "2", "--capture-load-policy",
        json.dumps(dict(schema="prismaquant.verified_activation_load.v1",
                        max_buffer_bytes=size, max_scratch_bytes=size)),
        "--export-hessian-reference-policy", json.dumps(dict(schema="tessera.hessian_reference_load.v1",
            max_metadata_bytes=size, max_file_bytes=size, max_hessian_bytes=size))]


def test_next_preparation_overlaps_encode(row, guard, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_TESSERA_PREPARE_AHEAD", "1")
    encoding, preparing = threading.Event(), threading.Event()
    calls = Counter()
    factors, encodes, overlap = {}, [], []
    prepare = hessian.encoder_kwargs
    encode = render.encode_tessera_unit
    transfer = torch.Tensor.to
    measure = campaign.cell_measurement.measure_cells
    caller = threading.get_ident()
    staged, reservations, jobs = [], [], []
    schedule = getattr(RowStream, 'prepare_next', None)

    def observed_schedule(stream, *args):
        schedule(stream, *args)
        jobs.append(stream._preparation.future)

    if schedule is not None:
        monkeypatch.setattr(RowStream, 'prepare_next', observed_schedule)

    def observed_transfer(tensor, *args, **kwargs):
        if (threading.get_ident() == caller or not kwargs.get('non_blocking')
                or tensor.shape not in ((HEIGHT, WIDTH), (ROWS, WIDTH))):
            return transfer(tensor, *args, **kwargs)
        reservations.append(guard.last['future_allocation_bytes'])
        result = transfer(tensor, *args, **kwargs).clone()
        staged.append(result)
        return result

    def observed_measure(**kwargs):
        if kwargs['qnames'] == [UNITS[1]] and staged:
            assert kwargs['weights'][0] is staged[0]
            assert kwargs['activations'][0] is staged[1]
        return measure(**kwargs)

    def observed_prepare(source, name, *args, **kwargs):
        calls[name] += 1
        if name == UNITS[1]:
            assert encoding.wait(5), "The first encoder did not enter."
            preparing.set()
        factors[name] = prepare(source, name, *args, **kwargs)
        return factors[name]

    def observed_encode(weight, fmt, **kwargs):
        name = UNITS[len(encodes)]
        encodes.append(name)
        if name == UNITS[0]:
            encoding.set()
            overlap.append(preparing.wait(5))
            if overlap[-1]:
                jobs[0].result(timeout=30)
        assert kwargs["activation_kwargs"] is factors[name]
        return encode(weight, fmt, **kwargs)

    monkeypatch.setattr(hessian, "encoder_kwargs", observed_prepare)
    monkeypatch.setattr(render, "encode_tessera_unit", observed_encode)
    monkeypatch.setattr(torch.Tensor, 'to', observed_transfer)
    monkeypatch.setattr(campaign.cell_measurement, 'measure_cells', observed_measure)
    assert campaign.main(row) == 0
    assert overlap == [True], "The next preparation did not start during encode."
    assert len(staged) >= 2
    payload = pickle.loads(Path(row[row.index('--out') + 1]).read_bytes())
    plan = payload['provenance']['selected_source_preparation']['resources']
    window = plan['stream_phases']['stream_window']
    reserved = sum(window[key] for key in ('ahead_input_bytes', 'ahead_factor_bytes',
        'ahead_transfer_scratch_bytes', 'ahead_factorization_scratch_bytes'))
    assert min(reservations) >= reserved > 0
    assert encodes == UNITS
    assert calls == dict.fromkeys(UNITS, 1)


@pytest.fixture
def slot():
    def build(device='cpu', check=None):
        assert callable(getattr(RowStream, 'prepare_next', None))
        stream = object.__new__(RowStream)
        matrix = torch.eye(WIDTH)
        identity = hessian.calibration_identity('slot', [torch.zeros(1, ROWS)], fit_tokens=ROWS)
        source = hessian.activation_source({'a': matrix}, identity)
        weights = torch.ones(HEIGHT, WIDTH, pin_memory=device == 'cuda')
        rows = torch.ones(ROWS, WIDTH, pin_memory=device == 'cuda')
        stream._weights = {'a': weights}
        stream._live = {'a': SimpleNamespace(source=source, inputs=rows, hessian=matrix,
            identities={'hessian': require_checkpoint_identity_api().tensor_identity(matrix)})}
        stream._hessian_identity = identity
        stream.capture = {'sha256': 'capture'}
        stream._device = device
        stream.prepare_ahead_bytes = 5 * matrix.numel() * matrix.element_size()
        stream._resource_check = check
        stream._memo_capacity = 1
        stream._memo = OrderedDict()
        stream._inflight = {}
        stream._preparation = None
        return stream
    return build


@pytest.mark.parametrize('change', ['capture', 'source', 'unit', 'H content', 'H shape', 'shape', 'plane'])
def test_prepared_input_refuses_drift(slot, change):
    stream = slot()
    entry = stream.entry('a')
    # Use a memo hit. It must not bypass the live Hessian check.
    factors = stream.encoder_kwargs('a', ScalePlaneKind.CHANNEL)
    stream.prepare_next('a', ScalePlaneKind.CHANNEL)
    try:
        stream._preparation.future.result(timeout=30)
        if change == 'capture':
            stream.capture['sha256'] = 'other capture'
        if change == 'source':
            entry.source = hessian.activation_source({'a': entry.hessian}, stream._hessian_identity)
        if change == 'H content':
            entry.hessian.data[0, 0] = 7
        if change == 'H shape':
            entry.hessian.resize_(WIDTH, WIDTH // 2)
        if change == 'shape':
            entry.inputs.resize_(1, ROWS * WIDTH)
        match = {'capture': 'source', 'source': 'source', 'unit': 'unit key',
                 'H content': 'Hessian', 'H shape': 'Hessian', 'shape': 'inputs', 'plane': 'plane'}
        with pytest.raises(render.HessianContractError, match=match[change]):
            stream.take_prepared('b' if change == 'unit' else 'a')
            stream.encoder_kwargs('a', ScalePlaneKind.LUT)
        assert stream._memo['a', ScalePlaneKind.CHANNEL] is factors
    finally:
        stream.release_prepared()


def test_slot_keeps_one_job_until_release(slot, monkeypatch):
    from concurrent.futures import Future

    queued = []
    reserve = []
    stream = slot(check=lambda label, *, reserve_bytes: reserve.append((label, reserve_bytes)))

    def submit(fn):
        assert reserve == [('before_row_stream_prepare_ahead', stream.prepare_ahead_bytes)]
        future = Future()
        queued.append((fn, future))
        return future

    monkeypatch.setattr(ENGINE, 'submit', submit)
    stream.prepare_next('a', ScalePlaneKind.CHANNEL)
    try:
        stream.prepare_next('a', ScalePlaneKind.CHANNEL)
        assert len(queued) == 1
        fn, future = queued[0]
        assert future.set_running_or_notify_cancel()
        stream.prepare_next('a', ScalePlaneKind.CHANNEL)
        assert len(queued) == 1
        future.set_result(fn())
        stream.prepare_next('a', ScalePlaneKind.CHANNEL)
        assert len(queued) == 1
        weight, rows = stream.take_prepared('a')
        assert weight is stream._weights['a']
        assert rows is stream.entry('a').inputs
        assert stream.encoder_kwargs('a', ScalePlaneKind.CHANNEL) is future.result()[-2]
        owner = stream._preparation
    finally:
        if len(queued) == 1 and queued[0][1].running():
            queued[0][1].set_exception(RuntimeError('The test did not finish the job.'))
        stream.release_prepared()
    assert stream._preparation is None
    assert owner.result is owner.future is None
    assert reserve[-1] == ('after_row_stream_prepare_ahead', 0)


def test_campaign_refuses_failed_preparation(row, monkeypatch):
    assert callable(getattr(RowStream, 'prepare_next', None))
    monkeypatch.setenv('PRISMAQUANT_TESSERA_PREPARE_AHEAD', '1')
    started, encoding = threading.Event(), threading.Event()
    prepare, encode = hessian.encoder_kwargs, render.encode_tessera_unit
    calls = []

    def fail(source, name, *args, **kwargs):
        if name == UNITS[1]:
            assert encoding.wait(30), 'The first encoder did not enter.'
            started.set()
            raise RuntimeError('Injected preparation failure')
        return prepare(source, name, *args, **kwargs)

    def observe(*args, **kwargs):
        calls.append(1)
        encoding.set()
        assert started.wait(30), 'Preparation did not start.'
        return encode(*args, **kwargs)

    monkeypatch.setattr(hessian, 'encoder_kwargs', fail)
    monkeypatch.setattr(render, 'encode_tessera_unit', observe)
    with pytest.raises(RuntimeError, match='Injected preparation failure'):
        campaign.main(row)
    assert calls == [1]


@pytest.mark.parametrize('exit_kind', ['error', 'deadline', 'close'])
def test_campaign_joins_preparation_before_release(row, guard, monkeypatch, exit_kind):
    import concurrent.futures
    assert callable(getattr(RowStream, 'release_prepared', None))
    monkeypatch.setenv('PRISMAQUANT_TESSERA_PREPARE_AHEAD', '1')
    started, allow, finished = [threading.Event() for _ in range(3)]
    wait = concurrent.futures.wait
    draining = []

    def join_worker(futures, *args, **kwargs):
        if draining and draining[0] in futures:
            allow.set()
        return wait(futures, *args, **kwargs)

    monkeypatch.setattr(concurrent.futures, 'wait', join_worker)
    prepare, encode = hessian.encoder_kwargs, render.encode_tessera_unit
    release, close_entry = RowStream.release_prepared, RowStream._close_entry
    clock = [0.0]
    drained = []
    first = True
    if exit_kind == 'deadline':
        row += ['--deadline-seconds', '1']
        monkeypatch.setattr(campaign, 'time', SimpleNamespace(
            time=lambda: clock[0], monotonic=time.monotonic, perf_counter=time.perf_counter))

    def block(source, name, *args, **kwargs):
        if name != UNITS[1] or started.is_set():
            return prepare(source, name, *args, **kwargs)
        started.set()
        assert allow.wait(30), 'Cleanup did not release the worker.'
        try:
            return prepare(source, name, *args, **kwargs)
        finally:
            finished.set()

    def stop(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            assert started.wait(30), 'Preparation did not start.'
            if exit_kind == 'error':
                raise ValueError('Injected encode failure')
            if exit_kind == 'close':
                raise render.HessianContractError('Injected encode refusal')
            clock[0] = 2.0
        return encode(*args, **kwargs)

    def join(stream):
        owner = stream._preparation
        if not started.is_set() or owner is None or owner.name != UNITS[1]:
            return release(stream)
        assert not finished.is_set(), 'The worker finished before cleanup.'
        assert owner.future.running()
        assert guard.last['future_allocation_bytes'] >= stream.prepare_ahead_bytes
        draining.append(owner.future)
        release(stream)
        assert finished.is_set(), 'Cleanup did not join the worker.'
        assert stream._preparation is None
        assert owner.future is owner.result is None
        assert guard.last['future_allocation_bytes'] == 0
        drained.append(True)
        draining.clear()

    def check_entry(stream, entry):
        if entry.name == UNITS[1] and started.is_set():
            assert finished.is_set(), 'Cleanup released the entry before it joined.'
        close_entry(stream, entry)

    monkeypatch.setattr(hessian, 'encoder_kwargs', block)
    monkeypatch.setattr(render, 'encode_tessera_unit', stop)
    monkeypatch.setattr(RowStream, 'release_prepared', join)
    monkeypatch.setattr(RowStream, '_close_entry', check_entry)
    try:
        if exit_kind == 'deadline':
            assert campaign.main(row) == 0
        else:
            with pytest.raises(RuntimeError, match='Injected encode'):
                campaign.main(row)
        assert drained == [True]
    finally:
        allow.set()


@pytest.mark.parametrize('batch_size', [1, 2])
def test_prepared_and_serial_campaigns_match(row, monkeypatch, tmp_path, batch_size):
    def snapshot(args):
        def arg(flag):
            return Path(args[args.index(flag) + 1])
        payload = pickle.loads(arg('--out').read_bytes())
        costs = payload['costs']
        for entries in costs.values():
            for cost in entries.values():
                cost.pop('encode_seconds', None)
        wires = {path.name: path.read_bytes()
                 for path in (arg('--cache-dir') / 'wire').glob('*.tessera')}
        manifest = json.loads(arg('--checkpoint').read_text())
        assert wires
        return costs, wires, manifest['identity']

    row += ['--anchor-batch-size', str(batch_size)]
    monkeypatch.setenv('PRISMAQUANT_TESSERA_PREPARE_AHEAD', '0')
    assert campaign.main(row) == 0
    expected = snapshot(row)
    candidate = list(row)
    for flag, name in [('--out', 'ahead.pkl'), ('--cache-dir', 'ahead'),
                       ('--checkpoint', 'ahead.json')]:
        candidate[candidate.index(flag) + 1] = str(tmp_path / name)
    monkeypatch.setenv('PRISMAQUANT_TESSERA_PREPARE_AHEAD', '1')
    assert campaign.main(candidate) == 0
    assert snapshot(candidate) == expected

    def unexpected_encode(*args, **kwargs):
        pytest.fail('Resume encoded a unit again.')

    monkeypatch.setattr(render, 'encode_tessera_unit', unexpected_encode)
    monkeypatch.setattr(render, 'encode_tessera_units', unexpected_encode)
    assert campaign.main(candidate) == 0
    assert snapshot(candidate) == expected


def test_scalar_refuses_incompatible_served_recipe(monkeypatch, tmp_path):
    from prismaquant.tessera_formats import TesseraFormatError

    monkeypatch.setattr(render, 'encode_tessera_unit',
                        lambda *a, **k: pytest.fail('The encoder used an incompatible recipe.'))
    with pytest.raises(TesseraFormatError, match='structure'):
        campaign.cell_measurement.measure_cells(qnames=['a'], weights=[torch.ones(HEIGHT, WIDTH)],
            activations=[torch.ones(ROWS, WIDTH)], format_name=FORMAT, cache=None,
            wire_dir=tmp_path, structures={'a': 'incompatible'})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='This test needs CUDA.')
@pytest.mark.parametrize('consume', [True, False], ids=['encode', 'unused'])
def test_cuda_prepared_storage_and_events(slot, monkeypatch, consume):
    checks, waits, records = [], [], []
    stream = slot('cuda', check=lambda label, *, reserve_bytes: checks.append(reserve_bytes))
    wait_event = torch.cuda.Stream.wait_event
    record_event = torch.cuda.Stream.record_event
    synchronize = torch.cuda.Stream.synchronize
    consumer = torch.cuda.Stream()
    input_stream = torch.cuda.Stream()
    checked = []

    def observe_wait(owner, event):
        waits.append((owner, event))
        wait_event(owner, event)

    def observe_record(owner, event=None):
        actual = record_event(owner, event)
        records.append((owner, actual))
        return actual

    def observe_sync(owner):
        pending = stream._preparation
        if pending is not None and owner in (pending.stream, pending.consumer):
            assert pending.future.done()
            result = pending.future.result()
            assert result[0] is stream.entry('a')
            assert result[-2]['ldl'].is_cuda
            assert checks[-1] == stream.prepare_ahead_bytes
            assert result[6].is_cuda and result[7].is_cuda
            checked.append(owner)
        synchronize(owner)

    monkeypatch.setattr(torch.cuda.Stream, 'wait_event', observe_wait)
    monkeypatch.setattr(torch.cuda.Stream, 'record_event', observe_record)
    monkeypatch.setattr(torch.cuda.Stream, 'synchronize', observe_sync)
    monkeypatch.setattr(torch.cuda, 'synchronize',
                        lambda *a, **k: pytest.fail('Preparation synchronized the device.'))
    with torch.cuda.stream(input_stream):
        stream._weights['a'] = torch.ones(HEIGHT, WIDTH, device='cuda')
        stream.prepare_next('a', ScalePlaneKind.CHANNEL)
    owner = stream._preparation
    try:
        owner.future.result(timeout=30)
        assert (owner.stream, dict(records)[input_stream]) in waits
        if consume:
            with torch.cuda.stream(consumer):
                weight, rows = stream.take_prepared('a')
                kwargs = stream.encoder_kwargs('a', ScalePlaneKind.CHANNEL)
                assert (consumer, dict(records)[owner.stream]) in waits
                encoded = weight @ rows.T
                finite = torch.isfinite(kwargs['ldl']).all()
                del weight, rows, kwargs
        else:
            assert owner.result is None
    finally:
        stream.release_prepared()
    assert checked == ([owner.stream, consumer] if consume else [owner.stream])
    if consume:
        assert torch.equal(encoded.cpu(), torch.full((HEIGHT, ROWS), float(WIDTH)))
        assert finite.item()
    assert checks[-1] == 0
    assert stream._preparation is None
    assert owner.future is owner.result is None


def test_profile_campaign_encode(row, monkeypatch, tmp_path):
    phase = os.environ.get('TL3B_PREPARE_AHEAD_PROFILE',
                           'after' if hasattr(RowStream, 'prepare_next') else 'before')
    output = Path(os.environ.get('TL3B_OUT', str(tmp_path)))
    monkeypatch.setenv('PRISMAQUANT_TESSERA_PREPARE_AHEAD', '1' if phase == 'after' else '0')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        assert campaign.main(row) == 0
    profile.export_chrome_trace(str(output / f'prepare-ahead-{phase}.json'))
