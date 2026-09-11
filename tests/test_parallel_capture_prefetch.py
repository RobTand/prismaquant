"""N readers load the verified capture; one consumer keeps the serial contract."""
import json
import threading

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import canonical_fields, identity as build_identity

UNITS = tuple(f'unit-{index:02d}' for index in range(12))


def policy():
    return dict(schema='prismaquant.verified_activation_load.v1',
                max_buffer_bytes=4*1024**2, max_scratch_bytes=1024**2)


def build_roster(tmp_path, units):
    """A capture wide enough that readers and the consumer actually overlap."""
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    (source/'model.safetensors').write_bytes(b'bounded source fixture')
    census = dict(model=str(source), counts={n: 5 for n in units},
                  max_abs={n: 4. for n in units},
                  unit_shapes={n: [3, 2] for n in units}, layer_stride=1,
                  anchor_groups={f'u:{n}': [n] for n in units}, **canonical_fields())
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    capture_id = build_identity(census_path, calibration={'fit_ids_sha256': 'draw'},
                                max_act_rows=2)
    acts = {n: torch.tensor([[1., 2.], [3., float(index)]]) for index, n in enumerate(units)}
    hessians = {n: torch.eye(2)*(index+1) for index, n in enumerate(units)}
    root = tmp_path/'capture'
    record = cc.publish_capture(root, census_path=census_path, identity=capture_id,
                                acts=acts, hessians=hessians, counts=census['counts'],
                                maxima=census['max_abs'])
    return root, census, capture_id, acts, hessians, record, tuple(units)


@pytest.fixture
def roster(tmp_path):
    return build_roster(tmp_path, UNITS)


@pytest.fixture
def wide_roster(tmp_path):
    """Enough entries that the in-flight window turns over many times."""
    return build_roster(tmp_path, tuple(f'wide-{index:03d}' for index in range(64)))


def prefetch(roster, *, threads, monkeypatch, resource_check=None, names=None,
             release_file_pages=False):
    root, census, capture_id, _acts, _hessians, record, units = roster
    monkeypatch.setenv('PRISMAQUANT_CAPTURE_READ_THREADS', str(threads))
    execution = {}
    values, receipt = cc.prefetch_capture(record['path'], expected_identity=capture_id,
        census=census, names=list(units if names is None else names), device='cpu',
        expected_sha256=record['sha256'], verified_load_policy=policy(),
        load_execution=execution, resource_check=resource_check,
        release_file_pages=release_file_pages)
    return values, receipt, execution


def test_reader_count_is_one_unless_the_environment_asks(monkeypatch):
    monkeypatch.delenv('PRISMAQUANT_CAPTURE_READ_THREADS', raising=False)
    assert cc.capture_read_threads() == 1
    monkeypatch.setenv('PRISMAQUANT_CAPTURE_READ_THREADS', 'eight')
    assert cc.capture_read_threads() == 1
    monkeypatch.setenv('PRISMAQUANT_CAPTURE_READ_THREADS', '0')
    assert cc.capture_read_threads() == 1
    monkeypatch.setenv('PRISMAQUANT_CAPTURE_READ_THREADS', '8')
    assert cc.capture_read_threads() == 8


@pytest.mark.parametrize('threads', [4, 8])
def test_parallel_readers_reproduce_the_serial_load_execution(roster, monkeypatch, threads):
    serial_values, serial_receipt, serial = prefetch(roster, threads=1, monkeypatch=monkeypatch)
    values, receipt, execution = prefetch(roster, threads=threads, monkeypatch=monkeypatch)
    assert execution == serial
    assert execution['loaded_entries'] == len(UNITS)
    assert receipt == serial_receipt
    assert list(values[0]) == list(serial_values[0]) == sorted(UNITS)
    for name in UNITS:
        assert torch.equal(values[0][name], serial_values[0][name])
        assert torch.equal(values[1][name], serial_values[1][name])
    assert (values[2], values[3]) == (serial_values[2], serial_values[3])


def test_ordered_identity_is_the_name_order_not_the_completion_order(roster, monkeypatch):
    """A reader that finishes late must not move its link in the chain."""
    original = cc._verified_capture_entry
    delay = {name: 0.02*(len(UNITS)-index) for index, name in enumerate(sorted(UNITS))}

    def slowed(path, name, **kwargs):
        import time
        time.sleep(delay[name])
        return original(path, name, **kwargs)

    monkeypatch.setattr(cc, '_verified_capture_entry', slowed)
    _values, _receipt, execution = prefetch(roster, threads=8, monkeypatch=monkeypatch)
    monkeypatch.undo()
    _serial_values, _serial_receipt, serial = prefetch(roster, threads=1, monkeypatch=monkeypatch)
    assert execution['ordered_load_identities_sha256'] == serial['ordered_load_identities_sha256']


def test_merge_load_execution_is_not_a_substitute_for_the_fold(roster, monkeypatch):
    """The chain is not associative over partials; the fold is the contract."""
    _values, _receipt, serial = prefetch(roster, threads=1, monkeypatch=monkeypatch)
    root, census, capture_id, _a, _h, _record, _units = roster
    total = cc._load_execution(policy(), capture_id, None)
    for name in sorted(UNITS):
        partial = cc._load_execution(policy(), capture_id, None)
        _payload, entry = cc._verified_capture_entry(root/f'inputs/{name}.pt', name,
            expected_sha256=cc.sha256(root/f'inputs/{name}.pt'), census=census,
            max_rows=capture_id['max_act_rows'], policy=policy(), execution=partial)
        cc.merge_load_execution(total, partial)
        del _payload, entry
    assert total['loaded_entries'] == serial['loaded_entries']
    assert total['ordered_load_identities_sha256'] != serial['ordered_load_identities_sha256']


def test_per_unit_guard_calls_keep_the_serial_order_and_bracket_their_reads(roster, monkeypatch):
    observed = []
    lock = threading.Lock()

    def resource_check(label, *, reserve_bytes=0):
        with lock:
            observed.append((label, reserve_bytes))

    prefetch(roster, threads=8, monkeypatch=monkeypatch, resource_check=resource_check,
             release_file_pages=True)
    outer = [label for label, _ in observed if label.startswith(('before_capture_prefetch:',
                                                                'after_capture_prefetch:'))]
    expected = []
    for name in sorted(UNITS):
        expected += [f'before_capture_prefetch:{name}', f'after_capture_prefetch:{name}']
    assert outer == expected
    # Each unit's own file work finishes before the consumer charges that unit.
    labels = [label for label, _ in observed]
    for name in sorted(UNITS):
        inner = [index for index, label in enumerate(labels) if label.endswith(f':{name}.pt')]
        charge = labels.index(f'before_capture_prefetch:{name}')
        assert inner and max(inner) < charge, (name, inner, charge)
    # The consumer's own charge is at least the serial reservation; readers
    # that are still live add theirs on top, which is the point of the sum.
    floor = {name: 2*cc._capture_storage_bytes(name, roster[1], roster[2]['max_act_rows'])
             for name in UNITS}
    for label, reserve in observed:
        if label.startswith('before_capture_prefetch:'):
            assert reserve >= floor[label.split(':', 1)[1]]


def test_concurrent_reservations_are_charged_together(roster, monkeypatch):
    """Two live readers present the sum, so the guard prices the real peak."""
    seen = []
    guard = cc._ConcurrentReservation(lambda label, *, reserve_bytes=0:
                                      seen.append((label, reserve_bytes)))
    ready, go = threading.Event(), threading.Event()

    def other():
        guard.check('other', reserve_bytes=1000)
        ready.set()
        go.wait(5)
        guard.release()

    worker = threading.Thread(target=other)
    worker.start()
    ready.wait(5)
    guard.check('mine', reserve_bytes=7)
    go.set()
    worker.join(5)
    assert ('mine', 1007) in seen
    guard.check('after-release', reserve_bytes=7)
    assert seen[-1] == ('after-release', 7)


def test_a_refusal_leaves_the_previous_reservation_in_place():
    charged = []

    def refusing(label, *, reserve_bytes=0):
        charged.append(reserve_bytes)
        if label == 'refused':
            raise RuntimeError('capture budget refused')
        return None

    guard = cc._ConcurrentReservation(refusing)
    guard.check('admitted', reserve_bytes=100)
    with pytest.raises(RuntimeError, match='budget refused'):
        guard.check('refused', reserve_bytes=900)
    guard.check('again', reserve_bytes=0)
    assert charged == [100, 900, 0]


def test_one_corrupt_entry_refuses_and_keeps_nothing_resident(roster, monkeypatch):
    import gc
    import weakref
    root = roster[0]
    target = root/'inputs/unit-07.pt'
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0xFF
    target.write_bytes(bytes(raw))
    loaded = []
    original = cc._validate_tensors

    def watched(name, payload, *args, **kwargs):
        x, h = original(name, payload, *args, **kwargs)
        loaded.append((weakref.ref(x), weakref.ref(h)))
        return x, h

    monkeypatch.setattr(cc, '_validate_tensors', watched)
    with pytest.raises(RuntimeError, match='checksum|identity|storage|geometry|owners'):
        prefetch(roster, threads=8, monkeypatch=monkeypatch)
    monkeypatch.undo()
    gc.collect()
    assert loaded, 'the refusal happened before any entry was decoded'
    assert all(reference() is None for pair in loaded for reference in pair)


def test_a_refusing_guard_does_not_wedge_the_readers(roster, monkeypatch):
    """A consumer that never returns its slots must not leave readers waiting."""
    calls = {'n': 0}

    def resource_check(label, *, reserve_bytes=0):
        calls['n'] += 1
        if label.startswith('before_capture_prefetch:unit-02'):
            raise RuntimeError('capture budget refused mid-prefetch')

    done = threading.Event()
    error = []

    def run():
        try:
            prefetch(roster, threads=4, monkeypatch=monkeypatch, resource_check=resource_check)
        except BaseException as failure:  # noqa: BLE001 - recorded for the assertion
            error.append(failure)
        finally:
            done.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert done.wait(60), 'parallel prefetch did not return after a refusal'
    worker.join(10)
    assert not worker.is_alive()
    assert error and 'refused mid-prefetch' in str(error[0])


@pytest.mark.parametrize('threads', [2, 3])
def test_a_narrow_window_never_wedges_under_a_racy_reader(wide_roster, monkeypatch, threads):
    """The in-flight window must not depend on which reader wakes first.

    A window held by the READERS -- each taking a permit the single consumer
    returns after it consumes an entry -- has no liveness bound: permits are
    fungible and granted in wakeup order, so one worker can complete several
    later entries (each holding its permit until consumed) while the entry the
    consumer is actually waiting for sits in another worker that never wins a
    permit. With T permits and T workers, T completed-unconsumed entries wedge
    the load permanently. The window therefore lives on the CONSUMER: it
    submits entry i+T only after consuming entry i, so the entry it wants next
    is always already running and no reader ever waits.

    A narrow window, many entries, a jittery reader and a slow consumer is the
    shape that exercises the turnover.
    """
    import random
    import time
    read_original = cc._verified_capture_entry
    validate_original = cc._validate_tensors

    def jittered(path, name, **kwargs):
        time.sleep(random.uniform(0, 0.003))
        return read_original(path, name, **kwargs)

    def slowed(name, payload, *args, **kwargs):
        time.sleep(0.002)
        return validate_original(name, payload, *args, **kwargs)

    monkeypatch.setattr(cc, '_verified_capture_entry', jittered)
    monkeypatch.setattr(cc, '_validate_tensors', slowed)
    for attempt in range(4):
        done, failure = threading.Event(), []

        def run():
            try:
                prefetch(wide_roster, threads=threads, monkeypatch=monkeypatch)
            except BaseException as error:  # noqa: BLE001 - reported by the assertion
                failure.append(error)
            finally:
                done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert done.wait(60), f'parallel prefetch wedged at threads={threads}, attempt {attempt}'
        worker.join(10)
        assert not failure, failure


def test_serial_default_leaves_the_existing_path_in_place(roster, monkeypatch):
    entered = []
    monkeypatch.setattr(cc, '_parallel_prefetch_capture',
                        lambda *a, **k: entered.append(True))
    monkeypatch.delenv('PRISMAQUANT_CAPTURE_READ_THREADS', raising=False)
    root, census, capture_id, acts, _hessians, record, _units = roster
    execution = {}
    values, _ = cc.prefetch_capture(record['path'], expected_identity=capture_id,
        census=census, names=list(UNITS), device='cpu', expected_sha256=record['sha256'],
        verified_load_policy=policy(), load_execution=execution)
    assert entered == []
    assert torch.equal(values[0]['unit-00'], acts['unit-00'])
