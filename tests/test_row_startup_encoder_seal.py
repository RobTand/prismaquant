"""Overlap the producer's exact cold source seal, never replace it (PQ #1742)."""

from concurrent.futures import Future
from contextlib import ExitStack
from functools import lru_cache
from types import SimpleNamespace

import pytest

from prismaquant import io_engine
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_campaign_wire as campaign_wire


class ControlledSubmit:
    def __init__(self):
        self.future = Future()
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return self.future

    def run(self):
        fn, args = self.calls[0]
        try:
            self.future.set_result(fn(*args))
        except BaseException as error:
            self.future.set_exception(error)


def _start(monkeypatch, *, census=False, capture=False):
    submit = ControlledSubmit()
    monkeypatch.setattr(io_engine, "ENGINE", submit)
    scope = ExitStack()
    args = SimpleNamespace(census_out="census.json" if census else None,
                           capture_calibration_out="capture" if capture else None)
    ahead = campaign._start_encoder_source_seal_ahead(args, scope)
    return ahead, submit, scope


def test_start_submits_without_waiting_and_uses_the_same_producer_seal(monkeypatch):
    calls = []
    digest = "f" * 64

    @lru_cache(maxsize=1)
    def source_seal():
        calls.append("producer")
        return digest

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api',
                        lambda: SimpleNamespace(encoder_source_sha256=source_seal))
    ahead, submit, scope = _start(monkeypatch)
    assert ahead is not None
    assert len(submit.calls) == 1
    assert not submit.future.done()  # startup has not blocked on the seal
    assert calls == []
    submit.run()
    assert ahead.wait() == digest
    # The unchanged run-identity call and producer bind now see their own cache.
    assert campaign_wire.require_checkpoint_identity_api().encoder_source_sha256() == digest
    assert calls == ["producer"]
    scope.close()


@pytest.mark.parametrize("census,capture", [(True, False), (False, True), (True, True)])
def test_nonpricing_heads_do_not_start_an_unused_source_hash(monkeypatch, census, capture):
    ahead, submit, scope = _start(monkeypatch, census=census, capture=capture)
    assert ahead is None
    assert submit.calls == []
    scope.close()


def test_source_seal_failure_reaches_identity_and_cleanup_does_not_mask_it(monkeypatch):
    failure = ValueError("producer source seal failed")

    def fail():
        raise failure

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api',
                        lambda: SimpleNamespace(encoder_source_sha256=fail))
    ahead, submit, scope = _start(monkeypatch)
    assert ahead is not None
    submit.run()
    with pytest.raises(ValueError, match="producer source seal failed") as seen:
        ahead.wait()
    assert seen.value is failure
    scope.close()  # drain, without replacing an earlier error during unwind


def test_pricing_scopes_have_no_shared_future(monkeypatch):
    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api',
                        lambda: SimpleNamespace(encoder_source_sha256=lambda: "a" * 64))
    first, a, scope_a = _start(monkeypatch)
    assert first is not None
    a.run()
    second, b, scope_b = _start(monkeypatch)
    assert second is not None
    assert a.future is not b.future
    b.run()
    assert first.wait() == second.wait() == "a" * 64
    scope_a.close()
    scope_b.close()


@pytest.mark.parametrize('row_head,identity_bytes', [
    ('load-all', 0), ('load-all', 1 << 20), ('stream', 0)])
def test_main_joins_the_encoder_seal_before_identity_consumers(
        monkeypatch, tmp_path, row_head, identity_bytes):
    from prismaquant import autoscale, cost_streaming
    from test_tessera_row_stream import stream_fixture

    campaign, argv, _state = stream_fixture(monkeypatch, tmp_path)
    events = []
    ahead_started = False
    joined = False
    start = campaign._start_encoder_source_seal_ahead
    wait = campaign._EncoderSourceSealAhead.wait
    build = cost_streaming.build_streamed_causal_lm
    identity = campaign.campaign_checkpoint_identity
    metadata_plan = campaign._campaign_identity_metadata_plan
    bind_identities = campaign._campaign_bound_identities
    resources = autoscale.selected_anchor_resources

    def resources_with_identity_hold(*args, **kwargs):
        result = resources(*args, **kwargs)
        result['phases']['resident_anchors']['campaign_identity_hold_scratch_bytes'] = 0
        return result

    def observed_start(*args, **kwargs):
        nonlocal ahead_started
        ahead = start(*args, **kwargs)
        assert ahead is not None
        ahead_started = True
        events.append('submitted')
        return ahead

    def observed_build(*args, **kwargs):
        assert ahead_started
        assert not joined
        events.append('model')
        return build(*args, **kwargs)

    def observed_wait(self):
        nonlocal joined
        result = wait(self)
        joined = True
        events.append('joined')
        return result

    def observed_identity(*args, **kwargs):
        assert joined
        result = identity(*args, **kwargs)
        events.append(('identity', result['encoder_source_sha256']))
        return result

    def observed_metadata_plan(*args, **kwargs):
        assert joined
        events.append('metadata_plan')
        return metadata_plan(*args, **kwargs)

    def observed_bind_identities(*args, **kwargs):
        assert joined
        events.append('bound_identities')
        return bind_identities(*args, **kwargs)

    monkeypatch.setattr(campaign, '_start_encoder_source_seal_ahead', observed_start)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', observed_build)
    monkeypatch.setattr(campaign._EncoderSourceSealAhead, 'wait', observed_wait)
    monkeypatch.setattr(campaign, 'campaign_checkpoint_identity', observed_identity)
    monkeypatch.setattr(campaign, '_campaign_identity_metadata_plan', observed_metadata_plan)
    monkeypatch.setattr(campaign, '_campaign_bound_identities', observed_bind_identities)
    monkeypatch.setattr(autoscale, 'selected_anchor_resources', resources_with_identity_hold)
    assert campaign.main([
        *argv, '--row-head', row_head,
        '--campaign-identity-bytes', str(identity_bytes)]) == 0
    assert events[:3] == ['submitted', 'model', 'joined']
    assert events.count('metadata_plan') == int(row_head == 'stream' or identity_bytes > 0)
    assert events.count('bound_identities') == int(row_head == 'load-all' and identity_bytes > 0)
    identities = [item[1] for item in events if isinstance(item, tuple)]
    assert identities
    assert set(identities) == {
        campaign_wire.require_checkpoint_identity_api().encoder_source_sha256()}
