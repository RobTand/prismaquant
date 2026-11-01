"""Characterize both spies against factories frozen from f4d9fa1d3c16."""
from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
import inspect

import pytest

from prismaquant import digests
from prismaquant import production_weight_cache as cache
import test_pwc_source_framing_1764 as pwc_framing
import test_source_framing_1752 as source_framing


def _frozen_source_spy(module, monkeypatch):
    batches = []

    class RecordingProfile:
        def __init__(self):
            self.owner = digests.LengthFramedSourceSha256()
            self.rows = []
            batches.append(self.rows)

        def update(self, name, payload):
            self.rows.append((name, payload))
            self.owner.update(name, payload)

        def hexdigest(self):
            return self.owner.hexdigest()

    monkeypatch.setattr(module, "LengthFramedSourceSha256", RecordingProfile, raising=False)
    return batches


def _frozen_pwc_spy(monkeypatch):
    batches = []

    class RecordingProfile:
        def __init__(self):
            self.owner = digests.LengthFramedSourceSha256()
            self.rows = []
            batches.append(self.rows)

        def update(self, name, raw):
            self.rows.append((name, raw))
            self.owner.update(name, raw)

        def hexdigest(self):
            return self.owner.hexdigest()

    monkeypatch.setattr(cache, "LengthFramedSourceSha256", RecordingProfile, raising=False)
    return batches


@pytest.fixture(params=[
    (module_name, version)
    for module_name in (*source_framing.MODULES, cache.__name__)
    for version in ("frozen", "shared")
])
def recording_case(request):
    module_name, version = request.param
    module = import_module(module_name)
    if module is cache:
        factory = _frozen_pwc_spy if version == "frozen" else pwc_framing._spy

        def install(patch):
            return factory(patch)

        keyword = "raw"
    else:
        factory = _frozen_source_spy if version == "frozen" else source_framing._spy

        def install(patch):
            return factory(module, patch)

        keyword = "payload"
    return module, install, keyword


def test_recording_grouping_identity_order_and_returns(recording_case, monkeypatch):
    module, install, keyword = recording_case
    owners = []
    result = object()
    update_result = object()

    class Owner:
        def __init__(self):
            assert len(batches) == len(owners)
            self.index = len(owners)
            self.received = []
            owners.append(self)

        def update(self, name, payload):
            recorded = batches[self.index][-1]
            assert recorded[0] is name
            assert recorded[1] is payload
            self.received.append(recorded)
            return update_result

        def hexdigest(self):
            return result

    batches = install(monkeypatch)
    monkeypatch.setattr(digests, "LengthFramedSourceSha256", Owner)
    first = module.LengthFramedSourceSha256()
    second = module.LengthFramedSourceSha256()
    assert batches == [[], []]
    assert batches[0] is first.rows and batches[1] is second.rows
    assert first.rows is not second.rows
    assert first.owner is owners[0] and second.owner is owners[1]
    name = object()
    payload = bytearray(b"\x00\xff\r\n")
    other_name = object()
    other_payload = memoryview(b"other")
    assert first.update(name, payload) is None
    assert second.update(name=other_name, **{keyword: other_payload}) is None
    assert first.update(name, payload) is None
    assert batches == [owners[0].received, owners[1].received]
    assert len(batches[0]) == 2 and len(batches[1]) == 1
    assert first.hexdigest() is result
    assert first.hexdigest() is result
    assert second.hexdigest() is result
    assert len(batches[0]) == 2 and len(batches[1]) == 1


@pytest.mark.parametrize("stage", ("constructor", "update", "hexdigest"))
def test_recording_preserves_delegate_failure(recording_case, monkeypatch, stage):
    module, install, _ = recording_case
    failure = RuntimeError(f"synthetic {stage} refusal")
    name = object()
    payload = object()
    calls = []

    class Owner:
        def __init__(self):
            calls.append("constructor")
            if stage == "constructor":
                assert batches == []
                raise failure

        def update(self, actual_name, actual_payload):
            calls.append("update")
            assert batches[0][-1][0] is actual_name is name
            assert batches[0][-1][1] is actual_payload is payload
            raise failure

        def hexdigest(self):
            calls.append("hexdigest")
            raise failure

    batches = install(monkeypatch)
    monkeypatch.setattr(digests, "LengthFramedSourceSha256", Owner)
    with pytest.raises(RuntimeError, match=f"^synthetic {stage} refusal$") as error:
        profile = module.LengthFramedSourceSha256()
        if stage == "update":
            profile.update(name, payload)
        elif stage == "hexdigest":
            profile.hexdigest()
    assert error.value is failure
    assert type(error.value) is RuntimeError
    assert calls == (["constructor"] if stage == "constructor" else ["constructor", stage])
    assert batches == ([] if stage == "constructor" else
                       [[(name, payload)]] if stage == "update" else [[]])


def test_recording_keeps_local_update_keyword_signature(recording_case, monkeypatch):
    module, install, keyword = recording_case
    batches = install(monkeypatch)
    profile = module.LengthFramedSourceSha256()
    assert str(inspect.signature(type(profile).update)) == f"(self, name, {keyword})"
    assert str(inspect.signature(profile.update)) == f"(name, {keyword})"
    wrong_keyword = "raw" if keyword == "payload" else "payload"
    with pytest.raises(TypeError) as error:
        profile.update(name="name", **{wrong_keyword: b"bytes"})
    message = str(error.value).replace("_frozen_source_spy", "_spy").replace(
        "_frozen_pwc_spy", "_spy")
    assert message == ("_spy.<locals>.RecordingProfile.update() got an unexpected "
                       f"keyword argument '{wrong_keyword}'")
    assert batches == [[]]
    assert profile.update(name="name", **{keyword: b"bytes"}) is None
    assert batches == [[("name", b"bytes")]]


@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize("refuse", (False, True))
def test_recording_restores_patched_target(recording_case, monkeypatch, present, refuse):
    module, install, _ = recording_case
    original = object()
    if present:
        monkeypatch.setattr(module, "LengthFramedSourceSha256", original, raising=False)
    else:
        monkeypatch.delattr(module, "LengthFramedSourceSha256", raising=False)
    outcome = pytest.raises(RuntimeError, match="^caller refusal$") if refuse else nullcontext()
    with outcome:
        with pytest.MonkeyPatch.context() as patch:
            first_batches = install(patch)
            first_type = module.LengthFramedSourceSha256
            first = first_type()
            second_batches = install(patch)
            second_type = module.LengthFramedSourceSha256
            second = second_type()
            assert first_type is not second_type
            assert first_batches is not second_batches
            assert first_batches == [[]] and second_batches == [[]]
            assert first_batches[0] is first.rows
            assert second_batches[0] is second.rows
            if refuse:
                raise RuntimeError("caller refusal")
    if present:
        assert module.LengthFramedSourceSha256 is original
    else:
        assert not hasattr(module, "LengthFramedSourceSha256")


def test_recording_preserves_real_utf8_refusal(recording_case, monkeypatch):
    module, install, _ = recording_case
    batches = install(monkeypatch)
    profile = module.LengthFramedSourceSha256()
    expected = digests.LengthFramedSourceSha256()
    name = "\ud800"
    payload = b"untouched"
    with pytest.raises(UnicodeEncodeError) as direct_error:
        expected.update(name, payload)
    with pytest.raises(UnicodeEncodeError) as recorded_error:
        profile.update(name, payload)
    assert str(recorded_error.value) == str(direct_error.value)
    assert batches == [[(name, payload)]]
    assert profile.hexdigest() == expected.hexdigest()
