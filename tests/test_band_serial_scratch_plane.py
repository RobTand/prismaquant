"""The band-serial byte-identity tests, on the production cotangent plane (PQ #1263).

A campaign row stores its cotangent plane in an ``ExactCotangentScratch``
(``PRISMAQUANT_STAGE_B_COTANGENT_ROOT``, ``cost_streaming.py``
``checkpoint_cotangent_sink``), and its streamed handoff writer reads each
final slot from that plane: through the tee ring, or back from the scratch.
The equality tests this module imports run on a dict plane in their own
modules. Here each runs again with the scratch switched on, so the handoff
a producer writes from the production plane is checked, byte for byte,
against the chain rebuild it replaces.

A fixture that only sets the variables would pass unchanged if the core
stopped reading them, so each test must also build a scratch plane, and
every handoff written from one must account for each entry through the tee
ring or a scratch read.
"""
from __future__ import annotations

import pytest

from test_band_serial_batched_regime import (  # noqa: F401 (module fixtures)
    campaign1,
    campaign4,
    test_a_band_serial_producer_runs_the_campaign_regime,
    test_the_batched_capture_plane_is_the_batched_chain_plane,
    test_the_control_at_batch_one_still_holds,
)
from test_band_serial_spill import (  # noqa: F401 (module fixture)
    campaign,
    test_band_serial_under_the_spill_equals_the_chain_rebuild,
)
from test_quantum_band_serial import (  # noqa: F401
    test_band_serial_payloads_and_planes_equal_the_chain_rebuild,
)
# The band-serial tests run offline, as test_quantum_band_serial's do.
from test_joint_cost_quantum_runtime import _offline_tier_policy  # noqa: E402,F401


@pytest.fixture(autouse=True)
def cotangent_scratch_plane(tmp_path, monkeypatch):
    from prismaquant.joint_quantum_handoff import HandoffStream
    from prismaquant.perturbed_x_cache import ExactCotangentScratch

    root = tmp_path / "cotangent-scratch"
    root.mkdir()
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_ROOT", str(root))
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES", str(1 << 30))
    built, streams = [], []
    real_init, real_exit = ExactCotangentScratch.__init__, HandoffStream.__exit__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    def leave(self, *args):
        streams.append((self._scratch is not None, dict(self.telemetry)))
        return real_exit(self, *args)

    monkeypatch.setattr(ExactCotangentScratch, "__init__", init)
    monkeypatch.setattr(HandoffStream, "__exit__", leave)
    yield
    assert built, "no cotangent scratch was built: the test ran on a dict plane"
    scratch_streams = [telemetry for on_scratch, telemetry in streams if on_scratch]
    assert scratch_streams, f"no handoff was written from a scratch plane: {streams}"
    for telemetry in scratch_streams:
        assert telemetry["error"] is None, telemetry
        assert (telemetry["tee_hits"] + telemetry["scratch_reads"]
                == telemetry["entries"]), telemetry
