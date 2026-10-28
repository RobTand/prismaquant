"""Quantum wire phases, invocation arguments and admitted row demand."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

HANDOFF_LOAD_PHASE = "handoff-load"
CHECKPOINT_LOAD_PHASE = "checkpoint-load"


@dataclass(frozen=True)
class ChainSourcePhase:
    layer: int

    @property
    def name(self) -> str:
        return executable_source_phase_name(self.layer)


@dataclass(frozen=True)
class ChainBoundaryPhase:
    layer: int

    @property
    def name(self) -> str:
        return executable_bound_phase_name(self.layer)


@dataclass(frozen=True)
class OwnSourcePhase:
    layer: int

    @property
    def name(self) -> str:
        return executable_own_source_phase_name(self.layer)


@dataclass(frozen=True)
class RenderPhase:
    window: int

    @property
    def name(self) -> str:
        return executable_render_phase_name(self.window)


@dataclass(frozen=True)
class ReplayPhase:
    window: int
    probe: int

    @property
    def name(self) -> str:
        return executable_replay_phase_name(self.window, self.probe)


@dataclass(frozen=True)
class SpillPhase:
    probe: int

    @property
    def name(self) -> str:
        return executable_spill_phase_name(self.probe)


@dataclass(frozen=True)
class NamedPhase:
    name: str


QuantumPhase = (ChainSourcePhase | ChainBoundaryPhase | OwnSourcePhase |
                RenderPhase | ReplayPhase | SpillPhase | NamedPhase)


_PHASE_FORMS = (
    (re.compile(r"chain-(\d{3,})-source"), ChainSourcePhase),
    (re.compile(r"chain-(\d{3,})-bound"), ChainBoundaryPhase),
    (re.compile(r"own-(\d{3,})-source"), OwnSourcePhase),
    (re.compile(r"render-(\d{2,})"), RenderPhase),
    (re.compile(r"replay-(\d{2,})-p(\d+)"), ReplayPhase),
    (re.compile(r"spill-p(\d+)"), SpillPhase),
)


def parse_quantum_phase(name: str) -> QuantumPhase:
    for pattern, phase_type in _PHASE_FORMS:
        if match := pattern.fullmatch(name):
            return phase_type(*(int(value) for value in match.groups()))
    return NamedPhase(name)



def quantum_source_layer_order(chain_layers: Sequence[int],
                               layer: int) -> tuple[int, ...]:
    """The source layers one quantum installs, in install order (PQ #1095).

    Its chain, descending from the checkpoint boundary, then its own layer.
    The quantum installs and prefetches exactly these layers
    (``joint_cost_quantum._install_with_settlement`` with
    ``source_read_plan.chain_prefetch_window``), and the executable readset
    declares one source phase for each, in this order. A band-serial
    consumer walks no chain, so its order is its own layer alone.
    """
    return tuple(int(c) for c in chain_layers) + (int(layer),)


def executable_source_phase_name(layer: int) -> str:
    """The executable-manifest phase staging one chain layer's source."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a source phase needs a layer, not {layer!r}")
    return f"chain-{layer:03d}-source"


def executable_bound_phase_name(layer: int) -> str:
    """The executable-manifest phase staging one chain layer's boundary."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a bound phase needs a layer, not {layer!r}")
    return f"chain-{layer:03d}-bound"


def executable_own_source_phase_name(layer: int) -> str:
    """The executable-manifest phase staging the quantum's own source."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"an own-source phase needs a layer, not {layer!r}")
    return f"own-{layer:03d}-source"


def executable_replay_phase_name(window_index: int | None, probe: int) -> str:
    """The executable-manifest phase for one replay probe pass.

    ``None`` names window zero: the zero-pending resume path still reads
    the own boundary once per probe with no window loop running, and those
    reads stage under the window-zero phases by this convention.
    """
    window = 0 if window_index is None else int(window_index)
    if window < 0 or type(probe) is not int or isinstance(probe, bool) \
            or probe < 0:
        raise ValueError(
            f"a replay phase needs a window and probe, not "
            f"{window_index!r}/{probe!r}")
    return f"replay-{window:02d}-p{probe}"


#: The replay modes an executable read plan is sealed for (PQ #1011).
#: ``windowed`` re-reads the own boundary run once per (window, probe);
#: ``spill`` reads it once per probe, into the one-pass replay spill.
REPLAY_MODES = ("windowed", "spill")


def normalize_replay_mode(value) -> str:
    """``windowed`` (the default, also for ``None``) or ``spill``."""
    mode = "windowed" if value is None else value
    if mode not in REPLAY_MODES:
        raise ValueError(f"replay mode must be one of {REPLAY_MODES}, "
                         f"not {value!r}")
    return mode


def executable_spill_phase_name(probe: int) -> str:
    """The executable-manifest phase for one probe's spill capture.

    Under the one-pass spill (PQ #994/#1011) each probe reads the own
    boundary run once, during its capture; the window replays read nothing.
    """
    if type(probe) is not int or isinstance(probe, bool) or probe < 0:
        raise ValueError(f"a spill phase needs a probe, not {probe!r}")
    return f"spill-p{probe}"


def executable_render_phase_name(window_index: int) -> str:
    """The executable-manifest phase staging one retained window's prepared renders.

    One per already-defined retained window, sealed immediately before that
    window's replay phases and entered in before_window before the PWC
    retained load (PQ #917). Source phases stay source-only; rendered inputs
    get their own consumption phases.
    """
    if type(window_index) is not int or isinstance(window_index, bool) \
            or window_index < 0:
        raise ValueError(f"a render phase needs a window, not {window_index!r}")
    return f"render-{window_index:02d}"


def quantum_executable_phase_names(chain_layers: Sequence[int], layer: int,
                                    *, n_probes: int,
                                    replay_windows: int,
                                    render_phases: bool = False,
                                    replay_mode: str | None = None) -> tuple[str, ...]:
    """The frozen executable staging order (PQ #862): calibration head,
    the checkpoint plane once, then per chain layer descending its source
    extents and its boundary entries, the quantum's own source extents,
    then per retained window and probe the replay boundary reads.

    With ``render_phases`` (PQ #917), each retained window additionally
    stages one ``render-{window:02d}`` phase immediately before that
    window's replay phases: the window's selected prepared renders get
    their own consumption phases while source phases stay source-only.
    Without it the historical sequencing-only order reproduces unchanged.

    ``replay_mode="spill"`` (PQ #1011) seals the one-pass spill's real
    consumption order instead: each probe's capture reads the own boundary
    run once, inside the first window's retained lifetime, and every window
    replay reads nothing. After the own source come ``render-00`` (with
    ``render_phases``), one ``spill-p{probe}`` phase per probe, then the
    remaining windows' render phases. The windowed default is unchanged.

    Every name is reported by the runtime through the existing semantic
    reporter as its bytes are consumed -- a staging contract declares
    exactly this list. Chunk names are deliberately not staging phases:
    durable units keep their existing currency and commit under the read
    phase in effect when the journal lands.
    """
    chain = [int(c) for c in chain_layers]
    if any(type(c) is not int or isinstance(c, bool) for c in chain_layers):
        raise ValueError("chain layers must be integers, "
                         f"not {list(chain_layers)!r}")
    if len(set(chain)) != len(chain):
        raise ValueError("chain layers repeat: refusing")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"an executable readset needs a layer, not {layer!r}")
    for label, value in (("probe count", n_probes),
                         ("replay windows", replay_windows)):
        if type(value) is not int or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be positive, not {value!r}")
    if type(render_phases) is not bool:
        raise ValueError(f"render phases must be a flag, not {render_phases!r}")
    replay_mode = normalize_replay_mode(replay_mode)
    names = ["head", CHECKPOINT_LOAD_PHASE]
    for boundary in chain:
        names.append(executable_source_phase_name(boundary))
        names.append(executable_bound_phase_name(boundary))
    names.append(executable_own_source_phase_name(layer))
    if replay_mode == "spill":
        for window_index in range(replay_windows):
            if render_phases:
                names.append(executable_render_phase_name(window_index))
            if window_index == 0:
                names.extend(executable_spill_phase_name(probe)
                             for probe in range(n_probes))
        return tuple(names)
    for window_index in range(replay_windows):
        if render_phases:
            names.append(executable_render_phase_name(window_index))
        for probe in range(n_probes):
            names.append(executable_replay_phase_name(window_index, probe))
    return tuple(names)


#: The tail's one term (PQ #1190), which :func:`compute_phase_work` appends to
#: the last declared phase's work.
TAIL_WORK = {"kind": "tail", "units": 1,
             "work": ("the quantum's tail after its last window, which "
                      "declares no phase and commits no unit")}
def phase_work_entries(name: str, fact: Mapping, annotations: Mapping):
    """The entries a phase's pass counts its work by, from its read-plan facts.

    A phase's entry count, less incoming plane entries moved into it by
    ``band_serial.streamed_incoming`` or ``checkpoint_incoming.streamed_incoming``:
    those add read bytes, rather than capture groups or chain batches.
    ``None`` when the plan does not count the phase's entries.
    """
    entries = fact.get("entries") if isinstance(fact, Mapping) else None
    if type(entries) is not int:
        return entries
    moved = 0
    for key in ("band_serial", "checkpoint_incoming"):
        owner = annotations.get(key) if isinstance(annotations, Mapping) else None
        streamed = owner.get("streamed_incoming") if isinstance(owner, Mapping) else None
        count = streamed.get(name, 0) if isinstance(streamed, Mapping) else 0
        if type(count) is not int or not 0 <= count <= entries:
            return None
        moved += count
    return entries - moved if moved <= entries else None


def compute_phase_work(name: str, *, replay_mode: str, entries,
                       n_probes, capture_batch,
                       runs_tail: bool = False) -> list[dict] | None:
    """The compute one phase's pass runs, or ``None`` for a phase with none.

    Each term is ``{"kind", "units", "work"}``; ``units`` is ``None`` when
    the read plan or the row does not count them. The order follows
    ``joint_statistics_replay.observe_and_project_retained_windows``: a spill
    row captures each probe under ``spill-pP`` and replays the first window
    for that probe under the same phase, and replays every later window under
    its ``render-NN`` phase. A resume that has committed window 0 still
    captures under ``spill-pP`` and replays nothing there (PQ #1172), so the
    replay term of ``spill-pP`` is an upper bound for it. ``render-00`` of a
    spill row, and every ``render-NN`` of a windowed row, only read.

    ``runs_tail`` marks the row's last declared phase: the quantum's tail
    after its last window runs on its clock (PQ #1187), so :data:`TAIL_WORK`
    is its last term, with or without a pass of its own (PQ #1190).
    """
    def counted(*values):
        return all(type(value) is int and value > 0 for value in values)

    tail = [dict(TAIL_WORK)] if runs_tail else []
    phase = parse_quantum_phase(name)
    if isinstance(phase, ChainBoundaryPhase):
        return [{"kind": "chain-roll",
                 "units": entries * n_probes if counted(entries, n_probes) else None,
                 "work": (f"chain layer {phase.layer}'s backward roll: "
                          "one row per probe and stored batch")}, *tail]
    if isinstance(phase, SpillPhase) and replay_mode == "spill":
        probe = phase.probe
        return [{"kind": "spill-capture",
                 "units": (-(-entries // capture_batch)
                           if counted(entries, capture_batch) else None),
                 "work": (f"probe {probe}'s spill capture: one group per "
                          "capture_batch stored batches")},
                {"kind": "spill-replay", "units": 1,
                 "work": f"the first window's spill replay for probe {probe}"},
                *tail]
    if isinstance(phase, RenderPhase) and replay_mode == "spill" and phase.window >= 1:
        return [{"kind": "spill-replay",
                 "units": n_probes if counted(n_probes) else None,
                 "work": (f"window {phase.window}'s spill replay for "
                          "each probe")}, *tail]
    if isinstance(phase, ReplayPhase) and replay_mode == "windowed":
        return [{"kind": "windowed-replay",
                 "units": entries if counted(entries) else None,
                 "work": (f"window {phase.window}'s replay for probe "
                          f"{phase.probe}: one backward per stored "
                          "batch")}, *tail]
    return tail or None


#: The tail leg's telemetry name. It is NOT a read-plan phase: published
#: ``manifest_phase_ranges`` drops cumulative==previous phases, so a
#: zero-byte tail would be absent from the sealed residency plan -- and
#: reporting a name the plan does not carry resets ``remaining`` to start.
#: The tail checkpoint work commits durable units under forward-last
#: instead; this string survives only for explicit runtime log lines.
ADJOINT_TAIL_PHASE = "tail"


def adjoint_forward_phase_name(layer: int) -> str:
    """The read-plan phase staging layer ``layer``'s source extents for the
    forward capture walk (ascending)."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a forward phase needs a nonnegative layer, not {layer!r}")
    return f"forward-{layer:03d}"


def adjoint_chain_phase_name(layer: int) -> str:
    """The read-plan phase staging layer ``layer``'s source extents for the
    reverse chain walk (descending). One spelling shared by the builder, the
    capture's progress reports, and the tests -- two spellings of the chain
    convention is how a consumer ends up committing a name the window cannot
    match."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a chain phase needs a nonnegative layer, not {layer!r}")
    return f"chain-{layer:03d}"


def adjoint_read_plan_phase_names(num_layers: int) -> tuple[str, ...]:
    """The frozen Stage A consumption order (PQ #837): head, forward
    ascending, reverse descending. The builder seals the manifest in this
    order, the capture reports in this order, and the dispatch lane
    declares progress in this order -- the published v2 linear-progress
    rule refuses anything else. There is deliberately no tail phase (see
    ``ADJOINT_TAIL_PHASE``)."""
    if type(num_layers) is not int or isinstance(num_layers, bool) or num_layers < 1:
        raise ValueError(f"a read plan needs a positive layer count, not {num_layers!r}")
    return (("head",)
            + tuple(adjoint_forward_phase_name(layer) for layer in range(num_layers))
            + tuple(adjoint_chain_phase_name(layer)
                    for layer in reversed(range(num_layers))))


QUANTUM_ENTRY_POINT = "prismaquant.joint_cost_quantum"
PROGRESS_GRACE_FLAG = "--progress-grace-derivation"


@dataclass(frozen=True)
class QuantumFileBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class QuantumManifestInvocation:
    record_path: str
    output_root: str

    def argv(self) -> list[str]:
        return ["python3", "-m", QUANTUM_ENTRY_POINT, "--quantum", self.record_path,
                "--output-root", self.output_root]


@dataclass(frozen=True)
class QuantumBandArguments:
    handoff: QuantumFileBinding | None = None
    emit_handoff: bool = False
    progress_grace: str | None = None

    def argv(self) -> list[str]:
        argv = []
        if self.handoff is not None:
            argv += ["--adjoint-handoff", str(self.handoff.path),
                     "--adjoint-handoff-sha256", str(self.handoff.sha256)]
        if self.emit_handoff:
            argv.append("--emit-adjoint-handoff")
        if self.progress_grace is not None:
            argv += [PROGRESS_GRACE_FLAG, self.progress_grace]
        return argv


@dataclass(frozen=True)
class QuantumRunInputs:
    quantum: QuantumFileBinding
    plan: QuantumFileBinding
    prepared: QuantumFileBinding
    adjoint_slice: QuantumFileBinding
    manifest_sha256: str
    output_root: str

    def argv(self, *, allowed_tiers: str, band: QuantumBandArguments | None = None) -> list[str]:
        band_arguments = [] if band is None else band.argv()
        return [
            "python3", "-m", QUANTUM_ENTRY_POINT,
            "--quantum", str(self.quantum.path),
            "--quantum-sha256", self.quantum.sha256,
            "--plan", str(self.plan.path),
            "--plan-sha256", self.plan.sha256,
            "--prepared", str(self.prepared.path),
            "--prepared-sha256", self.prepared.sha256,
            "--adjoint-slice", str(self.adjoint_slice.path),
            "--adjoint-slice-sha256", self.adjoint_slice.sha256,
            "--data-manifest-sha256", self.manifest_sha256,
            "--allowed-tiers", allowed_tiers,
            "--resume", "--output-root", str(self.output_root), *band_arguments]


@dataclass(frozen=True)
class QuantumRowDemand:
    memory_gib: str
    gpu_gib: str
    cpus: int

    @classmethod
    def from_policy(cls, policy, *, environment, source_prefetch=None, operator_prefetch=None):
        if policy is None:
            return cls("104", "80", 10)
        limits = policy["limits"]
        memory, gpu = (f"{limits[key] / 1024 ** 3:g}"
                       for key in ("physical_bytes", "gpu_bytes"))
        cpus = max(int(environment.get("PRISMAQUANT_LAYER_READ_THREADS", 1)),
                   int(source_prefetch["prefetch_workers"]) + 1,
                   int(operator_prefetch["prefetch_workers"]) + 1)
        return cls(memory, gpu, cpus)


@dataclass(frozen=True)
class QuantumPBEnvelope:
    manifest_path: str
    progress: Sequence[tuple[str, int]]
    demand: QuantumRowDemand
    tags: Sequence[str]
    priority: int
    command: Sequence[str]
    environment: Sequence[tuple[str, str]] = ()
    produced_output_template: str | None = None
    container_image: str | None = None

    def argv(self, *, python: str, pbrun: str, container_image_flag: str,
             dev_mode_assignment: str) -> list[str]:
        argv = [python, str(pbrun)]
        for tag in self.tags:
            argv += ["--tag", str(tag)]
        argv += ["--data-manifest", str(self.manifest_path),
                 "--residency", "stage", "--residency-ram", "auto"]
        for name, grace in self.progress:
            argv += ["--progress-phase", f"{name}={grace}"]
        argv += ["--priority", str(self.priority),
                 "--demand", f"gpu=1,mem_gb={self.demand.memory_gib}",
                 "--gpu-memory-gb", self.demand.gpu_gib,
                 "--cpus", str(self.demand.cpus)]
        if self.produced_output_template is not None:
            argv += ["--produced-output-template", str(self.produced_output_template)]
        if self.container_image is not None:
            argv += [container_image_flag, self.container_image]
        for name, value in self.environment:
            argv += ["--env", f"{name}={value}"]
        return [*argv, "--env", dev_mode_assignment, "--detach", "--", *self.command]
