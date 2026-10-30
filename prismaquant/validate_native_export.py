#!/usr/bin/env python3
"""validate_native_export.py — load a compressed-tensors checkpoint via
vLLM and do a single forward + greedy decode. Binary check: either
vLLM accepts the format and produces tokens, or it doesn't.

Usage (from inside a vllm-node container):
    python -m prismaquant.validate_native_export \\
        --model dq-runs-new/qwen36-fresh/exported \\
        --prompt "The capital of France is" \\
        --max-new-tokens 16

The script can optionally upgrade the container's flashinfer to a
serving-profile-pinned version before loading; this is needed for some vLLM builds
that ship with a flashinfer that can't dispatch the NVFP4 MoE backend
on Blackwell. Pass `--no-flashinfer-upgrade` to skip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_FLASHINFER_PACKAGES = ("flashinfer-python", "flashinfer-cubin")

#: The quantization this gate loads with (``LLM(..., quantization=...)`` in
#: `_run_arm`). A module constant so the lane check below reads the same value
#: the load uses, rather than a second spelling of it that can drift.
NATIVE_QUANTIZATION = "compressed-tensors"


def _artifact_quant_method(model_dir: str | Path) -> str | None:
    """The artifact's declared ``quantization_config.quant_method``, if any.

    Reads the top-level ``config.json`` entry and the multimodal nesting
    under ``text_config`` -- the same two places the streaming loader reads
    (``layer_streaming``) -- so a Tessera ``text_config`` cannot walk past.
    """
    try:
        cfg = json.loads(Path(model_dir, "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    candidates = [cfg.get("quantization_config")]
    nested = cfg.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested.get("quantization_config"))
    for qc in candidates:
        if isinstance(qc, dict) and qc.get("quant_method"):
            return str(qc["quant_method"])
    return None


def require_native_lane_artifact(model_dir: str | Path) -> str | None:
    """Refuse an artifact this gate cannot load, naming the lane that can.

    This is the NATIVE lane's load gate, not a lane-generic one -- exactly as
    the GGUF lane has its own ``llama-completion`` smoke instead
    (RobTand/prismaquant#119). Loading a foreign-lane artifact with
    ``quantization="compressed-tensors"`` dies inside vLLM on bytes it was
    never going to dispatch; refusing up front says which gate to run
    instead. Returns the declared method on pass (``None`` when the artifact
    declares none, preserving the dense-smoke behavior).
    """
    found = _artifact_quant_method(model_dir)
    if found is not None and found != NATIVE_QUANTIZATION:
        raise SystemExit(
            f"[validate] REFUSE: {model_dir} declares "
            f"quantization_config.quant_method={found!r}, but this gate loads "
            f"with quantization={NATIVE_QUANTIZATION!r}: it is the "
            f"compressed-tensors lane's load smoke, not a lane-generic one. "
            f"A {found!r} artifact is served by its own lane and its own "
            f"gates (see `python -m prismaquant.lane_spec --help`); running "
            f"it here would fail inside vLLM on bytes this quantization "
            f"was never going to dispatch.")
    return found


def maybe_upgrade_flashinfer(
    version: str,
    *,
    package_names: tuple[str, ...] = _DEFAULT_FLASHINFER_PACKAGES,
    env: dict[str, str] | None = None,
) -> None:
    """Raise flashinfer-python and flashinfer-cubin to at least `version` and
    set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass the AOT-cache pin
    that lags behind PyPI. No-op if the installed version is already >= target.

    The profile version is a FLOOR, never an exact pin, and this comparison is
    the whole reason why. It was `== version`, which made a newer install look
    wrong and pip-installed the older one over it. Measured 2026-08-14 on the
    Qwen3.8-27B native-export gate: `gridbook:0.8.6-clean-dde15e0` ships
    flashinfer 0.6.18, the `vllm_packed_moe` profile pins 0.6.8.post1, so the
    gate DOWNGRADED a working container and the engine died with
    `ImportError: cannot import name 'set_autotune_process_group' from
    'flashinfer.autotuner'` — vLLM 0.26 needs the newer API. The pin's original
    purpose was the opposite problem (images too old to dispatch the NVFP4 MoE
    backend on Blackwell), and a floor satisfies that intent without being able
    to break a container that was already fine.
    """
    for key, value in (env or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"}).items():
        os.environ.setdefault(key, value)

    def _parts(v: str) -> tuple[int, ...]:
        # "0.6.8.post1" -> (0, 6, 8); trailing .postN/.devN are ignored, so a
        # post-release never reads as older than its own base version.
        out = []
        for chunk in str(v).split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            if not digits or not chunk[:1].isdigit():
                break
            out.append(int(digits))
        return tuple(out)

    try:
        import flashinfer
        installed = getattr(flashinfer, "__version__", "0.0")
        if installed == version or _parts(installed) >= _parts(version):
            print(f"[validate] flashinfer {installed} already satisfies the "
                  f"{version} floor — not touching it", flush=True)
            return
    except ImportError:
        pass
    package_specs = [f"{name}=={version}" for name in package_names]
    print(f"[validate] upgrading {', '.join(package_names)} to {version}",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "-q",
        *package_specs,
    ])


def _flashinfer_runtime_package(target_profile: str):
    from .serving_profiles import load_serving_profile

    profile = load_serving_profile(target_profile)
    spec = profile.runtime_package("flashinfer")
    if spec is None:
        return (
            "0.6.8.post1",
            _DEFAULT_FLASHINFER_PACKAGES,
            {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        )
    return (
        spec.version or "0.6.8.post1",
        spec.pip_packages or _DEFAULT_FLASHINFER_PACKAGES,
        spec.env_dict() or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
    )


def _resolve_validation_target_profile(
    model_dir: str | Path,
    requested: str | None,
) -> str:
    """Resolve the serving profile for an exported checkpoint smoke.

    A profile this build cannot name falls back to the `default=` target below.
    `DeadVendoredOverrideError` does not (#201): it says the architecture is
    known and its modelling path is dead, and a validator that answers by
    fallback on a checkpoint it was just told it cannot reason about is
    validating the wrong thing.
    """
    from .model_profiles.registry import DeadVendoredOverrideError, detect_profile
    from .serving_profiles import resolve_target_profile

    try:
        profile = detect_profile(str(model_dir))
    except DeadVendoredOverrideError:
        raise
    except Exception:
        profile = None
    return resolve_target_profile(
        profile,
        requested,
        default="vllm_packed_moe",
    )


def summarize_quantization_config(cfg_path: Path) -> None:
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {})
    print(f"[validate] quant_method: {qc.get('quant_method', '<missing>')}")
    print(f"[validate] format:       {qc.get('format', '<missing>')}")
    for gn, g in qc.get("config_groups", {}).items():
        w = g.get("weights", {})
        print(f"[validate]   {gn}: bits={w.get('num_bits')} "
              f"strategy={w.get('strategy')} group={w.get('group_size')} "
              f"format={g.get('format')} n_targets={len(g.get('targets', []))}")
    print(f"[validate]   ignore: {len(qc.get('ignore', []))} entries")


def _speculative_config_uses_embedded_mtp(spec: dict) -> bool:
    method = str(spec.get("method") or "").lower()
    return method == "mtp" or method.endswith("_mtp")


def _install_route_sweep_hooks(llm, sweep_path: Path | None):
    """Attach the per-module forward counters before the generate runs."""
    if sweep_path is None:
        return None
    from .compressed_route_sweep import install_route_sweep_hooks

    apply_model = getattr(llm, "apply_model", None)
    if apply_model is None:
        print("[validate] route sweep: this vLLM build has no "
              "LLM.apply_model; the sweep will record that and the gate "
              "reads it as NOT VERIFIED", flush=True)
        return None
    counts = apply_model(install_route_sweep_hooks)
    print(f"[validate] route sweep: dispatch counters on {counts} module(s) "
          f"per rank", flush=True)
    return counts


def _write_route_sweep(llm, sweep_path: Path | None, hooks, *, model_dir: Path,
                       spec, enforce_eager: bool, args, prompt_ran: bool):
    """Write this arm's served-route sweep beside the smoke's verdict.

    The sweep is taken from THE SAME loaded engine that just generated, so the
    routes recorded are the routes that ran -- principle 8's execution
    identity, not a second load that might resolve differently.
    """
    if sweep_path is None:
        return None
    from .compressed_route_sweep import sweep_via_apply_model, write_sweep

    load = {
        "model_dir": str(model_dir),
        "quantization": NATIVE_QUANTIZATION,
        "enforce_eager": bool(enforce_eager),
        "speculative_config": spec is not None,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "hooks_attached": hooks,
    }
    record = sweep_via_apply_model(llm, load=load, prompt_ran=prompt_ran)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    write_sweep(str(sweep_path), record)
    print(f"[validate] route sweep written: {sweep_path}", flush=True)
    print("[validate]   close route.sweep with: python -m "
          "prismaquant.shipcard_cli fill-route-sweep <shipcard.json> "
          f"--sweep {sweep_path}", flush=True)
    return sweep_path


def _route_sweep_path(args, arm: str) -> Path | None:
    """Where this arm writes its served-route sweep, if asked for one.

    Only the EAGER arm writes one.  The sweep's dispatch counts come from
    forward hooks, and a CUDA-graph replay does not run them, so a graph-arm
    sweep would report zeros that mean "not observed" while looking exactly
    like "priced but never ran".  Refusing to write it is how that stays
    unambiguous (`compressed_route_sweep_gate` reads a non-eager load as NOT
    VERIFIED for the same reason).
    """
    if not getattr(args, "route_sweep_out", None) or arm != "eager":
        return None
    return Path(args.route_sweep_out)


def _graph_image() -> str:
    """Observe the container image, stamping unavailable identity evidence per D32."""
    import re
    from .dev_mode import NOT_COMPUTED, seal_check

    try:
        observed = (Path("/proc/self/cgroup").read_text()
                    + Path("/proc/self/mountinfo").read_text())
        containers = set(re.findall(r"/docker(?:/containers/|/|-)([0-9a-f]{64})(?:/|\.scope|$)",
                                    observed, re.MULTILINE))
        if len(containers) != 1:
            raise ValueError("cannot identify this process's Docker container")
        container_id = containers.pop()
        container = json.loads(subprocess.check_output(
            ["docker", "container", "inspect", container_id], text=True))[0]
        seal_check("container_id", container_id, container["Id"],
                   where="native_export.graph")
        if not container["State"]["Running"]:
            raise ValueError("observed container is not running")
        image = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", container["Image"]], text=True))[0]
        seal_check("image_id", container["Image"], image["Id"],
                   where="native_export.graph")
        digests = {ref for ref in image.get("RepoDigests", [])
                   if re.fullmatch(r".+@sha256:[0-9a-f]{64}", ref)}
        reference = container["Config"]["Image"]
        if reference in digests:
            return reference
        if len(digests) == 1:
            return digests.pop()
        raise ValueError("container image has no unique observed manifest digest")
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            subprocess.CalledProcessError) as exc:
        seal_check("image", "observed container manifest digest", NOT_COMPUTED,
                   where=f"native_export.graph: {exc}")
        return NOT_COMPUTED


def _graph_tessera_source_sha256() -> str:
    """Hash installed Python sources using Tessera #702's sha256sum recipe."""
    from .lane_spec import single_lane_hook

    hook = single_lane_hook("graph_tessera_source_sha256")
    if hook is None:
        from .dev_mode import NOT_COMPUTED, seal_check

        seal_check(
            "tessera_src_sha256",
            "installed Tessera Python source digest",
            NOT_COMPUTED,
            where="native_export.graph: no lane provides graph_tessera_source_sha256",
        )
        return NOT_COMPUTED
    return hook()


def _graph_config_json(value):
    """Keep observed config values, including enum values, as strict JSON data."""
    from dataclasses import asdict, is_dataclass
    from enum import Enum
    from collections.abc import Mapping
    from .digests import canonical_json

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return _graph_config_json(value.value)
    if isinstance(value, Mapping):
        return {key: _graph_config_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_graph_config_json(item) for item in value]
    return canonical_json(value, where="native_export.graph: observed config")


def _graph_rank_observation(model, *, model_dir: str) -> dict:
    """Read the worker's image and TP group; never infer NCCL transport."""
    from vllm.distributed import get_tp_group
    from .digests import file_sha256hex

    group = get_tp_group()
    size, rank = group.world_size, group.rank_in_group
    if type(size) is not int or size < 1 or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("cannot observe the worker's tensor-parallel group")
    if size != 1:
        raise ValueError("observed fabric is unavailable above one rank; an NCCL environment request is not transport evidence")
    return {"rank": rank, "tensor_parallel_size": size,
            "image": _graph_image(), "fabric": "none",
            "tessera_src_sha256": _graph_tessera_source_sha256(),
            "model_config_sha256": file_sha256hex(Path(model_dir) / "config.json")}


def _graph_serve_scope(llm, model_dir: Path, supplied: dict, *, image: str,
                       source_sha256: str, config_sha256: str) -> tuple[dict, dict, list]:
    """Observe the actual loaded engine, separately from its supplied config."""
    import re
    from collections.abc import Mapping
    from functools import partial
    from .digests import canonical_json_bytes, file_sha256hex

    config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    model = getattr(config, "model_config", None)
    loaded = getattr(model, "model", None)
    if not isinstance(loaded, str) or Path(loaded).resolve() != model_dir.resolve():
        raise ValueError("the observed loaded model differs from the requested artifact")
    if file_sha256hex(model_dir / "config.json") != config_sha256:
        raise ValueError("the artifact config.json bytes changed during load")
    if not isinstance(image, str) or re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("the graph smoke has no observed content-addressed image")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("the graph smoke has no observed Tessera source digest")
    scope = {"image": image, "model_config_sha256": config_sha256,
             "tessera_src_sha256": source_sha256, "compilation_config": supplied}
    for name, parent, attribute in (
        ("max_model_len", "model_config", "max_model_len"),
        ("max_num_seqs", "scheduler_config", "max_num_seqs"),
        ("tensor_parallel_size", "parallel_config", "tensor_parallel_size"),
        ("speculative_tokens", "speculative_config", "num_speculative_tokens"),
    ):
        owner = getattr(config, parent, None)
        value = (0 if name == "speculative_tokens" and config is not None
                 and hasattr(config, parent) and owner is None
                 else getattr(owner, attribute, None))
        if type(value) is not int or value < (0 if name == "speculative_tokens" else 1):
            raise ValueError(f"cannot derive {name} from the observed engine config")
        scope[name] = value
    compilation = _graph_config_json(getattr(config, "compilation_config", None))
    priority = _graph_config_json(getattr(getattr(config, "kernel_config", None), "ir_op_priority", None))
    if not isinstance(compilation, Mapping) or not isinstance(compilation.get("custom_ops"), list) or not all(isinstance(item, str) for item in compilation["custom_ops"]):
        raise ValueError("missing observed resolved compilation/custom_ops settings")
    if not isinstance(priority, (Mapping, list, str)):
        raise ValueError("missing observed ir_op_priority settings")
    if any(name not in compilation or canonical_json_bytes(compilation[name], where="observed compilation") != canonical_json_bytes(value, where="supplied compilation")
           for name, value in supplied.items()):
        raise ValueError("observed resolved compilation_config differs from the supplied input")
    resolved = {"compilation_config": compilation, "ir_op_priority": priority}
    apply_model = getattr(llm, "apply_model", None)
    if not callable(apply_model):
        raise ValueError("cannot observe graph smoke ranks through LLM.apply_model")
    ranks = apply_model(partial(_graph_rank_observation, model_dir=str(model_dir)))
    size = scope["tensor_parallel_size"]
    if not isinstance(ranks, list) or len(ranks) != size:
        raise ValueError("missing every-rank graph smoke observation")
    seen = set()
    for row in ranks:
        if not isinstance(row, Mapping) or type(row.get("rank")) is not int or not 0 <= row["rank"] < size or row["rank"] in seen or row.get("tensor_parallel_size") != size:
            raise ValueError("malformed graph smoke rank observation")
        seen.add(row["rank"])
        for name in ("image", "model_config_sha256", "tessera_src_sha256"):
            if row.get(name) != scope[name]:
                raise ValueError(f"rank {row['rank']} observed {name} disagrees with the engine")
        fabric = row.get("fabric")
        allowed = ("none",) if size == 1 else ("socket", "roce")
        if not isinstance(fabric, str) or fabric not in allowed:
            raise ValueError(f"rank {row['rank']} has no observed fabric")
    scope["fabric"] = ranks[0]["fabric"]
    if any(row["fabric"] != scope["fabric"] for row in ranks):
        raise ValueError("rank fabric observations disagree")
    return scope, resolved, ranks


def _run_arm(args, model_dir: Path, spec: dict | None, *,
             enforce_eager: bool) -> dict:
    """One load+generate smoke. Returns a shipcard-shaped verdict block."""
    arm = "eager" if enforce_eager else "graph"
    print(f"[validate] starting vLLM ({arm} arm) ...", flush=True)
    sweep_path = _route_sweep_path(args, arm)
    if sweep_path is not None:
        # vLLM V1 runs EngineCore in a subprocess; `LLM.apply_model` reaches
        # the workers either way, but keeping it in-process is what the
        # archived Gridbook probe learned to do and it keeps the sweep's
        # objects the same objects the generate ran on.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = None
    try:
        import copy
        from .digests import bytes_sha256hex, canonical_json, file_sha256hex

        graph_metrics = {}
        compilation = None
        if not enforce_eager:
            path = getattr(args, "graph_receipt", None)
            if not path:
                raise ValueError("graph_receipt_path is required (--graph-receipt)")
            path = Path(path).resolve(strict=True)
            with path.open("rb") as handle:
                raw = handle.read(128 * 1024 * 1024 + 1)
            if len(raw) > 128 * 1024 * 1024:
                raise ValueError("graph receipt exceeds 128 MiB")
            graph_metrics = {"graph_receipt_path": str(path),
                             "graph_receipt_sha256": bytes_sha256hex(raw)}
            compilation = canonical_json(json.loads(args.compilation_config),
                                         where="native_export.graph: supplied compilation_config")
            if not isinstance(compilation, dict):
                raise ValueError("compilation_config must be an explicit JSON object")
            image = _graph_image()
            source_sha256 = _graph_tessera_source_sha256()
            config_sha256 = file_sha256hex(model_dir / "config.json")
            from .shipcard import (
                assert_weight_stat_attestation, compute_model_sha, load_shipcard,
                weight_stat_attestation,
            )

            identity = load_shipcard(model_dir / "shipcard.json")
            assert_weight_stat_attestation(identity, model_dir)
            model_sha_before = compute_model_sha(model_dir)
            if identity.get("model_sha") != model_sha_before:
                raise ValueError("artifact identity differs from its shipcard before the graph smoke")
            identity = {"weight_stat_attestation": weight_stat_attestation(model_dir)}
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(model_dir),
            quantization=NATIVE_QUANTIZATION,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=getattr(args, "max_num_seqs", 1),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            speculative_config=spec,
            **({"compilation_config": copy.deepcopy(compilation)} if not enforce_eager else {}),
        )
        if not enforce_eager:
            scope, resolved, ranks = _graph_serve_scope(
                llm, model_dir, compilation, image=image,
                source_sha256=source_sha256, config_sha256=config_sha256)
            graph_metrics["serve_scope"] = scope
            graph_metrics["graph_smoke_observation"] = {"resolved": resolved, "ranks": ranks,
                                                       "model_sha_before": model_sha_before}
        hooks = _install_route_sweep_hooks(llm, sweep_path)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        out = llm.generate([args.prompt], sp)
        print(f"[validate] generated ({arm}):", flush=True)
        texts = []
        for o in out:
            text = o.outputs[0].text
            texts.append(text)
            print(f"  prompt: {o.prompt!r}", flush=True)
            print(f"  output: {text!r}", flush=True)
        produced = sum(len(t) for t in texts)
        if not enforce_eager:
            assert_weight_stat_attestation(identity, model_dir)
            model_sha_after = compute_model_sha(model_dir)
            if model_sha_after != model_sha_before or file_sha256hex(model_dir / "config.json") != config_sha256:
                raise ValueError("artifact/config identity changed during the graph smoke")
            graph_metrics["graph_smoke_observation"].update(
                model_sha_after=model_sha_after,
                generation={"prompt": args.prompt, "outputs": texts})
        sweep_written = _write_route_sweep(
            llm, sweep_path, hooks,
            model_dir=model_dir, spec=spec, enforce_eager=enforce_eager,
            args=args, prompt_ran=produced > 0)
        metrics = {"arm": arm, "generated_chars": produced,
                   "enforce_eager": enforce_eager,
                   "max_new_tokens": args.max_new_tokens}
        metrics.update(graph_metrics)
        if sweep_written is not None:
            metrics["route_sweep"] = str(sweep_written)
        return {
            "passed": produced > 0,
            "detail": (f"{arm}: generated {produced} chars"
                       if produced else f"{arm}: generated NOTHING"),
            "metrics": metrics,
        }
    except Exception as exc:
        print(f"[validate] {arm} arm FAILED: {exc!r}", flush=True)
        return {
            "passed": False,
            "detail": f"{arm}: {type(exc).__name__}: {exc}",
            "metrics": {"arm": arm, "enforce_eager": enforce_eager},
        }
    finally:
        # --both-arms holds two engines in one process otherwise; on a
        # unified-memory box the second load then profiles against a pool the
        # first one still owns.
        if llm is not None:
            try:
                del llm
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _record_arm(args, model_dir: Path, spec: dict | None, verdict: dict) -> None:
    """Close the matching `native_export.*` shipcard slot."""
    if not args.shipcard:
        return
    from .shipcard import compute_model_sha, fill_if_requested, git_provenance
    from .shipcard_slots import make_record

    arm = verdict["metrics"]["arm"]
    try:
        model_sha = compute_model_sha(model_dir)
    except Exception:
        model_sha = None
    record = make_record(
        slot=f"native_export.{arm}",
        tool="validate_native_export.py",
        passed=bool(verdict["passed"]),
        model_sha=model_sha,
        metrics=verdict["metrics"],
        detail=verdict["detail"],
        spec_decode_detected=(verdict["metrics"]["serve_scope"]["speculative_tokens"] > 0
                              if arm == "graph" and "serve_scope" in verdict["metrics"]
                              else spec is not None),
        git_commit=git_provenance().get("commit"),
    )
    fill_if_requested(args.shipcard, f"native_export.{arm}", record)


def main():
    from .serving_profiles import serving_profile_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Compressed-tensors checkpoint directory.")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--compilation-config", default=None,
                    help="Exact JSON object supplied to the graph arm's vLLM engine.")
    ap.add_argument("--graph-receipt", default=None,
                    help="Existing equality receipt to retain with graph smoke observations. "
                         "A later smoke does not create original equality-producer linkage.")
    ap.add_argument("--target-profile", default=None,
                    choices=serving_profile_names(),
                    help="Serving profile whose runtime package pins should "
                         "be used for validation preflight. Defaults to the "
                         "exported model profile's configured serving "
                         "profile, then vllm_packed_moe.")
    ap.add_argument("--flashinfer-version", default=None,
                    help="Override the serving profile's flashinfer package "
                         "version before loading vLLM.")
    ap.add_argument("--no-flashinfer-upgrade", action="store_true",
                    help="Skip the flashinfer pre-flight upgrade.")
    ap.add_argument("--speculative-config", default=None,
                    help="JSON string for vLLM SpeculativeConfig. Use this "
                         "to exercise MTP heads, e.g. "
                         "'{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 3, "
                         "\"model\": \"<same model dir>\"}'.")
    ap.add_argument("--no-enforce-eager", action="store_true",
                    help="Allow vLLM compile/CUDA-graph execution instead of "
                         "forcing eager mode. Use after the eager smoke passes.")
    ap.add_argument("--both-arms", action="store_true",
                    help="Run the eager arm AND the graph arm in one "
                         "invocation and fill both shipcard slots. The "
                         "two-arm rule used to live only in this help text; "
                         "this is it in code.")
    ap.add_argument("--route-sweep-out", default=None,
                    help="Write the EAGER arm's served-route sweep here "
                         "(prismaquant.compressed_route_sweep/1). Principle "
                         "14's serve-side leg on this lane: the resolved "
                         "quantization method, scheme and kernel per module, "
                         "read off the live engine through vLLM's own "
                         "LLM.apply_model, plus a forward-hook dispatch "
                         "count. Close the card's route.sweep slot from it "
                         "with `python -m prismaquant.shipcard_cli "
                         "fill-route-sweep`.")
    ap.add_argument("--shipcard", default=None,
                    help="Path to the artifact's shipcard.json; the arm's "
                         "verdict is appended to native_export.<arm> "
                         "(see python -m prismaquant.shipcard_cli).")
    args = ap.parse_args()
    if args.max_num_seqs < 1 or args.tensor_parallel_size < 1:
        ap.error("--max-num-seqs and --tensor-parallel-size must be positive")
    if args.both_arms or args.no_enforce_eager:
        if not args.graph_receipt or args.compilation_config is None:
            ap.error("the graph arm requires --graph-receipt and --compilation-config")
        try:
            from .digests import canonical_json

            compilation = canonical_json(json.loads(args.compilation_config),
                                         where="--compilation-config")
            if not isinstance(compilation, dict):
                raise ValueError("expected a JSON object")
        except (TypeError, ValueError) as exc:
            ap.error(f"invalid --compilation-config: {exc}")

    model_dir = Path(args.model)
    target_profile = _resolve_validation_target_profile(
        model_dir,
        args.target_profile,
    )
    print(f"[validate] target profile: {target_profile}", flush=True)

    if not args.no_flashinfer_upgrade:
        version, package_names, env = _flashinfer_runtime_package(target_profile)
        if args.flashinfer_version:
            version = args.flashinfer_version
        maybe_upgrade_flashinfer(version, package_names=package_names, env=env)

    summarize_quantization_config(model_dir / "config.json")

    # The lane check runs before any vLLM import: a foreign-lane artifact
    # must refuse here, not inside the loader.
    require_native_lane_artifact(model_dir)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    spec = None
    if args.speculative_config:
        spec = json.loads(args.speculative_config)
        # If caller omitted "model", default to the same checkpoint for
        # draft-model style configs. MTP-family methods are the exception:
        # their extra heads travel with the target checkpoint, and vLLM expects
        # model to be absent/null so it can take the embedded-MTP path.
        if "model" not in spec and not _speculative_config_uses_embedded_mtp(spec):
            spec["model"] = str(model_dir)
        print(f"[validate] speculative config: {spec}", flush=True)

    if args.both_arms:
        arms = [True, False]
    else:
        arms = [not args.no_enforce_eager]

    verdicts = []
    for enforce_eager in arms:
        verdict = _run_arm(args, model_dir, spec, enforce_eager=enforce_eager)
        _record_arm(args, model_dir, spec, verdict)
        verdicts.append(verdict)

    failed = [v for v in verdicts if not v["passed"]]
    for verdict in verdicts:
        print(f"[validate] {'PASS' if verdict['passed'] else 'FAIL'} "
              f"{verdict['detail']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
