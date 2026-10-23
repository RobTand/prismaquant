"""CPU proof for canonical injection, strict leases, and real gain output."""
from __future__ import annotations
import argparse
from concurrent.futures import Future
import json
import os
import subprocess
import sys
import torch
from canonical_weight import CanonicalWeight
from footprint import FootprintRecorder
from mixed_injection import UnitSpec, inject_layer
from assemble_bands import assemble
from band_gain_reduce import actual_token_sha256, reduce_observations
from test_gain_price_contract import observations
from strict_leases import strict_input_leases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draw",required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = torch.nn.Sequential(torch.nn.Linear(16,8,bias=False).to(torch.bfloat16))
    source = model[0].weight.detach().clone()
    x = torch.arange(48,dtype=torch.float32).reshape(3,16).div(17).to(torch.bfloat16)
    decoded = CanonicalWeight(source.clone(),torch.linspace(0.97,1.03,8))
    spec = UnitSpec("0","attention","projection",None,"BF16","bf16_unquantized",module=model[0])
    clean = model(x).detach()
    expected = (clean.float()+2*(decoded.project(x.float())-x.float()@source.float().T)).to(clean.dtype)
    with torch.inference_mode(),inject_layer(model,[spec],{"0":decoded},weight=True,activation=False,amplitude=2):
        if not torch.equal(model(x),expected):
            raise AssertionError("The canonical injection folds or changes the epilogue")
    if not torch.equal(model(x),clean) or not torch.equal(model[0].weight,source):
        raise AssertionError("The canonical forward lease did not restore the model")
    try:
        with inject_layer(model,[spec],{"0":decoded},weight=True,activation=False):
            raise RuntimeError("explicit interruption")
    except RuntimeError:
        pass
    if not torch.equal(model(x),clean):
        raise AssertionError("Exception cleanup changed the model")
    future = Future()
    future.set_result((bytearray(31),{"format":"actual byte reader tuple"}))
    recorder = FootprintRecorder()
    row = recorder.record("canonical-ready",weights={"0":{"source":source,"options":[({"name":"T16"},decoded)]}},pending=[(None,None,None,future)],state={"input":x})
    storage = source.untyped_storage().nbytes()+decoded.values.untyped_storage().nbytes()+decoded.row_scales.untyped_storage().nbytes()+x.untyped_storage().nbytes()
    if row["unique_storage_bytes"] != storage or row["pending"]["done_buffer_bytes"] != 31:
        raise AssertionError("The component footprint or queued bytes differ")
    original_map = os.environ.pop("PRISMABUILD_RESIDENCY_MAP",None)
    try:
        try:
            with strict_input_leases(True):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("A strict numerical run accepted no residency map")
    finally:
        if original_map is not None:
            os.environ["PRISMABUILD_RESIDENCY_MAP"] = original_map
    from safetensors.torch import load_file
    tokens = next(value for value in load_file(args.draw).values() if tuple(value.shape) == (512,512))
    inputs = torch.cat((torch.tensor([[154822,154824]]).repeat(64,1),tokens[384:448]),dim=1).tolist()
    document = observations(inputs,1.0,(1,2))
    sha = actual_token_sha256(document)
    streams = []
    for stream in document["streams"]:
        base = {"schema":"pact.band_stream.v1","cohort":dict(document["cohort"],token_sha256=sha),
                "input_token_ids":inputs,**{key:stream[key] for key in ("band_start","band_stop","class","kind")}}
        for amplitude in stream["amplitudes"]:
            streams.append({**base,**amplitude,"null_replay":False})
        streams.append({**base,"amplitude":1,"null_replay":True,"per_sequence":stream["null_per_sequence"]})
    published = assemble(streams)
    gains = reduce_observations(published,bootstrap_draws=32)
    if gains["status"] != "measured" or any(abs(row["alpha_measured"]-1)>1e-12 for row in gains["gains"]):
        raise AssertionError("The actual assembler and gain producer disagree")
    try:
        assemble(streams[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("The assembler accepted a missing null stream")
    cli = subprocess.run([sys.executable,"band_replay.py","--help"],capture_output=True,text=True,timeout=20)
    if cli.returncode or "--stream-kind" not in cli.stdout:
        raise AssertionError(cli.stderr)
    result = {"schema":"pact.integration_CPU_proof.v1","passed":True,"gain_streams":len(gains["gains"]),
        "canonical_restoration":True,"footprint":recorder.finish(),"strict_missing_map_refused":True,
        "limit":"The token data are real. The energy and KL observations are synthetic. No GPU or price qualification follows."}
    with open(args.output,"w") as handle:
        json.dump(result,handle,allow_nan=False)
    print(json.dumps({"integration_CPU_passed":True,"gain_streams":len(gains["gains"])}),flush=True)


if __name__ == "__main__":
    main()
