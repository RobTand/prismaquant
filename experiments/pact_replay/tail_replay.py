"""Replay actual final clean boundaries through the existing model tail."""
from contextlib import ExitStack
import io
import json


def run_tail_frontier(runner,args,cohort,inputs,measure,guard,*,on_logits=None):
    import torch
    from g3_residency import read_file
    from frontier_replay import move_state
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts,StreamedForwardBoundaries
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    receipt = json.loads(read_file(args.boundaries))
    if receipt.get("schema") != "pact.prefixed_clean_frontier.v1" or receipt.get("layer_stop") != 45 or receipt.get("dry_run"):
        raise ValueError("The head needs complete actual clean boundaries at layer45")
    for key in ("sample_range","raw_tokens_per_sequence","prefix_ids","input_contract","local_prefix_rows","global_original_tokens","scored_positions_per_sequence"):
        if receipt["cohort"].get(key) != cohort[key]:
            raise ValueError("The head and boundary cohorts differ")
    samples = list(range(*cohort["sample_range"]))
    if receipt["sample_ids"] != samples or set(receipt["references"]) != {str(sample) for sample in samples}:
        raise ValueError("The head boundary sample population differs")
    states = torch.load(io.BytesIO(read_file(receipt["batch_state_path"])),map_location="cpu",weights_only=True)
    if set(states) != set(samples):
        raise ValueError("The head states omit or repeat samples")
    refs = {}
    for sample,ids in zip(samples,inputs):
        if not torch.equal(states[sample]["input_ids"],ids):
            raise ValueError("The actual head and boundary tokens differ")
        ref = reference_from_record(receipt["references"][str(sample)])
        coordinate = json.loads(ref.metadata_json)["identity"]["coordinates"]
        if coordinate["batch"] != sample or coordinate["boundary"] != 45 or tuple(ref.shape[:2]) != (1,514):
            raise ValueError("The head boundary coordinates or token shape differ")
        refs[sample] = ref
    with StreamedBoundaryArtifacts(receipt["boundary_config"]) as owner:
        owner.attach(receipt["session"],n_probes=0)
        cursor = 0
        replay_state = {"states":states,"boundary_owner":owner,"current":{}}
        def forward(ids):
            nonlocal cursor
            if cursor >= len(samples) or not torch.equal(ids,inputs[cursor]):
                raise ValueError("The head replay changes sample order")
            sample = samples[cursor]
            row = move_state(states[sample],runner.device)
            batch = StreamedForwardBoundaries(row["input_ids"],row["position_ids"],row["position_embeddings"],row["attention_mask"],[],None)
            with owner.prefetch([refs[sample]]) as window:
                guard("resident clean head call")
                hidden = owner.get(window,refs[sample]).to(runner.device,dtype=runner.dtype)
                replay_state["current"] = {"row":row,"hidden":hidden}
                with torch.inference_mode():
                    logits = runner.tail_logits(batch,hidden)
                replay_state["current"]["logits"] = logits
                if on_logits is not None:
                    on_logits(sample,logits)
                replay_state["current"].clear()
            cursor += 1
        measure(45,forward,replay_state)
        if cursor != len(samples):
            raise ValueError("The head replay omits samples")
    return {"schema":receipt["schema"],"layer_range":[45,46],"sample_ids":samples,
            "start_generation":receipt["session"],"foreign_entries_retired":0}
