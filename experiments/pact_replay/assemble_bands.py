"""Assemble actual amplitude and null streams for the existing gain producer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from band_gain_reduce import actual_token_sha256,COHORT

REPLAY_COMPARABILITY = ("layer_range", "sample_ids")


def replay_comparability(document):
    """Actual replay comparability: layer ranges and samples.

    These refuse on mismatch. Producer labels never enter this comparison."""
    replay = document.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("The band stream lacks its replay path record")
    try:
        return {field: replay[field] for field in REPLAY_COMPARABILITY}
    except KeyError:
        raise ValueError("The band stream replay path is incomplete")


def replay_provenance(document):
    """Advisory producer provenance: recorded session labels.

    A label stamps the assembled stream and continues. It never refuses, and
    it never proves actual replay comparability."""
    replay = document.get("replay")
    if not isinstance(replay, dict):
        return {}
    return {field: replay[field] for field in ("start_generation", "stop_generation")
            if field in replay}


def assemble(documents):
    return assemble_with_axes(documents)


def assemble_with_axes(documents, required_axes=None):
    groups, nulls = {}, {}
    tokens, required = None, set()
    for document in documents:
        if document.get("schema") != "pact.band_stream.v1":
            raise ValueError("A completed measured band stream is required")
        sha = actual_token_sha256(document)
        if document["cohort"].get("token_sha256") != sha:
            raise ValueError("The stream token digest differs from its actual tokens")
        for key,value in COHORT.items():
            if document["cohort"].get(key) != value:
                raise ValueError("The stream cohort differs")
        if tokens is None:
            tokens = document["input_token_ids"]
        elif tokens != document["input_token_ids"]:
            raise ValueError("The actual stream populations differ")
        band = document["band_start"],document["band_stop"]
        records = document["per_sequence"]
        if len(records) != 64 or {row["sample_id"] for row in records} != set(range(384,448)):
            raise ValueError("The measured stream sample population differs")
        identity = replay_comparability(document)
        provenance = replay_provenance(document)
        if document.get("null_replay"):
            if band in nulls:
                raise ValueError("The clean null replay is duplicated for the band")
            if document.get("injections"):
                raise ValueError("The clean null replay carries injections")
            nulls[band] = {"records": records, "replay": identity, "provenance": provenance}
            continue
        cls,kind = document["class"],document["kind"]
        key = (*band,cls,kind)
        required.add(cls)
        amplitudes = groups.setdefault(key,{"replay":identity,"provenance":[]})
        if amplitudes["replay"] != identity:
            raise ValueError("The amplitude replay path differs inside one measured stream")
        amplitudes["provenance"].append(provenance)
        amplitude = document["amplitude"]
        if amplitude in amplitudes or amplitude not in (1,2,4):
            raise ValueError("A measured amplitude is duplicated or invalid")
        amplitudes[amplitude] = records
    streams = []
    for key,amplitudes in sorted(groups.items()):
        band = key[:2]
        pair = sorted(value for value in amplitudes if value not in ("replay", "provenance"))
        if pair not in ([1,2],[2,4],[1,2,4]) or band not in nulls:
            raise ValueError("A stream lacks its accepted amplitude pair or its band null replay")
        if amplitudes["replay"] != nulls[band]["replay"]:
            raise ValueError("The clean null replay path differs from its band streams")
        stamps = sorted({json.dumps(stamp, sort_keys=True) for stamp in [*amplitudes["provenance"], nulls[band]["provenance"]]})
        streams.append({"band_start":key[0],"band_stop":key[1],"class":key[2],"kind":key[3],
            "amplitudes":[{"amplitude":a,"per_sequence":amplitudes[a]} for a in pair],
            "null_per_sequence":nulls[band]["records"],
            "replay_provenance":[json.loads(stamp) for stamp in stamps]})
    if not streams or {key[:2] for key in groups} != set(nulls):
        raise ValueError("The stream set has missing or unused band null replays")
    from band_gain_reduce import normalize_declared_axes
    axes = None
    if required_axes is not None:
        normalize_declared_axes(required_axes)
        if isinstance(required_axes, dict):
            axes = dict(required_axes)
        else:
            raise ValueError("The declared axes must use the versioned required-axis declaration")
    result = {"schema":"pact.band_observations.v1","cohort":COHORT,"input_token_ids":tokens,
        "required_classes":sorted(required),"streams":streams}
    if axes is not None:
        result["required_axes"] = axes
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--streams",type=Path,nargs="+",required=True)
    p.add_argument("--required-axes",type=Path,
                   help="JSON file with the canonical pact.required_gain_axes.v1 declaration")
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    axes = json.loads(args.required_axes.read_bytes()) if args.required_axes else None
    result = assemble_with_axes([json.loads(path.read_bytes()) for path in args.streams], axes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(args.output)+".partial")
    temporary.write_text(json.dumps(result,allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(json.dumps({"output":str(args.output),"streams":len(result["streams"])}))


if __name__ == "__main__":
    main()
