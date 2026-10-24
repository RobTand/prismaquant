def write_pipeline_spec(spec: PipelineSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(spec.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
