def cmd_census(args) -> int:
    spec = campaign_rows.load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    census_path = workspace / "census.json"
    manifest = workspace / "census-manifest.json"
    row = campaign_rows.prismabuild_row(
        spec,
        ["--model", spec["model"],
         "--out", str(workspace / "census-unused.pkl"),
         "--cache-dir", str(workspace / "census-cache"),
         "--census-out", str(census_path),
         *spec["campaign_argv"]],
        mem_gb=campaign_rows.row_memory_gb(spec, [], {}),
        timeout_s=int(args.timeout_s),
        # Census exits before the pricing journal/reporter exists.  Its
        # explicit wall-clock deadline is the only bound it declares.
        progress_phases=(),
    )
    manifest.write_text(json.dumps([row], indent=2) + "\n")
    if explicit_campaign_options(spec['campaign_argv']).get("streaming", False):
        (workspace/'census-resources.json').write_text(json.dumps(
            campaign_rows.streamed_resource_plan(spec, {}, []), indent=2, sort_keys=True)+'\n')
    print(f"[dispatch] census manifest {manifest}")
    if args.submit:
        return _pbcampaign(manifest, wait_s=args.wait_s,
                           receipts=workspace / "census-receipts.json")
    return 0

def cmd_capture(args) -> int:
    """Submit one dependent full-scope capture through the existing PB adapter."""
    spec = campaign_rows.load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    census_path = workspace / "census.json"
    census = json.loads(census_path.read_text())
    if census.get("model") != spec["model"]:
        raise RuntimeError("capture census and spec name different models")
    manifest = workspace / "capture-manifest.json"
    row = campaign_rows.prismabuild_row(spec, ["--model", spec["model"],
        "--out", str(workspace / "capture-unused.pkl"),
        "--cache-dir", str(workspace / "capture-cache"),
        "--calibration-census", str(census_path),
        "--capture-calibration-out", str(workspace / "calibration-cache"),
        *spec["campaign_argv"]],
        mem_gb=campaign_rows.row_memory_gb(spec, sorted(census["counts"]), census),
        timeout_s=int(args.timeout_s),
        # Capture is likewise not an anchor-pricing row and makes no durable
        # anchor-counter reports.
        progress_phases=())
    manifest.write_text(json.dumps([row], indent=2) + "\n")
    if explicit_campaign_options(spec['campaign_argv']).get("streaming", False):
        (workspace/'capture-resources.json').write_text(json.dumps(
            campaign_rows.streamed_resource_plan(spec, census, sorted(census['counts'])),
            indent=2, sort_keys=True)+'\n')
    if args.submit:
        return _pbcampaign(manifest, wait_s=args.wait_s,
                           receipts=workspace / "capture-receipts.json")
    return 0


def cmd_plan(args) -> int:
    spec = campaign_rows.load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    census = json.loads((workspace / "census.json").read_text())
    if census.get("model") != spec["model"]:
        raise RuntimeError(
            f"census was taken on {census.get('model')!r}, the spec names "
            f"{spec['model']!r}")
    acquisition, acquisition_binding = _planned_campaign_acquisition(spec, args, census)
    calibration_cache = _calibration_cache_binding(
        getattr(args, "calibration_cache", None), workspace / "census.json")
    spec_options = explicit_campaign_options(spec['campaign_argv'])
    selected_source = spec_options.get('streaming', False)
    if selected_source and calibration_cache is None:
        raise RuntimeError('streaming anchor rows require a hash-bound complete calibration cache')
    proof = _planned_source_identity_cache(
        getattr(args, "source_identity_cache", None), spec.get("source_identity_cache"))
    if proof and not selected_source:
        raise RuntimeError('--source-identity-cache binds streaming anchor rows only')
    if selected_source and not proof:
        raise RuntimeError(SOURCE_IDENTITY_REQUIRED)
    source_identity_cache = _source_identity_cache_binding(
        proof, spec["model"], calibration_cache)
    seed_rows, seed_workspace = None, None
    if getattr(args, 'seed_workspace', None):
        if args.seed_checkpoint or args.seed_wire_dir:
            raise RuntimeError('seed workspace is exclusive with a global seed checkpoint/wire directory')
        seed_rows, seed_workspace = _seed_workspace_rows(args.seed_workspace,
            census=census, calibration_cache=calibration_cache)
    groups = census["anchor_groups"]
    if not groups:
        raise RuntimeError("census reports no anchor group to price")

    acquisition_plan = None
    if acquisition is not None:
        from prismaquant.tessera_campaign_table import campaign_acquisition_scope
        requested_keys, _ = campaign_acquisition_scope(acquisition, groups)
        active = [key for key in requested_keys if any(
            qs for name in groups[key] for qs in acquisition["requests"][name].values())]
        deferred = [key for key in requested_keys if key not in active]
        acquisition_plan = {"binding": acquisition_binding, "identity": acquisition["identity"],
            "requested_groups": requested_keys, "active_groups": active,
            "deferred_groups": [{"key": key, "members": sorted(groups[key]),
                "reason": "all_requested_families_deferred_no_measurement_work"} for key in deferred]}
        groups = {key: groups[key] for key in active}

    experts_per_row = getattr(args, "experts_per_row", 0)
    partition_records, partition_stacks = {}, {}
    if experts_per_row:
        from prismaquant.tessera_campaign_table import parse_rate_band
        band = parse_rate_band(spec_options.get('rate_band'))
        if (type(experts_per_row) is not int or experts_per_row <= 0
                or args.groups_per_row != 1 or getattr(args, "row_work_profile", None)
                or args.stack_sample is not None or getattr(args, "seed_workspace", None)
                or args.seed_checkpoint or args.seed_wire_dir
                or any(spec_options.get(field) for field in ('seed_checkpoint', 'seed_wire_dir', 'research_exact_member'))
                or band is None or band[0] != band[1]
                or spec_options.get('max_rounds', 0) != 1):
            raise RuntimeError("expert partition requires positive size, groups-per-row=1, pinned band, max-rounds=1, no sampling/seeds/work-profile")
        from prismaquant.tessera_expert_projection import carried_units
        _source, partition_records, partition_stacks = carried_units(census["expert_projection"])

    stack_sample: dict[str, dict] = {}
    if args.stack_sample is not None:
        size_source = getattr(args, "stack_sample_sizes", "probe") or "probe"
        if not args.probe:
            raise RuntimeError(
                "--stack-sample needs --probe: the stack row's currency is the "
                "packed probe's h_trace, whatever the draw is proportional to")
        from prismaquant.model_profiles import detect_profile
        stack_sample = sample_stack_groups(
            groups, load_probe_h_trace(args.probe), profile=detect_profile(spec["model"]),
            stack_sample=int(args.stack_sample), seed=int(args.stack_sample_seed),
            audit_rate=int(args.audit_rate), sizes=size_source, census=census)
        priced = sum(len(entry["sampled"]) for entry in stack_sample.values())
        frame = sum(len(groups[key]) for key in stack_sample)
        print(f"[dispatch] sampled {priced} of {frame} routed expert units "
              f"across {len(stack_sample)} stack(s), "
              f"{sum(len(e['audit']) for e in stack_sample.values())} audited")

    units_dir = workspace / "units"
    ordered = sorted(groups)
    bundles = list(enumerate(
        ordered[index:index + args.groups_per_row]
        for index in range(0, len(ordered), args.groups_per_row)))
    work_profile = None
    if getattr(args, 'row_work_profile', None):
        bundles, work_profile = work_profile_bundles(
            Path(args.row_work_profile), census_path=workspace / 'census.json',
            groups=groups, campaign_argv=spec['campaign_argv'],
            groups_per_row=args.groups_per_row)

    expanded_bundles = []
    for index, bundle in bundles:
        if experts_per_row and bundle[0].startswith("s:"):
            from prismaquant.tessera_campaign_selection import EXPERT_PARTITION_SCHEMA
            by_expert = {}
            for name in groups[bundle[0]]:
                if name not in partition_records:
                    raise RuntimeError(f"expert partition has no producer record for {name}")
                by_expert.setdefault(partition_records[name]["expert"], []).append(name)
            expert_ids = sorted(by_expert)
            count = (len(expert_ids) + experts_per_row - 1) // experts_per_row
            for part_index in range(count):
                selected_experts = expert_ids[part_index * experts_per_row:(part_index + 1) * experts_per_row]
                part = {"schema": EXPERT_PARTITION_SCHEMA, "experts_per_row": experts_per_row,
                        "index": part_index, "count": count, "rate_q256": band[0],
                        "members": sorted(name for expert in selected_experts for name in by_expert[expert])}
                expanded_bundles.append((index, bundle, part))
        else:
            expanded_bundles.append((index, bundle, None))

    rows: list[dict] = []
    planned: list[dict] = []
    selection_writes: list[tuple[Path, str]] = []
    for index, bundle, part in expanded_bundles:
        row_id = f"row-{index:04d}" + (f"-p{part['index']:04d}" if part is not None else "")
        entries = []
        for key in bundle:
            entry = {"key": key, "members": sorted(groups[key])}
            if key in stack_sample:
                entry.update(stack_sample[key])
            if part is not None:
                entry["partition"] = part
            entries.append(entry)
        selection = {
            # A file that samples says so in its schema; one that does not
            # stays byte-identical to what every row before 2026-09-06 read.
            "schema": ("prismaquant.tessera_campaign_units.v3" if part is not None
                       else UNITS_SCHEMA_V2 if stack_sample else UNITS_SCHEMA),
            "model": spec["model"],
            "layer_stride": census["layer_stride"],
            "groups": entries,
        }
        units_path = units_dir / f"{row_id}.json"
        selection_writes.append((units_path, json.dumps(selection, indent=2, sort_keys=True) + "\n"))
        row_dir = workspace / "rows" / row_id
        from prismaquant.tessera_campaign_table import expert_partition_members
        from prismaquant.tessera_campaign_selection import selection_priced_units
        if part is not None:
            expert_partition_members(selection, unit_records=partition_records,
                stack_of=partition_stacks, rate_band=spec_options.get('rate_band'),
                max_rounds=spec_options.get('max_rounds', 0))
        members = sorted(selection_priced_units(selection)[0])
        if acquisition is not None:
            from prismaquant.tessera_campaign_table import campaign_acquisition_row_scope
            campaign_acquisition_row_scope(acquisition, census["anchor_groups"], selected=members)
        argv = [
            "--model", spec["model"],
            "--out", str(row_dir / "cost.pkl"),
            "--cache-dir", str(row_dir / "cache"),
            "--checkpoint", str(row_dir / "cost.anchors.json"),
            "--units", str(units_path),
            "--calibration-census", str(workspace / "census.json"),
            *spec["campaign_argv"],
        ]
        if calibration_cache:
            argv += ["--calibration-cache", calibration_cache["path"],
                     "--calibration-cache-sha256", calibration_cache["sha256"]]
        if source_identity_cache:
            argv += ["--source-identity-cache", source_identity_cache["path"],
                     "--source-identity-cache-sha256", source_identity_cache["sha256"]]
        row_seed = (_seed_for_selection(seed_rows, bundle, selection)
                    if seed_rows is not None else None)
        if row_seed is not None:
            argv += ['--seed-checkpoint', row_seed['checkpoint'],
                     '--seed-wire-dir', row_seed['wire_dir']]
        if args.seed_checkpoint:
            argv += ["--seed-checkpoint", str(args.seed_checkpoint)]
            if args.seed_wire_dir:
                argv += ["--seed-wire-dir", str(args.seed_wire_dir)]
        # The demand is derived from the argv the row will run, seed flags
        # included, exactly as ``verify_row_demand`` re-derives it at submit:
        # a seed makes the row run the load-all head, and a plan read off the
        # spec's argv alone priced that row at its stream window instead
        # (PQ #1686).
        row_spec = {**spec, "campaign_argv": argv}
        row = campaign_rows.prismabuild_row(spec, argv,
                   mem_gb=campaign_rows.row_memory_gb(row_spec, members, census, selected_source=selected_source),
                   timeout_s=(None if args.timeout_s is None else int(args.timeout_s)))
        if part is not None:
            effective_argv = _inner_campaign_argv(row)
            effective_options = explicit_campaign_options(effective_argv)
            expert_partition_members(selection, unit_records=partition_records,
                stack_of=partition_stacks, rate_band=effective_options.get('rate_band'),
                max_rounds=effective_options.get('max_rounds', 0),
                seeded=any(effective_options.get(field) for field in ('seed_checkpoint', 'seed_wire_dir', 'research_exact_member')))
        if acquisition is not None:
            _check_acquisition_binding(_campaign_acquisition_argv(_inner_campaign_argv(row)),
                acquisition_binding, refusal=RuntimeError, where="acquisition row argv")
        rows.append(row)
        predicted_work = {}
        if work_profile is not None and not bundle[0].startswith('s:'):
            timing = work_profile['profile']
            pricing = math.fsum(timing['group_pricing_seconds'][key] for key in bundle)
            startup = timing['startup_seconds']
            predicted_work = {'predicted_work': {
                'startup_seconds': startup, 'pricing_seconds': pricing,
                'startup_fraction': 1.0 / (1.0 + pricing / startup)}}
        planned.append({"row_id": row_id, "groups": bundle, "members": sorted(members),
                        "dir": str(row_dir), "units": str(units_path),
                        **predicted_work,
                        **({'expert_partition_selection': selection} if part is not None else {}),
                        **({'seed': row_seed} if row_seed is not None else {}),
                        **({'resources': campaign_rows.streamed_resource_plan(row_spec, census, members,
                            selected_source=True)} if selected_source else {})})

    # PB alone admits and places these independently retryable rows according
    # to their actual source/capture preparation and resident encoding demand.
    # All this decides is which rows it is handed: a row too wide for the box
    # is declined here rather than submitted for an admission that cannot
    # come, and the rest of the plan goes on being work.
    per_box = int(args.rows_per_box)
    row_memory_gb = {entry["row_id"]: int(row["demand"]["mem_gb"])
                     for entry, row in zip(planned, rows)}
    admissible, inadmissible = partition_rows_by_fit(
        row_memory_gb, per_box, spec.get("box_memory_gb"))
    members_by_row = {entry["row_id"]: entry["members"] for entry in planned}
    for record in inadmissible:
        record["members"] = members_by_row[record["row_id"]]
    admitted = set(admissible)
    for entry in planned:
        entry["admissible"] = entry["row_id"] in admitted

    manifest = workspace / "manifest.json"
    plan = {
        "schema": PLAN_SCHEMA,
        "model": spec["model"],
        # The spec this plan was derived from, so ``check`` and ``submit`` can
        # re-derive every row's demand without being told again. A plan
        # written before this field exists is checked with an explicit
        # ``--spec``.
        "spec": str(args.spec),
        "census": str(workspace / "census.json"),
        "calibration_cache": calibration_cache,
        **({"acquisition": acquisition_plan} if acquisition_plan is not None else {}),
        **({"source_identity_cache": source_identity_cache}
           if source_identity_cache is not None else {}),
        "manifest": str(manifest),
        "groups_per_row": int(args.groups_per_row),
        **({'row_work_profile': work_profile} if work_profile is not None else {}),
        "rows_per_box": per_box,
        "row_memory_gb": row_memory_gb,
        # The reservation those demands carry, stated once for the whole plan
        # because it is a per-row constant, with where it came from. Zero
        # means the spec declared none, and every row's phase plan records the
        # same thing in its own ``baseline_policy``, so a reader cannot
        # mistake an absent reservation for a covered one.
        "process_baseline_bytes": campaign_rows.process_baseline(spec)[0],
        "process_baseline_policy": campaign_rows.process_baseline(spec)[1],
        # The other term outside the phase deltas: the margin the row's own
        # guard holds back from the cap. Recorded because a demand that does
        # not carry it is admitted and then refused.
        "guard_margin_bytes": campaign_rows.guard_margin_bytes(),
        # The rows the manifest does not hold, at the demand they were derived
        # at. A reader of the plan sees the whole layout; a reader of the
        # manifest sees only what was submitted.
        "inadmissible_rows": inadmissible,
        **({'seed_workspace': seed_workspace} if seed_workspace is not None else {}),
        "seed_checkpoint": (None if not args.seed_checkpoint
                            else str(args.seed_checkpoint)),
        # The draw itself, whole: which experts stand for their stack, under
        # what inclusion probability, from which probe and which seed. It is
        # here as well as in every units file because the plan is the thing a
        # reader audits, and an estimate built on a sample is only checkable
        # against the pi it was drawn under.
        "stack_sample": {
            "size": (None if args.stack_sample is None else int(args.stack_sample)),
            "seed": int(args.stack_sample_seed),
            "audit_rate": int(args.audit_rate),
            "probe": (None if not args.probe else str(args.probe)),
            # Which per-expert vector the draw was proportional to, written
            # only when it is not the probe's Fisher vector -- so a plan made
            # without the flag is byte-identical to the ones already on disk,
            # and an absent field means ``probe`` exactly as an absent
            # ``sizes`` block on a record does.
            **({} if (getattr(args, "stack_sample_sizes", "probe") or "probe")
               == "probe" else {"sizes": args.stack_sample_sizes}),
            "stacks": stack_sample,
        },
        "rows": planned,
    }

    # Every published row names the bytes PrismaBuild's prewarm loop warms it
    # from, so a subset copied out of manifest.json carries them too. Only the
    # producer knows a row's read set, and a row whose read set cannot be
    # derived refuses the plan here.
    published_rows, manifest_writes = planned_data_manifests(
        workspace, plan,
        {str(path): json.loads(text) for path, text in selection_writes},
        [row for entry, row in zip(planned, rows) if entry["admissible"]])

    # A refused fit check, or a row with no derivable read set, must not
    # rewrite selections still named by an existing published manifest.
    # Derive every row and every data manifest before publishing bytes.
    units_dir.mkdir(parents=True, exist_ok=True)
    for units_path, selection_text in selection_writes:
        units_path.write_text(selection_text)
    for path, blob in manifest_writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    manifest.write_text(json.dumps(published_rows, indent=2) + "\n")
    (workspace / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[dispatch] planned {len(rows)} rows over {len(ordered)} anchor "
          f"groups, {len(admissible)} submitted -> {manifest}")
    return 0
