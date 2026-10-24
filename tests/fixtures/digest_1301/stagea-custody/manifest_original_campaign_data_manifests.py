import json

import os

def publish_row_manifests(campaign_manifest, workspace, *, out_dir, out_manifest, sizes_cache=None, stat=False, dry_run=False):
    src = campaign_manifest
    rows = json.load(open(src))
    if not isinstance(rows, list):
        raise ReadsetRefused(f"{src}: expected a list of pbcampaign rows")
    src_sha = file_sha256hex(src)

    sizes = None if stat else load_sizes(sizes_cache)
    campaign = CachedSizeCampaign(workspace, sizes)
    produced_by = deterministic_provenance(
        workspace, campaign, "stat" if stat else sizes_cache)

    out_rows = []
    report = []
    for row in rows:
        rid = row_id_of(row)
        if rid is None:
            raise ReadsetRefused(f"{src}: a row's argv names no single units/row-XXXX.json")
        if rid not in campaign.rows:
            raise ReadsetRefused(f"{src}: {rid} is not a row of {workspace}/plan.json")
        man = build_campaign_row_manifest(campaign, rid, produced_by, row.get("argv"))
        path = os.path.join(out_dir, f"{rid}.data-manifest.json")
        blob = json.dumps(man, indent=1, sort_keys=False).encode() + b"\n"
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        new_row = dict(row)
        new_row["data_manifest"] = path
        out_rows.append(new_row)
        report.append({
            "row_id": rid, "path": path,
            "sha256": bytes_sha256hex(blob),
            "manifest_bytes": len(blob),
            "entry_count": man["entry_count"],
            "total_bytes": man["total_bytes"],
            "captures": man["annotations"]["counts"]["captures"],
            "weight_extents": man["annotations"]["counts"]["weight_extents"],
            "seeds": man["annotations"]["counts"]["seeds"],
            "capture_bytes": man["annotations"]["bytes"]["captures"],
            "weight_bytes": man["annotations"]["bytes"]["weight_extents"],
            "seed_bytes": man["annotations"]["bytes"]["seeds"],
        })

    out_blob = json.dumps(out_rows, indent=1).encode() + b"\n"
    if not dry_run:
        tmp = out_manifest + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(out_blob)
        os.replace(tmp, out_manifest)

    summary = {
        "source_manifest": src,
        "source_manifest_sha256": src_sha,
        "out_manifest": out_manifest,
        "out_manifest_sha256": bytes_sha256hex(out_blob),
        "out_dir": out_dir,
        "rows": len(report),
        "total_bytes": sum(r["total_bytes"] for r in report),
        "dry_run": dry_run,
        "produced_by": produced_by,
        "manifests": report,
    }
    return summary

