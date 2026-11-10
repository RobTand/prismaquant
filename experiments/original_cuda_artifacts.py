"""Bind actual original-control sidecars without launching qualification."""
from pathlib import Path


def publish_control_artifacts(node_id, out, profile_evidence, *, inputs, max_artifact_bytes):
    """Publish actual sidecars through the single native result boundary."""
    from prismaquant.native_results import publish_result
    from prismaquant.source_generation import ORIGINAL_RUNTIME_DEPENDENCIES
    controls = list((out / 'controls').glob('*.json'))
    if len(controls) != 1:
        raise RuntimeError('successful node requires exactly one actual control artifact')
    paths = dict(control=controls[0], execution=out / 'test-result.json',
        action_result=out / 'action-result.json', netdata_sparky=out / 'netdata-sparky.json',
        netdata_sparklina=out / 'netdata-sparklina.json', torch_trace=Path(profile_evidence['path']))
    return publish_result(out / 'publication.json', producer_id=node_id, inputs=inputs,
                          artifacts=paths, dependencies=ORIGINAL_RUNTIME_DEPENDENCIES,
                          max_artifact_bytes=max_artifact_bytes)

