"""The actual adaptive proposal must serialize its canonical surface."""
import json

from prismaquant.tessera_allocator import (
    adaptive_trellis_rate_surface, build_tessera_allocator_candidate,
)


def _records():
    return tuple(build_tessera_allocator_candidate(
        'u', (256, 256), family='TESSERA_E4M3_K1', body_rate_q256=q,
        layout='tight', schedule=None, alphabets=None,
        predicted_dloss=value, target_profile='research',
    ) for q, value in ((256, 8.0), (2048, 1.0)))


def test_adaptive_proposal_serializes_the_existing_canonical_surface():
    proposal = adaptive_trellis_rate_surface('TESSERA_E4M3_K1', _records())
    result = proposal.as_dict()
    assert result['surface'] == json.loads(proposal.surface.identity())
    assert result['surface']['proposed_q256']
    assert len(result['identity_sha256']) == 64
