"""Research proposal identity checks, independent of proposal generation."""
from .schemas import Contract
from .digests import DIRECT_ASCII_LAX

SCHEMA = 'prismaquant.tessera_sampled_stack_proposal.v1'
_require = Contract(ValueError).require


def selected_assignment_sha256(assignment):
    """Bind the expanded member assignment, not its optimizer super-item IDs."""
    _require(isinstance(assignment, dict) and assignment and
             all(isinstance(k, str) and isinstance(v, str) for k, v in assignment.items()),
             'selected research assignment must be complete qname/format pairs')
    return DIRECT_ASCII_LAX.sha256(assignment)


def require_research_proposal_assignment(proposal, assignment, canonicalize_assignment, *, pilot_joint_binding=None):
    """Bind an export request to its reviewed research proposal; later gates own eligibility."""
    _require(isinstance(proposal, dict) and proposal.get('schema') == SCHEMA
             and proposal.get('status') == 'research_proposal'
             and proposal.get('production_export_authority') is False
             and proposal.get('validation_export_eligible') is None
             and proposal.get('research_validation_permitted') is True,
             'research export needs a pending-integrity sampled proposal')
    _require(proposal.get('pilot', {}).get('status') == 'diagnostic_pilot',
             'research proposal lost diagnostic pilot identity')
    expected = proposal.get('expanded_assignment')
    actual = canonicalize_assignment(assignment)
    _require(actual == expected and selected_assignment_sha256(actual) ==
             proposal.get('selected_assignment_sha256'),
             'research export assignment differs from expanded sampled proposal')
    if pilot_joint_binding is not None:
        _require(proposal.get('input_bindings', {}).get('pilot_joint_cost') == pilot_joint_binding,
                 'research selected cache pilot cost binding differs')
    return {'schema': 'prismaquant.tessera_sampled_validation_export_binding.v1',
            'proposal_schema': SCHEMA,
            'selected_assignment_sha256': proposal['selected_assignment_sha256'],
            'validation_export_eligible': None,
            'production_export_authority': False}
