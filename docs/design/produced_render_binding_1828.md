# Produced-render binding prerequisite

Issue #1828 is a CPU prerequisite for #870, not the completed rendered-weight
writer or reader migration. It does not enable a production path.

## Reuse the admitted-owner lifecycle

`prismaquant.produced_render_publication.ProducedRenderPublication` extends
`BoundaryProducedPublication`. It inherits binding, prewrite credit, descriptor
validation, mover publication, reader context, retirement and release. It does
not create another cache, dispatcher, durable ledger or PB client implementation.

Binding reads the producer's sealed template and live admitted claim through
the existing public PB APIs. The template must authorize the `renders` slot as
`payload` and must not be write-only: the producer needs its rendered weights.
The shared binding path checks these lane constraints before declaring or
admitting an instance. Direct construction checks the same constraints. The
boundary adapter's existing lane policy is unchanged.

`render_batch_id_for(qname, fmt)` uses the digest owner's existing
`DIRECT_ASCII_STRICT` profile to hash a versioned, unambiguous JSON coordinate
containing the full admitted owner action key, template digest, qualified Linear
name and registry-canonical format name. Format normalization matches the weight
writer: strip surrounding whitespace, uppercase, then call
`canonical_format_name`. Aliases that name the same cache entry share its batch
ID; whitespace-only formats refuse. Linear names are not sanitized or truncated.
Rebinding and retrying the same owner preserve the logical batch ID; PB still
authenticates the attempt. This ID is not a payload digest, generation identifier
or substitute for PB's instance and descriptor checks. The inherited boundary
`batch_id_for` API remains unchanged.

## Evidence and remaining integration

The CPU fixture creates a real private PB queue, sealed producer request,
queue claim and declared template. Its broker-shaped control record comes from
the existing fixture helper, not a running broker or cgroup. It checks matching
and foreign attempts, restart identity, format alias/whitespace normalization,
coordinate separation, invalid lane policy before instance filing,
prewrite before the first fixture byte, descriptor validation and abort after
all planned fixture paths are absent. Shared method identity checks prevent a
second publication or retirement implementation.

These checks do not qualify a rendered-weight cache entry, a reader residency
map, a mover round trip or a production campaign. #870 still requires:

- the existing `ProductionWeightCache` writer to claim credit before tensor
  preparation or serialization, without charging an existing valid entry again;
- bounded, byte-identical output publication and authenticated descriptors;
- the existing resident-prefetch reader to validate PB-produced bindings through
  the public queue seam; and
- restart, retention and retirement evidence for that integrated path.

No format, serialized tensor bytes, numerical method, calibration contract,
serving gate, runtime pin or production default changes here. GPU-bound cache
fill and resident-prefetch requirements remain unchanged. This control-plane
prerequisite makes no speed, KL, bpp, residency or GPU qualification claim; those
comparisons belong to the integrated path on the same calibration contract.
