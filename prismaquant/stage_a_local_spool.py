"""Former name of :mod:`prismaquant.produced_output_spool`.

The client was never specific to Stage A; every declared-output writer now
uses it under its generic name. These aliases keep branches written against
the old import working until they merge.
"""
from .produced_output_spool import (  # noqa: F401
    MAX_ENV,
    ROOT_ENV,
    ProducedOutputSpool as BoundaryOutputSpool,
    ProducedOutputSpoolRefused as BoundarySpoolRefused,
)
