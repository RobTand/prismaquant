"""The adjoint-capture lane alias (contract §5.1/§4.1).

The plan block and the adjoint read manifest name this entry point
(``prismaquant.joint_adjoint_capture``); the executing module is
``prismaquant.joint_cost_stage_a``. Importing either runs the same stage.
"""
from .joint_cost_stage_a import *  # noqa: F401,F403
from .joint_cost_stage_a import main

if __name__ == "__main__":
    raise SystemExit(main())
