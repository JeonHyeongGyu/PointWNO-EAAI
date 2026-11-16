"""Public entry point of the PointWNO-EAAI release (EAAI-26-17015).

This package contains no model code of its own. It imports the unchanged research code under ``code/`` (the exact files
that trained the released checkpoints; their SHA-256 values are recorded in each checkpoint's ``run_meta.json``) and
exposes a small interface:

    from pointwno import build_model, load_checkpoint, count_params, run_config
    model = build_model("V2J2", seed=0)        # canonical PointWNO (db2, J = 2), 21,801,731 parameters
    model = build_model("FNOM", seed=0)        # parameter-matched PointFNO (34 x 26 modes), 21,801,731 parameters
"""
from .api import ARMS, REPO, build_model, count_params, load_checkpoint, run_config, make_vrg  # noqa: F401
