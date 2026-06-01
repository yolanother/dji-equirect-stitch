# =============================================================================
#  panostitch — calibrated DJI dual-fisheye -> equirect 360 stitching
#  Use of this source code is governed by the LICENSE file in the repo root.
# =============================================================================
"""Calibrated dual-fisheye -> equirectangular stitching, telemetry
world-stabilization, and optimal-seam blending. See the repository README for
the calibration findings and usage."""
from .lens import (KBModel, reproject_rectilinear, reproject_equirect,
                   equirect_to_perspective)
from .stitch import (DualFisheyeRig, load_rig, render_equirect, render_layers,
                     stabilize, rotate_equirect, parse_srt_attitude,
                     calibrate, stitch_video)

__all__ = [
    "KBModel", "reproject_rectilinear", "reproject_equirect",
    "equirect_to_perspective", "DualFisheyeRig", "load_rig", "render_equirect",
    "render_layers", "stabilize", "rotate_equirect", "parse_srt_attitude",
    "calibrate", "stitch_video",
]
__version__ = "0.1.0"
