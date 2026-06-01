#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  Splat Forge - calibrated DJI dual-fisheye -> equirect stitcher
#
#  Copyright (c) DoubTech / Splat Forge.
#  Use of this source code is governed by the LICENSE file in the repository
#  root. See that file for the full terms.
# =============================================================================
"""Stitch a DJI Avata 360 OSV's two fisheye lenses into one equirectangular
panorama, with a lens model CALIBRATED to match (and on the seam, beat) the
DJI-Studio stitch.

GEOMETRY (verified against ground-truth frames)
-----------------------------------------------
The Avata 360 mounts two ~200deg lenses VERTICALLY:
  * lens A (top)    points up   -> the sky hemisphere -> TOP half of equirect.
  * lens B (bottom) points down -> the ground (where the field / bench /
    fence detail lives) -> BOTTOM half of equirect.
The two lenses each see a little past 90deg, so they OVERLAP around the
horizon (equirect lat ~ 0) — that overlap band is the seam we blend.

World frame: +Y up, +Z forward, +X right.
  equirect (u,v): lon = u/W*2pi - pi ; lat = pi/2 - v/H*pi  (v=0 is the top).
  ray d = (cos lat sin lon, sin lat, cos lat cos lon).

Each lens is a :class:`~fisheye_lens.KBModel` (Kannala-Brandt) whose axis is
+Z in its own frame. A per-lens rotation maps a world ray into the lens frame;
lens A's base is Rx(-90deg) (world +Y -> lens +Z), lens B's base is Rx(+90deg)
(world -Y -> lens +Z). Calibration refines each lens's intrinsics
(f, cx, cy, k1..k4) AND a small per-lens rotation delta + a global yaw so the
rendered equirect lines up with the DJI reference.

This module is the SHARED math for two deliverables: this offline (high
quality) stitcher, and the live-preview GLSL shader (a GPU port of
``render_equirect``). Calibration is done here; the fitted model is saved to
JSON for both consumers.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np

import sys

from .lens import KBModel  # noqa: E402


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 rotation (cv2.Rodrigues wrapper)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return R


@dataclass
class DualFisheyeRig:
    """A calibrated two-lens fisheye rig that stitches to equirect.

    :ivar lensA: top-lens Kannala-Brandt model.
    :ivar lensB: bottom-lens Kannala-Brandt model.
    :ivar rvecA: small rotation delta (Rodrigues) on top of lens A's
        Rx(-90deg) base, mapping world rays into lens A's frame.
    :ivar rvecB: rotation delta on top of lens B's Rx(+90deg) base.
    :ivar yaw_deg: global yaw (about world +Y) to match DJI's heading.
    :ivar blend_deg: feather half-width (deg) around each lens's FOV edge.
    """
    lensA: KBModel
    lensB: KBModel
    rvecA: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rvecB: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0
    blend_deg: float = 12.0
    # Relative mounting of lens B vs lens A. Back-to-back 360 sensors are
    # often rolled ~180deg and sometimes mirrored relative to each other;
    # a continuous optimiser can't discover a 180deg flip, so these capture
    # the discrete part of the mount (coarse-searched), with rvecB the fine
    # residual on top.
    b_roll_deg: float = 0.0      # roll of lens B about its own (+Z) axis
    b_mirror: bool = False       # lens B image is left-right mirrored
    # Global output handedness. Self-consistency calibration is BLIND to a
    # global mirror (mirroring both lenses keeps them consistent), so the
    # fit can land on a mirrored-but-consistent solution. Only comparison to
    # the real world (DJI) breaks the tie — verified the DJI Avata pipeline
    # needs this flip. Applied to the final equirect only (not the overlap
    # consistency, which is mirror-invariant).
    flip_x: bool = False

    @staticmethod
    def default(in_size: int = 3840, fov_deg: float = 200.0) -> "DualFisheyeRig":
        """Equidistant starting rig (k=0) for a square fisheye of ``in_size``."""
        return DualFisheyeRig(
            lensA=KBModel.equidistant(in_size, fov_deg),
            lensB=KBModel.equidistant(in_size, fov_deg),
        )

    # base rotations: world axis of each lens -> +Z lens axis.
    # Rx(+90) maps world +Y (up) -> +Z; Rx(-90) maps world -Y (down) -> +Z.
    def _RA(self) -> np.ndarray:
        base = _rodrigues([math.pi / 2, 0, 0])       # +Y (up)   -> +Z
        return _rodrigues(self.rvecA) @ base

    def _RB(self) -> np.ndarray:
        base = _rodrigues([-math.pi / 2, 0, 0])      # -Y (down) -> +Z
        roll = _rodrigues([0, 0, math.radians(self.b_roll_deg)])  # about lens axis
        return roll @ _rodrigues(self.rvecB) @ base

    def _Ryaw(self) -> np.ndarray:
        return _rodrigues([0, math.radians(self.yaw_deg), 0])


def _equirect_rays(out_w: int, out_h: int) -> np.ndarray:
    """(H,W,3) world rays for an equirect grid (+Y up, +Z forward)."""
    lon = (np.arange(out_w) + 0.5) / out_w * 2 * math.pi - math.pi
    lat = math.pi / 2 - (np.arange(out_h) + 0.5) / out_h * math.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


def _lens_layer(img: np.ndarray, m: KBModel, R: np.ndarray,
                dirs: np.ndarray, blend_deg: float):
    """Sample one lens for every equirect ray. Returns (bgr, weight).

    ``R`` maps a world ray into the lens frame (axis +Z). Weight feathers
    from 1 inside the FOV to 0 at the FOV edge over ``blend_deg``.
    """
    d = dirs @ R.T                                   # world -> lens frame
    u, v, valid = m.project(d)
    theta = np.degrees(np.arccos(np.clip(d[..., 2], -1, 1)))
    edge = m.fov_deg / 2.0
    w = np.clip((edge - theta) / max(blend_deg, 1e-3), 0.0, 1.0)
    w = np.where(valid, w, 0.0).astype(np.float32)
    u = np.where(w > 0, u, -1).astype(np.float32)
    v = np.where(w > 0, v, -1).astype(np.float32)
    bgr = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return bgr.astype(np.float32), w[..., None]


def render_layers(imgA, imgB, rig: DualFisheyeRig, out_w, out_h):
    """Sample both lenses for every equirect ray. Returns (a, wa, b, wb)."""
    dirs = _equirect_rays(out_w, out_h) @ rig._Ryaw().T
    imgB_use = imgB[:, ::-1] if rig.b_mirror else imgB   # un-mirror sensor B
    a, wa = _lens_layer(imgA, rig.lensA, rig._RA(), dirs, rig.blend_deg)
    b, wb = _lens_layer(imgB_use, rig.lensB, rig._RB(), dirs, rig.blend_deg)
    return a, wa, b, wb


def _multiband_blend(a: np.ndarray, b: np.ndarray, m: np.ndarray,
                     levels: int = 5) -> np.ndarray:
    """Laplacian-pyramid (multiband) blend of layers ``a``/``b`` by mask ``m``
    (weight for ``a``, in [0,1]). Low frequencies blend over a wide band and
    high frequencies over a narrow one, so the seam vanishes even in flat sky
    where a simple feather leaves a visible mark. This is where we beat the
    DJI stitch on the horizon seam."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    ga, gb, gm = [a], [b], [m.astype(np.float32)]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1])); gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))
    out = ga[-1] * gm[-1][..., None] + gb[-1] * (1 - gm[-1][..., None])
    for i in range(levels - 1, -1, -1):
        sz = (ga[i].shape[1], ga[i].shape[0])
        la = ga[i] - cv2.pyrUp(ga[i + 1], dstsize=sz)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=sz)
        mi = gm[i][..., None]
        out = cv2.pyrUp(out, dstsize=sz) + la * mi + lb * (1 - mi)
    return np.clip(out, 0, 255).astype(np.uint8)


def _flow_align(a, b, wa, wb, m):
    """Optical-flow seam alignment: warp each lens toward the other in the
    overlap so parallax-displaced features coincide before blending — this
    removes the ghosting (doubled edges) a plain blend leaves. Lens A is
    warped toward B by ``(1-m)`` of the flow and B toward A by ``m``, so they
    meet weighted by the blend mask; outside the overlap the flow is zero."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    # Confine the warp to the SOLID overlap interior: erode the
    # both-lenses-strong region so the warp tapers to zero well before
    # either lens's black FOV boundary (warping near the boundary pulls the
    # black in). Then clamp the flow so a bad match can't yank a feature far.
    solid = ((wa[..., 0] > 0.25) & (wb[..., 0] > 0.25)).astype(np.float32)
    solid = cv2.erode(solid, np.ones((9, 9), np.float32))
    ov = cv2.GaussianBlur(solid, (0, 0), 6)[..., None]
    ga = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f_ab = dis.calc(ga, gb, None)      # a's pixel p -> b's pixel p + f_ab
        f_ba = dis.calc(gb, ga, None)      # b's pixel p -> a's pixel p + f_ba
    except Exception:  # noqa: BLE001
        f_ab = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 25, 5, 7, 1.5, 0)
        f_ba = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 4, 25, 5, 7, 1.5, 0)
    f_ab = np.clip(f_ab, -20, 20) * ov
    f_ba = np.clip(f_ba, -20, 20) * ov
    H, W = ga.shape
    xs, ys = np.meshgrid(np.arange(W).astype(np.float32), np.arange(H).astype(np.float32))
    mm = m[..., None]

    def warp(img, s):
        return cv2.remap(img, xs + s[..., 0], ys + s[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Morph each lens toward the other: where b dominates (m->0) warp a fully
    # into b's geometry via f_ba; where a dominates warp b via f_ab. Correct
    # backward-remap direction so the two views CONVERGE (not separate).
    return warp(a, (1 - mm) * f_ba), warp(b, mm * f_ab)


def _optimal_seam(diff: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Min-cost top-to-bottom seam ROW per column through the overlap band.

    Dynamic program left-to-right: the seam may step at most +/-1 row per
    column (stays continuous) and pays the lens disagreement ``diff`` where
    it cuts. It therefore threads through where the two lenses AGREE (grass,
    sky) and detours around high-disagreement objects (buildings), so each
    object lands wholly on one side -> no doubling."""
    H, W = diff.shape
    INF = np.float32(1e9)
    cost = np.where(overlap, diff, INF).astype(np.float32)
    dp = np.empty((H, W), np.float32); dp[:, 0] = cost[:, 0]
    bk = np.zeros((H, W), np.int8)
    for x in range(1, W):
        prev = dp[:, x - 1]
        up = np.empty(H, np.float32); up[:-1] = prev[1:]; up[-1] = INF   # r'=r+1
        dn = np.empty(H, np.float32); dn[1:] = prev[:-1]; dn[0] = INF    # r'=r-1
        stack = np.stack([dn, prev, up])
        idx = np.argmin(stack, axis=0)
        dp[:, x] = cost[:, x] + np.take_along_axis(stack, idx[None], 0)[0]
        bk[:, x] = idx.astype(np.int8) - 1
    seam = np.zeros(W, np.int32); seam[-1] = int(np.argmin(dp[:, -1]))
    for x in range(W - 2, -1, -1):
        seam[x] = int(np.clip(seam[x + 1] + bk[seam[x + 1], x + 1], 0, H - 1))
    return seam


def _seam_mask(a, b, wa, wb, feather: float = 3.0, seam_w: int = 1280) -> np.ndarray:
    """Build a blend mask (weight for lens A) from the optimal seam: 1 above
    the seam, 0 below, softened by a narrow feather for anti-aliasing.

    The DP runs at a reduced width (``seam_w``) and the resulting mask is
    upscaled — the seam is a smooth path, so this decouples the O(W) DP cost
    from the output resolution (fast at 6k/8k equirect)."""
    H, W = a.shape[:2]
    sw = min(W, seam_w); sh = max(2, int(round(H * sw / W)))
    a_s = cv2.resize(a, (sw, sh)); b_s = cv2.resize(b, (sw, sh))
    wa_s = cv2.resize(wa[..., 0], (sw, sh)); wb_s = cv2.resize(wb[..., 0], (sw, sh))
    overlap = (wa_s > 0.05) & (wb_s > 0.05)
    ga = cv2.cvtColor(np.clip(a_s, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(np.clip(b_s, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = cv2.GaussianBlur(np.abs(ga - gb), (0, 0), 1.5)
    seam = _optimal_seam(diff, overlap)
    rows = np.arange(sh)[:, None]
    m_s = (rows <= seam[None, :]).astype(np.float32)
    m = cv2.resize(m_s, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), feather)


def render_equirect(imgA: np.ndarray, imgB: np.ndarray, rig: DualFisheyeRig,
                    out_w: int = 3840, out_h: int = 1920,
                    multiband: int = 5, flow_blend: bool = False,
                    seam: bool = False) -> np.ndarray:
    """Stitch the two lens images into an equirect (uint8 BGR).

    ``multiband`` = Laplacian-pyramid levels for seam blending (0 = feather,
    used by the calibration inner loop for speed). ``flow_blend`` adds
    optical-flow seam alignment to kill parallax ghosting (best quality;
    slower — for final output / conversion, not the live preview).
    """
    a, wa, b, wb = render_layers(imgA, imgB, rig, out_w, out_h)
    if seam:
        m = _seam_mask(a, b, wa, wb)        # optimal seam routes around objects
    else:
        m = (wa[..., 0] / (wa[..., 0] + wb[..., 0] + 1e-6))
    if flow_blend:
        a, b = _flow_align(a, b, wa, wb, m)
    if multiband > 0:
        out = _multiband_blend(a, b, m, multiband)
    else:
        out = np.clip((a.astype(np.float32) * wa + b.astype(np.float32) * wb)
                      / (wa + wb + 1e-6), 0, 255).astype(np.uint8)
    if rig.flip_x:
        out = np.ascontiguousarray(out[:, ::-1])
    return out


# --------------------------------------------------------------------------- #
#  Telemetry stabilization (world-lock from the SRT attitude).
# --------------------------------------------------------------------------- #
def _Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate_equirect(eq: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Resample an equirect under a 3x3 world rotation ``R`` (wraps in lon)."""
    H, W = eq.shape[:2]
    lon = (np.arange(W) + 0.5) / W * 2 * math.pi - math.pi
    lat = math.pi / 2 - (np.arange(H) + 0.5) / H * math.pi
    lon, lat = np.meshgrid(lon, lat); cl = np.cos(lat)
    d = np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], -1) @ R.T
    nlon = np.arctan2(d[..., 0], d[..., 2]); nlat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = ((nlon / (2 * math.pi) + 0.5) * W).astype(np.float32)
    v = ((0.5 - nlat / math.pi) * H).astype(np.float32)
    return cv2.remap(eq, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def stabilize(eq: np.ndarray, yaw: float, pitch: float, roll: float,
              yaw_ref: float = 0.0, trim: float = 0.0, level: bool = True) -> np.ndarray:
    """World-stabilize an equirect from the drone attitude (deg).

    Yaw is referenced to the clip's start heading (``yaw_ref``) so forward =
    where the drone faced at clip start (DJI's convention), plus a constant
    mount ``trim``. ``level`` cancels the small pitch/roll so the horizon
    stays flat. Result: the world is held fixed as the drone turns — which
    is what helps frame-to-frame SfM matching and powers the preview/convert.
    """
    R = _Ry(math.radians((yaw - yaw_ref) + trim))
    if level:
        R = _Rx(math.radians(-pitch)) @ _Rz(math.radians(-roll)) @ R
    return rotate_equirect(eq, R)


def parse_srt_attitude(srt_path: Path) -> list[tuple[float, float, float, float]]:
    """Parse a DJI .SRT into [(t_sec, gb_yaw, gb_pitch, gb_roll), ...]."""
    import re
    txt = Path(srt_path).read_text(errors="ignore")
    ts = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d+)\s*-->")
    py = re.compile(r"gb_yaw\s*:\s*(-?\d+\.?\d*)", re.I)
    pp = re.compile(r"gb_pitch\s*:\s*(-?\d+\.?\d*)", re.I)
    pr = re.compile(r"gb_roll\s*:\s*(-?\d+\.?\d*)", re.I)
    out = []
    for b in re.split(r"\n\s*\n", txt):
        mt = ts.search(b); my = py.search(b)
        if not (mt and my):
            continue
        h, m, s, ms = map(int, mt.groups())
        t = h * 3600 + m * 60 + s + ms / 1000.0
        out.append((t, float(my.group(1)),
                    float(pp.search(b).group(1)) if pp.search(b) else 0.0,
                    float(pr.search(b).group(1)) if pr.search(b) else 0.0))
    out.sort()
    return out


def _att_at(att: list, t: float):
    """Nearest (yaw,pitch,roll) for time ``t`` from a parsed attitude list."""
    if not att:
        return 0.0, 0.0, 0.0
    best = min(att, key=lambda c: abs(c[0] - t))
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
#  Calibration: fit the rig so its render matches the DJI equirect.
# --------------------------------------------------------------------------- #
def _feat(bgr: np.ndarray, blur: float = 3.0) -> np.ndarray:
    """A grading-invariant, SMOOTH alignment feature: heavily-blurred,
    zero-mean / unit-variance luma. Blurring removes the high-frequency
    local minima that make raw photometric fits stall; normalisation makes
    it robust to DJI's brightness/gain/colour grading."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    g -= g.mean()
    return g / (g.std() + 1e-6)


def _pack(rig: DualFisheyeRig) -> np.ndarray:
    a, b = rig.lensA, rig.lensB
    return np.array([
        a.f, a.cx, a.cy, a.k1, a.k2, a.k3, a.k4,
        b.f, b.cx, b.cy, b.k1, b.k2, b.k3, b.k4,
        *rig.rvecA, *rig.rvecB, rig.yaw_deg,
    ], dtype=np.float64)


def _unpack(p: np.ndarray, fov_deg: float, blend_deg: float) -> DualFisheyeRig:
    return DualFisheyeRig(
        lensA=KBModel(p[0], p[1], p[2], p[3], p[4], p[5], p[6], fov_deg),
        lensB=KBModel(p[7], p[8], p[9], p[10], p[11], p[12], p[13], fov_deg),
        rvecA=list(p[14:17]), rvecB=list(p[17:20]), yaw_deg=float(p[20]),
        blend_deg=blend_deg,
    )


def _consistency_resid(rig: DualFisheyeRig, AB, cw, ch,
                       blur: float = 3.0, band_deg: float = 30.0) -> np.ndarray:
    """Residual measuring how much the two lenses DISAGREE in their overlap.

    The >180deg lenses overlap in a band around the horizon (equirect lat~0,
    the middle rows — we always render with the rig's default orientation so
    that band is fixed). A correct rig makes lens A and lens B render the
    SAME thing there. Per frame we compare their blurred luma over a FIXED
    horizon band (fixed length — required by least_squares), each normalised
    over the overlapping pixels to kill the per-sensor exposure/colour
    difference, and WEIGHTED by the per-pixel overlap min(wa,wb) so only the
    genuinely-overlapping pixels drive the fit. Driving this to zero removes
    the seam tearing/ghosting; it's independent of where the rig points."""
    half = max(2, int(ch * band_deg / 180.0))
    r0, r1 = ch // 2 - half, ch // 2 + half
    out = []
    for A, B in AB:
        a, wa, b, wb = render_layers(A, B, rig, cw, ch)
        sl = slice(r0, r1)
        ga = cv2.GaussianBlur(cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                              .astype(np.float32), (0, 0), blur)[sl]
        gb = cv2.GaussianBlur(cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                              .astype(np.float32), (0, 0), blur)[sl]
        w = np.minimum(wa[sl, :, 0], wb[sl, :, 0])
        m = w > 0.05
        if m.sum() >= 50:
            ga = (ga - ga[m].mean()) / (ga[m].std() + 1e-6)
            gb = (gb - gb[m].mean()) / (gb[m].std() + 1e-6)
        out.append(((ga - gb) * np.sqrt(np.clip(w, 0, 1))).ravel().astype(np.float32))
    return np.concatenate(out)


def calibrate(pairs, init: DualFisheyeRig, fov_deg: float,
              calib_w: int = 768) -> tuple[DualFisheyeRig, float]:
    """Calibrate the FIXED inter-sensor geometry by self-consistency.

    The two lenses are rigidly mounted, so their relative orientation +
    intrinsics are a one-time camera property. We fit them so the lenses
    AGREE in their overlap band (no DJI reference needed, yaw-independent):
      1. coarse-search the discrete sensor-B mount (roll 0/180/... + mirror)
         — a 180deg flip is unreachable by a continuous optimiser;
      2. fine-tune lens intrinsics (both KB models) + the relative rotation
         + continuous roll, coarse-to-fine over blur.
    ``pairs`` may be (A,B) or (A,B,dji) tuples — only A,B are used.
    """
    from scipy.optimize import least_squares
    cw, ch = calib_w, calib_w // 2
    AB = [(p[0], p[1]) for p in pairs]

    # 1. discrete sensor-B orientation search (roll x mirror)
    best = None
    for mir in (False, True):
        for roll in range(0, 360, 30):
            init.b_mirror = mir; init.b_roll_deg = float(roll)
            init.rvecB = [0.0, 0.0, 0.0]
            e = float(np.mean(_consistency_resid(init, AB[:2], cw, ch, 4.0) ** 2))
            if best is None or e < best[0]:
                best = (e, mir, float(roll))
    init.b_mirror, init.b_roll_deg = best[1], best[2]
    init.rvecB = [0.0, 0.0, 0.0]
    print(f"[stitch] sensor-B mount: mirror={best[1]} roll={best[2]:.0f}deg "
          f"(overlap cost {best[0]:.4f}); fine-tuning ...", flush=True)

    # 2. continuous fine-tune: [fA,cxA,cyA,k1..4A, fB,cxB,cyB,k1..4B, rvecB(3), roll]
    def pack(r):
        a, b = r.lensA, r.lensB
        return np.array([a.f, a.cx, a.cy, a.k1, a.k2, a.k3, a.k4,
                         b.f, b.cx, b.cy, b.k1, b.k2, b.k3, b.k4,
                         *r.rvecB, r.b_roll_deg], float)

    def unpack(p):
        return DualFisheyeRig(
            KBModel(p[0], p[1], p[2], p[3], p[4], p[5], p[6], fov_deg),
            KBModel(p[7], p[8], p[9], p[10], p[11], p[12], p[13], fov_deg),
            rvecB=list(p[14:17]), b_roll_deg=float(p[17]),
            b_mirror=init.b_mirror, blend_deg=init.blend_deg)

    p0 = pack(init)
    s = init.lensA.f
    span = np.array([s*.3, 300, 300, .6, .6, .6, .6,
                     s*.3, 300, 300, .6, .6, .6, .6,
                     .4, .4, .4, 20])
    lo, hi = p0 - span, p0 + span
    rms = 0.0
    for blur in (6.0, 3.0, 1.5):
        def resid(p, blur=blur):
            return _consistency_resid(unpack(p), AB, cw, ch, blur)
        res = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                            x_scale="jac", diff_step=3e-3, max_nfev=120,
                            ftol=1e-6, verbose=1)
        p0 = res.x
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        print(f"[stitch]   blur={blur}: overlap RMS={rms:.4f}", flush=True)
    return unpack(p0), rms


def _load_triples(calib_dir: Path, times: list[int]):
    out = []
    for t in times:
        A = cv2.imread(str(calib_dir / f"lensA_t{t}.jpg"))
        B = cv2.imread(str(calib_dir / f"lensB_t{t}.jpg"))
        D = cv2.imread(str(calib_dir / f"dji_t{t}.jpg"))
        if A is not None and B is not None and D is not None:
            out.append((A, B, D))
    return out


def load_rig(path: Path) -> DualFisheyeRig:
    """Load a calibrated rig from rig.json."""
    j = json.loads(Path(path).read_text())
    return DualFisheyeRig(KBModel(**j["lensA"]), KBModel(**j["lensB"]),
                          j.get("rvecA", [0, 0, 0]), j.get("rvecB", [0, 0, 0]),
                          j.get("yaw_deg", 0.0), j.get("blend_deg", 12.0),
                          j.get("b_roll_deg", 0.0), j.get("b_mirror", False),
                          j.get("flip_x", False))


def _ffbin(name: str) -> str:
    import shutil as _sh
    for c in (name, f"/c/ProgramData/chocolatey/bin/{name}"):
        p = _sh.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return name


def stitch_video(osv: Path, srt: Path | None, rig: DualFisheyeRig, out_dir: Path,
                 fps: float = 2.0, width: int = 3840, start: float = 0.0,
                 duration: float | None = None, trim: float = 0.0,
                 stabilize_on: bool = True, prefix: str = "eq_",
                 mp4: Path | None = None) -> int:
    """Convert a DJI OSV (+SRT) to stabilized equirect frames using the
    calibrated rig + optimal-seam blend + telemetry world-stabilization.

    The OSV's two fisheye streams are extracted at ``fps``, stitched and
    (optionally) stabilized per frame from the SRT attitude (clip-start
    reference + ``trim``), written as ``<prefix>NNNNN.jpg`` in ``out_dir``.
    Returns the frame count. This is the in-house replacement for the DJI
    Studio equirect export — the input to the SfM/3DGS pipeline.
    """
    import subprocess
    import tempfile
    ff = _ffbin("ffmpeg")
    att = parse_srt_attitude(srt) if (srt and Path(srt).exists()) else []
    yaw_ref = att[0][1] if att else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        ta = Path(td) / "a"; tb = Path(td) / "b"; ta.mkdir(); tb.mkdir()
        base = [ff, "-y", "-ss", str(start), "-i", str(osv)]
        dur = (["-t", str(duration)] if duration else [])
        subprocess.run(base + dur + ["-map", "0:v:0", "-vf", f"fps={fps}",
                       "-q:v", "2", str(ta / "f_%05d.jpg")], check=True, capture_output=True)
        subprocess.run(base + dur + ["-map", "0:v:1", "-vf", f"fps={fps}",
                       "-q:v", "2", str(tb / "f_%05d.jpg")], check=True, capture_output=True)
        fa = sorted(ta.glob("f_*.jpg")); fb = sorted(tb.glob("f_*.jpg"))
        n = min(len(fa), len(fb))
        print(f"[stitch-video] {n} frame(s) @ {fps}fps -> {out_dir}", flush=True)
        for k in range(n):
            A = cv2.imread(str(fa[k])); B = cv2.imread(str(fb[k]))
            eq = render_equirect(A, B, rig, width, width // 2, multiband=5, seam=True)
            if stabilize_on and att:
                t = start + k / max(fps, 1e-6)
                yaw, pit, rol = _att_at(att, t)
                eq = stabilize(eq, yaw, pit, rol, yaw_ref, trim, level=True)
            cv2.imwrite(str(out_dir / f"{prefix}{k:05d}.jpg"), eq,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{n}", flush=True)
        if mp4:
            subprocess.run([ff, "-y", "-framerate", str(fps), "-i",
                            str(out_dir / f"{prefix}%05d.jpg"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
                           check=True, capture_output=True)
            print(f"[stitch-video] wrote {mp4}", flush=True)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="fit a rig to lensA/lensB/dji triples")
    pc.add_argument("--calib-dir", required=True, type=Path)
    pc.add_argument("--times", default="20,60,100,140,180")
    pc.add_argument("--fov", type=float, default=200.0)
    pc.add_argument("--out", type=Path, default=Path("calib/rig.json"))
    pc.add_argument("--calib-w", type=int, default=768)

    pr = sub.add_parser("render", help="stitch one triple with a saved rig")
    pr.add_argument("--rig", type=Path, default=None,
                    help="rig.json (default: equidistant prior)")
    pr.add_argument("--lensA", required=True, type=Path)
    pr.add_argument("--lensB", required=True, type=Path)
    pr.add_argument("--out", required=True, type=Path)
    pr.add_argument("--width", type=int, default=3840)
    pr.add_argument("--fov", type=float, default=200.0)
    pr.add_argument("--compare", type=Path, default=None,
                    help="DJI equirect to diff against (writes _cmp.jpg).")

    pv = sub.add_parser("stitch-video", help="OSV+SRT -> stabilized equirect frames (+mp4)")
    pv.add_argument("--osv", required=True, type=Path)
    pv.add_argument("--srt", type=Path, default=None)
    pv.add_argument("--rig", type=Path, default=Path("calib/rig.json"))
    pv.add_argument("--out-dir", required=True, type=Path)
    pv.add_argument("--prefix", default="eq_")
    pv.add_argument("--fps", type=float, default=2.0)
    pv.add_argument("--width", type=int, default=3840)
    pv.add_argument("--start", type=float, default=0.0)
    pv.add_argument("--duration", type=float, default=None)
    pv.add_argument("--trim", type=float, default=0.0, help="forward-yaw trim (deg)")
    pv.add_argument("--no-stabilize", action="store_true")
    pv.add_argument("--mp4", type=Path, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "stitch-video":
        stitch_video(args.osv, args.srt, load_rig(args.rig), args.out_dir,
                     fps=args.fps, width=args.width, start=args.start,
                     duration=args.duration, trim=args.trim,
                     stabilize_on=not args.no_stabilize, prefix=args.prefix,
                     mp4=args.mp4)
        return

    if args.cmd == "calibrate":
        times = [int(x) for x in args.times.split(",")]
        pairs = _load_triples(args.calib_dir, times)
        if not pairs:
            raise SystemExit(f"no triples in {args.calib_dir}")
        in_size = pairs[0][0].shape[0]
        init = DualFisheyeRig.default(in_size, args.fov)
        print(f"[stitch] calibrating on {len(pairs)} frame(s), "
              f"init equidistant f={init.lensA.f:.1f} ...", flush=True)
        rig, rms = calibrate(pairs, init, args.fov, args.calib_w)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "lensA": asdict(rig.lensA), "lensB": asdict(rig.lensB),
            "rvecA": rig.rvecA, "rvecB": rig.rvecB,
            "yaw_deg": rig.yaw_deg, "blend_deg": rig.blend_deg,
            "b_roll_deg": rig.b_roll_deg, "b_mirror": rig.b_mirror,
            "flip_x": rig.flip_x,
        }, indent=2))
        print(f"[stitch] RMS grad residual={rms:.5f}  -> {args.out}", flush=True)
        return

    # render
    if args.rig and args.rig.exists():
        j = json.loads(args.rig.read_text())
        rig = DualFisheyeRig(KBModel(**j["lensA"]), KBModel(**j["lensB"]),
                             j.get("rvecA", [0, 0, 0]), j.get("rvecB", [0, 0, 0]),
                             j.get("yaw_deg", 0.0), j.get("blend_deg", 12.0),
                             j.get("b_roll_deg", 0.0), j.get("b_mirror", False),
                             j.get("flip_x", False))
    else:
        A0 = cv2.imread(str(args.lensA))
        rig = DualFisheyeRig.default(A0.shape[0], args.fov)
    A = cv2.imread(str(args.lensA)); B = cv2.imread(str(args.lensB))
    _seam_out = True   # optimal-seam blend for the final output (de-ghosts)
    if args.compare and args.compare.exists():
        # yaw is the rig pointing (per-frame) — auto-align it to DJI just for
        # a fair side-by-side, by maximising blurred-luma correlation.
        dj = cv2.resize(cv2.imread(str(args.compare)), (args.width, args.width // 2))
        djf = _feat(cv2.resize(dj, (512, 256)))
        best_y, best_e = rig.yaw_deg, 1e9
        for y in range(-180, 180, 5):
            rig.yaw_deg = float(y)
            e = float(np.mean((_feat(render_equirect(A, B, rig, 512, 256)) - djf) ** 2))
            if e < best_e:
                best_e, best_y = e, float(y)
        rig.yaw_deg = best_y
    out = render_equirect(A, B, rig, args.width, args.width // 2, seam=_seam_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[stitch] wrote {args.out}", flush=True)
    if args.compare and args.compare.exists():
        cmp = np.vstack([out, cv2.resize(cv2.imread(str(args.compare)),
                                         (args.width, args.width // 2))])
        cv2.imwrite(str(args.out.with_name(args.out.stem + "_cmp.jpg")), cmp,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[stitch] wrote comparison (ours / DJI stacked)", flush=True)


if __name__ == "__main__":
    main()
