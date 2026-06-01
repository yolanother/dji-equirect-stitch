#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  Splat Forge - DJI 360 fisheye lens model + reprojection
#
#  Copyright (c) DoubTech / Splat Forge.
#  Use of this source code is governed by the LICENSE file in the repository
#  root. See that file for the full terms.
# =============================================================================
"""Kannala-Brandt fisheye lens model for the DJI Avata 360 / Osmo 360 lenses,
plus reprojection to rectilinear (pinhole) and equirectangular.

WHY
---
The ~200deg Avata lenses do NOT follow the equidistant model (r = f*theta);
our earlier equidistant rectification left residual distortion so SfM views
never reprojected consistently -> blurry splats. The Kannala-Brandt (KB)
model
    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    r = f * theta_d
is the standard model for >180deg lenses (OpenCV cv::fisheye). With k=0 it
reduces to equidistant, so this is a strict superset of what we had. The k
coefficients are CALIBRATED (see scripts/fisheye_calibrate.py / plumb-line)
so that world-straight lines (the SRAC track edges) reproject straight.

This module is the shared, dependency-light core: a KB model and the two
reprojections used everywhere downstream (rectilinear views for SfM, equirect
for preview / proper 360 output). A previewer CLI dumps a rectilinear + an
equirect render of one frame so calibration can be checked visually.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class KBModel:
    """Kannala-Brandt fisheye intrinsics for one lens.

    :param f: focal length in pixels (r = f * theta_d).
    :param cx: principal point x (px).
    :param cy: principal point y (px).
    :param k1..k4: KB distortion polynomial coefficients (0 = equidistant).
    :param fov_deg: full field of view (deg) used only to clip invalid rays.
    """
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    fov_deg: float = 200.0

    @staticmethod
    def equidistant(in_size: int, fov_deg: float = 200.0) -> "KBModel":
        """A k=0 (equidistant) starting model for a square fisheye image."""
        f = (in_size / 2.0) / math.radians(fov_deg / 2.0)
        return KBModel(f=f, cx=in_size / 2.0, cy=in_size / 2.0, fov_deg=fov_deg)

    def theta_to_r(self, theta: np.ndarray) -> np.ndarray:
        """Forward distortion: incidence angle theta -> image radius r (px)."""
        t2 = theta * theta
        theta_d = theta * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))
        return self.f * theta_d

    def project(self, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D rays (..,3, +Z = lens axis) to fisheye pixels.

        :returns: (u, v, valid) where valid marks rays inside the FOV.
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        n = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(np.clip(z / n, -1.0, 1.0))
        phi = np.arctan2(y, x)
        r = self.theta_to_r(theta)
        u = self.cx + r * np.cos(phi)
        v = self.cy + r * np.sin(phi)
        valid = theta <= math.radians(self.fov_deg / 2.0)
        return u.astype(np.float32), v.astype(np.float32), valid


def reproject_rectilinear(img: np.ndarray, m: KBModel, out_fov_deg: float,
                          out_size: int, R: np.ndarray | None = None) -> np.ndarray:
    """Render a rectilinear (pinhole) view through the KB model.

    :param R: optional 3x3 rotation aiming the view (cam +Z -> world dir).
    """
    f_pin = (out_size / 2.0) / math.tan(math.radians(out_fov_deg / 2.0))
    o = out_size / 2.0
    j, i = np.meshgrid(np.arange(out_size), np.arange(out_size))
    v = np.stack([(j - o) / f_pin, (i - o) / f_pin, np.ones_like(j, float)], axis=-1)
    if R is not None:
        v = v @ R.T
    u, vv, valid = m.project(v)
    u[~valid] = -1; vv[~valid] = -1
    return cv2.remap(img, u, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def reproject_equirect(img: np.ndarray, m: KBModel, out_w: int) -> np.ndarray:
    """Render an equirect of the lens hemisphere (lon in [-pi,pi], lat in
    [-pi/2,pi/2]); the lens axis points to lat=+pi/2 (image top)."""
    out_h = out_w // 2
    lon = (np.arange(out_w) + 0.5) / out_w * 2 * math.pi - math.pi
    lat = (np.arange(out_h) + 0.5) / out_h * math.pi - math.pi / 2
    lon, lat = np.meshgrid(lon, lat)
    # Ray with lens axis = +Z: zenith (lat=+90) along +Z.
    cz = np.sin(lat)
    cx = np.cos(lat) * np.cos(lon)
    cy = np.cos(lat) * np.sin(lon)
    dirs = np.stack([cx, cy, cz], axis=-1)
    u, vv, valid = m.project(dirs)
    u[~valid] = -1; vv[~valid] = -1
    return cv2.remap(img, u, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def equirect_to_perspective(eq: np.ndarray, out_fov_deg: float, out_size: int,
                            elev_deg: float, az_deg: float) -> np.ndarray:
    """Render a rectilinear view from an equirectangular image.

    Equirect convention: u in [0,W) -> lon in [-pi,pi]; v in [0,H) -> lat in
    [+pi/2 (top) .. -pi/2 (bottom)]; forward (lon=0,lat=0) = +Z, up = +Y.
    The view axis is aimed by (elevation, azimuth). NOTE the sign
    convention (verified empirically): POSITIVE elevation looks DOWN
    (+55 = steeply down at the ground; +90 would be nadir), negative
    looks up. az rotates around the vertical (Y) axis. Avoid exactly
    +/-90 (pole singularity in the look-at basis); use up to ~+80.

    :param eq: equirectangular image (H x 2H).
    :param out_fov_deg: output pinhole horizontal FOV (deg).
    :param out_size: output side length (px).
    :param elev_deg: view elevation (deg; 0 = horizon, +90 ~ straight down).
    :param az_deg: view azimuth (deg around the up axis).
    :returns: rendered rectilinear BGR image.
    """
    H, W = eq.shape[:2]
    f = (out_size / 2.0) / math.tan(math.radians(out_fov_deg / 2.0))
    o = out_size / 2.0
    j, i = np.meshgrid(np.arange(out_size), np.arange(out_size))
    # Camera rays: +X right, +Y down, +Z forward.
    x = (j - o) / f
    y = (i - o) / f
    z = np.ones_like(x)
    v = np.stack([x, y, z], axis=-1)
    el = math.radians(elev_deg); az = math.radians(az_deg)
    # Pitch up by elev (about camera X), then yaw by az (about world up Y).
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(el), math.sin(el)],
                   [0, -math.sin(el), math.cos(el)]])
    Ry = np.array([[math.cos(az), 0, math.sin(az)],
                   [0, 1, 0],
                   [-math.sin(az), 0, math.cos(az)]])
    d = v @ Rx.T @ Ry.T
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    lat = np.arcsin(np.clip(-dy / n, -1, 1))   # +Y is down in cam -> up = -dy
    lon = np.arctan2(dx, dz)
    u = (lon / (2 * math.pi) + 0.5) * W
    vv = (0.5 - lat / math.pi) * H
    return cv2.remap(eq, u.astype(np.float32), vv.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def main(argv: list[str] | None = None) -> None:
    """CLI: preview one fisheye frame as rectilinear + equirect through a model."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="A raw fisheye JPEG.")
    ap.add_argument("--model", type=Path, default=None,
                    help="KB model JSON (default: equidistant for the image size).")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--fov-in", type=float, default=200.0)
    ap.add_argument("--rect-fov", type=float, default=110.0)
    ap.add_argument("--rect-size", type=int, default=1200)
    ap.add_argument("--equirect-w", type=int, default=2048)
    args = ap.parse_args(argv)

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]
    s = min(h, w)
    if h != w:
        img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]

    if args.model and args.model.exists():
        m = KBModel(**json.loads(args.model.read_text()))
    else:
        m = KBModel.equidistant(s, args.fov_in)
    print(f"[lens] model: f={m.f:.1f} c=({m.cx:.0f},{m.cy:.0f}) "
          f"k=({m.k1:.4g},{m.k2:.4g},{m.k3:.4g},{m.k4:.4g}) fov={m.fov_deg}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rect = reproject_rectilinear(img, m, args.rect_fov, args.rect_size)
    cv2.imwrite(str(args.out_dir / f"{args.image.stem}_rectilinear.jpg"), rect,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    eq = reproject_equirect(img, m, args.equirect_w)
    cv2.imwrite(str(args.out_dir / f"{args.image.stem}_equirect.jpg"), eq,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Persist the model used so calibration can iterate on it.
    (args.out_dir / "lens_model.json").write_text(json.dumps(asdict(m), indent=2))
    print(f"[lens] wrote {args.image.stem}_rectilinear.jpg + _equirect.jpg + "
          f"lens_model.json to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
