# dji-equirect-stitch

**Calibrated DJI dual-fisheye → equirectangular 360 stitching — in Python and the browser, with no proprietary tooling.**

> 📷 **Built and tested on the DJI Avata 360.** The bundled `calib/rig.json` is
> calibrated for that camera, and every result below is from real Avata 360
> footage. The method generalizes to other back-to-back dual-fisheye 360
> cameras (Insta360, etc.) with a fresh calibration.

![Before / after — raw DJI Avata 360 dual-fisheye to stitched, stabilized equirect](docs/images/before_after.jpg)

`dji-equirect-stitch` turns the two raw fisheye lenses of a DJI Avata 360 (and similar
back-to-back 360 cameras) into a clean, stabilized equirectangular panorama
that matches — and on the seam, beats — the DJI-Studio stitch. It is built for
reuse across web and Python projects: a Python library + CLI, a Three.js/WebGL
real-time stitcher, a React 360 video player, and a single shared, calibrated
lens model.

> Built to remove the DJI-Studio dependency from a Gaussian-splat
> reconstruction pipeline: feed raw OSV → get a sharp, stabilized equirect →
> photogrammetry/3DGS, entirely in-house.

---

## Why this exists / what we learned

Getting a 360 camera's two fisheye lenses to stitch as well as the vendor's app
is deceptively hard. The hard-won findings, all encoded in the library:

1. **The lens model.** Each ~200° lens follows the **Kannala–Brandt** model
   (`r = f·θ·(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`), *not* the equidistant
   approximation (`r = f·θ`). Using equidistant leaves residual distortion that
   blurs downstream reconstruction. We calibrate `f, cx, cy, k₁…k₄` per lens.

2. **Calibrate by self-consistency, not against a reference.** The two lenses
   overlap by ~20° at the horizon (each sees >180°). We fit the per-lens
   intrinsics **and the relative rotation between the sensors** by making the
   two lenses **agree in that overlap** — no vendor equirect needed, and it's
   independent of where the rig points.

3. **The sensors are mounted 180° apart.** Back-to-back 360 sensors are usually
   rolled ~180° relative to each other. A continuous optimizer can't discover a
   180° flip, so we **discrete-search** the relative roll/mirror first. (For the
   Avata: `b_roll = 180°`, no mirror.)

4. **Self-consistency is blind to a *global* mirror.** Mirroring *both* lenses
   keeps them consistent, so the fit can silently land on a mirrored solution.
   Only comparison to the real world breaks the tie. We resolve handedness with
   a SIFT-inlier vote against a reference frame and bake in `flip_x`. *(This one
   cost us — the calibration looked perfect while the whole world was
   left-right reversed.)*

5. **DJI is world-stabilized, not forward-locked.** The vendor equirect holds
   the world fixed as the drone turns, driven by the flight telemetry
   (`gb_yaw/pitch/roll` in the SRT). We reproduce it from the SRT alone
   (`yaw = gb_yaw + offset`, referenced to each clip's start), which is also
   exactly what makes **frame-to-frame SfM matching** reliable (consistent
   orientation) — a free win for photogrammetry.

6. **Seam ghosting is parallax — fix it with an optimal seam, not a blend.**
   The lenses are a few cm apart, so near objects appear at two positions in
   the overlap; blending across it (even multiband) *doubles* them. Optical-flow
   morphing made it worse (smearing across the distorted lens edges). The robust
   fix is a **min-cost seam** (DP seam-carve through the lens-disagreement
   field): the seam threads where the lenses agree (grass/sky) and detours
   around objects, so each object comes wholly from one lens — **no doubling.**

## Test results

| Check | Result |
|---|---|
| Stitch vs DJI-Studio equirect (held-out clip) | aligns — road/buildings/horizon match after telemetry stabilization |
| Global handedness | SIFT-inlier vote 6/0 in favor of `flip_x` (flipped: 389–2476 inliers @ 0.017 residual; un-flipped: 0–2) |
| Relative sensor roll | overlap cost 1.46 → 0.21 at `b_roll=180°` |
| Telemetry yaw model | `gb_yaw + offset` reaches 0.156 vs the unconstrained per-frame optimum 0.142 → telemetry drives orientation |
| Seam (parallax ghost) | multiband leaves doubled buildings; **optimal seam → crisp single copies** |
| Raw OSV → stabilized equirect, end-to-end | clean, level, sharp (see `docs/images/osv_stitched_equirect.jpg`) |

`docs/images/`: `ours_vs_dji.jpg` (ours/DJI stacked), `seam_multiband_vs_optimal.jpg`
(de-ghosting), `osv_stitched_equirect.jpg` (full OSV→equirect).

---

## Install

### Python
```bash
pip install dji-equirect-stitch          # (after first release)
# or, from this repo:
pip install ./python
```

### JavaScript / React (Three.js)
```bash
npm install dji-equirect-stitch          # (after first npm publish)
```

---

## Usage

### Python — stitch a single frame
```python
import cv2
from dji_equirect_stitch import load_rig, render_equirect

rig = load_rig("calib/rig.json")
A = cv2.imread("lensA.jpg"); B = cv2.imread("lensB.jpg")  # the two fisheye streams
equirect = render_equirect(A, B, rig, out_w=5760, seam=True)   # optimal-seam blend
cv2.imwrite("equirect.jpg", equirect)
```

### Python — convert a whole OSV to stabilized equirect frames
```python
from dji_equirect_stitch import load_rig, stitch_video
stitch_video("DJI_….OSV", "DJI_….SRT", load_rig("calib/rig.json"),
             out_dir="frames/", fps=2, width=5760, mp4="equirect.mp4")
```
or the CLI:
```bash
dji-stitch stitch-video --osv clip.OSV --srt clip.SRT --rig calib/rig.json \
  --out-dir frames/ --fps 2 --width 5760 --mp4 equirect.mp4
```

### Python — calibrate a new camera
```bash
# extract (lensA, lensB, [reference-equirect]) frame triples, then:
dji-stitch calibrate --calib-dir triples/ --out calib/rig.json
```

### React — live, pannable 360 player for raw OSV (GPU stitch)
```jsx
import { OsvPlayer } from "dji-equirect-stitch";
import rig from "./calib/rig.json";

<OsvPlayer osv="/clip.OSV" srt="/clip.SRT" rig={rig} stabilize />
```
The player decodes both fisheye streams and stitches + stabilizes them live on
the GPU (Three.js shader port of the Python `render_equirect` math), so you can
pan around during playback at capture quality. *(Three.js stitcher + player:
see `js/` — in active development; tracked in the project's task board.)*


### React — view a trained Gaussian splat

```jsx
import { SplatViewerPlayer } from "dji-equirect-stitch";

<SplatViewerPlayer src="/scene.ply" style={{ height: 480 }} />
```

Or the framework-agnostic core (same injected-THREE / option / event / `dispose()`
shape as `Equirect360`):

```js
import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { SplatViewer } from "dji-equirect-stitch";

const viewer = new SplatViewer(mountEl, {
  three: THREE,                  // inject the host THREE (shared instance)
  gaussianSplats3D: GaussianSplats3D,
  initialView: { position: [0, 0, 5], lookAt: [0, 0, 0], up: [0, -1, 0] },
});
viewer.on("ready", () => viewer.setView({ position: [2, 1, 4] }));
await viewer.load("/scene.ply");   // .ply | .splat | .ksplat
// ... later
viewer.dispose();
```

A runnable demo (CDN import-map, no build step) lives at
[`examples/splat-viewer.html`](examples/splat-viewer.html) — serve the `js/`
parent over HTTP and drop a real `.ply` next to it.

**Rendering** is delegated to the mature, MIT-licensed
[`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D)
renderer (a real Gaussian-splat rasterizer is out of scope here). It is a hard
dependency of the JS package; it requires **`three >= 0.160`**, which is
stricter than the equirect viewer's `>= 0.150` — pin a recent `three` if you
use the splat viewer.

**Supported formats (v1): `.ply`, `.splat`, `.ksplat`** — what the renderer
ingests natively.

**`.spz` / `.sog`:** our Python side (`scripts/splat_export.py`,
`splat-transform`) can emit these compressed formats, but there is no robust
in-browser SPZ/SOG decoder in this renderer yet, so **pre-convert them to
`.ply`/`.splat` with `splat-transform`** before loading:

```bash
npx splat-transform scene.spz scene.ply
```

When a real browser SPZ/SOG decoder is available, plug it in via the
**`opts.decoders` extension point** (no library fork needed):

```js
new SplatViewer(mountEl, {
  three: THREE,
  decoders: {
    ".spz": async (url) => {
      const buf = await (await fetch(url)).arrayBuffer();
      return decodeSpzToSplatBuffer(buf);   // your decoder → ArrayBuffer | path
    },
  },
});
await viewer.load("/scene.spz");
```

The core also exposes `setView`/`getView`, an `on('frame'|'ready'|'view', cb)`
emitter, and an `opts.viewerOptions(three, gs3d, self)` hook plus overridable
`_makeViewer()`/`_resolveSource()` for deeper customisation — matching
`Equirect360`'s extensibility.

---

## Repository layout
```
python/dji-equirect-stitch/   Python library: lens.py (Kannala-Brandt), stitch.py
                     (calibration, render, stabilize, stitch_video) + CLI
js/                  npm package: Three.js stitcher (shared shader math),
                     React <OsvPlayer/> + <Equirect360/>, dual-stream decoder
calib/rig.json       the calibrated DJI Avata rig (reference / example)
examples/            runnable browser demos (e.g. splat-viewer.html)
docs/images/         test-result screenshots
```

The Python and JS stitchers share the **same calibrated `rig.json`** and the
same projection math, so a frame stitched on the server matches one stitched in
the browser.

---

## Releasing

- **Python:** tag a version, `python -m build ./python`, `twine upload`.
- **npm:** `cd js && npm version <x> && npm publish`.
- **GitHub:** push, create a release for the tag; this repo is consumed as a
  git submodule (`git submodule add <url> libs/dji-equirect-stitch`) or via
  `pip install` / `npm install`.

## License
See [LICENSE](LICENSE).
