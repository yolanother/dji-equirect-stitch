// =============================================================================
//  dji-equirect-stitch (browser) — public API
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================
export { Equirect360 } from "./Equirect360.js";
export { STITCH_FRAGMENT, rigUniforms, stabRotation } from "./stitchShader.js";
export { PanoStitchMaterial } from "./PanoStitchMaterial.js";
export { SplatViewer } from "./SplatViewer.js";

// React components (optional peer: react). Tree-shaken out if unused.
export { Equirect360Player } from "./Equirect360Player.jsx";
export { OsvPlayer } from "./OsvPlayer.jsx";
export { SplatViewer as SplatViewerPlayer } from "./SplatViewer.jsx";
