// =============================================================================
//  panostitch — WebGL/Three.js dual-fisheye stitch shader
//  Browser port of the Python render_equirect projection (see
//  python/panostitch/stitch.py render_layers + lens.py KBModel.project).
//  Same calibrated rig.json drives both, so server and browser stitches match.
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================

/**
 * GLSL fragment shader: given a view ray (world dir, +Y up, +Z forward) it
 * applies the global mirror + telemetry stabilization, projects into each
 * fisheye lens via the Kannala-Brandt model, and feathers the two together.
 * Feed `vDir` from the geometry (a sphere for a 360 viewer, or compute it from
 * equirect uv for an offscreen equirect render).
 */
export const STITCH_FRAGMENT = /* glsl */ `
precision highp float;
uniform sampler2D texA;     // top-lens fisheye (stream 0)
uniform sampler2D texB;     // bottom-lens fisheye (stream 1)
uniform mat3 uStab;         // stabilization rotation (world)
uniform mat3 uRA;           // world -> lens A frame
uniform mat3 uRB;           // world -> lens B frame
uniform vec3 uFcA;          // lens A: f, cx, cy (pixels)
uniform vec3 uFcB;
uniform vec4 uKA;           // lens A: k1..k4
uniform vec4 uKB;
uniform float uFovHalfDeg;  // half FOV (deg), e.g. 100 for a 200deg lens
uniform float uTexSize;     // fisheye texture size (px), e.g. 3840
uniform float uBlendDeg;    // feather width near the FOV edge
uniform bool  uFlipX;       // global handedness correction
varying vec3 vDir;

// Kannala-Brandt forward projection: lens-frame ray (+Z axis) -> texel uv.
vec2 kbProject(vec3 d, vec3 fc, vec4 k, out float thetaDeg) {
  float n = length(d);
  float theta = acos(clamp(d.z / n, -1.0, 1.0));
  float phi = atan(d.y, d.x);
  float t2 = theta * theta;
  float thd = theta * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
  float r = fc.x * thd;
  thetaDeg = degrees(theta);
  return vec2(fc.y + r * cos(phi), fc.z + r * sin(phi)) / uTexSize;
}

void main() {
  vec3 d = normalize(vDir);
  if (uFlipX) d.x = -d.x;
  d = uStab * d;

  float ta, tb;
  vec2 ua = kbProject(uRA * d, uFcA, uKA, ta);
  vec2 ub = kbProject(uRB * d, uFcB, uKB, tb);

  float wa = clamp((uFovHalfDeg - ta) / uBlendDeg, 0.0, 1.0);
  float wb = clamp((uFovHalfDeg - tb) / uBlendDeg, 0.0, 1.0);
  vec3 ca = texture2D(texA, ua).rgb;
  vec3 cb = texture2D(texB, ub).rgb;
  gl_FragColor = vec4((ca * wa + cb * wb) / (wa + wb + 1e-4), 1.0);
}
`;

// --- rig -> shader uniforms -------------------------------------------------
const deg2rad = (d) => (d * Math.PI) / 180;
const matmul = (A, B) => A.map((_, i) =>
  [0, 1, 2].map((j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));
const Rx = (a) => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
const Ry = (a) => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
const Rz = (a) => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];

/** Rodrigues 3-vector -> 3x3 rotation. */
function rodrigues([x, y, z]) {
  const t = Math.hypot(x, y, z);
  if (t < 1e-9) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [kx, ky, kz] = [x / t, y / t, z / t];
  const c = Math.cos(t), s = Math.sin(t), C = 1 - c;
  return [
    [c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s],
    [ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s],
    [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C],
  ];
}
const flat = (m) => [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]]; // column-major for Three.Matrix3

/**
 * Build the static shader uniforms from a calibrated rig.json object.
 * Pass the per-frame stabilization rotation separately (see stabRotation()).
 */
export function rigUniforms(rig) {
  const RA = matmul(rodrigues(rig.rvecA || [0, 0, 0]), Rx(Math.PI / 2));         // +Y -> +Z
  const RB = matmul(matmul(Rz(deg2rad(rig.b_roll_deg || 0)),
                           rodrigues(rig.rvecB || [0, 0, 0])), Rx(-Math.PI / 2)); // -Y -> +Z
  const fov = rig.lensA.fov_deg || 200;
  return {
    uRA: flat(RA), uRB: flat(RB),
    uFcA: [rig.lensA.f, rig.lensA.cx, rig.lensA.cy],
    uFcB: [rig.lensB.f, rig.lensB.cx, rig.lensB.cy],
    uKA: [rig.lensA.k1, rig.lensA.k2, rig.lensA.k3, rig.lensA.k4],
    uKB: [rig.lensB.k1, rig.lensB.k2, rig.lensB.k3, rig.lensB.k4],
    uFovHalfDeg: fov / 2, uTexSize: 3840, uBlendDeg: rig.blend_deg || 12,
    uFlipX: !!rig.flip_x,
  };
}

/** Per-frame world-stabilization rotation from SRT attitude (deg). */
export function stabRotation({ yaw, pitch, roll }, yawRef = 0, trim = 0, level = true) {
  let R = Ry(deg2rad(yaw - yawRef + trim));
  if (level) R = matmul(matmul(Rx(deg2rad(-pitch)), Rz(deg2rad(-roll))), R);
  return flat(R);
}
