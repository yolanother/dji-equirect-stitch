// =============================================================================
//  panostitch — Three.js material that stitches two fisheye textures live
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================
import * as THREE from "three";
import { STITCH_FRAGMENT, rigUniforms, stabRotation } from "./stitchShader.js";

const VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  // For a unit sphere viewed from the center, the position IS the ray dir.
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

/**
 * A THREE.ShaderMaterial that stitches lens A/B fisheye textures into the
 * equirect 360 on the GPU, using a calibrated rig. Apply it to an inward-facing
 * sphere and put the camera at the center for a pannable 360 viewer.
 *
 *   const mat = new PanoStitchMaterial(rigJson);
 *   mat.setTextures(texA, texB);
 *   mat.setAttitude({ yaw, pitch, roll }, yawRef, trim);   // per frame
 */
export class PanoStitchMaterial extends THREE.ShaderMaterial {
  constructor(rig) {
    const u = rigUniforms(rig);
    super({
      side: THREE.BackSide,
      vertexShader: VERT,
      fragmentShader: STITCH_FRAGMENT,
      uniforms: {
        texA: { value: null }, texB: { value: null },
        uStab: { value: new THREE.Matrix3() },
        uRA: { value: new THREE.Matrix3().fromArray(u.uRA) },
        uRB: { value: new THREE.Matrix3().fromArray(u.uRB) },
        uFcA: { value: new THREE.Vector3(...u.uFcA) },
        uFcB: { value: new THREE.Vector3(...u.uFcB) },
        uKA: { value: new THREE.Vector4(...u.uKA) },
        uKB: { value: new THREE.Vector4(...u.uKB) },
        uFovHalfDeg: { value: u.uFovHalfDeg }, uTexSize: { value: u.uTexSize },
        uBlendDeg: { value: u.uBlendDeg }, uFlipX: { value: u.uFlipX },
      },
    });
    this._rig = rig;
  }

  setTextures(texA, texB) {
    this.uniforms.texA.value = texA;
    this.uniforms.texB.value = texB;
  }

  setAttitude(att, yawRef = 0, trim = 0, level = true) {
    this.uniforms.uStab.value.fromArray(stabRotation(att, yawRef, trim, level));
  }
}
