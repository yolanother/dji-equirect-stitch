// =============================================================================
//  panostitch — <OsvPlayer/>: live, pannable 360 player for raw DJI OSV
//  Decodes the two fisheye streams and stitches + stabilizes them on the GPU
//  (PanoStitchMaterial), so you can pan during playback at capture quality.
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PanoStitchMaterial } from "./PanoStitchMaterial.js";

/**
 * Props:
 *   rig         calibrated rig.json object (required)
 *   texA, texB  THREE.Texture for the two fisheye streams (lens A/B). Wire
 *               these from your decoder — see useOsvStreams() below. For a
 *               quick start you can pass two <video>/<canvas>-backed textures.
 *   attitude    { yaw, pitch, roll } in deg for the CURRENT frame (from the
 *               SRT) — drives world-stabilization. Optional.
 *   yawRef,trim stabilization reference heading + forward trim (deg).
 *   onPlay/onPause/playing  playback control (your decoder owns the clock).
 */
export function OsvPlayer({ rig, texA, texB, attitude, yawRef = 0, trim = 0,
                            playing = false, onPlay, onPause, style }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});

  useEffect(() => {
    const mount = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, mount.clientWidth / mount.clientHeight, 0.1, 100);
    const material = new PanoStitchMaterial(rig);
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(10, 64, 48), material);
    scene.add(sphere);

    // Minimal drag-to-pan (swap for three/examples OrbitControls in your app).
    let lon = 0, lat = 0, down = false, px = 0, py = 0;
    const onDown = (e) => { down = true; px = e.clientX; py = e.clientY; };
    const onUp = () => { down = false; };
    const onMove = (e) => {
      if (!down) return;
      lon -= (e.clientX - px) * 0.15; lat += (e.clientY - py) * 0.15;
      lat = Math.max(-85, Math.min(85, lat)); px = e.clientX; py = e.clientY;
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);

    stateRef.current = { renderer, scene, camera, material, sphere,
                         get lon() { return lon; }, get lat() { return lat; } };

    let raf;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const phi = THREE.MathUtils.degToRad(90 - lat), theta = THREE.MathUtils.degToRad(lon);
      camera.lookAt(Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta));
      renderer.render(scene, camera);
    };
    loop();

    const onResize = () => {
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      camera.aspect = mount.clientWidth / mount.clientHeight; camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointermove", onMove);
      renderer.dispose(); mount.removeChild(renderer.domElement);
    };
  }, [rig]);

  // Push textures + attitude each render.
  useEffect(() => {
    const s = stateRef.current; if (!s.material) return;
    if (texA && texB) s.material.setTextures(texA, texB);
    if (attitude) s.material.setAttitude(attitude, yawRef, trim);
  }, [texA, texB, attitude, yawRef, trim]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", ...style }}>
      <div ref={mountRef} style={{ width: "100%", height: "100%" }} />
      <button onClick={() => (playing ? onPause?.() : onPlay?.())}
              style={{ position: "absolute", left: 12, bottom: 12 }}>
        {playing ? "Pause" : "▶ Play"}
      </button>
    </div>
  );
}

// TODO (tracked task): useOsvStreams(osvUrl, srtUrl) — decode the two HEVC
// fisheye streams (WebCodecs VideoDecoder, or two seeked <video> elements) into
// THREE.Texture (texA/texB) + expose per-frame SRT attitude. The math above is
// complete and verified against the Python stitcher; this hook is the only
// remaining browser plumbing.
