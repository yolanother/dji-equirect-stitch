// =============================================================================
//  dji-equirect-stitch — <SplatViewer/>: React wrapper over the SplatViewer core
//  Mounts an extensible 3D Gaussian-splat viewer and loads a trained splat
//  (.ply / .splat / .ksplat) from `src`. Mirrors <Equirect360Player/>: all
//  core options pass through and `onReady(viewer)` hands back the instance.
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================
import React, { useEffect, useRef } from "react";
import { SplatViewer as SplatViewerCore } from "./SplatViewer.js";

/**
 * React component that renders a 3D Gaussian-splat scene from a URL. All
 * {@link SplatViewerCore} options pass through; `onReady(viewer)` hands back
 * the core instance for full extensibility (setView, decoders, etc.).
 *
 *   <SplatViewer src="/scene.ply" autoRotate onReady={(v) => v.setView(...)} />
 *
 * @param {object} props
 * @param {string} props.src Splat URL (`.ply` / `.splat` / `.ksplat`).
 * @param {Function} [props.onReady] `(viewer) => void` once the scene loads.
 * @param {object} [props.style] Inline style merged onto the mount div.
 * @param {object} [props.rest] Any remaining props forward to the core opts.
 * @returns {JSX.Element} The viewer mount element.
 */
export function SplatViewer({ src, onReady, style, ...opts }) {
  const mountRef = useRef(null);
  const viewerRef = useRef(null);
  useEffect(() => {
    const v = new SplatViewerCore(mountRef.current, opts);
    viewerRef.current = v;
    onReady && onReady(v);
    if (src) v.load(src).catch((err) => console.error("SplatViewer.load failed:", err));
    return () => v.dispose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const v = viewerRef.current; if (!v || !src) return;
    v.load(src).catch((err) => console.error("SplatViewer.load failed:", err));
  }, [src]);
  return <div ref={mountRef} style={{ width: "100%", height: "100%", ...style }} />;
}
