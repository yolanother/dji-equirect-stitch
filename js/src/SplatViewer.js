// =============================================================================
//  dji-equirect-stitch — SplatViewer: framework-agnostic, extensible 3D
//  Gaussian-splat scene viewer. Loads a trained splat (.ply / .splat / .ksplat)
//  from a URL and renders it in a THREE.js scene via the MIT-licensed
//  @mkkellogg/gaussian-splats-3d renderer, behind the SAME injected-THREE /
//  option-flag / method-event / dispose() shape as Equirect360 — so the OSV
//  360 player and the splat viewer share one consistent player API. Usable
//  directly from vanilla JS / htm and wrapped by <SplatViewer/> for React.
//  Use of this source code is governed by the LICENSE file in the repo root.
// =============================================================================
import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const DEFAULTS = {
  three: null,            // inject the host's THREE (peer) to share one instance
  gaussianSplats3D: null, // inject the renderer module (peer) — defaults to import
  cameraFov: 65,
  cameraNear: 0.1, cameraFar: 500,
  initialView: {          // camera position + lookAt target in splat space
    position: [0, 0, 5],
    lookAt: [0, 0, 0],
    up: [0, -1, 0],       // splats are typically Y-down; flip for upright view
  },
  background: 0x000000,
  controls: true,         // let the underlying Viewer drive orbit controls
  sharedMemoryForWorkers: false, // requires COOP/COEP headers; off by default
  // Pluggable decoder map: { '.ext': async (url, ctx) => path|ArrayBuffer }.
  // EXTENSION POINT for formats the renderer can't ingest natively (e.g. .spz,
  // .sog). v1 ships no decoders — see README: pre-convert with splat-transform.
  decoders: {},
  viewerOptions: null,    // (three, gs3d, self) => object — raw Viewer opts hook
  onFrame: null,          // (viewer, dt) => void
  onReady: null,          // (viewer) => void
};

/** Formats the underlying renderer ingests natively. */
const NATIVE_FORMATS = [".ply", ".splat", ".ksplat"];

/**
 * Extensible 3D Gaussian-splat viewer. Construct with a mount element + options,
 * then `load(url)` a trained splat scene. Mirrors {@link Equirect360}'s
 * contract: inject your host's THREE via `opts.three`, drive option flags,
 * subscribe with `on('frame'|'ready', cb)`, reposition with `setView/getView`,
 * and release everything with `dispose()`.
 *
 * Extension points:
 *  - `opts.three` / `opts.gaussianSplats3D` — share peer module instances.
 *  - `opts.decoders` — map of `{ '.ext': async (url, ctx) => path }` to add
 *    support for formats the renderer can't read natively (e.g. plug an SPZ/SOG
 *    decoder here; v1 ships none — pre-convert via `splat-transform`).
 *  - `opts.viewerOptions` — `(three, gs3d, self) => object` to fully customise
 *    the wrapped `GaussianSplats3D.Viewer` options.
 *  - `opts.onFrame` / `opts.onReady` callbacks + the `setView`/`getView` API.
 *  - Subclass-friendly: override `_makeViewer()` / `_resolveSource()`.
 */
export class SplatViewer {
  /**
   * @param {HTMLElement} mount Container element the canvas is appended to.
   * @param {object} [opts] Options merged over the defaults.
   * @param {object} [opts.three] Host THREE instance to share (peer).
   * @param {object} [opts.gaussianSplats3D] Host renderer module to share (peer).
   * @param {number} [opts.cameraFov] Vertical field of view, degrees.
   * @param {object} [opts.initialView] `{ position, lookAt, up }` start camera.
   * @param {number|null} [opts.background] Scene clear color, or null for none.
   * @param {boolean} [opts.controls] Enable the renderer's orbit controls.
   * @param {boolean} [opts.sharedMemoryForWorkers] Use SharedArrayBuffer workers
   *   (needs COOP/COEP cross-origin-isolation headers); default false.
   * @param {Object<string,Function>} [opts.decoders] Pluggable format decoders.
   * @param {Function} [opts.viewerOptions] `(three, gs3d, self) => object` hook.
   * @param {Function} [opts.onFrame] `(viewer, dt) => void` per-frame callback.
   * @param {Function} [opts.onReady] `(viewer) => void` post-load callback.
   */
  constructor(mount, opts = {}) {
    this.opts = { ...DEFAULTS, ...opts };
    this.mount = mount;
    this.THREE = this.opts.three || THREE;
    this.GS3D = this.opts.gaussianSplats3D || GaussianSplats3D;
    this.view = {
      position: [...this.opts.initialView.position],
      lookAt: [...this.opts.initialView.lookAt],
    };
    this._handlers = {};
    this._ready = false;
    this._last = performance.now ? performance.now() : 0;
    this._init();
  }

  /** @private Build the camera, the wrapped Viewer, and start the loop. */
  _init() {
    const T = this.THREE, mount = this.mount;
    const w = mount.clientWidth || 1, h = mount.clientHeight || 1;
    this.camera = new T.PerspectiveCamera(this.opts.cameraFov, w / h, this.opts.cameraNear, this.opts.cameraFar);
    this.camera.position.fromArray(this.view.position);
    this.camera.up.fromArray(this.opts.initialView.up);
    this.camera.lookAt(new T.Vector3().fromArray(this.view.lookAt));
    this.viewer = this._makeViewer();
    this.renderer = this.viewer.renderer;
    if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode !== mount) {
      mount.appendChild(this.renderer.domElement);
    }
    if (this.renderer && this.opts.background != null && this.renderer.setClearColor) {
      this.renderer.setClearColor(new T.Color(this.opts.background), 1);
    }
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    if (this.viewer.start) this.viewer.start();
    this._loop();
  }

  /**
   * Override to customise the wrapped renderer. Builds a self-driven
   * `GaussianSplats3D.Viewer` using our injected THREE + camera; merges
   * `opts.viewerOptions(three, gs3d, self)` when provided (extension point).
   * @returns {object} The constructed `GaussianSplats3D.Viewer`.
   */
  _makeViewer() {
    const T = this.THREE, GS = this.GS3D, mount = this.mount;
    const base = {
      camera: this.camera,
      useBuiltInControls: this.opts.controls,
      rootElement: mount,
      selfDrivenMode: false,            // we pump update()/render() in our loop
      sharedMemoryForWorkers: this.opts.sharedMemoryForWorkers,
      gpuAcceleratedSort: true,
      renderMode: GS.RenderMode ? GS.RenderMode.OnChange : undefined,
    };
    const extra = this.opts.viewerOptions ? this.opts.viewerOptions(T, GS, this) : null;
    return new GS.Viewer({ ...base, ...(extra || {}) });
  }

  /**
   * Resolve a source URL to something the renderer can ingest, applying a
   * pluggable decoder when the extension isn't natively supported.
   * @param {string} url Source splat URL.
   * @returns {Promise<string|ArrayBuffer>} Native path/URL or decoded buffer.
   * @throws {Error} If the format is unsupported and no decoder is registered.
   */
  async _resolveSource(url) {
    const ext = ("." + url.split("?")[0].split("#")[0].split(".").pop()).toLowerCase();
    if (NATIVE_FORMATS.includes(ext)) return url;
    const decoder = this.opts.decoders && this.opts.decoders[ext];
    if (decoder) return decoder(url, { three: this.THREE, gs3d: this.GS3D, viewer: this });
    throw new Error(
      `SplatViewer: unsupported format "${ext}". Native: ${NATIVE_FORMATS.join(", ")}. ` +
      `For .spz/.sog, pre-convert to .ply/.splat with splat-transform, or register ` +
      `opts.decoders["${ext}"].`
    );
  }

  // --- source API ---
  /**
   * Load and render a trained Gaussian-splat scene from a URL.
   * @param {string} url Splat URL (`.ply` / `.splat` / `.ksplat`, or any
   *   extension you have registered a decoder for in `opts.decoders`).
   * @param {object} [sceneOptions] Extra per-scene options forwarded to the
   *   renderer's `addSplatScene` (e.g. `position`, `scale`, `rotation`).
   * @returns {Promise<SplatViewer>} Resolves once the scene is loaded.
   * @throws {Error} If the format is unsupported and undecodable.
   */
  async load(url, sceneOptions = {}) {
    const source = await this._resolveSource(url);
    const opt = { showLoadingUI: false, ...sceneOptions };
    if (typeof source === "string") {
      await this.viewer.addSplatScene(source, opt);
    } else {
      await this.viewer.addSplatSceneFromBuffer(source, opt);
    }
    this._ready = true;
    this.opts.onReady && this.opts.onReady(this);
    this._emit("ready", this);
    return this;
  }

  /** @returns {boolean} Whether a splat scene has finished loading. */
  isReady() { return this._ready; }

  // --- camera API ---
  /**
   * Reposition the camera in splat space.
   * @param {object} [view] `{ position?: [x,y,z], lookAt?: [x,y,z] }`.
   * @returns {SplatViewer} this, for chaining.
   */
  setView({ position, lookAt } = {}) {
    const T = this.THREE;
    if (position) { this.view.position = [...position]; this.camera.position.fromArray(position); }
    if (lookAt) {
      this.view.lookAt = [...lookAt];
      this.camera.lookAt(new T.Vector3().fromArray(lookAt));
      if (this.viewer.controls && this.viewer.controls.target) {
        this.viewer.controls.target.fromArray(lookAt);
      }
    }
    this._emit("view", this.getView());
    return this;
  }

  /** @returns {{position:number[], lookAt:number[]}} Current camera view. */
  getView() {
    return { position: [...this.camera.position.toArray()], lookAt: [...this.view.lookAt] };
  }

  /** Resize the renderer + camera to the mount's current size. */
  resize() {
    const m = this.mount;
    const w = m.clientWidth || 1, h = m.clientHeight || 1;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    if (this.renderer && this.renderer.setSize) this.renderer.setSize(w, h);
  }

  // --- frame loop ---
  /** @private Drive the wrapped Viewer's update/render and emit `frame`. */
  _loop() {
    this._raf = requestAnimationFrame(() => this._loop());
    const now = performance.now ? performance.now() : 0;
    const dt = (now - this._last) / 1000; this._last = now;
    if (this.viewer.update) this.viewer.update();
    if (this.viewer.render) this.viewer.render();
    this.opts.onFrame && this.opts.onFrame(this, dt);
    this._emit("frame", dt);
  }

  // --- events ---
  /**
   * Subscribe to an event. Events: `'frame'` (dt seconds), `'ready'` (this),
   * `'view'` (the current view).
   * @param {('frame'|'ready'|'view')} ev Event name.
   * @param {Function} cb Listener.
   * @returns {SplatViewer} this, for chaining.
   */
  on(ev, cb) { (this._handlers[ev] ||= []).push(cb); return this; }

  /** @private Dispatch an event to its listeners. */
  _emit(ev, arg) { (this._handlers[ev] || []).forEach((cb) => cb(arg)); }

  /** Tear down the loop, listeners, renderer, and wrapped Viewer. */
  dispose() {
    cancelAnimationFrame(this._raf);
    window.removeEventListener("resize", this._onResize);
    if (this.viewer) {
      if (this.viewer.stop) this.viewer.stop();
      if (this.viewer.dispose) this.viewer.dispose();
    }
    if (this.renderer && this.renderer.domElement &&
        this.renderer.domElement.parentNode === this.mount) {
      this.mount.removeChild(this.renderer.domElement);
    }
    this._handlers = {};
    this._ready = false;
  }
}
