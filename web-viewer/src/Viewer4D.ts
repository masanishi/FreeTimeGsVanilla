/**
 * Viewer4D — PlayCanvas-based 4D Gaussian Splat Viewer
 *
 * Loads canonical.ply via GSplatComponent, uploads velocity/timing data as
 * GPU textures, and uses setWorkBufferModifier() GLSL to compute per-splat
 * temporal motion and opacity on the GPU every frame.
 *
 * FreeTimeGS equations (CVPR 2025):
 *   Position:  mu_x(t) = mu_x + v * (t - mu_t)
 *   Temporal:  sigma(t) = exp(-0.5 * ((t - mu_t) / s)^2)
 *   Combined:  opacity(t) = base_opacity * sigma(t)
 */

import * as pc from 'playcanvas';
import { type Params4D, loadParams4D } from './Params4DLoader';

// ───────────────────────── GLSL Work-Buffer Modifier ─────────────────────────
// These run on the GPU for every splat in the copy-to-workbuffer pass.
// Available globals: splat.index (uint), splat.uv (ivec2)
//
// uDebugMode:
//   0 = normal rendering
//   1 = color by temporal opacity (green=high, red=low)
//   2 = color by displacement magnitude (blue=none, red=max)
const MODIFIER_GLSL = /* glsl */ `
uniform sampler2D uVelocityTex;
uniform sampler2D uTimingTex;
uniform float uNormalizedTime;
uniform float uTemporalThreshold;
uniform float uParamsTexWidth;
uniform float uMaxSplatScale;
uniform float uMaxDisplacement;
uniform float uEnableDisplacement;
uniform float uDebugMode;
uniform float uAlphaMode;  // 0=full modulation, 1=binary (no fade), 2=sqrt (gentle)
uniform float uMaxAspectRatio;  // max allowed aspect ratio (0=disabled)

float g_temporalOpacity;
float g_dispLen;

void modifySplatCenter(inout vec3 center) {
    float fi = float(splat.index);
    ivec2 uv = ivec2(int(mod(fi, uParamsTexWidth)), int(floor(fi / uParamsTexWidth)));
    vec4 vel    = texelFetch(uVelocityTex, uv, 0);
    vec4 timing = texelFetch(uTimingTex,   uv, 0);

    float mu_t     = timing.r;
    float duration = timing.g;
    float dt       = uNormalizedTime - mu_t;

    // Temporal Gaussian: sigma(t) = exp(-0.5 * ((t - mu_t) / s)^2)
    g_temporalOpacity = exp(-0.5 * (dt * dt) / (duration * duration + 1e-8));

    // Linear motion: mu_x(t) = mu_x + v * (t - mu_t)
    vec3 displacement = vel.xyz * dt;
    g_dispLen = length(displacement);
    // Clamp displacement magnitude instead of hard-culling
    if (g_dispLen > uMaxDisplacement) {
        displacement = displacement * (uMaxDisplacement / g_dispLen);
        g_dispLen = uMaxDisplacement;
    }
    // Only apply displacement if enabled
    if (uEnableDisplacement > 0.5) {
        center += displacement;
    }
}

void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter,
                              inout vec4 rotation, inout vec3 scale) {
    if (g_temporalOpacity < uTemporalThreshold) {
        scale = vec3(0.0);   // cull invisible splats
        return;
    }
    // Clamp scale to prevent oversized splats
    scale = min(scale, vec3(uMaxSplatScale));
    // Aspect ratio filter: cull needle-like splats
    if (uMaxAspectRatio > 0.5) {
        float sMax = max(scale.x, max(scale.y, scale.z));
        float sMin = min(scale.x, min(scale.y, scale.z));
        if (sMax > sMin * uMaxAspectRatio) {
            scale = vec3(0.0);  // cull needle
        }
    }
}

void modifySplatColor(vec3 center, inout vec4 color) {
    // Alpha modulation mode:
    //   0 = full modulation (physically correct): alpha *= temporalOpacity
    //   1 = binary (no fade): alpha unchanged if above threshold
    //   2 = sqrt (gentle fade): alpha *= sqrt(temporalOpacity)
    if (uAlphaMode < 0.5) {
        color.a *= g_temporalOpacity;
    } else if (uAlphaMode < 1.5) {
        // binary: no alpha modulation (only culling via scale=0 in RotationScale)
    } else {
        color.a *= sqrt(g_temporalOpacity);
    }

    // Debug coloring (overwrites RGB, keeps modulated alpha)
    if (uDebugMode > 0.5 && uDebugMode < 1.5) {
        // Mode 1: temporal opacity heatmap (red=0 → green=1)
        color.rgb = vec3(1.0 - g_temporalOpacity, g_temporalOpacity, 0.0);
    } else if (uDebugMode > 1.5 && uDebugMode < 2.5) {
        // Mode 2: displacement magnitude heatmap (blue=0 → red=max)
        float d = clamp(g_dispLen / uMaxDisplacement, 0.0, 1.0);
        color.rgb = vec3(d, 0.0, 1.0 - d);
    }
}
`;

// ───────────────────────── WGSL Work-Buffer Modifier (WebGPU) ────────────────
const MODIFIER_WGSL = /* wgsl */ `
var uVelocityTex: texture_2d<f32>;
var uTimingTex: texture_2d<f32>;
uniform uNormalizedTime: f32;
uniform uTemporalThreshold: f32;
uniform uParamsTexWidth: f32;
uniform uMaxSplatScale: f32;
uniform uMaxDisplacement: f32;
uniform uEnableDisplacement: f32;
uniform uDebugMode: f32;
uniform uAlphaMode: f32;
uniform uMaxAspectRatio: f32;

var<private> g_temporalOpacity: f32;
var<private> g_dispLen: f32;

fn modifySplatCenter(center: ptr<function, vec3f>) {
  let fi = f32(splat.index);
  let width = uniform.uParamsTexWidth;
  let x = i32(fi - floor(fi / width) * width);
  let y = i32(floor(fi / width));
  let uv = vec2i(x, y);
    let vel    = textureLoad(uVelocityTex, uv, 0);
    let timing = textureLoad(uTimingTex,   uv, 0);

    let mu_t     = timing.r;
    let duration = timing.g;
    let dt       = uniform.uNormalizedTime - mu_t;

    // Temporal Gaussian: sigma(t) = exp(-0.5 * ((t - mu_t) / s)^2)
    g_temporalOpacity = exp(-0.5 * (dt * dt) / (duration * duration + 1e-8));

    // Linear motion: mu_x(t) = mu_x + v * (t - mu_t)
    var displacement = vel.xyz * dt;
    g_dispLen = length(displacement);
    // Clamp displacement magnitude instead of hard-culling
    if (g_dispLen > uniform.uMaxDisplacement) {
        displacement = displacement * (uniform.uMaxDisplacement / g_dispLen);
        g_dispLen = uniform.uMaxDisplacement;
    }
    // Only apply displacement if enabled
    if (uniform.uEnableDisplacement > 0.5) {
        *center = *center + displacement;
    }
}

fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f,
                            rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
    if (g_temporalOpacity < uniform.uTemporalThreshold) {
        *scale = vec3f(0.0);   // cull invisible splats
        return;
    }
    // Clamp scale to prevent oversized splats
    *scale = min(*scale, vec3f(uniform.uMaxSplatScale));
    // Aspect ratio filter: cull needle-like splats
    if (uniform.uMaxAspectRatio > 0.5) {
        let sMax = max((*scale).x, max((*scale).y, (*scale).z));
        let sMin = min((*scale).x, min((*scale).y, (*scale).z));
        if (sMax > sMin * uniform.uMaxAspectRatio) {
            *scale = vec3f(0.0);  // cull needle
        }
    }
}

fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    // Alpha modulation mode:
    //   0 = full modulation (physically correct): alpha *= temporalOpacity
    //   1 = binary (no fade): alpha unchanged if above threshold
    //   2 = sqrt (gentle fade): alpha *= sqrt(temporalOpacity)
    if (uniform.uAlphaMode < 0.5) {
        (*color).a = (*color).a * g_temporalOpacity;
    } else if (uniform.uAlphaMode < 1.5) {
        // binary: no alpha modulation (only culling via scale=0 in RotationScale)
    } else {
        (*color).a = (*color).a * sqrt(g_temporalOpacity);
    }

    // Debug coloring (overwrites RGB, keeps modulated alpha)
    if (uniform.uDebugMode > 0.5 && uniform.uDebugMode < 1.5) {
        // Mode 1: temporal opacity heatmap (red=0 → green=1)
        (*color).r = 1.0 - g_temporalOpacity;
        (*color).g = g_temporalOpacity;
        (*color).b = 0.0;
    } else if (uniform.uDebugMode > 1.5 && uniform.uDebugMode < 2.5) {
        // Mode 2: displacement magnitude heatmap (blue=0 → red=max)
        let d = clamp(g_dispLen / uniform.uMaxDisplacement, 0.0, 1.0);
        (*color).r = d;
        (*color).g = 0.0;
        (*color).b = 1.0 - d;
    }
}
`;

// ───────────────────────── UI Helpers ────────────────────────────────────────
function $(id: string) { return document.getElementById(id)!; }

// ───────────────────────── Viewer Class ─────────────────────────────────────
export class Viewer4D {
  public app!: pc.AppBase;
  private camera!: pc.Entity;
  public splatEntity: pc.Entity | null = null;
  public params: Params4D | null = null;

  // Animation state
  private currentFrame = 0;
  private playing = false;
  private totalFrames = 60;
  private playFps = 30;
  private accumTime = 0;

  // Orbit camera state
  private orbitYaw = 0;
  private orbitPitch = -15;
  private orbitDistance = 5;
  private orbitTarget = new pc.Vec3(0, 0, 0);
  private dragging = false;
  private panning = false;
  private lastMx = 0;
  private lastMy = 0;

  // Camera presets
  private cameraPresets: Array<{
    name: string;
    position: number[];
    worldMatrix?: number[];
    fov?: number;
  }> = [];

  // Scene-specific viewer parameters (from scene_meta.json)
  private sceneViewerParams: {
    maxSplatScale?: number;
    maxDisplacement?: number;
    maxAspectRatio?: number;
    temporalThreshold?: number;
  } = {};

  // FPS counter
  private fpsFrames = 0;
  private fpsTime = 0;
  private fpsDisplay = 0;

  // --- Public entry point ---
  async init() {
    await this.createApp();
    this.setupCamera();
    this.setupMouseControls();

    // URL parameter: ?data=data_sh0 to load from /data_sh0/ instead of /data/
    const urlParams = new URLSearchParams(window.location.search);
    const dataDir = '/' + (urlParams.get('data') || 'data');
    console.log(`[Viewer4D] Data directory: ${dataDir}`);

    // Load scene metadata, data, and splat asset in parallel
    const [meta, params] = await Promise.all([
      fetch(`${dataDir}/scene_meta.json`).then(r => r.ok ? r.json() : null).catch(() => null),
      loadParams4D(`${dataDir}/params_4d.bin`),
      this.loadSplatAsset(`${dataDir}/canonical.ply`),
    ]);
    this.params = params;
    this.totalFrames = params.totalFrames;

    // Apply scene-specific viewer parameters if available
    if (meta?.viewerParams) {
      this.sceneViewerParams = meta.viewerParams;
      console.log('[Viewer4D] Scene viewer params loaded:', this.sceneViewerParams);
    }

    // Auto-position camera from scene metadata
    // Prefer lookatCenter (computed from camera rays) over center (Gaussian median)
    if (meta) {
      const center = meta.lookatCenter || meta.center;
      if (center) {
        this.orbitTarget.set(center[0], center[1], center[2]);
        const radius = meta.radius || 2.0;
        this.orbitDistance = radius * 3.0;
        this.updateCameraFromOrbit();
        console.log(`[Viewer4D] Camera auto-positioned: center=(${center.map((v: number) => v.toFixed(2))}), radius=${radius.toFixed(2)}`);
      }
    }

    // Load camera presets
    if (meta && meta.cameraPresets) {
      this.cameraPresets = meta.cameraPresets;
      this.populateCameraPresets();
    }

    // Update seekbar range
    const seekbar = $('seekbar') as HTMLInputElement;
    seekbar.max = String(this.totalFrames - 1);

    // Attach 4D modifier once gsplat component is ready
    // DEBUG: add ?static to URL to disable 4D animation (scale clamp still active)
    const useModifier = !window.location.search.includes('static');
    if (useModifier) {
      this.attach4DModifier();
    } else {
      console.warn('[Viewer4D] Static mode — 4D animation disabled, scale clamp active');
      this.attachStaticModifier();
    }

    // Start render loop
    this.app.on('update', this.onUpdate, this);
    this.setupUIControls();

    // Hide loading — remove from DOM after fade-out so page search won't match
    const overlay = $('loading-overlay');
    overlay.classList.add('hidden');
    overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });
    $('dbg-splats').textContent = params.numSplats.toLocaleString();
  }

  // ──────────── PlayCanvas App ──────────────────────────────────────────────
  private async createApp() {
    const canvas = $('application-canvas') as HTMLCanvasElement;

    const device = await pc.createGraphicsDevice(canvas, {
      deviceTypes: ['webgpu', 'webgl2'],
      antialias: false,
    });
    console.log(`[Viewer4D] Graphics device: ${device.isWebGPU ? 'WebGPU' : 'WebGL2'}`);
    device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

    const opts = new pc.AppOptions();
    opts.graphicsDevice = device;
    opts.mouse = new pc.Mouse(document.body);
    opts.touch = new pc.TouchDevice(document.body);
    opts.componentSystems = [
      pc.RenderComponentSystem,
      pc.CameraComponentSystem,
      pc.LightComponentSystem,
      pc.ScriptComponentSystem,
      pc.GSplatComponentSystem,
    ];
    opts.resourceHandlers = [
      pc.TextureHandler,
      pc.ContainerHandler,
      pc.ScriptHandler,
      pc.GSplatHandler,
    ];

    const app = new pc.AppBase(canvas);
    app.init(opts);
    app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    app.start();

    const resize = () => app.resizeCanvas();
    window.addEventListener('resize', resize);
    app.on('destroy', () => window.removeEventListener('resize', resize));

    this.app = app;
  }

  // ──────────── Camera Presets ────────────────────────────────────────────────
  private populateCameraPresets() {
    const select = $('camera-preset') as HTMLSelectElement;
    for (const preset of this.cameraPresets) {
      const opt = document.createElement('option');
      opt.value = preset.name;
      opt.textContent = preset.name;
      select.appendChild(opt);
    }
  }

  private applyCameraPreset(preset: {
    position: number[];
    worldMatrix?: number[];
    fov?: number;
  }) {
    if (preset.worldMatrix) {
      // Directly set world transform from COLMAP-derived matrix
      const m = new pc.Mat4();
      m.set(preset.worldMatrix);
      const p = new pc.Vec3();
      const r = new pc.Quat();
      m.getTranslation(p);
      r.setFromMat4(m);
      this.camera.setPosition(p);
      this.camera.setRotation(r);

      // Use training FOV directly (no auto-widening: the worldMatrix already
      // points the camera correctly at the training subject)
      if (this.camera.camera && preset.fov) {
        this.camera.camera.fov = preset.fov;
      }
    } else {
      // Fallback: use position + lookAt scene center
      const pos = preset.position;
      this.camera.setPosition(pos[0], pos[1], pos[2]);
      this.camera.lookAt(this.orbitTarget);

      if (this.camera.camera && preset.fov) {
        this.camera.camera.fov = preset.fov;
      }
    }

    // Update orbit state to match (so mouse drag continues from this viewpoint)
    const pos = preset.position;
    const dx = pos[0] - this.orbitTarget.x;
    const dy = pos[1] - this.orbitTarget.y;
    const dz = pos[2] - this.orbitTarget.z;
    this.orbitDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    this.orbitPitch = Math.asin(dy / this.orbitDistance) * (180 / Math.PI);
    this.orbitYaw = Math.atan2(dx, dz) * (180 / Math.PI);
  }

  // ──────────── Camera ──────────────────────────────────────────────────────
  private setupCamera() {
    const camera = new pc.Entity('camera');
    camera.addComponent('camera', {
      clearColor: new pc.Color(1, 1, 1),
      fov: 60,
      nearClip: 0.01,   // Match trainer near_plane=0.01
      farClip: 1000,
    });
    this.app.root.addChild(camera);
    this.camera = camera;
    this.updateCameraFromOrbit();
  }

  private updateCameraFromOrbit() {
    const yawRad = (this.orbitYaw * Math.PI) / 180;
    const pitchRad = (this.orbitPitch * Math.PI) / 180;
    const cosPitch = Math.cos(pitchRad);

    const x = this.orbitTarget.x + this.orbitDistance * cosPitch * Math.sin(yawRad);
    const y = this.orbitTarget.y + this.orbitDistance * Math.sin(pitchRad);
    const z = this.orbitTarget.z + this.orbitDistance * cosPitch * Math.cos(yawRad);

    this.camera.setPosition(x, y, z);
    this.camera.lookAt(this.orbitTarget);
  }

  // ──────────── Mouse Controls ──────────────────────────────────────────────
  private setupMouseControls() {
    const canvas = $('application-canvas') as HTMLCanvasElement;

    canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) this.dragging = true;
      if (e.button === 2) this.panning = true;
      this.lastMx = e.clientX;
      this.lastMy = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.dragging = false;
      this.panning = false;
    });

    window.addEventListener('mousemove', (e) => {
      const dx = e.clientX - this.lastMx;
      const dy = e.clientY - this.lastMy;
      this.lastMx = e.clientX;
      this.lastMy = e.clientY;

      if (this.dragging) {
        this.orbitYaw -= dx * 0.3;
        this.orbitPitch = Math.max(-89, Math.min(89, this.orbitPitch - dy * 0.3));
        this.updateCameraFromOrbit();
      } else if (this.panning) {
        // Pan in camera-local XY plane
        const right = new pc.Vec3();
        const up = new pc.Vec3();
        this.camera.getWorldTransform().getX(right);
        this.camera.getWorldTransform().getY(up);
        const panSpeed = this.orbitDistance * 0.002;
        this.orbitTarget.add(right.mulScalar(-dx * panSpeed));
        this.orbitTarget.add(up.mulScalar(dy * panSpeed));
        this.updateCameraFromOrbit();
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbitDistance *= 1 + e.deltaY * 0.001;
      this.orbitDistance = Math.max(0.2, Math.min(100, this.orbitDistance));
      this.updateCameraFromOrbit();
    }, { passive: false });

    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  // ──────────── GSplat Asset Loading ────────────────────────────────────────
  private loadSplatAsset(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const asset = new pc.Asset('canonical', 'gsplat', { url });
      // Disable Morton-order reorder so that splat.index == PLY vertex index.
      // This is critical: our params_4d.bin texture is indexed by original PLY
      // vertex order. PlayCanvas default reorder would scramble the mapping.
      asset.data = { ...asset.data, reorder: false };
      this.app.assets.add(asset);
      this.app.assets.load(asset);

      asset.on('load', () => {
        $('loading-text').textContent = 'シーン構築中…';
        const entity = new pc.Entity('splat');
        entity.addComponent('gsplat', {
          asset,
          unified: true,
        });
        this.app.root.addChild(entity);
        this.splatEntity = entity;
        resolve();
      });
      asset.on('error', (err: string) => reject(new Error(err)));
    });
  }

  // ──────────── 4D Modifier ─────────────────────────────────────────────────
  private attach4DModifier() {
    if (!this.splatEntity || !this.params) return;
    const gsplat = this.splatEntity.gsplat!;
    const device = this.app.graphicsDevice;
    const p = this.params;

    // Create velocity texture (RGBA32F)
    const velTex = new pc.Texture(device, {
      name: 'velocityTex',
      width: p.texWidth,
      height: p.texHeight,
      format: pc.PIXELFORMAT_RGBA32F,
      mipmaps: false,
      minFilter: pc.FILTER_NEAREST,
      magFilter: pc.FILTER_NEAREST,
      addressU: pc.ADDRESS_CLAMP_TO_EDGE,
      addressV: pc.ADDRESS_CLAMP_TO_EDGE,
    });
    const velPixels = velTex.lock() as Float32Array;
    velPixels.set(p.velocityData);
    velTex.unlock();

    // Create timing texture (RGBA32F) — (mu_t, duration, 0, 0)
    const timTex = new pc.Texture(device, {
      name: 'timingTex',
      width: p.texWidth,
      height: p.texHeight,
      format: pc.PIXELFORMAT_RGBA32F,
      mipmaps: false,
      minFilter: pc.FILTER_NEAREST,
      magFilter: pc.FILTER_NEAREST,
      addressU: pc.ADDRESS_CLAMP_TO_EDGE,
      addressV: pc.ADDRESS_CLAMP_TO_EDGE,
    });
    const timPixels = timTex.lock() as Float32Array;
    timPixels.set(p.timingData);
    timTex.unlock();

    // Bind modifier and uniforms
    try {
      gsplat.setWorkBufferModifier({ glsl: MODIFIER_GLSL, wgsl: MODIFIER_WGSL });
      gsplat.workBufferUpdate = pc.WORKBUFFER_UPDATE_ALWAYS;
      gsplat.setParameter('uVelocityTex', velTex);
      gsplat.setParameter('uTimingTex', timTex);
      gsplat.setParameter('uParamsTexWidth', p.texWidth);
      // Use scene-specific params if available, otherwise use defaults
      // that match the Python trainer (no artificial caps)
      const vp = this.sceneViewerParams;
      gsplat.setParameter('uTemporalThreshold', vp.temporalThreshold ?? 0.01);
      gsplat.setParameter('uMaxSplatScale', vp.maxSplatScale ?? 0.3);
      gsplat.setParameter('uMaxDisplacement', vp.maxDisplacement ?? 5.0);
      gsplat.setParameter('uEnableDisplacement', 1.0);
      gsplat.setParameter('uDebugMode', 0.0);
      gsplat.setParameter('uAlphaMode', 0.0);  // Full modulation: matches Python reference
      gsplat.setParameter('uMaxAspectRatio', vp.maxAspectRatio ?? 50.0);  // Match convert filter
      gsplat.setParameter('uNormalizedTime', 0.0);
      console.log('[Viewer4D] Work-buffer modifier attached');
    } catch (e) {
      console.error('[Viewer4D] Failed to attach work-buffer modifier:', e);
      console.warn('[Viewer4D] Falling back to static (no 4D animation)');
    }
  }

  // ──────────── Static Modifier (scale clamp only) ──────────────────────────
  private attachStaticModifier() {
    if (!this.splatEntity) return;
    const gsplat = this.splatEntity.gsplat!;

    const STATIC_GLSL = /* glsl */ `
uniform float uMaxSplatScale;
void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter,
                              inout vec4 rotation, inout vec3 scale) {
    scale = min(scale, vec3(uMaxSplatScale));
}
void modifySplatColor(vec3 center, inout vec4 color) {}
`;
    const STATIC_WGSL = /* wgsl */ `
uniform uMaxSplatScale: f32;
fn modifySplatCenter(center: ptr<function, vec3f>) {}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f,
                            rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
    *scale = min(*scale, vec3f(uniform.uMaxSplatScale));
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {}
`;
    try {
      gsplat.setWorkBufferModifier({ glsl: STATIC_GLSL, wgsl: STATIC_WGSL });
      gsplat.workBufferUpdate = pc.WORKBUFFER_UPDATE_ALWAYS;
      gsplat.setParameter('uMaxSplatScale', 0.05);
      console.log('[Viewer4D] Static modifier (scale clamp) attached');
    } catch (e) {
      console.error('[Viewer4D] Failed to attach static modifier:', e);
    }
  }

  // ──────────── Animation Loop ──────────────────────────────────────────────
  private onUpdate = (dt: number) => {
    // FPS counter
    this.fpsFrames++;
    this.fpsTime += dt;
    if (this.fpsTime >= 0.5) {
      this.fpsDisplay = Math.round(this.fpsFrames / this.fpsTime);
      this.fpsFrames = 0;
      this.fpsTime = 0;
      $('dbg-fps').textContent = String(this.fpsDisplay);
    }

    // Auto-play
    if (this.playing) {
      this.accumTime += dt;
      const interval = 1 / this.playFps;
      if (this.accumTime >= interval) {
        this.accumTime -= interval;
        this.currentFrame = (this.currentFrame + 1) % this.totalFrames;
        this.applyFrame(this.currentFrame);
      }
    }
  };

  private applyFrame(frame: number) {
    this.currentFrame = frame;
    const t = this.totalFrames > 1 ? frame / (this.totalFrames - 1) : 0;

    if (this.splatEntity?.gsplat) {
      this.splatEntity.gsplat.setParameter('uNormalizedTime', t);
    }

    // Update UI
    ($('seekbar') as HTMLInputElement).value = String(frame);
    $('frame-label').textContent = `${frame} / ${this.totalFrames - 1}`;
    $('dbg-frame').textContent = `${frame} / ${this.totalFrames - 1}`;
    $('dbg-time').textContent = t.toFixed(3);
  }

  // ──────────── UI Controls ─────────────────────────────────────────────────
  private setupUIControls() {
    const playBtn = $('btn-play') as HTMLButtonElement;
    const seekbar = $('seekbar') as HTMLInputElement;

    playBtn.addEventListener('click', () => {
      this.playing = !this.playing;
      playBtn.textContent = this.playing ? '⏸' : '▶';
      this.accumTime = 0;
    });

    seekbar.addEventListener('input', () => {
      this.applyFrame(parseInt(seekbar.value, 10));
    });

    // Camera preset selector
    const cameraSelect = $('camera-preset') as HTMLSelectElement;
    cameraSelect.addEventListener('change', () => {
      const name = cameraSelect.value;
      if (!name) {
        // "自由視点" selected — restore default FOV and orbit
        if (this.camera.camera) {
          this.camera.camera.fov = 60;
        }
        this.updateCameraFromOrbit();
        return;
      }
      const preset = this.cameraPresets.find(p => p.name === name);
      if (preset) this.applyCameraPreset(preset);
    });

    // Temporal threshold slider
    const thresholdSlider = $('threshold-slider') as HTMLInputElement;
    const thresholdValue = $('threshold-value');
    thresholdSlider.addEventListener('input', () => {
      const val = parseFloat(thresholdSlider.value);
      thresholdValue.textContent = val.toFixed(3);
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uTemporalThreshold', val);
      }
    });

    // Playback FPS slider
    const fpsSlider = $('fps-slider') as HTMLInputElement;
    const fpsValue = $('fps-value');
    fpsSlider.addEventListener('input', () => {
      this.playFps = parseInt(fpsSlider.value, 10);
      fpsValue.textContent = String(this.playFps);
    });

    // Max splat scale slider
    const scaleSlider = $('scale-slider') as HTMLInputElement;
    const scaleValue = $('scale-value');
    scaleSlider.addEventListener('input', () => {
      const val = parseFloat(scaleSlider.value);
      scaleValue.textContent = val.toFixed(3);
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uMaxSplatScale', val);
      }
    });

    // Max displacement slider
    const dispSlider = $('disp-slider') as HTMLInputElement;
    const dispValue = $('disp-value');
    dispSlider.addEventListener('input', () => {
      const val = parseFloat(dispSlider.value);
      dispValue.textContent = val.toFixed(3);
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uMaxDisplacement', val);
      }
    });

    // Enable/disable displacement checkbox
    const enableDispCheckbox = $('enable-disp-checkbox') as HTMLInputElement;
    enableDispCheckbox.addEventListener('change', () => {
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uEnableDisplacement', enableDispCheckbox.checked ? 1.0 : 0.0);
      }
    });

    // Debug mode selector
    const debugModeSelect = $('debug-mode-select') as HTMLSelectElement;
    debugModeSelect.addEventListener('change', () => {
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uDebugMode', parseFloat(debugModeSelect.value));
      }
    });

    // Alpha mode selector
    const alphaModeSelect = $('alpha-mode-select') as HTMLSelectElement;
    alphaModeSelect.addEventListener('change', () => {
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uAlphaMode', parseFloat(alphaModeSelect.value));
      }
    });

    // Max aspect ratio slider
    const aspectSlider = $('aspect-slider') as HTMLInputElement;
    const aspectValue = $('aspect-value');
    aspectSlider.addEventListener('input', () => {
      const val = parseFloat(aspectSlider.value);
      aspectValue.textContent = val.toFixed(1);
      if (this.splatEntity?.gsplat) {
        this.splatEntity.gsplat.setParameter('uMaxAspectRatio', val === 0 ? 0 : val);
      }
    });

    // Background color selector
    const bgSelect = $('bg-color-select') as HTMLSelectElement;
    bgSelect.addEventListener('change', () => {
      if (!this.camera.camera) return;
      const colors: Record<string, pc.Color> = {
        white: new pc.Color(1, 1, 1),
        black: new pc.Color(0, 0, 0),
        gray: new pc.Color(0.5, 0.5, 0.5),
      };
      this.camera.camera.clearColor = colors[bgSelect.value] || colors.white;
    });

    // Reset button
    const resetBtn = $('btn-reset') as HTMLButtonElement;
    resetBtn.addEventListener('click', () => {
      // Default values (match scene params or trainer defaults)
      const vp = this.sceneViewerParams;
      const defaults: Record<string, number | string | boolean> = {
        'threshold-slider': vp.temporalThreshold ?? 0.01,
        'scale-slider': vp.maxSplatScale ?? 0.3,
        'disp-slider': vp.maxDisplacement ?? 5.0,
        'fps-slider': 30,
        'aspect-slider': vp.maxAspectRatio ?? 50,
      };
      for (const [id, val] of Object.entries(defaults)) {
        const el = $(id) as HTMLInputElement;
        el.value = String(val);
        el.dispatchEvent(new Event('input'));
      }
      // Checkboxes
      (($('enable-disp-checkbox') as HTMLInputElement)).checked = true;
      ($('enable-disp-checkbox') as HTMLInputElement).dispatchEvent(new Event('change'));
      // Selects
      (($('debug-mode-select') as HTMLSelectElement)).value = '0';
      ($('debug-mode-select') as HTMLSelectElement).dispatchEvent(new Event('change'));
      (($('alpha-mode-select') as HTMLSelectElement)).value = '0';
      ($('alpha-mode-select') as HTMLSelectElement).dispatchEvent(new Event('change'));
      (($('bg-color-select') as HTMLSelectElement)).value = 'white';
      ($('bg-color-select') as HTMLSelectElement).dispatchEvent(new Event('change'));
      (($('camera-preset') as HTMLSelectElement)).value = '';
      ($('camera-preset') as HTMLSelectElement).dispatchEvent(new Event('change'));
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        playBtn.click();
      } else if (e.code === 'ArrowRight') {
        this.applyFrame(Math.min(this.currentFrame + 1, this.totalFrames - 1));
      } else if (e.code === 'ArrowLeft') {
        this.applyFrame(Math.max(this.currentFrame - 1, 0));
      }
    });
  }
}
