/**
 * ============================================================
 *  WebGPU 渲染器 (renderer.js)
 * ============================================================
 *
 *  负责可视化：
 *    1. 模具渲染：透明三角网格（Lambert 哑光）
 *    2. 三角形边线渲染：FEA 风格三角网格边线
 *    3. 相机控制：鼠标拖拽旋转 + 滚轮缩放
 *
 *  不再使用 SDF — 模型纯由三角网格表示。
 */

import moldShader from './shaders/moldRender.wgsl?raw';

// 归一化模型单位/秒。默认速率下调至原来的 1/4，使颗粒库存与壁厚增长更平缓。
const DEPOSITION_RATE = 0.0005;
const THICKNESS_COLOR_REFERENCE = 0.02;
// Eight solver passes per 1/60 s frame preserve the coarse-grained flow while
// keeping the 80^3 field within real-time range on mid-range GPUs.
const GRANULE_GPU_ITERATIONS = 8;
const GRANULE_STATS_INTERVAL = 6;
const GPU_FLOAT_STAT_SCALE = 65536;

// 与 moldRender.wgsl 中 colorOpacity 一致：未沉积区域的基础模具色
const MOLD_BASE_COLOR = [0.4, 0.5, 0.7];

// 与 moldRender.wgsl 中 thicknessColor() 完全一致的热力图色映射（蓝→青→黄→红）
function thicknessColor(value) {
  const t = Math.max(0, Math.min(1, value));
  if (t < 0.33) return mix3([0.08, 0.32, 0.95], [0.0, 0.9, 0.85], t / 0.33);
  if (t < 0.66) return mix3([0.0, 0.9, 0.85], [1.0, 0.88, 0.1], (t - 0.33) / 0.33);
  return mix3([1.0, 0.88, 0.1], [0.95, 0.08, 0.04], (t - 0.66) / 0.34);
}
function mix3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/**
 * 三角形边线着色器 — 绘制 FEA 风格的三角网格边线
 * 复用模具顶点缓冲（position offset=0, stride=24），
 * 使用独立的边线索引缓冲（line-list 拓扑）。
 */
const EDGE_SHADER = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  modelMatrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) pos: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec3<f32>,
};

@vertex
fn vsMain(v: VertexInput) -> VertexOutput {
  var o: VertexOutput;
  o.position = uniforms.viewProj * uniforms.modelMatrix * vec4<f32>(v.pos, 1.0);
  o.localPos = v.pos;
  return o;
}

fn edgeGrain(position: vec3<f32>) -> f32 {
  let cell = floor(position * 120.0);
  return fract(sin(dot(cell, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let grain = mix(0.88, 1.02, edgeGrain(input.localPos));
  let matteBlueGray = vec3<f32>(0.22, 0.31, 0.38) * grain;
  return vec4<f32>(matteBlueGray, 0.50);
}`;

const GRANULE_SHADER = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  modelMatrix: mat4x4<f32>,
  surfaceParams: vec4<f32>,
  cameraPosition: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> granuleMass: array<f32>;
@group(0) @binding(2) var<storage, read> cavityMask: array<u32>;

struct VertexInput {
  @location(0) center: vec3<f32>,
  @location(1) gridIndex: u32,
};
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPosition: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) @interpolate(flat) visibleFlag: f32,
};

fn gridCoordinates(index: u32, resolution: i32) -> vec3<i32> {
  let layer = resolution * resolution;
  let z = i32(index) / layer;
  let remainder = i32(index) - z * layer;
  let y = remainder / resolution;
  return vec3<i32>(remainder - y * resolution, y, z);
}

fn sampleMass(coordinates: vec3<i32>, resolution: i32, fallback: f32) -> f32 {
  if (any(coordinates < vec3<i32>(0)) || any(coordinates >= vec3<i32>(resolution))) {
    return fallback;
  }
  let index = u32(
    coordinates.x
    + coordinates.y * resolution
    + coordinates.z * resolution * resolution
  );
  if (cavityMask[index] == 0u) { return fallback; }
  return granuleMass[index];
}

@vertex
fn vsMain(v: VertexInput, @builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let cornerIndex = vertexIndex % 6u;
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
  );
  let faceNormals = array<vec3<i32>, 6>(
    vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
    vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1)
  );
  let resolution = i32(uniforms.surfaceParams.x);
  let isoLevel = uniforms.surfaceParams.z;
  let coordinates = gridCoordinates(v.gridIndex, resolution);
  let currentMass = granuleMass[v.gridIndex];
  var hasSurface = false;
  var boundaryVector = vec3<f32>(0.0);
  for (var faceIndex = 0u; faceIndex < 6u; faceIndex++) {
    let neighbor = coordinates + faceNormals[faceIndex];
    var neighborIsCavity = false;
    var neighborMass = 1.0;
    if (all(neighbor >= vec3<i32>(0)) && all(neighbor < vec3<i32>(resolution))) {
      let neighborIndex = u32(
        neighbor.x + neighbor.y * resolution + neighbor.z * resolution * resolution
      );
      neighborIsCavity = cavityMask[neighborIndex] != 0u;
      if (neighborIsCavity) { neighborMass = granuleMass[neighborIndex]; }
    }
    if (!neighborIsCavity) {
      boundaryVector += vec3<f32>(faceNormals[faceIndex]);
      hasSurface = true;
    } else if (neighborMass < isoLevel) {
      hasSurface = true;
    }
  }

  let gradient = vec3<f32>(
    sampleMass(coordinates + vec3<i32>(1, 0, 0), resolution, currentMass)
      - sampleMass(coordinates - vec3<i32>(1, 0, 0), resolution, currentMass),
    sampleMass(coordinates + vec3<i32>(0, 1, 0), resolution, currentMass)
      - sampleMass(coordinates - vec3<i32>(0, 1, 0), resolution, currentMass),
    sampleMass(coordinates + vec3<i32>(0, 0, 1), resolution, currentMass)
      - sampleMass(coordinates - vec3<i32>(0, 0, 1), resolution, currentMass)
  );
  let gradientLength = length(gradient);
  let boundaryLength = length(boundaryVector);
  let fallbackNormal = select(
    vec3<f32>(0.0, 1.0, 0.0),
    normalize(boundaryVector),
    boundaryLength > 1e-4
  );
  let smoothNormal = select(fallbackNormal, normalize(-gradient), gradientLength > 1e-4);
  let referenceAxis = select(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    abs(smoothNormal.y) > 0.92
  );
  let tangent = normalize(cross(referenceAxis, smoothNormal));
  let bitangent = cross(smoothNormal, tangent);
  // Boundary cells sit against the mold wall. Their visual patch must stay
  // inside the wall instead of spanning a full cell around the voxel center.
  let boundaryCell = boundaryLength > 1e-4;
  let surfaceOffset = uniforms.surfaceParams.y * select(0.44, 0.30, boundaryCell);
  let patchHalfSize = uniforms.surfaceParams.y * select(0.53, 0.36, boundaryCell);
  let corner = corners[cornerIndex];
  let localPosition = v.center
    + smoothNormal * surfaceOffset
    + (tangent * corner.x + bitangent * corner.y) * patchHalfSize;
  let world = uniforms.modelMatrix * vec4<f32>(localPosition, 1.0);
  var o: VertexOutput;
  o.position = uniforms.viewProj * world;
  o.worldPosition = world.xyz;
  o.worldNormal = normalize((uniforms.modelMatrix * vec4<f32>(smoothNormal, 0.0)).xyz);
  o.visibleFlag = select(0.0, 1.0, currentMass >= isoLevel && hasSurface);
  return o;
}

@fragment
fn fsMain(v: VertexOutput) -> @location(0) vec4<f32> {
  if (v.visibleFlag < 0.5) { discard; }
  let normal = normalize(v.worldNormal);
  let lightDirection = normalize(vec3<f32>(0.35, 0.82, 0.44));
  let viewDirection = normalize(uniforms.cameraPosition.xyz - v.worldPosition);
  let halfDirection = normalize(lightDirection + viewDirection);
  let diffuse = 0.42 + 0.48 * max(dot(normal, lightDirection), 0.0);
  let specular = pow(max(dot(normal, halfDirection), 0.0), 40.0) * 0.42;
  let fresnel = pow(1.0 - abs(dot(normal, viewDirection)), 3.0) * 0.18;
  let baseColor = vec3<f32>(0.08, 0.48, 0.74);
  let color = baseColor * diffuse + vec3<f32>(specular + fresnel);
  return vec4<f32>(color, uniforms.surfaceParams.w);
}`;

const GRANULE_COMPUTE_SHADER = `
struct Params {
  gravityAndAngle: vec4<f32>,
  flowAndScale: vec4<f32>,
  angularVelocityAndDomain: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> massIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> massOut: array<f32>;
@group(0) @binding(3) var<storage, read> cavityMask: array<u32>;
@group(0) @binding(4) var<storage, read> activeCells: array<u32>;

@compute @workgroup_size(128)
fn stepGranules(@builtin(global_invocation_id) gid: vec3<u32>) {
  let activeIndex = gid.x;
  let activeCount = u32(params.flowAndScale.w);
  if (activeIndex >= activeCount) { return; }
  let index = activeCells[activeIndex];
  if (cavityMask[index] == 0u) {
    massOut[index] = 0.0;
    return;
  }

  let resolution = i32(params.flowAndScale.z);
  let layer = resolution * resolution;
  let z = i32(index) / layer;
  let remainder = i32(index) - z * layer;
  let y = remainder / resolution;
  let x = remainder - y * resolution;
  let center = vec3<i32>(x, y, z);
  let directions = array<vec3<i32>, 26>(
    vec3<i32>(-1, -1, -1), vec3<i32>(0, -1, -1), vec3<i32>(1, -1, -1),
    vec3<i32>(-1,  0, -1), vec3<i32>(0,  0, -1), vec3<i32>(1,  0, -1),
    vec3<i32>(-1,  1, -1), vec3<i32>(0,  1, -1), vec3<i32>(1,  1, -1),
    vec3<i32>(-1, -1,  0), vec3<i32>(0, -1,  0), vec3<i32>(1, -1,  0),
    vec3<i32>(-1,  0,  0),                              vec3<i32>(1,  0,  0),
    vec3<i32>(-1,  1,  0), vec3<i32>(0,  1,  0), vec3<i32>(1,  1,  0),
    vec3<i32>(-1, -1,  1), vec3<i32>(0, -1,  1), vec3<i32>(1, -1,  1),
    vec3<i32>(-1,  0,  1), vec3<i32>(0,  0,  1), vec3<i32>(1,  0,  1),
    vec3<i32>(-1,  1,  1), vec3<i32>(0,  1,  1), vec3<i32>(1,  1,  1)
  );

  let currentMass = massIn[index];
  let sinRepose = params.gravityAndAngle.w;
  let denominator = max(1.0 - sinRepose, 1e-4);
  // Granular beds exhibit slow subcritical creep. Without this term a
  // voxel solver has a dead band: shallow slopes never move because no
  // single 26-neighbor direction exceeds the repose threshold.
  let subcriticalCreep = 0.48;
  let avalancheExponent = 0.62;
  let globalTilt = length(params.gravityAndAngle.xz);
  let globalAvalanche = 1.0 + 5.0 * clamp(
    (globalTilt - sinRepose) / denominator,
    0.0,
    1.0
  );
  var outgoing = 0.0;
  var incoming = 0.0;
  for (var directionIndex = 0u; directionIndex < 26u; directionIndex++) {
    let delta = directions[directionIndex];
    let neighbor = center + delta;
    if (any(neighbor < vec3<i32>(0)) || any(neighbor >= vec3<i32>(resolution))) {
      continue;
    }
    let neighborIndex = u32(neighbor.x + neighbor.y * resolution + neighbor.z * layer);
    if (cavityMask[neighborIndex] == 0u) { continue; }

    let midpoint = vec3<f32>(center) + vec3<f32>(delta) * 0.5;
    let domainSize = params.angularVelocityAndDomain.w;
    let position = midpoint / f32(resolution - 1) * domainSize - vec3<f32>(domainSize * 0.5);
    let omega = params.angularVelocityAndDomain.xyz;
    let centrifugal = -cross(omega, cross(omega, position)) / 9.81;
    let effectiveDown = normalize(params.gravityAndAngle.xyz + centrifugal);
    let descent = dot(normalize(vec3<f32>(delta)), effectiveDown);
    if (descent > sinRepose) {
      let drive = pow((descent - sinRepose) / denominator, avalancheExponent);
      outgoing += min(currentMass, 1.0 - massIn[neighborIndex])
        * drive * globalAvalanche * params.flowAndScale.x / 26.0;
    } else if (descent > 0.0) {
      // Keep the repose angle as the fast-avalanche threshold, while
      // allowing a slow downhill redistribution below it.
      let drive = descent * subcriticalCreep;
      outgoing += min(currentMass, 1.0 - massIn[neighborIndex])
        * drive * params.flowAndScale.x / 26.0;
    } else if (descent < -sinRepose) {
      let drive = pow((-descent - sinRepose) / denominator, avalancheExponent);
      incoming += min(massIn[neighborIndex], 1.0 - currentMass)
        * drive * globalAvalanche * params.flowAndScale.x / 26.0;
    } else if (descent < 0.0) {
      let drive = (-descent) * subcriticalCreep;
      incoming += min(massIn[neighborIndex], 1.0 - currentMass)
        * drive * params.flowAndScale.x / 26.0;
    }
  }
  let limitedOutgoing = min(outgoing, currentMass);
  let availableIncoming = max(0.0, 1.0 - currentMass + limitedOutgoing);
  let limitedIncoming = min(incoming, availableIncoming);
  massOut[index] = clamp(
    (currentMass - limitedOutgoing + limitedIncoming) * params.flowAndScale.y,
    0.0,
    1.0
  );
}`;

const GPU_STATS_SHADER = `
struct StatsParams {
  resolution: u32,
  domainSize: f32,
  activeCount: u32,
  triangleCount: u32,
};
struct Stats {
  totalMass: atomic<u32>,
  contactArea: atomic<u32>,
  maxThickness: atomic<u32>,
  reserved: atomic<u32>,
};
@group(0) @binding(0) var<storage, read> mass: array<f32>;
@group(0) @binding(1) var<storage, read> activeCells: array<u32>;
@group(0) @binding(2) var<storage, read> cavityMask: array<u32>;
@group(0) @binding(3) var<storage, read> triangleSamples: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> triangleAreas: array<f32>;
@group(0) @binding(5) var<storage, read_write> contactWeights: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: Stats;
@group(0) @binding(7) var<uniform> params: StatsParams;
@group(0) @binding(8) var<storage, read_write> contactHistory: array<f32>;
// The compute solver ping-pongs between two mass buffers. Sampling both
// buffers closes the gap when a particle crosses a surface voxel between
// two rendered frames.
@group(0) @binding(9) var<storage, read> previousMass: array<f32>;

const TRIANGLE_SAMPLE_COUNT: u32 = 5u;

fn sampleMass(center: vec3<i32>) -> f32 {
  let resolution = i32(params.resolution);
  let layer = resolution * resolution;
  var result = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    let z = center.z + dz;
    if (z < 0 || z >= resolution) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = center.y + dy;
      if (y < 0 || y >= resolution) { continue; }
      for (var dx = -1; dx <= 1; dx++) {
        let x = center.x + dx;
        if (x < 0 || x >= resolution) { continue; }
        let index = u32(x + y * resolution + z * layer);
        // The mass buffers are already populated only for cavity cells.
        // Requiring cavityMask here again rejects valid particles next to a
        // wall because the triangle sample itself is on the solid side of
        // the boundary. Contact sampling must use nearby mass, regardless of
        // which side of the voxelized boundary owns that cell.
        result = max(result, max(mass[index], previousMass[index]));
      }
    }
  }
  return clamp(result, 0.0, 1.0);
}

@compute @workgroup_size(128)
fn collectStats(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index < params.activeCount) {
    let value = max(0.0, mass[activeCells[index]]);
    atomicAdd(&stats.totalMass, u32(value * 1024.0));
  }
  if (index < params.triangleCount) {
    let scale = f32(params.resolution - 1) / params.domainSize;
    let half = params.domainSize * 0.5;
    var weight = 0.0;
    // A triangle can straddle several voxels. Sampling only its centroid
    // produces zero-weight holes when the centroid falls in an empty cell.
    for (var sampleIndex = 0u; sampleIndex < TRIANGLE_SAMPLE_COUNT; sampleIndex++) {
      let p = triangleSamples[index * TRIANGLE_SAMPLE_COUNT + sampleIndex].xyz;
      let center = vec3<i32>(
        clamp(i32(round((p.x + half) * scale)), 0, i32(params.resolution - 1)),
        clamp(i32(round((p.y + half) * scale)), 0, i32(params.resolution - 1)),
        clamp(i32(round((p.z + half) * scale)), 0, i32(params.resolution - 1))
      );
      weight = max(weight, sampleMass(center));
    }
    // Keep a short bridge for fast flips. The history is deliberately
    // decayed, so a surface that is no longer reached cannot grow forever.
    let effectiveWeight = max(weight, contactHistory[index] * 0.6);
    contactWeights[index] = effectiveWeight;
    contactHistory[index] = max(weight, contactHistory[index] * 0.6);
    atomicAdd(&stats.contactArea, u32(max(0.0, triangleAreas[index] * effectiveWeight) * 65536.0));
  }
}
`;

const GPU_THICKNESS_SHADER = `
struct ThicknessParams {
  triangleCount: u32,
  thicknessIncrement: f32,
  statScale: f32,
  reserved: f32,
};
struct Stats {
  totalMass: atomic<u32>,
  contactArea: atomic<u32>,
  maxThickness: atomic<u32>,
  reserved: atomic<u32>,
};
@group(0) @binding(0) var<storage, read> contactWeights: array<f32>;
@group(0) @binding(1) var<storage, read_write> thickness: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: Stats;
@group(0) @binding(3) var<uniform> params: ThicknessParams;

@compute @workgroup_size(128)
fn accumulateThickness(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.triangleCount) { return; }
  let value = thickness[index * 3u] + params.thicknessIncrement * contactWeights[index];
  thickness[index * 3u] = value;
  thickness[index * 3u + 1u] = value;
  thickness[index * 3u + 2u] = value;
  atomicMax(&stats.maxThickness, u32(max(0.0, value) * params.statScale));
}
`;

/**
 * WebGPU 渲染器类
 */
export class Renderer {
  constructor() {
    this.device = null;
    this.context = null;
    this.canvas = null;
    this.viewCubeCanvas = null;
    this.viewCubeContext = null;
    this.viewCubeHitRegions = [];
    this.viewCubeHoveredFace = null;
    this.viewCubeBasis = null;
    this.format = null;

    // ─── 相机参数 ───
    this.cameraPos = [0, 0.5, 3.5];
    this.rotX = -0.3;
    this.rotY = 0.5;
    this.zoom = 3.5;

    // ─── 模具 GPU 资源 ───
    this.moldPipeline = null;
    this.moldVertexBuffer = null;
    this.moldIndexBuffer = null;
    this.moldBindGroup = null;
    this.moldUniformBuffer = null;
    this.moldOverlayBindGroup = null;
    this.moldOverlayUniformBuffer = null;
    this.moldOverlayPipeline = null;
    this.moldIndexCount = 0;
    this.moldVertices = null;
    this.moldIndices = null;
    this.triangleThickness = null;
    this.triangleCentroids = null;
    this.triangleAreas = null;
    this.triangleSampleBuffer = null;
    this.triangleSamplePoints = null;
    this.triangleSampleGridIndices = null;
    this.triangleSampleCount = 0;
    this.triangleAreaBuffer = null;
    this.contactWeightsBuffer = null;
    this.contactHistoryBuffer = null;
    this.thicknessVertexBuffer = null;
    this.showThicknessColors = true;

    // ─── 三角形边线（FEA 风格）───
    this.edgePipeline = null;
    this.edgeBindGroup = null;
    this.edgeIndexBuffer = null;
    this.edgeIndexCount = 0;
    this.edgeUniformBuffer = null;

    // ─── 鼠标交互 ───
    this.isDragging = false;
    this.lastMouse = [0, 0];
    this.onCameraChange = null;

    // ─── 滚塑颗粒自由表面 ───
    this.granulePipeline = null;
    this.granuleUniformBuffer = null;
    this.granuleVertexBuffer = null;
    this.granuleGridIndexBuffer = null;
    this.granuleBindGroup = null;
    this.granuleRenderBindGroups = null;
    this.granuleCount = 0;
    this.granuleComputePipeline = null;
    this.granuleComputeUniformBuffer = null;
    this.granuleMassBuffers = null;
    this.granuleComputeBindGroups = null;
    this.granuleMaskBuffer = null;
    this.granuleReadbackBuffer = null;
    this.granuleMassCpu = null;
    this.granuleCurrentBuffer = 0;
    this.granuleGridResolution = 0;
    this.granuleGridDomainSize = 0;
    this.granuleCellCount = 0;
    this.granuleReadbackPending = false;
    this.granuleReadbackPromise = null;
    this.granuleReadbackGeneration = 0;
    this.granuleReadbackFrame = 0;
    this.gpuStatsPipeline = null;
    this.gpuStatsParamsBuffer = null;
    this.gpuStatsBuffer = null;
    this.gpuStatsReadbackBuffer = null;
    this.contactWeightsReadbackBuffer = null;
    this.gpuStatsBindGroups = null;
    this.gpuThicknessPipeline = null;
    this.gpuThicknessParamsBuffer = null;
    this.gpuThicknessBindGroup = null;
    this.gpuStatsPending = false;
    this.gpuStatsPromise = null;
    this.lastGpuStats = { totalMass: 0, contactArea: 0, maxThickness: 0 };
    this.thicknessReadbackBuffer = null;
    this.gpuStatsMassReadbackBuffer = null;
    this.contactHistoryCpu = null;
    this.granuleVolumeRatio = 1;
    this.pendingGranuleMassScale = 1;
    this.granuleInitialMass = 0;
    this.granuleTargetMass = 0;
    this.granuleStepCounter = 0;
    this.lastStatsStep = -Infinity;
    this.lastGpuStats = { totalMass: 0, contactArea: 0, maxThickness: 0 };

    // ─── 共享资源 ───
    this.depthTexture = null;
  }

  async init(device, canvas, viewCubeCanvas = null) {
    this.device = device;
    this.canvas = canvas;
    this.viewCubeCanvas = viewCubeCanvas;
    this.viewCubeContext = viewCubeCanvas?.getContext('2d') || null;
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'premultiplied' });

    this._setupMouse();
    this._setupViewCube();
    this._createPipelines();
    this._resizeDepthTexture();
  }

  _resizeDepthTexture() {
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this._resizeDepthTexture();
  }

  // ─── 鼠标交互 ─────────────────────────────────────────────

  _setupMouse() {
    const c = this.canvas;
    c.addEventListener('mousedown', e => {
      this.isDragging = true;
      this.lastMouse = [e.clientX, e.clientY];
    });
    window.addEventListener('mouseup', () => this.isDragging = false);
    window.addEventListener('mousemove', e => {
      if (!this.isDragging) return;
      this.rotY -= (e.clientX - this.lastMouse[0]) * 0.005;
      this.rotX += (e.clientY - this.lastMouse[1]) * 0.005;
      this.rotX = Math.max(-1.5, Math.min(1.5, this.rotX));
      this.lastMouse = [e.clientX, e.clientY];
      this.onCameraChange?.();
    });
    c.addEventListener('wheel', e => {
      this.zoom *= e.deltaY > 0 ? 1.1 : 0.9;
      this.zoom = Math.max(1, Math.min(20, this.zoom));
      e.preventDefault();
      this.onCameraChange?.();
    }, { passive: false });
  }

  // ─── 相机矩阵 ─────────────────────────────────────────────

  _cameraMatrix() {
    const cx = this.zoom * Math.cos(this.rotX) * Math.sin(this.rotY);
    const cy = this.zoom * Math.sin(this.rotX);
    const cz = this.zoom * Math.cos(this.rotX) * Math.cos(this.rotY);
    this.cameraPos = [cx, cy + 0.3, cz];

    const eye = this.cameraPos, target = [0, 0, 0], up = [0, 1, 0];

    const f = 1 / Math.tan(Math.PI / 8);
    const aspect = this.canvas.width / this.canvas.height;
    const near = 0.1, far = 50;
    const rangeInv = 1 / (near - far);
    const proj = [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, far * rangeInv, -1,
      0, 0, near * far * rangeInv, 0,
    ];

    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    this._drawViewCube(x, y, z);
    const view = [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
    ];

    return mulMat(proj, view);
  }

  _setupViewCube() {
    const canvas = this.viewCubeCanvas;
    if (!canvas) return;

    const faceAtEvent = event => {
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * 112 / Math.max(1, rect.width);
      const y = (event.clientY - rect.top) * 112 / Math.max(1, rect.height);
      for (let index = this.viewCubeHitRegions.length - 1; index >= 0; index--) {
        const region = this.viewCubeHitRegions[index];
        if (pointInPolygon2D([x, y], region.polygon)) return region;
      }
      return null;
    };

    canvas.addEventListener('mousemove', event => {
      const face = faceAtEvent(event);
      const nextLabel = face?.label || null;
      canvas.style.cursor = face ? 'pointer' : 'default';
      if (nextLabel === this.viewCubeHoveredFace) return;
      this.viewCubeHoveredFace = nextLabel;
      if (this.viewCubeBasis) this._drawViewCube(...this.viewCubeBasis);
    });
    canvas.addEventListener('mouseleave', () => {
      canvas.style.cursor = 'default';
      if (this.viewCubeHoveredFace === null) return;
      this.viewCubeHoveredFace = null;
      if (this.viewCubeBasis) this._drawViewCube(...this.viewCubeBasis);
    });
    canvas.addEventListener('click', event => {
      const face = faceAtEvent(event);
      if (!face) return;
      this.rotX = face.view.rotX;
      this.rotY = face.view.rotY;
      this.onCameraChange?.();
    });
  }

  _drawViewCube(cameraRight, cameraUp, cameraBack) {
    if (!this.viewCubeCanvas || !this.viewCubeContext) return;
    this.viewCubeBasis = [cameraRight, cameraUp, cameraBack];
    const canvas = this.viewCubeCanvas;
    const context = this.viewCubeContext;
    const size = 112;
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    const targetSize = Math.round(size * pixelRatio);
    if (canvas.width !== targetSize || canvas.height !== targetSize) {
      canvas.width = targetSize;
      canvas.height = targetSize;
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, size, size);

    const center = [56, 54];
    const scale = 28;
    const vertices = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ];
    const projected = vertices.map(vertex => [
      center[0] + dot(vertex, cameraRight) * scale,
      center[1] - dot(vertex, cameraUp) * scale,
    ]);
    const faces = [
      { label: '右', normal: [1, 0, 0], textU: [0, 0, -1], textV: [0, -1, 0], indices: [1, 5, 6, 2], view: { rotX: 0, rotY: Math.PI / 2 } },
      { label: '左', normal: [-1, 0, 0], textU: [0, 0, 1], textV: [0, -1, 0], indices: [0, 3, 7, 4], view: { rotX: 0, rotY: -Math.PI / 2 } },
      { label: '上', normal: [0, 1, 0], textU: [1, 0, 0], textV: [0, 0, 1], indices: [3, 2, 6, 7], view: { rotX: 1.5, rotY: 0 } },
      { label: '下', normal: [0, -1, 0], textU: [1, 0, 0], textV: [0, 0, -1], indices: [0, 4, 5, 1], view: { rotX: -1.5, rotY: 0 } },
      { label: '前', normal: [0, 0, 1], textU: [1, 0, 0], textV: [0, -1, 0], indices: [4, 7, 6, 5], view: { rotX: 0, rotY: 0 } },
      { label: '后', normal: [0, 0, -1], textU: [-1, 0, 0], textV: [0, -1, 0], indices: [0, 1, 2, 3], view: { rotX: 0, rotY: Math.PI } },
    ].filter(face => dot(face.normal, cameraBack) > 0.015)
      .map(face => ({
        ...face,
        depth: dot(face.normal, cameraBack),
        polygon: face.indices.map(index => projected[index]),
      }))
      .sort((a, b) => a.depth - b.depth);

    this.viewCubeHitRegions = faces;
    context.lineJoin = 'round';
    context.font = '600 15px Segoe UI, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    for (const face of faces) {
      const hovered = face.label === this.viewCubeHoveredFace;
      const lighting = 0.5 + 0.5 * Math.max(0, dot(face.normal, [0.35, 0.75, 0.55]));
      const gray = Math.round(72 + lighting * 48);
      context.beginPath();
      context.moveTo(face.polygon[0][0], face.polygon[0][1]);
      for (let index = 1; index < face.polygon.length; index++) {
        context.lineTo(face.polygon[index][0], face.polygon[index][1]);
      }
      context.closePath();
      context.shadowColor = 'rgba(0, 0, 0, 0.35)';
      context.shadowBlur = 7;
      context.shadowOffsetY = 2;
      context.fillStyle = hovered ? 'rgba(53, 151, 224, 0.96)' : `rgba(${gray}, ${gray + 5}, ${gray + 10}, 0.94)`;
      context.fill();
      context.shadowColor = 'transparent';
      context.strokeStyle = hovered ? '#9bd7ff' : 'rgba(225, 233, 242, 0.82)';
      context.lineWidth = hovered ? 1.8 : 1.15;
      context.stroke();

      const labelX = face.polygon.reduce((sum, point) => sum + point[0], 0) / face.polygon.length;
      const labelY = face.polygon.reduce((sum, point) => sum + point[1], 0) / face.polygon.length;
      const textU = [
        dot(face.textU, cameraRight) * scale * 2,
        -dot(face.textU, cameraUp) * scale * 2,
      ];
      const textV = [
        dot(face.textV, cameraRight) * scale * 2,
        -dot(face.textV, cameraUp) * scale * 2,
      ];
      context.save();
      context.transform(
        textU[0] / 56,
        textU[1] / 56,
        textV[0] / 56,
        textV[1] / 56,
        labelX,
        labelY,
      );
      context.fillStyle = '#ffffff';
      context.strokeStyle = 'rgba(25, 31, 39, 0.68)';
      context.lineWidth = 2;
      context.strokeText(face.label, 0, 0);
      context.fillText(face.label, 0, 0);
      context.restore();
    }
  }

  // ─── 管线创建 ─────────────────────────────────────────────

  _createPipelines() {
    const dev = this.device;

    // ─── 模具 uniform 缓冲 ───
    this.moldUniformBuffer = dev.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.moldOverlayUniformBuffer = dev.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ─── 模具渲染管线 ───
    const mMod = dev.createShaderModule({ code: moldShader });
    this.moldPipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: mMod, entryPoint: 'vsMain',
        buffers: [
          {
            arrayStride: 24,
            attributes: [
              { shaderLocation: 0, offset: 0,  format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
            ],
          },
          {
            arrayStride: 4,
            attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }],
          },
        ],
      },
      fragment: {
        module: mMod, entryPoint: 'fsMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.moldOverlayPipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: mMod, entryPoint: 'vsMain',
        buffers: [
          {
            arrayStride: 24,
            attributes: [
              { shaderLocation: 0, offset: 0,  format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
            ],
          },
          {
            arrayStride: 4,
            attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }],
          },
        ],
      },
      fragment: {
        module: mMod, entryPoint: 'fsMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // Thickness is an explicit overlay. It must be visible even when the
      // particle bed already occupies the same depth range.
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
    });

    // ─── 三角形边线 uniform 缓冲 ───
    this.edgeUniformBuffer = dev.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ─── 三角形边线渲染管线 ───
    const eMod = dev.createShaderModule({ code: EDGE_SHADER });
    this.edgePipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: eMod, entryPoint: 'vsMain',
        buffers: [{
          arrayStride: 24,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: {
        module: eMod, entryPoint: 'fsMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'line-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
    });

    this.granuleUniformBuffer = dev.createBuffer({
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const gMod = dev.createShaderModule({ code: GRANULE_SHADER });
    this.granulePipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: gMod, entryPoint: 'vsMain',
        buffers: [{
          arrayStride: 12,
          stepMode: 'instance',
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }, {
          arrayStride: 4,
          stepMode: 'instance',
          attributes: [{ shaderLocation: 1, offset: 0, format: 'uint32' }],
        }],
      },
      fragment: {
        module: gMod, entryPoint: 'fsMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // Keep particle depth writes enabled so translucent patches do not
      // accumulate over one another and look like the particle count grows.
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.granuleComputeUniformBuffer = dev.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const computeModule = dev.createShaderModule({ code: GRANULE_COMPUTE_SHADER });
    this.granuleComputePipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'stepGranules' },
    });

    this.gpuStatsParamsBuffer = dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const statsModule = dev.createShaderModule({ code: GPU_STATS_SHADER });
    this.gpuStatsPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: statsModule, entryPoint: 'collectStats' },
    });
    this.gpuThicknessParamsBuffer = dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const thicknessModule = dev.createShaderModule({ code: GPU_THICKNESS_SHADER });
    this.gpuThicknessPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: thicknessModule, entryPoint: 'accumulateThickness' },
    });
  }

  // ─── 模具网格加载 ──────────────────────────────────────────

  loadMoldMesh(meshData) {
    const dev = this.device;
    const sourceVertices = meshData.vertices instanceof Float32Array ? meshData.vertices : new Float32Array(meshData.vertices);
    const sourceIndices = meshData.indices instanceof Uint32Array ? meshData.indices : new Uint32Array(meshData.indices);
    // 展开索引，使每个三角形拥有独立顶点，从而无需 primitive_index 扩展也能逐面着色。
    const v = new Float32Array(sourceIndices.length * 6);
    const i = new Uint32Array(sourceIndices.length);
    for (let index = 0; index < sourceIndices.length; index++) {
      const source = sourceIndices[index] * 6;
      v.set(sourceVertices.subarray(source, source + 6), index * 6);
      i[index] = index;
    }

    this.moldVertexBuffer?.destroy();
    this.moldIndexBuffer?.destroy();
    this.thicknessVertexBuffer?.destroy();
    this.triangleSampleBuffer?.destroy();
    this.triangleAreaBuffer?.destroy();
    this.contactWeightsBuffer?.destroy();
    this.contactHistoryBuffer?.destroy();
    this.triangleSamplePoints = null;
    this.triangleSampleGridIndices = null;

    this.moldVertices = v;
    this.moldIndices = i;
    const triangleCount = i.length / 3;
    this.triangleThickness = new Float32Array(triangleCount);
    this.lastGpuStats = { totalMass: 0, contactArea: 0, maxThickness: 0 };
    this.triangleCentroids = new Float32Array(triangleCount * 3);
    this.triangleAreas = new Float32Array(triangleCount);
    for (let t = 0; t < triangleCount; t++) {
      for (let axis = 0; axis < 3; axis++) {
        this.triangleCentroids[t * 3 + axis] = (
          v[i[t * 3] * 6 + axis] + v[i[t * 3 + 1] * 6 + axis] + v[i[t * 3 + 2] * 6 + axis]
        ) / 3;
      }
      const a = i[t * 3] * 6;
      const b = i[t * 3 + 1] * 6;
      const c = i[t * 3 + 2] * 6;
      const abx = v[b] - v[a], aby = v[b + 1] - v[a + 1], abz = v[b + 2] - v[a + 2];
      const acx = v[c] - v[a], acy = v[c + 1] - v[a + 1], acz = v[c + 2] - v[a + 2];
      const crossX = aby * acz - abz * acy;
      const crossY = abz * acx - abx * acz;
      const crossZ = abx * acy - aby * acx;
      this.triangleAreas[t] = 0.5 * Math.hypot(crossX, crossY, crossZ);
    }
    this.thicknessVertexData = new Float32Array(i.length);
    this.thicknessVertexBuffer = dev.createBuffer({
      size: Math.max(4, this.thicknessVertexData.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    dev.queue.writeBuffer(this.thicknessVertexBuffer, 0, this.thicknessVertexData);

    // Store six points per triangle. Paired normal offsets bridge the render
    // surface to the cavity voxel layer without a costly volumetric search.
    const sampleCount = 5;
    const sampleData = new Float32Array(triangleCount * sampleCount * 4);
    for (let t = 0; t < triangleCount; t++) {
      const ia = i[t * 3] * 6;
      const ib = i[t * 3 + 1] * 6;
      const ic = i[t * 3 + 2] * 6;
      const a = v.subarray(ia, ia + 3);
      const b = v.subarray(ib, ib + 3);
      const c = v.subarray(ic, ic + 3);
      const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const faceNormal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
      ];
      const normalLength = Math.hypot(...faceNormal);
      if (normalLength > 1e-8) {
        faceNormal[0] /= normalLength;
        faceNormal[1] /= normalLength;
        faceNormal[2] /= normalLength;
      }
      const centroid = this.triangleCentroids.subarray(t * 3, t * 3 + 3);
      const offsetPoint = (point, distance) => [
        point[0] + faceNormal[0] * distance,
        point[1] + faceNormal[1] * distance,
        point[2] + faceNormal[2] * distance,
      ];
      const edgeAB = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
      const edgeBC = [(b[0] + c[0]) * 0.5, (b[1] + c[1]) * 0.5, (b[2] + c[2]) * 0.5];
      const edgeCA = [(c[0] + a[0]) * 0.5, (c[1] + a[1]) * 0.5, (c[2] + a[2]) * 0.5];
      // Samples on both sides of the face bridge the render surface to the
      // cavity voxel layer without a costly 7x7x7 search for every triangle.
      const points = [
        centroid,
        offsetPoint(centroid, 0.035),
        offsetPoint(centroid, -0.035),
        offsetPoint(centroid, 0.07),
        offsetPoint(centroid, -0.07),
      ];
      for (let sampleIndex = 0; sampleIndex < points.length; sampleIndex++) {
        sampleData.set(points[sampleIndex], (t * sampleCount + sampleIndex) * 4);
      }
    }
    this.triangleSamplePoints = sampleData;
    this.triangleSampleCount = sampleCount;
    this.triangleSampleBuffer = dev.createBuffer({
      size: Math.max(16, sampleData.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.triangleSampleBuffer, 0, sampleData);
    this.triangleAreaBuffer = dev.createBuffer({
      size: Math.max(4, this.triangleAreas.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.triangleAreaBuffer, 0, this.triangleAreas);
    this.contactWeightsBuffer = dev.createBuffer({
      size: Math.max(4, this.triangleThickness.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.contactHistoryBuffer = dev.createBuffer({
      size: Math.max(4, this.triangleThickness.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.contactHistoryBuffer, 0, new Float32Array(triangleCount));

    this.moldVertexBuffer = dev.createBuffer({ size: v.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.moldVertexBuffer, 0, v);

    this.moldIndexBuffer = dev.createBuffer({ size: i.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.moldIndexBuffer, 0, i);

    this.moldIndexCount = i.length;
    this.moldBindGroup = dev.createBindGroup({
      layout: this.moldPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.moldUniformBuffer } }],
    });
    this.moldOverlayBindGroup = dev.createBindGroup({
      layout: this.moldOverlayPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.moldOverlayUniformBuffer } }],
    });

    this._buildEdgeBuffer(i);
  }

  loadGranuleVoxels(params) {
    const positions = params?.positions;
    const gridIndices = params?.gridIndices;
    this.granuleVertexBuffer?.destroy();
    this.granuleGridIndexBuffer?.destroy();
    this.granuleMaskBuffer?.destroy();
    this.granuleReadbackBuffer?.destroy();
    this.gpuStatsBuffer?.destroy();
    this.gpuStatsReadbackBuffer?.destroy();
    this.contactWeightsReadbackBuffer?.destroy();
    this.thicknessReadbackBuffer?.destroy();
    this.gpuStatsMassReadbackBuffer?.destroy();
    this.granuleMassBuffers?.forEach(buffer => buffer.destroy());
    this.granuleBindGroup = null;
    this.granuleRenderBindGroups = null;
    this.granuleComputeBindGroups = null;
    this.gpuStatsBindGroups = null;
    this.gpuThicknessBindGroup = null;
    this.granuleReadbackGeneration++;
    this.granuleReadbackPending = false;
    this.granuleReadbackPromise = null;
    this.gpuStatsPromise = null;
    this.gpuStatsPending = false;
    this.contactHistoryCpu = null;
    this.granuleStepCounter = 0;
    this.lastStatsStep = -Infinity;
    this.granuleCount = positions?.length ? positions.length / 3 : 0;
    if (!this.granuleCount || !gridIndices?.length) return;

    this.granuleGridResolution = params.gridResolution;
    this.granuleGridDomainSize = params.gridDomainSize;
    this.granuleCellCount = this.granuleGridResolution ** 3;
    const byteSize = this.granuleCellCount * Float32Array.BYTES_PER_ELEMENT;

    this.granuleVertexBuffer = this.device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.granuleVertexBuffer, 0, positions);

    this.granuleGridIndexBuffer = this.device.createBuffer({
      size: gridIndices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.granuleGridIndexBuffer, 0, gridIndices);

    const mask = new Uint32Array(this.granuleCellCount);
    for (const gridIndex of gridIndices) mask[gridIndex] = 1;
    this.granuleMaskBuffer = this.device.createBuffer({
      size: mask.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.granuleMaskBuffer, 0, mask);

    this.granuleMassBuffers = [0, 1].map(() => this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
    this.granuleReadbackBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.gpuStatsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.gpuStatsReadbackBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.contactWeightsReadbackBuffer = this.device.createBuffer({
      size: Math.max(4, this.triangleThickness?.byteLength || 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.thicknessReadbackBuffer = this.device.createBuffer({
      size: Math.max(4, this.thicknessVertexData?.byteLength || 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.gpuStatsMassReadbackBuffer = this.device.createBuffer({
      size: Math.max(4, this.granuleCellCount * Float32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    if (this.triangleSamplePoints && this.triangleSampleCount > 0) {
      this.triangleSampleGridIndices = new Uint32Array(
        (this.triangleThickness?.length || 0) * this.triangleSampleCount,
      );
      const scale = (this.granuleGridResolution - 1) / this.granuleGridDomainSize;
      const half = this.granuleGridDomainSize * 0.5;
      for (let index = 0; index < this.triangleSampleGridIndices.length; index++) {
        const base = index * 4;
        const x = Math.max(0, Math.min(this.granuleGridResolution - 1,
          Math.round((this.triangleSamplePoints[base] + half) * scale)));
        const y = Math.max(0, Math.min(this.granuleGridResolution - 1,
          Math.round((this.triangleSamplePoints[base + 1] + half) * scale)));
        const z = Math.max(0, Math.min(this.granuleGridResolution - 1,
          Math.round((this.triangleSamplePoints[base + 2] + half) * scale)));
        this.triangleSampleGridIndices[index] = x + y * this.granuleGridResolution
          + z * this.granuleGridResolution * this.granuleGridResolution;
      }
      this.contactHistoryCpu = new Float32Array(this.triangleThickness.length);
    }

    const renderLayout = this.granulePipeline.getBindGroupLayout(0);
    this.granuleRenderBindGroups = this.granuleMassBuffers.map(buffer => this.device.createBindGroup({
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.granuleUniformBuffer } },
        { binding: 1, resource: { buffer } },
        { binding: 2, resource: { buffer: this.granuleMaskBuffer } },
      ],
    }));

    const computeLayout = this.granuleComputePipeline.getBindGroupLayout(0);
    this.granuleComputeBindGroups = [
      this.device.createBindGroup({
        layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.granuleComputeUniformBuffer } },
          { binding: 1, resource: { buffer: this.granuleMassBuffers[0] } },
          { binding: 2, resource: { buffer: this.granuleMassBuffers[1] } },
          { binding: 3, resource: { buffer: this.granuleMaskBuffer } },
          { binding: 4, resource: { buffer: this.granuleGridIndexBuffer } },
        ],
      }),
      this.device.createBindGroup({
        layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.granuleComputeUniformBuffer } },
          { binding: 1, resource: { buffer: this.granuleMassBuffers[1] } },
          { binding: 2, resource: { buffer: this.granuleMassBuffers[0] } },
          { binding: 3, resource: { buffer: this.granuleMaskBuffer } },
          { binding: 4, resource: { buffer: this.granuleGridIndexBuffer } },
        ],
      }),
    ];

    if (this.triangleSampleBuffer && this.triangleAreaBuffer && this.contactWeightsBuffer && this.contactHistoryBuffer && this.gpuStatsBuffer) {
      const statsLayout = this.gpuStatsPipeline.getBindGroupLayout(0);
      this.gpuStatsBindGroups = this.granuleMassBuffers.map((buffer, index) => this.device.createBindGroup({
        layout: statsLayout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.granuleGridIndexBuffer } },
          { binding: 2, resource: { buffer: this.granuleMaskBuffer } },
          { binding: 3, resource: { buffer: this.triangleSampleBuffer } },
          { binding: 4, resource: { buffer: this.triangleAreaBuffer } },
          { binding: 5, resource: { buffer: this.contactWeightsBuffer } },
          { binding: 6, resource: { buffer: this.gpuStatsBuffer } },
          { binding: 7, resource: { buffer: this.gpuStatsParamsBuffer } },
          { binding: 8, resource: { buffer: this.contactHistoryBuffer } },
          { binding: 9, resource: { buffer: this.granuleMassBuffers[1 - index] } },
        ],
      }));
      this.gpuThicknessBindGroup = this.device.createBindGroup({
        layout: this.gpuThicknessPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.contactWeightsBuffer } },
          { binding: 1, resource: { buffer: this.thicknessVertexBuffer } },
          { binding: 2, resource: { buffer: this.gpuStatsBuffer } },
          { binding: 3, resource: { buffer: this.gpuThicknessParamsBuffer } },
        ],
      });
    }

    this._granulePositions = positions;
    this._granuleGridIndices = gridIndices;
    this.resetGranules(params);
  }

  resetGranules(params) {
    if (!this.granuleMassBuffers || !this._granulePositions || !this._granuleGridIndices) return;
    const gravityDown = params.gravityDown || [0, -1, 0];
    const order = Array.from({ length: this.granuleCount }, (_, index) => index);
    order.sort((a, b) => {
      const ai = a * 3, bi = b * 3;
      const aHeight = -(
        this._granulePositions[ai] * gravityDown[0]
        + this._granulePositions[ai + 1] * gravityDown[1]
        + this._granulePositions[ai + 2] * gravityDown[2]
      );
      const bHeight = -(
        this._granulePositions[bi] * gravityDown[0]
        + this._granulePositions[bi + 1] * gravityDown[1]
        + this._granulePositions[bi + 2] * gravityDown[2]
      );
      return aHeight - bHeight;
    });

    const mass = new Float32Array(this.granuleCellCount);
    let remainingCells = this.granuleCount * Math.max(0, Math.min(1, params.volumeFraction ?? 0));
    this.granuleInitialMass = remainingCells;
    this.granuleTargetMass = remainingCells;
    for (const cavityIndex of order) {
      if (remainingCells <= 0) break;
      const amount = Math.min(1, remainingCells);
      mass[this._granuleGridIndices[cavityIndex]] = amount;
      remainingCells -= amount;
    }
    this.device.queue.writeBuffer(this.granuleMassBuffers[0], 0, mass);
    this.device.queue.writeBuffer(this.granuleMassBuffers[1], 0, mass);
    if (this.contactWeightsBuffer && this.contactHistoryBuffer && this.triangleThickness) {
      const emptyContact = new Float32Array(this.triangleThickness.length);
      this.device.queue.writeBuffer(this.contactWeightsBuffer, 0, emptyContact);
      this.device.queue.writeBuffer(this.contactHistoryBuffer, 0, emptyContact);
      this.contactHistoryCpu = emptyContact;
    }
    this.granuleMassCpu = mass;
    this.granuleReadbackGeneration++;
    this.granuleCurrentBuffer = 0;
    this.granuleBindGroup = this.granuleRenderBindGroups[0];
    this.granuleVolumeRatio = 1;
    this.pendingGranuleMassScale = 1;
    this.granuleReadbackFrame = 0;
    this.granuleStepCounter = 0;
    this.lastStatsStep = -Infinity;
  }

  setGranuleVolumeRatio(ratio) {
    const next = Math.max(0, Math.min(1, ratio));
    if (next <= 1e-9 && this.granuleMassBuffers) {
      const empty = new Float32Array(this.granuleCellCount);
      this.device.queue.writeBuffer(this.granuleMassBuffers[0], 0, empty);
      this.device.queue.writeBuffer(this.granuleMassBuffers[1], 0, empty);
      this.granuleMassCpu = empty;
      this.granuleTargetMass = 0;
      this.pendingGranuleMassScale = 1;
      this.granuleVolumeRatio = 0;
      this.granuleReadbackGeneration++;
      return;
    }
    if (this.granuleVolumeRatio > 1e-9) {
      this.pendingGranuleMassScale *= next / this.granuleVolumeRatio;
    } else if (next <= 0) {
      this.pendingGranuleMassScale = 0;
    }
    this.granuleVolumeRatio = next;
    this.granuleTargetMass = this.granuleInitialMass * next;
  }

  stepGranules(params, iterations = GRANULE_GPU_ITERATIONS, collectStats = true) {
    if (!this.granuleComputeBindGroups || this.granuleCellCount === 0) return;
    const iterationCount = Math.max(1, iterations);
    const totalScale = Math.max(0, this.pendingGranuleMassScale);
    const scalePerIteration = totalScale === 0 ? 0 : totalScale ** (1 / iterationCount);
    this.pendingGranuleMassScale = 1;

    const gravity = params.gravityDown || [0, -1, 0];
    const angularVelocity = params.angularVelocity || [0, 0, 0];
    const reposeRadians = (params.reposeAngle ?? 35) * Math.PI / 180;
    const uniform = new Float32Array([
      gravity[0], gravity[1], gravity[2], Math.sin(reposeRadians),
      1.6, scalePerIteration, this.granuleGridResolution, this.granuleCount,
      angularVelocity[0], angularVelocity[1], angularVelocity[2], this.granuleGridDomainSize,
    ]);
    this.device.queue.writeBuffer(this.granuleComputeUniformBuffer, 0, uniform);

    const encoder = this.device.createCommandEncoder();
    for (let iteration = 0; iteration < iterationCount; iteration++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.granuleComputePipeline);
      pass.setBindGroup(0, this.granuleComputeBindGroups[this.granuleCurrentBuffer]);
      pass.dispatchWorkgroups(Math.ceil(this.granuleCount / 128));
      pass.end();
      this.granuleCurrentBuffer = 1 - this.granuleCurrentBuffer;
    }
    this.device.queue.submit([encoder.finish()]);
    this.granuleBindGroup = this.granuleRenderBindGroups[this.granuleCurrentBuffer];

    this.granuleStepCounter++;
    if (collectStats && this.granuleStepCounter - this.lastStatsStep >= GRANULE_STATS_INTERVAL) {
      this.lastStatsStep = this.granuleStepCounter;
      this._requestGpuStats();
    }
  }

  waitForGranuleReadback() {
    return this.gpuStatsPromise || Promise.resolve();
  }

  _requestGpuStats() {
    if (this.gpuStatsBindGroups && this.gpuStatsBuffer && this.gpuStatsReadbackBuffer) {
      return this._requestGpuStatsOnGpu();
    }
    return this._requestGpuStatsOnCpu();
  }

  _requestGpuStatsOnGpu() {
    if (this.gpuStatsPending) return this.gpuStatsPromise || Promise.resolve();
    this.gpuStatsPending = true;
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setUint32(0, this.granuleGridResolution, true);
    view.setFloat32(4, this.granuleGridDomainSize, true);
    view.setUint32(8, this.granuleCount, true);
    view.setUint32(12, this.triangleThickness?.length || 0, true);
    this.device.queue.writeBuffer(this.gpuStatsParamsBuffer, 0, params);
    this.device.queue.writeBuffer(this.gpuStatsBuffer, 0, new Uint32Array(4));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.gpuStatsPipeline);
    pass.setBindGroup(0, this.gpuStatsBindGroups[this.granuleCurrentBuffer]);
    pass.dispatchWorkgroups(Math.ceil(Math.max(this.granuleCount, this.triangleThickness?.length || 0) / 128));
    pass.end();
    encoder.copyBufferToBuffer(this.gpuStatsBuffer, 0, this.gpuStatsReadbackBuffer, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    const promise = this.gpuStatsReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const values = new Uint32Array(this.gpuStatsReadbackBuffer.getMappedRange().slice(0));
      this.gpuStatsReadbackBuffer.unmap();
      const totalMass = values[0] / 1024;
      const contactArea = values[1] / 65536;
      if (this.granuleTargetMass > 1e-5 && totalMass <= 0) {
        console.warn('[Granules] GPU stats returned zero; falling back to CPU readback.');
        this.gpuStatsBindGroups = null;
        this.gpuStatsPending = false;
        this.gpuStatsPromise = null;
        return this._requestGpuStatsOnCpu();
      }
      this.lastGpuStats = { totalMass, contactArea, maxThickness: this.lastGpuStats.maxThickness || 0 };
      this._scheduleMassCorrection(totalMass);
    }).catch(error => {
      console.warn('[Granules] GPU stats readback failed; falling back to CPU:', error);
      this.gpuStatsBindGroups = null;
      this.gpuStatsPending = false;
      this.gpuStatsPromise = null;
      return this._requestGpuStatsOnCpu();
    }).finally(() => {
      this.gpuStatsPending = false;
    });
    this.gpuStatsPromise = promise;
    return promise;
  }

  _requestGpuStatsOnCpu() {
    if (this.gpuStatsPending || !this.gpuStatsMassReadbackBuffer || !this.granuleMassBuffers) {
      return this.gpuStatsPromise || Promise.resolve();
    }

    // Read back the current density field once per solver batch. This keeps
    // contact detection deterministic across adapters and avoids the silent
    // zero result produced by storage atomics on some WebGPU drivers.
    this.gpuStatsPending = true;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.granuleMassBuffers[this.granuleCurrentBuffer],
      0,
      this.gpuStatsMassReadbackBuffer,
      0,
      this.granuleCellCount * Float32Array.BYTES_PER_ELEMENT,
    );
    this.device.queue.submit([encoder.finish()]);

    const promise = this.gpuStatsMassReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const mass = new Float32Array(this.gpuStatsMassReadbackBuffer.getMappedRange().slice(0));
      this.gpuStatsMassReadbackBuffer.unmap();
      let totalMass = 0;
      for (const index of this._granuleGridIndices || []) totalMass += Math.max(0, mass[index] || 0);

      const triangleCount = this.triangleThickness?.length || 0;
      const sampleCount = this.triangleSampleCount || 0;
      const contacts = new Float32Array(triangleCount);
      const history = this.contactHistoryCpu || new Float32Array(triangleCount);
      const resolution = this.granuleGridResolution;
      const layer = resolution * resolution;
      const sampleMass = (cellIndex) => {
        const z = Math.floor(cellIndex / layer);
        const remainder = cellIndex - z * layer;
        const y = Math.floor(remainder / resolution);
        const x = remainder - y * resolution;
        let value = 0;
        // Match the GPU contact shader: particles can be diagonally adjacent
        // to a surface voxel, especially on large planar faces.
        for (let dz = -1; dz <= 1; dz++) {
          const sampleZ = z + dz;
          if (sampleZ < 0 || sampleZ >= resolution) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const sampleY = y + dy;
            if (sampleY < 0 || sampleY >= resolution) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const sampleX = x + dx;
              if (sampleX < 0 || sampleX >= resolution) continue;
              value = Math.max(value, mass[sampleX + sampleY * resolution + sampleZ * layer] || 0);
            }
          }
        }
        return Math.max(0, Math.min(1, value));
      };

      let contactArea = 0;
      for (let triangle = 0; triangle < triangleCount; triangle++) {
        let weight = 0;
        const offset = triangle * sampleCount;
        for (let sample = 0; sample < sampleCount; sample++) {
          weight = Math.max(weight, sampleMass(this.triangleSampleGridIndices?.[offset + sample] || 0));
        }
        const effectiveWeight = Math.max(weight, history[triangle] * 0.6);
        contacts[triangle] = effectiveWeight;
        history[triangle] = Math.max(weight, history[triangle] * 0.6);
        contactArea += this.triangleAreas[triangle] * effectiveWeight;
      }
      this.contactHistoryCpu = history;
      if (this.contactWeightsBuffer) this.device.queue.writeBuffer(this.contactWeightsBuffer, 0, contacts);
      if (this.contactHistoryBuffer) this.device.queue.writeBuffer(this.contactHistoryBuffer, 0, history);
      this.granuleMassCpu = mass;
      this.lastGpuStats = {
        totalMass,
        contactArea,
        maxThickness: this.lastGpuStats.maxThickness || 0,
      };
      this._scheduleMassCorrection(totalMass);
    }).catch(error => {
      console.warn('[Granules] GPU mass readback failed:', error);
    }).finally(() => {
      this.gpuStatsPending = false;
    });
    this.gpuStatsPromise = promise;
    return promise;
  }

  _scheduleMassCorrection(actualMass) {
    const targetMass = this.granuleTargetMass;
    if (!(targetMass > 1e-5) || !(actualMass > 1e-5)) return;
    const ratio = targetMass / actualMass;
    if (!Number.isFinite(ratio) || Math.abs(ratio - 1) < 0.002) return;
    // Apply conservation gently over the next GPU iterations. This prevents
    // high angular velocity from accumulating mass while avoiding a visible
    // one-frame collapse when the contact area changes abruptly.
    const correction = Math.max(0.96, Math.min(1.04, ratio));
    this.pendingGranuleMassScale *= correction;
  }

  exportGranuleState(maxPoints = 60000) {
    if (!this._granulePositions || !this._granuleGridIndices || !this.granuleMassCpu) return null;
    const positions = [];
    const masses = [];
    for (let cavityIndex = 0; cavityIndex < this.granuleCount; cavityIndex++) {
      const mass = this.granuleMassCpu[this._granuleGridIndices[cavityIndex]] || 0;
      if (mass <= 1e-5) continue;
      const base = cavityIndex * 3;
      positions.push([
        this._granulePositions[base],
        this._granulePositions[base + 1],
        this._granulePositions[base + 2],
      ]);
      masses.push(mass);
    }
    if (positions.length > maxPoints) {
      const sampledPositions = [];
      const sampledMasses = [];
      for (let index = 0; index < maxPoints; index++) {
        const source = Math.floor(index * positions.length / maxPoints);
        sampledPositions.push(positions[source]);
        sampledMasses.push(masses[source]);
      }
      return this._statePayload(sampledPositions, sampledMasses);
    }
    return this._statePayload(positions, masses);
  }

  async prepareGranuleState() {
    await this._requestGranuleReadback();
  }

  _statePayload(particlePositions, particleMass) {
    const vertices = [];
    for (let index = 0; index < this.moldVertices.length; index += 6) {
      vertices.push([this.moldVertices[index], this.moldVertices[index + 1], this.moldVertices[index + 2]]);
    }
    const faces = [];
    for (let index = 0; index < this.moldIndices.length; index += 3) {
      faces.push([this.moldIndices[index], this.moldIndices[index + 1], this.moldIndices[index + 2]]);
    }
    return { schemaVersion: 1, vertices, faces, particlePositions, particleMass };
  }

  _requestGranuleReadback() {
    if (this.granuleReadbackPending) return this.granuleReadbackPromise;
    if (!this.granuleReadbackBuffer) return Promise.resolve();
    this.granuleReadbackPending = true;
    const generation = this.granuleReadbackGeneration;
    const readbackBuffer = this.granuleReadbackBuffer;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.granuleMassBuffers[this.granuleCurrentBuffer],
      0,
      readbackBuffer,
      0,
      this.granuleCellCount * Float32Array.BYTES_PER_ELEMENT,
    );
    this.device.queue.submit([encoder.finish()]);
    const readbackPromise = readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const mapped = readbackBuffer.getMappedRange();
      const nextMass = new Float32Array(mapped.slice(0));
      readbackBuffer.unmap();
      if (generation !== this.granuleReadbackGeneration) {
        this.granuleReadbackPending = false;
        return;
      }

      let actualMass = 0;
      for (let index = 0; index < nextMass.length; index++) actualMass += nextMass[index];
      const targetMass = this.granuleTargetMass;
      let correctionScale = 1;
      if (targetMass > 1e-6 && actualMass > 1e-6) {
        correctionScale = targetMass / actualMass;
        if (Math.abs(correctionScale - 1) > 0.01) {
          for (let index = 0; index < nextMass.length; index++) {
            nextMass[index] = Math.min(1, Math.max(0, nextMass[index] * correctionScale));
          }
          this.device.queue.writeBuffer(this.granuleMassBuffers[0], 0, nextMass);
          this.device.queue.writeBuffer(this.granuleMassBuffers[1], 0, nextMass);
        }
      }
      this.granuleMassCpu = nextMass;
      this.granuleReadbackPending = false;
    }).catch(error => {
      if (generation === this.granuleReadbackGeneration) {
        console.warn('[Granules] GPU readback failed:', error);
        this.granuleReadbackPending = false;
      }
    });
    this.granuleReadbackPromise = readbackPromise;
    return readbackPromise;
  }

  resetThickness() {
    if (!this.triangleThickness || !this.thicknessVertexData) return;
    this.triangleThickness.fill(0);
    this.thicknessVertexData.fill(0);
    this.lastGpuStats.maxThickness = 0;
    if (this.gpuStatsBuffer) this.device.queue.writeBuffer(this.gpuStatsBuffer, 0, new Uint32Array(4));
    if (this.device && this.thicknessVertexBuffer) {
      this.device.queue.writeBuffer(this.thicknessVertexBuffer, 0, this.thicknessVertexData);
    }
    if (this.device && this.contactWeightsBuffer && this.contactHistoryBuffer) {
      const emptyContact = new Float32Array(this.triangleThickness.length);
      this.device.queue.writeBuffer(this.contactWeightsBuffer, 0, emptyContact);
      this.device.queue.writeBuffer(this.contactHistoryBuffer, 0, emptyContact);
      this.contactHistoryCpu = emptyContact;
    }
  }

  setThicknessColorsVisible(visible) {
    this.showThicknessColors = Boolean(visible);
  }

  async accumulateThickness(elapsedSeconds, availableVolume = Infinity, depositionRate = DEPOSITION_RATE) {
    if (
      !this.triangleThickness
      || !this.gpuThicknessBindGroup
      || !(elapsedSeconds > 0)
      || !(availableVolume > 0)
    ) {
      return { depositedVolume: 0, maxThickness: this.lastGpuStats.maxThickness || this._getMaxThickness() };
    }

    await this.waitForGranuleReadback();
    const contactArea = Math.max(0, this.lastGpuStats.contactArea || 0);
    if (!(contactArea > 0)) {
      return { depositedVolume: 0, maxThickness: this.lastGpuStats.maxThickness || this._getMaxThickness() };
    }

    const requestedIncrement = Math.max(0, depositionRate) * elapsedSeconds;
    const requestedVolume = contactArea * requestedIncrement;
    const maxStepVolume = availableVolume * 0.08;
    const depositedVolume = Math.min(requestedVolume, maxStepVolume, availableVolume);
    const thicknessIncrement = depositedVolume / contactArea;
    const params = new DataView(new ArrayBuffer(16));
    params.setUint32(0, this.triangleThickness.length, true);
    params.setFloat32(4, thicknessIncrement, true);
    params.setFloat32(8, GPU_FLOAT_STAT_SCALE, true);
    params.setFloat32(12, THICKNESS_COLOR_REFERENCE, true);
    this.device.queue.writeBuffer(this.gpuThicknessParamsBuffer, 0, params.buffer);
    this.device.queue.writeBuffer(this.gpuStatsBuffer, 0, new Uint32Array(4));

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.gpuThicknessPipeline);
    pass.setBindGroup(0, this.gpuThicknessBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.triangleThickness.length / 128));
    pass.end();
    encoder.copyBufferToBuffer(this.gpuStatsBuffer, 0, this.gpuStatsReadbackBuffer, 0, 16);
    encoder.copyBufferToBuffer(
      this.thicknessVertexBuffer,
      0,
      this.thicknessReadbackBuffer,
      0,
      this.thicknessVertexData.byteLength,
    );
    this.device.queue.submit([encoder.finish()]);

    let maxThickness = this.lastGpuStats.maxThickness || 0;
    try {
      await this.gpuStatsReadbackBuffer.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(this.gpuStatsReadbackBuffer.getMappedRange().slice(0));
      this.gpuStatsReadbackBuffer.unmap();
      await this.thicknessReadbackBuffer.mapAsync(GPUMapMode.READ);
      const thicknessValues = new Float32Array(this.thicknessReadbackBuffer.getMappedRange().slice(0));
      this.thicknessReadbackBuffer.unmap();
      maxThickness = 0;
      for (let index = 0; index < thicknessValues.length; index += 3) {
        maxThickness = Math.max(maxThickness, thicknessValues[index] || 0);
      }
      this.lastGpuStats.maxThickness = maxThickness;
    } catch (error) {
      console.warn('[Thickness] GPU max readback failed:', error);
    }
    return { depositedVolume, maxThickness };
  }

  async prepareExport() {
    if (!this.thicknessReadbackBuffer || !this.thicknessVertexBuffer || !this.thicknessVertexData) return;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.thicknessVertexBuffer,
      0,
      this.thicknessReadbackBuffer,
      0,
      this.thicknessVertexData.byteLength,
    );
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.thicknessReadbackBuffer.mapAsync(GPUMapMode.READ);
      this.thicknessVertexData = new Float32Array(this.thicknessReadbackBuffer.getMappedRange().slice(0));
      this.thicknessReadbackBuffer.unmap();
      if (this.triangleThickness) {
        for (let t = 0; t < this.triangleThickness.length; t++) {
          this.triangleThickness[t] = this.thicknessVertexData[t * 3];
        }
      }
    } catch (error) {
      console.warn('[Thickness] GPU export readback failed:', error);
    }
  }

  _sampleGranuleMass(x, y, z) {
    const resolution = this.granuleGridResolution;
    if (!resolution || !this.granuleMassCpu) return 0;
    const scale = (resolution - 1) / this.granuleGridDomainSize;
    const half = this.granuleGridDomainSize * 0.5;
    const centerX = Math.round((x + half) * scale);
    const centerY = Math.round((y + half) * scale);
    const centerZ = Math.round((z + half) * scale);
    let result = 0;
    for (let dz = -1; dz <= 1; dz++) {
      const iz = centerZ + dz;
      if (iz < 0 || iz >= resolution) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const iy = centerY + dy;
        if (iy < 0 || iy >= resolution) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const ix = centerX + dx;
          if (ix < 0 || ix >= resolution) continue;
          const index = ix + iy * resolution + iz * resolution * resolution;
          result = Math.max(result, this.granuleMassCpu[index] || 0);
        }
      }
    }
    return Math.max(0, Math.min(1, result));
  }

  _getMaxThickness() {
    if (!this.triangleThickness) return 0;
    let maxThickness = 0;
    for (const thickness of this.triangleThickness) {
      maxThickness = Math.max(maxThickness, thickness);
    }
    return maxThickness;
  }

  /**
   * 导出上色后的模具模型为 OBJ 文本（局部坐标，未施加仿真旋转）。
   *
   * 颜色与渲染着色器完全一致：
   *   - 已沉积区域：壁厚热力图（thicknessVertexData 中归一化的厚度值，
   *     与 moldRender.wgsl 的 thicknessColor() 同款蓝→青→黄→红映射）
   *   - 未沉积区域：基础模具色 MOLD_BASE_COLOR
   *
   * 顶点格式：`v x y z r g b`（OBJ 顶点色扩展，MeshLab/CloudCompare/three.js 等直接支持），
   * 并同时写入法线 `vn`。网格在加载时已按三角形展开，每个顶点只属于一个三角形，
   * 因此逐顶点颜色即逐三角形颜色，不存在共享顶点歧义。
   *
   * @returns {string|null} OBJ 文本；模具未加载时返回 null
   */
  exportColoredOBJ() {
    if (!this.moldVertices || this.moldVertices.length === 0 || this.moldIndexCount === 0) {
      return null;
    }
    const vertices = this.moldVertices;
    const indices = this.moldIndices;
    const thickness = this.thicknessVertexData;
    const vertexCount = vertices.length / 6;
    const triangleCount = indices.length / 3;
    const fmt = value => +value.toFixed(6);

    const lines = [
      '# Rotomolding simulation - colored mold mesh',
      '# v x y z r g b   (r,g,b = wall-thickness heatmap color)',
      `# vertices: ${vertexCount}, triangles: ${triangleCount}`,
    ];

    for (let c = 0; c < vertexCount; c++) {
      const base = c * 6;
      const scaled = thickness
        ? thickness[c] / Math.max(THICKNESS_COLOR_REFERENCE, this.lastGpuStats.maxThickness || THICKNESS_COLOR_REFERENCE)
        : 0;
      const color = scaled > 0.001 ? thicknessColor(scaled) : MOLD_BASE_COLOR;
      lines.push(
        `v ${fmt(vertices[base])} ${fmt(vertices[base + 1])} ${fmt(vertices[base + 2])} `
        + `${fmt(color[0])} ${fmt(color[1])} ${fmt(color[2])}`
      );
    }
    for (let c = 0; c < vertexCount; c++) {
      const base = c * 6;
      lines.push(`vn ${fmt(vertices[base + 3])} ${fmt(vertices[base + 4])} ${fmt(vertices[base + 5])}`);
    }
    for (let t = 0; t < triangleCount; t++) {
      const a = indices[t * 3] + 1, b = indices[t * 3 + 1] + 1, c = indices[t * 3 + 2] + 1;
      lines.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
    }
    return lines.join('\n') + '\n';
  }

  _buildEdgeBuffer(indices) {
    const dev = this.device;
    const edgeSet = new Set();
    const edges = [];

    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t], b = indices[t + 1], c = indices[t + 2];
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const key = Math.min(p, q) + ',' + Math.max(p, q);
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push(p, q);
        }
      }
    }

    this.edgeIndexBuffer?.destroy();
    const edgeIdx = new Uint32Array(edges);
    this.edgeIndexCount = edgeIdx.length;
    this.edgeIndexBuffer = dev.createBuffer({
      size: edgeIdx.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.edgeIndexBuffer, 0, edgeIdx);

    this.edgeBindGroup = dev.createBindGroup({
      layout: this.edgePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.edgeUniformBuffer } }],
    });
  }

  // ─── 渲染主函数 ────────────────────────────────────────────

  /**
   * @param {number} moldAngleY - 模具 Y 轴旋转角
   * @param {number} moldAngleX - 模具 X 轴旋转角
   */
  render(moldAngleY, moldAngleX) {
    const dev = this.device;

    // ─── 视图投影矩阵 ───
    const vp = this._cameraMatrix();

    // ─── 更新模具 uniform ───
    if (this.moldBindGroup) {
      const cy = Math.cos(moldAngleY || 0), sy = Math.sin(moldAngleY || 0);
      const cx = Math.cos(moldAngleX || 0), sx = Math.sin(moldAngleX || 0);
      const mU = new Float32Array(64);
      mU.set(vp, 0);
      mU[16] = cy;     mU[17] = 0;    mU[18] = -sy;    mU[19] = 0;
      mU[20] = sy*sx;  mU[21] = cx;   mU[22] = cy*sx;  mU[23] = 0;
      mU[24] = sy*cx;  mU[25] = -sx;  mU[26] = cy*cx;  mU[27] = 0;
      mU[28] = 0;      mU[29] = 0;    mU[30] = 0;      mU[31] = 1;
      mU[32] = cy;     mU[33] = sy*sx; mU[34] = sy*cx; mU[35] = 0;
      mU[36] = 0;      mU[37] = cx;    mU[38] = -sx;   mU[39] = 0;
      mU[40] = -sy;    mU[41] = cy*sx; mU[42] = cy*cx; mU[43] = 0;
      mU[44] = 0;      mU[45] = 0;     mU[46] = 0;     mU[47] = 1;
      mU[48] = this.cameraPos[0]; mU[49] = this.cameraPos[1]; mU[50] = this.cameraPos[2];
      mU[52] = 0.5; mU[53] = 0.8; mU[54] = 0.6;
      mU[56] = 0.4; mU[57] = 0.5; mU[58] = 0.7; mU[59] = 0.2;
      mU[60] = this.showThicknessColors ? 1 : 0;
      mU[61] = 1 / Math.max(THICKNESS_COLOR_REFERENCE, this.lastGpuStats.maxThickness || 0);
      dev.queue.writeBuffer(this.moldUniformBuffer, 0, mU.buffer);
      const overlayU = mU.slice();
      overlayU[60] = this.showThicknessColors ? 1 : 0;
      overlayU[62] = 1;
      dev.queue.writeBuffer(this.moldOverlayUniformBuffer, 0, overlayU.buffer);
    }

    // ─── 更新边线 uniform ───
    if (this.edgeBindGroup) {
      const cy = Math.cos(moldAngleY || 0), sy = Math.sin(moldAngleY || 0);
      const cx = Math.cos(moldAngleX || 0), sx = Math.sin(moldAngleX || 0);
      const eU = new Float32Array(32);
      eU.set(vp, 0);
      eU[16] = cy;     eU[17] = 0;    eU[18] = -sy;    eU[19] = 0;
      eU[20] = sy*sx;  eU[21] = cx;   eU[22] = cy*sx;  eU[23] = 0;
      eU[24] = sy*cx;  eU[25] = -sx;  eU[26] = cy*cx;  eU[27] = 0;
      eU[28] = 0;      eU[29] = 0;    eU[30] = 0;      eU[31] = 1;
      dev.queue.writeBuffer(this.edgeUniformBuffer, 0, eU.buffer);
    }

    if (this.granuleBindGroup) {
      const cy = Math.cos(moldAngleY || 0), sy = Math.sin(moldAngleY || 0);
      const cx = Math.cos(moldAngleX || 0), sx = Math.sin(moldAngleX || 0);
      const gU = new Float32Array(40);
      gU.set(vp, 0);
      gU[16] = cy; gU[17] = 0; gU[18] = -sy; gU[19] = 0;
      gU[20] = sy*sx; gU[21] = cx; gU[22] = cy*sx; gU[23] = 0;
      gU[24] = sy*cx; gU[25] = -sx; gU[26] = cy*cx; gU[27] = 0;
      gU[28] = 0; gU[29] = 0; gU[30] = 0; gU[31] = 1;
      gU[32] = this.granuleGridResolution;
      gU[33] = this.granuleGridDomainSize / Math.max(1, this.granuleGridResolution - 1);
      gU[34] = 0.12;
      gU[35] = 0.78;
      gU[36] = this.cameraPos[0]; gU[37] = this.cameraPos[1]; gU[38] = this.cameraPos[2];
      dev.queue.writeBuffer(this.granuleUniformBuffer, 0, gU);
    }

    // ─── 渲染通道 ───
    const colorTex = this.context.getCurrentTexture();
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorTex.createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1,
      },
    });

    // 1) 只画颗粒与空腔之间的自由表面
    if (this.granuleBindGroup && this.granuleCount > 0) {
      pass.setPipeline(this.granulePipeline);
      pass.setBindGroup(0, this.granuleBindGroup);
      pass.setVertexBuffer(0, this.granuleVertexBuffer);
      pass.setVertexBuffer(1, this.granuleGridIndexBuffer);
      pass.draw(6, this.granuleCount);
    }

    // 2) 画模具与累计壁厚
    if (this.moldBindGroup && this.moldIndexCount > 0) {
      pass.setPipeline(this.moldPipeline);
      pass.setBindGroup(0, this.moldBindGroup);
      pass.setVertexBuffer(0, this.moldVertexBuffer);
      pass.setVertexBuffer(1, this.thicknessVertexBuffer);
      pass.setIndexBuffer(this.moldIndexBuffer, 'uint32');
      pass.drawIndexed(this.moldIndexCount);
      if (this.showThicknessColors
        && this.moldOverlayBindGroup
        && (this.lastGpuStats.maxThickness || this._getMaxThickness()) > 1e-6) {
        pass.setPipeline(this.moldOverlayPipeline);
        pass.setBindGroup(0, this.moldOverlayBindGroup);
        pass.drawIndexed(this.moldIndexCount);
      }
    }

    // 3) 画三角形边线（FEA 风格网格）
    if (this.edgeBindGroup && this.edgeIndexCount > 0) {
      pass.setPipeline(this.edgePipeline);
      pass.setBindGroup(0, this.edgeBindGroup);
      pass.setVertexBuffer(0, this.moldVertexBuffer);
      pass.setIndexBuffer(this.edgeIndexBuffer, 'uint32');
      pass.drawIndexed(this.edgeIndexCount);
    }

    pass.end();
    dev.queue.submit([enc.finish()]);
  }
}

// ─── 向量/矩阵辅助 ─────────────────────────────────────────────

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(v) { const l = Math.sqrt(dot(v, v)); return l > 0 ? v.map(x => x / l) : [0, 0, 0]; }
function pointInPolygon2D(point, polygon) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const currentPoint = polygon[index];
    const previousPoint = polygon[previous];
    const intersects = (currentPoint[1] > point[1]) !== (previousPoint[1] > point[1])
      && point[0] < (previousPoint[0] - currentPoint[0])
        * (point[1] - currentPoint[1]) / (previousPoint[1] - currentPoint[1])
        + currentPoint[0];
    if (intersects) inside = !inside;
  }
  return inside;
}
function mulMat(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[i + k * 4] * b[k + j * 4]; r[i + j * 4] = s;
  }
  return r;
}
