/**
 * ============================================================
 *  滚塑仿真引擎 — GPU 颗粒流动模式 (simulation.js)
 * ============================================================
 *
 *  CPU 端负责腔体体素化、模具姿态与颗粒库存；颗粒邻域流动由
 *  renderer.js 中的 WebGPU 计算管线完成。
 */

/**
 * 仿真引擎类
 */
export class Simulation {
  constructor() {
    // ─── 网格 / 腔体数据 ───
    this.meshVertices = null;
    this.meshIndices = null;
    this.cavityPositions = null;      // Float32Array 交错 xyz
    this.cavityGridIndices = null;    // Uint32Array，对应 100^3 规则网格索引
    // A coarser voxel grid keeps the WebGPU solver and CPU contact fallback
    // responsive while retaining enough resolution for wall deposition.
    this.gridResolution = 80;
    this.gridDomainSize = 1.8;
    this.numCavityVoxels = 0;
    this.cavityVoxelVolume = 0;
    this.cavityVolume = 0;

    // ─── 旋转参数 ───
    this.rotationAngle = 0;
    this.rotationAngle2 = 0;
    this.rotationSpeed = 0.8;
    this.rotationSpeed2 = 0.3;
    this.simTime = 0;
    this.subSteps = 30;
    this.physicsStepSeconds = 1 / 60;
    this.dt = this.physicsStepSeconds / this.subSteps;
    this.lastStepDuration = this.physicsStepSeconds;

    // ─── 颗粒参数 ───
    this.initialVolumeFraction = 0.20;
    this.volumeFraction = this.initialVolumeFraction;
    this.initialGranuleVolume = 0;
    this.remainingGranuleVolume = 0;
    this.depositedVolume = 0;
    this.reposeAngle = 35;
    this.terminationCondition = 'particles';
    this.terminationCycles = 1;
  }

  async init() {}

  // ─── 网格加载 ─────────────────────────────────────────────

  /**
   * 加载三角网格并计算腔体体素
   * @param {Float32Array} vertices - 顶点数据（stride=6: xyz + normal）
   * @param {Uint32Array} indices - 三角形索引
   */
  loadMesh(vertices, indices, cavityData = null) {
    this.meshVertices = vertices;
    this.meshIndices = indices;
    if (cavityData?.cavityPositions && cavityData?.cavityGridIndices) {
      this.cavityPositions = cavityData.cavityPositions;
      this.cavityGridIndices = cavityData.cavityGridIndices;
      this.numCavityVoxels = cavityData.numCavityVoxels;
      this.cavityVoxelVolume = cavityData.cavityVoxelVolume;
      this.cavityVolume = cavityData.cavityVolume;
      this.resetGranuleInventory();
      return;
    }
    this._computeCavityFromMesh(vertices, indices);
  }

  getCavityData() {
    return {
      cavityPositions: this.cavityPositions,
      cavityGridIndices: this.cavityGridIndices,
      numCavityVoxels: this.numCavityVoxels,
      cavityVoxelVolume: this.cavityVoxelVolume,
      cavityVolume: this.cavityVolume,
    };
  }

  /**
   * 从三角网格计算腔体体素（射线交叉法）
   * 在当前规则网格中测试每个体素是否在网格内部
   */
  _computeCavityFromMesh(vertices, indices) {
    const res = this.gridResolution;
    const domainSize = this.gridDomainSize;
    const half = domainSize / 2;
    const cellSize = domainSize / res;
    const positions = [];
    const gridIndices = [];

    // 提取三角形（仅位置，跳过法线）
    const numTris = indices.length / 3;
    const stride = 6;
    const tris = [];
    for (let t = 0; t < numTris; t++) {
      const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
      tris.push({
        v0: [vertices[i0 * stride], vertices[i0 * stride + 1], vertices[i0 * stride + 2]],
        v1: [vertices[i1 * stride], vertices[i1 * stride + 1], vertices[i1 * stride + 2]],
        v2: [vertices[i2 * stride], vertices[i2 * stride + 1], vertices[i2 * stride + 2]],
      });
    }

    // 构建 16³ 空间加速网格
    const gridSize = 16;
    const accelCellSize = domainSize / gridSize;
    const spatialGrid = new Array(gridSize * gridSize * gridSize).fill(null).map(() => []);
    for (let t = 0; t < numTris; t++) {
      const tri = tris[t];
      const minB = [Infinity, Infinity, Infinity], maxB = [-Infinity, -Infinity, -Infinity];
      for (const v of [tri.v0, tri.v1, tri.v2]) {
        for (let d = 0; d < 3; d++) {
          minB[d] = Math.min(minB[d], v[d]);
          maxB[d] = Math.max(maxB[d], v[d]);
        }
      }
      const ix0 = Math.max(0, Math.floor((minB[0] + half) / accelCellSize));
      const ix1 = Math.min(gridSize - 1, Math.floor((maxB[0] + half) / accelCellSize));
      const iy0 = Math.max(0, Math.floor((minB[1] + half) / accelCellSize));
      const iy1 = Math.min(gridSize - 1, Math.floor((maxB[1] + half) / accelCellSize));
      const iz0 = Math.max(0, Math.floor((minB[2] + half) / accelCellSize));
      const iz1 = Math.min(gridSize - 1, Math.floor((maxB[2] + half) / accelCellSize));
      for (let iz = iz0; iz <= iz1; iz++) {
        for (let iy = iy0; iy <= iy1; iy++) {
          for (let ix = ix0; ix <= ix1; ix++) {
            spatialGrid[ix + iy * gridSize + iz * gridSize * gridSize].push(t);
          }
        }
      }
    }

    // 射线交叉法判断内外。seen 数组防止跨多个加速单元的三角形被重复计数。
    const seen = new Uint32Array(numTris);
    let rayStamp = 0;
    for (let iz = 0; iz < res; iz++) {
      for (let iy = 0; iy < res; iy++) {
        for (let ix = 0; ix < res; ix++) {
          const x = (ix / (res - 1)) * domainSize - half;
          const y = (iy / (res - 1)) * domainSize - half;
          const z = (iz / (res - 1)) * domainSize - half;

          const cx = Math.min(gridSize - 1, Math.max(0, Math.floor((x + half) / accelCellSize)));
          const cy = Math.min(gridSize - 1, Math.max(0, Math.floor((y + half) / accelCellSize)));
          const cz = Math.min(gridSize - 1, Math.max(0, Math.floor((z + half) / accelCellSize)));

          let crossings = 0;
          rayStamp++;
          for (let gx = cx; gx < gridSize; gx++) {
            for (let dcy = -1; dcy <= 1; dcy++) {
              for (let dcz = -1; dcz <= 1; dcz++) {
                const gcy = cy + dcy, gcz = cz + dcz;
                if (gcy < 0 || gcy >= gridSize || gcz < 0 || gcz >= gridSize) continue;
                const cell = spatialGrid[gx + gcy * gridSize + gcz * gridSize * gridSize];
                for (const t of cell) {
                  if (seen[t] === rayStamp) continue;
                  seen[t] = rayStamp;
                  // 轻微偏斜射线，避免恰好穿过两个三角形的共享边时被计数两次。
                  if (rayIntersectsTriangle([x, y, z], [1, 0.000123, 0.000371], tris[t])) crossings++;
                }
              }
            }
          }

          if (crossings % 2 === 1) {
            positions.push(x, y, z);
            gridIndices.push(ix + iy * res + iz * res * res);
          }
        }
      }
    }

    this.cavityPositions = new Float32Array(positions);
    this.cavityGridIndices = new Uint32Array(gridIndices);
    this.numCavityVoxels = positions.length / 3;
    this.cavityVoxelVolume = cellSize ** 3;
    this.cavityVolume = this.numCavityVoxels * this.cavityVoxelVolume;
    this.resetGranuleInventory();
  }

  resetGranuleInventory(volumeFraction = this.initialVolumeFraction) {
    this.initialVolumeFraction = Math.max(0, Math.min(1, volumeFraction));
    this.volumeFraction = this.initialVolumeFraction;
    this.initialGranuleVolume = this.cavityVolume * this.initialVolumeFraction;
    this.remainingGranuleVolume = this.initialGranuleVolume;
    this.depositedVolume = 0;
    this.simTime = 0;
    this.lastStepDuration = this.physicsStepSeconds;
  }

  resetProcess(volumeFraction = this.initialVolumeFraction) {
    this.simTime = 0;
    this.rotationAngle = 0;
    this.rotationAngle2 = 0;
    this.resetGranuleInventory(volumeFraction);
  }

  consumeGranules(requestedVolume) {
    if (!(requestedVolume > 0) || this.remainingGranuleVolume <= 0) return 0;

    const consumed = Math.min(requestedVolume, this.remainingGranuleVolume);
    this.remainingGranuleVolume -= consumed;
    this.depositedVolume += consumed;

    const epsilon = Math.max(1e-12, this.initialGranuleVolume * 1e-9);
    if (this.remainingGranuleVolume <= epsilon) {
      this.remainingGranuleVolume = 0;
      this.depositedVolume = this.initialGranuleVolume;
    }

    this.volumeFraction = this.cavityVolume > 0
      ? this.remainingGranuleVolume / this.cavityVolume
      : 0;
    return consumed;
  }

  // 世界重力变换到随模具旋转的局部坐标。
  getGravityUpLocal() {
    const ax = this.rotationAngle2;
    return [0, Math.cos(ax), -Math.sin(ax)];
  }

  // ─── 帧更新 ───────────────────────────────────────────────

  step() {
    const substepDt = this.dt;
    this.lastStepDuration = this.physicsStepSeconds;
    for (let sub = 0; sub < this.subSteps; sub++) {
      this.simTime += substepDt;
      this.rotationAngle += this.rotationSpeed * substepDt;
      this.rotationAngle2 += this.rotationSpeed2 * substepDt;
    }
  }

  getGranularParams() {
    const up = this.getGravityUpLocal();
    const ax = this.rotationAngle2;
    return {
      positions: this.cavityPositions,
      gridIndices: this.cavityGridIndices,
      gridResolution: this.gridResolution,
      gridDomainSize: this.gridDomainSize,
      gravityDown: [-up[0], -up[1], -up[2]],
      angularVelocity: [
        this.rotationSpeed2,
        Math.cos(ax) * this.rotationSpeed,
        -Math.sin(ax) * this.rotationSpeed,
      ],
      reposeAngle: this.reposeAngle,
      volumeFraction: this.initialVolumeFraction,
    };
  }

  getDepositionStats() {
    const initial = this.initialGranuleVolume;
    return {
      cavityVolume: this.cavityVolume,
      initialGranuleVolume: initial,
      remainingGranuleVolume: this.remainingGranuleVolume,
      depositedVolume: this.depositedVolume,
      remainingRatio: initial > 0 ? this.remainingGranuleVolume / initial : 0,
      isComplete: initial > 0 && this.remainingGranuleVolume <= 0,
    };
  }

  getRotationCycles() {
    const fullTurn = Math.PI * 2;
    const primarySpeed = Math.abs(this.rotationSpeed);
    const secondarySpeed = Math.abs(this.rotationSpeed2);
    const jointPeriodSeconds = getJointRotationPeriodSeconds(primarySpeed, secondarySpeed);
    return {
      primary: Math.abs(this.rotationAngle) / fullTurn,
      secondary: Math.abs(this.rotationAngle2) / fullTurn,
      joint: jointPeriodSeconds > 0 ? this.simTime / jointPeriodSeconds : 0,
      jointPeriodSeconds,
    };
  }

  isTerminationComplete() {
    if (this.terminationCondition === 'cycles') {
      const cycles = this.getRotationCycles();
      return cycles.jointPeriodSeconds > 0 && cycles.joint >= this.terminationCycles;
    }
    return this.getDepositionStats().isComplete;
  }

  setParams(params) {
    let inventoryReset = false;
    if (params.rotationSpeed !== undefined) this.rotationSpeed = params.rotationSpeed;
    if (params.rotationSpeed2 !== undefined) this.rotationSpeed2 = params.rotationSpeed2;
    if (params.terminationCondition !== undefined) {
      this.terminationCondition = params.terminationCondition === 'cycles' ? 'cycles' : 'particles';
    }
    if (params.terminationCycles !== undefined) {
      this.terminationCycles = Math.max(1, Math.floor(Number(params.terminationCycles) || 1));
    }
    if (
      params.volumeFraction !== undefined
      && Math.abs(params.volumeFraction - this.initialVolumeFraction) > 1e-8
    ) {
      this.resetGranuleInventory(params.volumeFraction);
      inventoryReset = true;
    }
    if (params.reposeAngle !== undefined) {
      this.reposeAngle = Math.max(0, Math.min(60, params.reposeAngle));
    }
    return { inventoryReset };
  }
}

// ─── 几何辅助（射线三角形交叉） ────────────────────────────

// The mold returns to its initial orientation when both axis angles are
// integer multiples of 2*pi. For rational speeds, this is 2*pi divided by
// the greatest common divisor of the two angular speeds.
function getJointRotationPeriodSeconds(primarySpeed, secondarySpeed) {
  if (!(primarySpeed > 0) && !(secondarySpeed > 0)) return Infinity;
  if (!(primarySpeed > 0)) return (Math.PI * 2) / secondarySpeed;
  if (!(secondarySpeed > 0)) return (Math.PI * 2) / primarySpeed;
  const scale = 1e6;
  let a = Math.round(primarySpeed * scale);
  let b = Math.round(secondarySpeed * scale);
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  const gcdSpeed = a / scale;
  return gcdSpeed > 0 ? (Math.PI * 2) / gcdSpeed : Infinity;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function rayIntersectsTriangle(origin, dir, tri) {
  const e1 = sub(tri.v1, tri.v0);
  const e2 = sub(tri.v2, tri.v0);
  const h = cross(dir, e2);
  const a = dot(e1, h);
  if (a > -1e-10 && a < 1e-10) return false;

  const f = 1.0 / a;
  const s = sub(origin, tri.v0);
  const u = f * dot(s, h);
  if (u < 0.0 || u > 1.0) return false;

  const q = cross(s, e1);
  const v = f * dot(dir, q);
  if (v < 0.0 || u + v > 1.0) return false;

  const t = f * dot(e2, q);
  return t > 1e-10;
}
