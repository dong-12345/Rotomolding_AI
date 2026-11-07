/**
 * ============================================================
 *  滚塑工艺仿真 — GPU 颗粒流动模式 (main.js)
 * ============================================================
 *
 *  本文件是整个应用的入口，负责：
 *    1. 初始化 WebGPU 设备与画布
 *    2. 加载 OpenCascade 内核 → 生成默认立方体网格
 *    3. 加载 STEP 模型 → 三角网格
 *    4. 启动仿真主循环（requestAnimationFrame）
 *    5. 绑定 UI 控件
 */

import { Simulation } from './simulation.js';
import { Renderer } from './renderer.js';
import { loadSTEPFile, initOpenCascade, generateOCCCube } from './stepLoader.js';

const THICKNESS_UPDATE_INTERVAL = 6;

class App {
  constructor() {
    this.sim = new Simulation();
    this.renderer = new Renderer();
    this.running = false;
    this.frameCount = 0;
    this.lastFpsTime = 0;
    this.fps = 0;
    this.currentSTEPFile = null;
    this.isMeshing = false;
    this.maxWallThickness = 0;
    this.wallThicknessDelaySeconds = 5;
    this.wallThicknessPendingSeconds = 0;
    this.wallThicknessPendingSteps = 0;
    this.wallRunElapsed = 0;
    this.realRunElapsed = 0;
    this.fastBatchSteps = 30;
    this.lastLoopTimestamp = 0;
    this.loopGeneration = 0;
    this.gpuAdapterName = '';
  }

  async init() {
    if (!navigator.gpu) {
      this.setStatus('错误: 此浏览器不支持 WebGPU。请使用 Chrome 113+ 或 Edge 113+。');
      return false;
    }

    const gpuStatus = window.rotomoldingDesktop?.gpu
      ? await window.rotomoldingDesktop.gpu.status()
      : null;
    const selectedGpu = gpuStatus?.selection || null;
    const powerPreference = selectedGpu?.vendor === 'intel' ? 'low-power' : 'high-performance';
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) {
      this.setStatus('错误: 无法获取 WebGPU 适配器。');
      return false;
    }

    const adapterInfo = adapter.info;
    verifySelectedGpu(selectedGpu, adapterInfo || {});
    this.gpuAdapterName = adapterInfo?.description || adapterInfo?.device || '';
    updateActiveGpuLabel(this.gpuAdapterName);

    let device;
    try {
      device = await adapter.requestDevice();
    } catch (e) {
      console.warn('Device request failed:', e);
      throw new Error('无法创建 WebGPU 设备');
    }

    const canvas = document.getElementById('canvas');
    const viewCube = document.getElementById('viewCube');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    device.addEventListener('uncapturederror', e => {
      console.error('[WebGPU] Uncaptured error:', e.error?.message || e.error);
    });
    device.lost.then(info => {
      console.error('[WebGPU] Device lost:', info.message);
    });

    await this.sim.init();
    await this.renderer.init(device, canvas, viewCube);
    this.updateParamsFromUI();

    this.renderer.onCameraChange = () => {
      if (!this.running) this.render();
    };

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      this.renderer.resize(canvas.width, canvas.height);
    });

    // 加载 OpenCascade 内核，用于生成精确的默认模具
    this.setStatus('正在加载 OpenCascade WASM (~30MB)...');
    try {
      await initOpenCascade(msg => this.setStatus(msg));
    } catch (e) {
      this.setStatus(`OCCT 加载失败: ${e.message}，使用备选方案`);
      console.error('Failed to load OCCT:', e.message);
    }

    this.setStatus('正在生成默认模具...');
    await this.loadDefaultMold();

    this.setStatus('就绪 - 点击"开始仿真"启动');
    return true;
  }

  // ─── 模具加载 ──────────────────────────────────────────────

  async loadDefaultMold() {
    try {
      const oc = await initOpenCascade();
      // 使用 OpenCascade 内核生成立方体
      const meshResult = generateOCCCube(oc, 0.8, 0.015);

      this.sim.loadMesh(meshResult.vertices, meshResult.indices);
      this.renderer.loadMoldMesh({ vertices: meshResult.vertices, indices: meshResult.indices });
      this.renderer.loadGranuleVoxels(this.sim.getGranularParams());
      document.getElementById('voxelGrid').textContent =
        `${this.sim.gridResolution}³ (${this.sim.gridResolution ** 3})`;
      document.getElementById('voxelCount').textContent = String(this.sim.numCavityVoxels);

      document.getElementById('gridRes').textContent =
        `${meshResult.indices.length / 3} 三角形`;

      this.setStatus(`方形模具已加载 (${this.sim.numCavityVoxels} 腔体体素)`);
      this.maxWallThickness = 0;
      this.wallRunElapsed = 0;
      this.wallThicknessPendingSeconds = 0;
      this.wallThicknessPendingSteps = 0;
      this.realRunElapsed = 0;
      this.updateDepositionStats();
      requestAnimationFrame(() => this.render());
    } catch (e) {
      this.setStatus(`模具加载失败: ${e.message}`);
      console.error('[Mold] Error:', e);
    }
  }

  async loadSTEPFile(file) {
    if (this.isMeshing) return;
    this.isMeshing = true;
    this.currentSTEPFile = file;
    const fileInput = document.getElementById('fileInput');
    const remeshButton = document.getElementById('btnRemesh');
    fileInput.disabled = true;
    remeshButton.disabled = true;
    try {
      const buffer = await file.arrayBuffer();
      const targetEdgeLength = parseFloat(document.getElementById('meshSize').value);
      // 提高三角化精度（linearDeflection 从 0.03 降到 0.008）
      const result = await loadSTEPFile(
        buffer,
        0.008,
        msg => this.setStatus(msg),
        targetEdgeLength,
      );

      this.setStatus('正在计算腔体体素...');

      this.sim.loadMesh(
        result.analysisVertices || result.vertices,
        result.analysisIndices || result.indices,
      );
      this.renderer.loadMoldMesh({ vertices: result.vertices, indices: result.indices });
      this.renderer.loadGranuleVoxels(this.sim.getGranularParams());
      document.getElementById('voxelGrid').textContent =
        `${this.sim.gridResolution}³ (${this.sim.gridResolution ** 3})`;
      document.getElementById('voxelCount').textContent = String(this.sim.numCavityVoxels);

      const triangleCount = result.indices.length / 3;
      const meshStats = result.statistics;
      document.getElementById('gridRes').textContent = meshStats
        ? `${triangleCount} 三角形 · h90% ${meshStats.p05.toFixed(3)}–${meshStats.p95.toFixed(3)}`
        : `${triangleCount} 三角形`;

      this.setStatus(`STEP 模型已加载 (${this.sim.numCavityVoxels} 腔体体素, ${triangleCount} 三角形)`);
      this.maxWallThickness = 0;
      this.wallRunElapsed = 0;
      this.wallThicknessPendingSeconds = 0;
      this.wallThicknessPendingSteps = 0;
      this.realRunElapsed = 0;
      this.updateDepositionStats();

      this.render();
    } catch (e) {
      console.error(e);
      this.setStatus(`错误: ${e.message}`);
    } finally {
      this.isMeshing = false;
      fileInput.disabled = false;
      remeshButton.disabled = !this.currentSTEPFile;
    }
  }

  // ─── 仿真主循环 ───────────────────────────────────────────

  updateParamsFromUI() {
    const result = this.sim.setParams({
      rotationSpeed: parseFloat(document.getElementById('rotationSpeed').value),
      rotationSpeed2: parseFloat(document.getElementById('rotationSpeed2').value),
      volumeFraction: parseInt(document.getElementById('volumeFraction').value) / 100,
      reposeAngle: parseFloat(document.getElementById('reposeAngle').value),
      terminationCondition: document.getElementById('terminationCondition').value,
      terminationCycles: parseInt(document.getElementById('terminationCycles').value, 10),
    });
    if (result.inventoryReset) {
      this.renderer.resetGranules(this.sim.getGranularParams());
      this.renderer.resetThickness();
      this.maxWallThickness = 0;
      this.wallRunElapsed = 0;
      this.wallThicknessPendingSeconds = 0;
      this.wallThicknessPendingSteps = 0;
      this.realRunElapsed = 0;
      this.updateDepositionStats();
    }
  }

  start() {
    if (this.running) return;
    this.updateParamsFromUI();
    this.running = true;
    this.loopGeneration++;
    const generation = this.loopGeneration;
    const now = performance.now();
    this.lastFpsTime = now;
    this.lastLoopTimestamp = now;
    this.frameCount = 0;

    const showProcess = document.getElementById('showProcess')?.checked ?? true;
    this.setStatus(showProcess ? '仿真运行中...' : '仿真加速计算中...');
    this._loop(generation);
  }

  pause() {
    this.running = false;
    this.loopGeneration++;
    const now = performance.now();
    if (this.lastLoopTimestamp > 0) {
      this.realRunElapsed += Math.max(0, (now - this.lastLoopTimestamp) / 1000);
      this.lastLoopTimestamp = now;
    }
    this.render();
    this.updateDepositionStats();
    this.setStatus('仿真已暂停');
  }

  reset() {
    this.running = false;
    this.loopGeneration++;
    this.frameCount = 0;
    this.fps = 0;
    this.lastLoopTimestamp = 0;
    this.wallRunElapsed = 0;
    this.realRunElapsed = 0;
    document.getElementById('fps').textContent = '0';

    const volumeFraction = parseInt(document.getElementById('volumeFraction').value) / 100;
    this.sim.resetProcess(volumeFraction);
    this.renderer.resetGranules(this.sim.getGranularParams());
    this.renderer.resetThickness();
    this.maxWallThickness = 0;
    this.wallThicknessPendingSeconds = 0;
    this.wallThicknessPendingSteps = 0;
    this.updateDepositionStats();
    this.render();
    this.setStatus('当前模具已重置为初始状态');
  }

  /**
   * 导出上色后的模具模型（壁厚热力图着色，局部坐标）为 OBJ 文件并触发下载
   */
  async exportOBJ() {
    await this.renderer.prepareExport();
    const obj = this.renderer.exportColoredOBJ();
    if (!obj) {
      this.setStatus('无可导出的模型，请先加载模具');
      return;
    }
    const baseName = this.currentSTEPFile
      ? this.currentSTEPFile.name.replace(/\.(step|stp)$/i, '')
      : 'default_mold';
    const blob = new Blob([obj], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${baseName}_thickness.obj`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    this.setStatus(`已导出 ${link.download} (${(blob.size / 1024).toFixed(1)} KB)`);
  }

  _loop(generation = this.loopGeneration) {
    if (!this.running || generation !== this.loopGeneration) return;
    this._runLoopFrame(generation);
  }

  async _runLoopFrame(generation) {
    if (!this.running || generation !== this.loopGeneration) return;

    let showProcess = document.getElementById('showProcess')?.checked ?? true;
    let shouldRender = false;
    let completed = false;
    try {
      showProcess = document.getElementById('showProcess')?.checked ?? true;
      const batchSteps = showProcess ? 1 : this.fastBatchSteps;
      this.updateParamsFromUI();

      for (let batch = 0; batch < batchSteps && this.running && generation === this.loopGeneration; batch++) {
        this.sim.step();
        const collectStats = this.sim.simTime >= this.wallThicknessDelaySeconds;
        this.renderer.stepGranules(this.sim.getGranularParams(), undefined, collectStats);
        await this.renderer.waitForGranuleReadback();
        if (!this.running || generation !== this.loopGeneration) break;
        let deposition = { depositedVolume: 0, maxThickness: this.maxWallThickness };
        if (this.sim.simTime >= this.wallThicknessDelaySeconds) {
          this.wallThicknessPendingSeconds += this.sim.lastStepDuration;
          this.wallThicknessPendingSteps++;
          if (this.wallThicknessPendingSteps >= THICKNESS_UPDATE_INTERVAL || this.sim.isTerminationComplete()) {
            deposition = await this.renderer.accumulateThickness(
              this.wallThicknessPendingSeconds,
              this.sim.remainingGranuleVolume,
            );
            this.sim.consumeGranules(deposition.depositedVolume);
            this.renderer.setGranuleVolumeRatio(this.sim.getDepositionStats().remainingRatio);
            this.maxWallThickness = deposition.maxThickness;
            this.wallThicknessPendingSeconds = 0;
            this.wallThicknessPendingSteps = 0;
          }
        }
        if (this.sim.isTerminationComplete()) {
          this.running = false;
          completed = true;
        }
      }
      this.wallRunElapsed = this.sim.simTime;
      shouldRender = showProcess || !this.running;
    } catch (e) {
      console.error('[Loop] Error:', e);
    }

    if (generation !== this.loopGeneration) return;
    const frameEnd = performance.now();
    this.realRunElapsed += Math.max(0, (frameEnd - this.lastLoopTimestamp) / 1000);
    this.lastLoopTimestamp = frameEnd;
    this.updateDepositionStats();
    if (shouldRender) this.render();
    if (completed) {
      this.setStatus(`仿真完成 - 滚塑颗粒已全部沉积（仿真用时 ${formatElapsedSeconds(this.realRunElapsed)}）`);
    }

    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsTime = now;
      document.getElementById('fps').textContent = this.fps;

      if (this.running) {
        const showProcess = document.getElementById('showProcess')?.checked ?? true;
        this.setStatus(showProcess ? '仿真运行中...' : '仿真加速计算中...');
      }
    }

    if (this.running && generation === this.loopGeneration) {
      requestAnimationFrame(() => this._loop(generation));
    }
  }

  render() {
    try {
      this.renderer.render(
        this.sim.rotationAngle,
        this.sim.rotationAngle2,
      );
    } catch (e) {
      this.setStatus(`渲染错误: ${e.message}`);
      console.error('[Render] Error:', e);
    }
  }

  updateDepositionStats() {
    const stats = this.sim.getDepositionStats();
    const gravityDown = this.sim.getGranularParams().gravityDown;
    const gravityTilt = Math.acos(
      Math.max(-1, Math.min(1, -gravityDown[1])),
    ) * 180 / Math.PI;
    const remainingPercent = stats.initialGranuleVolume > 0
      ? stats.remainingRatio * 100
      : 0;
    const values = {
      granuleRemaining: ['颗粒剩余', `${remainingPercent.toFixed(1)}%`],
      granuleVolume: ['剩余颗粒体积', stats.remainingGranuleVolume.toFixed(5)],
      depositedVolume: ['累计沉积体积', stats.depositedVolume.toFixed(5)],
      maxThickness: ['最大壁厚', this.maxWallThickness.toFixed(5)],
      simulationTime: ['仿真时间', `${this.sim.simTime.toFixed(2)} s`],
      simulationElapsed: ['仿真用时', formatElapsedSeconds(this.realRunElapsed)],
      gravityTilt: ['当前重力倾角', `${gravityTilt.toFixed(1)}°`],
      rotationCycles: ['模具联合周期', `${this.sim.getRotationCycles().joint.toFixed(2)} / ${this.sim.terminationCycles}`],
      wallDelay: ['壁厚计算', this.wallRunElapsed < this.wallThicknessDelaySeconds
        ? `等待 ${(this.wallThicknessDelaySeconds - this.wallRunElapsed).toFixed(1)} 秒`
        : '已开始'],
    };

    const statsPanel = document.getElementById('stats');
    for (const [id, [label, value]] of Object.entries(values)) {
      let output = document.getElementById(id);
      if (!output && statsPanel) {
        const row = document.createElement('div');
        row.append(`${label}: `);
        output = document.createElement('span');
        output.id = id;
        row.append(output);
        statsPanel.append(row);
      }
      if (output) output.textContent = value;
    }
  }

  setStatus(msg) {
    document.getElementById('status').textContent = msg;
    console.log('[App]', msg);
  }
}

function formatElapsedSeconds(seconds) {
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes} min ${String(remainingSeconds).padStart(2, '0')} s`;
}

function updateActiveGpuLabel(adapterName) {
  const label = document.getElementById('gpuActiveAdapter');
  if (!label) return;
  label.textContent = adapterName
    ? `当前 WebGPU 适配器：${adapterName}`
    : '当前 WebGPU 适配器：浏览器未公开型号。';
}

function normalizedGpuText(value) {
  return String(value || '').trim().toLowerCase();
}

function verifySelectedGpu(selection, adapterInfo) {
  if (!selection) return;
  const adapterVendor = normalizedGpuText(adapterInfo.vendor);
  const adapterDescription = normalizedGpuText(adapterInfo.description);
  if (selection.vendor && adapterVendor && !adapterVendor.includes(selection.vendor)) {
    throw new Error(`Selected GPU ${selection.name} (${selection.id}) was not assigned.`);
  }
  if (selection.requiresAdapterDescription && !adapterDescription) {
    throw new Error(`WebGPU cannot verify the selected GPU ${selection.name} (${selection.id}).`);
  }
  if (adapterDescription && !adapterDescription.includes(normalizedGpuText(selection.name))) {
    throw new Error(`Selected GPU ${selection.name} (${selection.id}) was not assigned.`);
  }
  if (!adapterVendor && !adapterDescription) {
    throw new Error(`WebGPU did not expose adapter identity, so GPU ${selection.name} (${selection.id}) cannot be verified.`);
  }
}

function gpuDeviceLabel(device) {
  const name = device.name || `GPU ${device.index}`;
  return `GPU ${device.index} · ${name} · ${device.id}`;
}

function populateGpuDeviceOptions(select, devices, selectedId) {
  const options = [new Option('自动选择 GPU', '')]
    .concat(devices.map(device => new Option(gpuDeviceLabel(device), device.id)));
  select.replaceChildren(...options);
  select.value = selectedId && devices.some(device => device.id === selectedId) ? selectedId : '';
}

async function initializeGpuPreferenceUI() {
  const section = document.getElementById('gpuPreferenceSection');
  const select = document.getElementById('gpuPreference');
  const hint = document.getElementById('gpuPreferenceHint');
  const apply = document.getElementById('btnApplyGpuPreference');
  const desktop = window.rotomoldingDesktop;
  section.hidden = false;

  if (!desktop?.gpu) {
    select.replaceChildren(new Option('仅桌面版支持明确设备选择', ''));
    select.disabled = true;
    apply.hidden = true;
    hint.textContent = '网页端 WebGPU 不提供按设备编号绑定功能。';
    return;
  }

  try {
    const status = await desktop.gpu.status();
    const devices = status.selectableDevices || [];
    populateGpuDeviceOptions(select, devices, status.selection?.id);
    const selected = status.selection;
    const details = [
      status.renderer ? `当前渲染器：${status.renderer}` : '',
      selected ? `已选择：${gpuDeviceLabel(selected)}` : '尚未选择特定 GPU。',
      '切换后将重启应用。批处理会验证实际 WebGPU 适配器；不匹配时任务不会运行。',
    ].filter(Boolean);
    hint.textContent = details.join(' ');
  } catch (error) {
    console.warn('Unable to initialize GPU preference UI:', error);
    hint.textContent = 'GPU 检测失败，将使用系统默认 GPU。';
  }

  apply.addEventListener('click', async () => {
    apply.disabled = true;
    try {
      const selectedOption = select.selectedOptions[0];
      const selectedText = selectedOption?.textContent || '自动选择 GPU';
      if (!window.confirm(`确认使用“${selectedText}”？应用后软件会重启。`)) {
        apply.disabled = false;
        return;
      }
      const result = await desktop.gpu.applyDevice(select.value);
      if (!result?.restarting) apply.disabled = false;
    } catch (error) {
      apply.disabled = false;
      console.error('Unable to apply GPU preference:', error);
      hint.textContent = `无法切换 GPU：${error.message}`;
    }
  });
}

const BATCH_STATUS_LABELS = {
  pending: '等待仿真',
  running: '正在仿真',
  paused: '已暂停',
  completed: '仿真完成',
  failed: '失败',
};

function batchOptionsFromUI() {
  return {
    meshSize: parseFloat(document.getElementById('meshSize').value),
    rotationSpeed: parseFloat(document.getElementById('rotationSpeed').value),
    rotationSpeed2: parseFloat(document.getElementById('rotationSpeed2').value),
    volumeFraction: parseInt(document.getElementById('volumeFraction').value, 10) / 100,
    reposeAngle: parseFloat(document.getElementById('reposeAngle').value),
    terminationCondition: document.getElementById('terminationCondition').value,
    terminationCycles: parseInt(document.getElementById('terminationCycles').value, 10),
  };
}

function renderBatchStatus(snapshot) {
  if (!snapshot) return;
  const counts = snapshot.counts || {};
  const system = snapshot.system || {};
  const total = snapshot.total || 0;
  const terminal = (counts.completed || 0) + (counts.failed || 0) === total && total > 0;
  const summary = snapshot.paused
    ? `批量仿真已暂停，已完成 ${counts.completed || 0}/${total}`
    : terminal
      ? `批量仿真已结束，完成 ${counts.completed || 0}/${total}${counts.failed ? `，失败 ${counts.failed}` : ''}`
      : `正在批量仿真：运行 ${counts.running || 0}，等待 ${counts.pending || 0}，完成 ${counts.completed || 0}/${total}`;
  document.getElementById('batchSummary').textContent = summary;
  document.getElementById('batchCpu').textContent = system.cpuPercent == null ? '采样中' : `${system.cpuPercent.toFixed(0)}%`;
  document.getElementById('batchGpu').textContent = system.gpuPercent == null ? '不可用' : `${system.gpuPercent.toFixed(0)}%`;
  document.getElementById('batchParallelism').textContent = `${snapshot.active || 0}/${snapshot.parallelism || 1}`;
  document.getElementById('batchOutput').textContent = snapshot.outputDirectory ? `结果目录：${snapshot.outputDirectory}` : '';

  const list = document.getElementById('batchJobList');
  const rows = (snapshot.jobs || []).map(job => {
    const row = document.createElement('div');
    row.className = 'batch-job-row';

    const name = document.createElement('span');
    name.className = 'batch-job-name';
    name.textContent = job.name;
    name.title = job.name;

    const status = document.createElement('span');
    status.className = `batch-job-status ${job.status}`;
    status.textContent = BATCH_STATUS_LABELS[job.status] || job.status;

    const detail = document.createElement('span');
    detail.className = 'batch-job-detail';
    const percentage = Number.isFinite(job.progress) ? ` ${Math.round(job.progress * 100)}%` : '';
    detail.textContent = job.error || `${job.stage || '等待中'}${percentage}`;
    detail.title = detail.textContent;
    row.append(name, status, detail);
    return row;
  });
  list.replaceChildren(...rows);

  document.getElementById('btnBatchPause').disabled = Boolean(snapshot.paused) || terminal;
  document.getElementById('btnBatchResume').disabled = !snapshot.paused || terminal;
}

async function startBatchSimulation(app) {
  const desktop = window.rotomoldingDesktop;
  const files = Array.from(document.getElementById('batchFileInput').files || []);
  if (files.length === 0) {
    app.setStatus('请先选择一个或多个 STEP 文件');
    return;
  }
  if (!desktop?.batch) {
    app.setStatus('批量并行仿真需要使用打包后的 Windows 桌面版');
    return;
  }

  try {
    const filePaths = await desktop.files.getPaths(files);
    if (filePaths.length === 0) throw new Error('未能读取所选 STEP 文件路径');
    document.getElementById('batchModal').hidden = false;
    const snapshot = await desktop.batch.start({ filePaths, options: batchOptionsFromUI() });
    renderBatchStatus(snapshot);
  } catch (error) {
    console.error('[Batch] Failed to start:', error);
    app.setStatus(`批量仿真启动失败: ${error.message}`);
  }
}

// ─── 启动 ──────────────────────────────────────────────────────

const app = new App();

initializeGpuPreferenceUI();

document.getElementById('btnDefault').addEventListener('click', () => {
  app.loadDefaultMold();
});
document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) app.loadSTEPFile(e.target.files[0]);
});
document.getElementById('btnRemesh').addEventListener('click', () => {
  if (app.currentSTEPFile) app.loadSTEPFile(app.currentSTEPFile);
});
document.getElementById('btnStart').addEventListener('click', () => app.start());
document.getElementById('btnPause').addEventListener('click', () => app.pause());
document.getElementById('btnReset').addEventListener('click', () => app.reset());
document.getElementById('btnExportOBJ').addEventListener('click', () => app.exportOBJ());
document.getElementById('btnBatchStart').addEventListener('click', () => startBatchSimulation(app));
document.getElementById('btnBatchClose').addEventListener('click', () => {
  document.getElementById('batchModal').hidden = true;
});
document.getElementById('btnBatchPause').addEventListener('click', () => {
  window.rotomoldingDesktop?.batch.pause();
});
document.getElementById('btnBatchResume').addEventListener('click', () => {
  window.rotomoldingDesktop?.batch.resume();
});

if (window.rotomoldingDesktop?.batch) {
  window.rotomoldingDesktop.batch.onStatus(snapshot => {
    renderBatchStatus(snapshot);
  });
}

const showThicknessColors = document.getElementById('showThicknessColors');
const terminationCondition = document.getElementById('terminationCondition');
const terminationCycles = document.getElementById('terminationCycles');
const updateTerminationUI = () => {
  const cycleMode = terminationCondition?.value === 'cycles';
  if (terminationCycles) terminationCycles.disabled = !cycleMode;
  app.updateParamsFromUI();
  if (!app.running) app.render();
};
terminationCondition?.addEventListener('change', updateTerminationUI);
terminationCycles?.addEventListener('input', updateTerminationUI);
const updateThicknessColorVisibility = () => {
  app.renderer.setThicknessColorsVisible(showThicknessColors.checked);
  document.getElementById('thicknessLegend').hidden = !showThicknessColors.checked;
  if (!app.running && app.renderer.device) app.render();
};
showThicknessColors.addEventListener('change', updateThicknessColorVisibility);
updateThicknessColorVisibility();
updateTerminationUI();

for (const [id, labelId] of [
  ['rotationSpeed', 'rotationSpeedLabel'],
  ['rotationSpeed2', 'rotationSpeed2Label'],
  ['volumeFraction', 'volumeFractionLabel'],
  ['reposeAngle', 'reposeAngleLabel'],
]) {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('input', () => {
      const label = document.getElementById(labelId);
      if (label) label.textContent = el.value;
      app.updateParamsFromUI();
      if (!app.running) {
        app.render();
      }
    });
  }
}

app.init().then(ok => {
  if (ok) requestAnimationFrame(() => app.render());
}).catch(err => {
  console.error('Init failed:', err);
  app.setStatus(`初始化失败: ${err.message}`);
});

const meshSize = document.getElementById('meshSize');
meshSize.addEventListener('input', () => {
  document.getElementById('meshSizeLabel').textContent = Number(meshSize.value).toFixed(3);
});
