const { app, BrowserWindow, ipcMain } = require('electron');
const { execFile, execFileSync } = require('child_process');
const fs = require('fs');
const fsp = fs.promises;
const os = require('os');
const path = require('path');

const GPU_PREFERENCE_ARGUMENT = '--gpu-preference=';
const GPU_PREFERENCE_OPTION = '--gpu';
const GPU_PREFERENCES = new Set(['default', 'low-power', 'high-performance']);
const GPU_DEVICE_ARGUMENT = '--gpu-device=';
const GPU_DEVICE_OPTION = '--gpu-device';
const GPU_INDEX_ARGUMENT = '--gpu-index=';
const GPU_INDEX_OPTION = '--gpu-index';
const GPU_LIST_OPTION = '--list-gpus';

function normalizeGpuHexId(value) {
  const text = String(value || '').trim().toLowerCase();
  if (!text) return '';
  const match = text.match(/(?:0x)?([0-9a-f]{1,8})$/i);
  return match ? match[1].replace(/^0+/, '') || '0' : '';
}

function gpuKey(vendorId, deviceId) {
  const vendor = normalizeGpuHexId(vendorId);
  const device = normalizeGpuHexId(deviceId);
  return vendor && device ? `${vendor}:${device}` : '';
}

function gpuVendorName(vendorId) {
  switch (normalizeGpuHexId(vendorId)) {
    case '10de': return 'nvidia';
    case '8086': return 'intel';
    case '1002':
    case '1022': return 'amd';
    default: return '';
  }
}

function windowsGpuInventorySync() {
  if (process.platform !== 'win32') return [];
  const command = [
    "$entries = Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Video\\*\\0000' -ErrorAction SilentlyContinue;",
    '$entries | ForEach-Object {',
    "  $id = [string]$_.MatchingDeviceId;",
    "  $vendor = if ($id -match 'VEN_([0-9A-Fa-f]{4})') { $Matches[1] } else { '' };",
    "  $device = if ($id -match 'DEV_([0-9A-Fa-f]{4})') { $Matches[1] } else { '' };",
    "  [PSCustomObject]@{ name = [string]$_.DriverDesc; vendorId = $vendor; deviceId = $device; pnpDeviceId = $id }",
    '} | ConvertTo-Json -Compress',
  ].join(' ');
  try {
    const output = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], {
      windowsHide: true,
      encoding: 'utf8',
      timeout: 5000,
    }).trim();
    if (!output) return [];
    const parsed = JSON.parse(output);
    return (Array.isArray(parsed) ? parsed : [parsed]).map((device, index) => ({
      index,
      name: String(device.name || ''),
      vendorId: normalizeGpuHexId(device.vendorId),
      deviceId: normalizeGpuHexId(device.deviceId),
      pnpDeviceId: String(device.pnpDeviceId || ''),
    })).filter(device => device.vendorId && device.deviceId);
  } catch (_) { return []; }
}

const windowsGpuInventory = windowsGpuInventorySync();

function argumentValue(option, equalsArgument) {
  const inline = process.argv.find(value => value.startsWith(equalsArgument));
  if (inline) return inline.slice(equalsArgument.length);
  const index = process.argv.indexOf(option);
  return index >= 0 ? process.argv[index + 1] || null : null;
}

function gpuSelectionFromArguments() {
  const requestedDevice = argumentValue(GPU_DEVICE_OPTION, GPU_DEVICE_ARGUMENT);
  if (requestedDevice) {
    const [vendorId, deviceId] = requestedDevice.split(':');
    const id = gpuKey(vendorId, deviceId);
    if (!id) throw new Error(`Invalid GPU device identifier: ${requestedDevice}`);
    return { id, source: 'device' };
  }
  const requestedIndex = argumentValue(GPU_INDEX_OPTION, GPU_INDEX_ARGUMENT);
  if (requestedIndex === null) return null;
  if (!/^\d+$/.test(requestedIndex)) throw new Error(`Invalid GPU index: ${requestedIndex}`);
  const device = windowsGpuInventory.find(candidate => candidate.index === Number(requestedIndex));
  if (!device) throw new Error(`GPU index ${requestedIndex} is not available. Run --list-gpus to inspect selectable devices.`);
  return { id: gpuKey(device.vendorId, device.deviceId), index: device.index, source: 'index' };
}

function gpuPreferenceFromArguments() {
  const argument = process.argv.find(value => value.startsWith(GPU_PREFERENCE_ARGUMENT));
  const optionIndex = process.argv.indexOf(GPU_PREFERENCE_OPTION);
  const preference = argument?.slice(GPU_PREFERENCE_ARGUMENT.length)
    || (optionIndex >= 0 ? process.argv[optionIndex + 1] : null);
  if (gpuSelection?.id) {
    const vendor = gpuSelection.id.split(':')[0];
    if (vendor === '10de' || vendor === '1002' || vendor === '1022') return 'high-performance';
    if (vendor === '8086') return 'low-power';
  }
  const fallback = process.argv.includes('sweep') ? 'high-performance' : 'default';
  return GPU_PREFERENCES.has(preference) ? preference : fallback;
}

const gpuSelection = gpuSelectionFromArguments();
const gpuPreference = gpuPreferenceFromArguments();
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch('disable-gpu-sandbox');
app.commandLine.appendSwitch('no-sandbox');
if (gpuPreference === 'low-power') app.commandLine.appendSwitch('force_low_power_gpu');
if (gpuPreference === 'high-performance') app.commandLine.appendSwitch('force_high_performance_gpu');

function selectedGpuDevice() {
  if (!gpuSelection?.id) return null;
  return windowsGpuInventory.find(device => gpuKey(device.vendorId, device.deviceId) === gpuSelection.id) || null;
}

function selectedGpuIsUnambiguous() {
  if (!gpuSelection?.id) return true;
  return windowsGpuInventory.filter(device => gpuKey(device.vendorId, device.deviceId) === gpuSelection.id).length === 1;
}

function gpuSelectionPayload() {
  const device = selectedGpuDevice();
  if (!gpuSelection || !device) return null;
  return {
    id: gpuSelection.id,
    index: device.index,
    name: device.name,
    vendorId: device.vendorId,
    deviceId: device.deviceId,
    vendor: gpuVendorName(device.vendorId),
    requiresAdapterDescription: windowsGpuInventory.filter(candidate => candidate.vendorId === device.vendorId).length > 1,
  };
}

function assertGpuSelectionIsAvailable() {
  if (!gpuSelection) return;
  const device = selectedGpuDevice();
  if (!device) throw new Error(`Requested GPU ${gpuSelection.id} is not available. Run --list-gpus to inspect selectable devices.`);
  if (!selectedGpuIsUnambiguous()) {
    throw new Error(`Requested GPU ${gpuSelection.id} is not uniquely identifiable. WebGPU cannot distinguish duplicate adapters with this driver.`);
  }
}

function relaunchArgumentsForGpuDevice(deviceId) {
  const valuesWithArguments = new Set([
    GPU_PREFERENCE_OPTION,
    GPU_DEVICE_OPTION,
    GPU_INDEX_OPTION,
  ]);
  const args = [];
  for (let index = 1; index < process.argv.length; index++) {
    const argument = process.argv[index];
    if (valuesWithArguments.has(argument)) {
      index++;
      continue;
    }
    if (
      argument.startsWith(GPU_PREFERENCE_ARGUMENT)
      || argument.startsWith(GPU_DEVICE_ARGUMENT)
      || argument.startsWith(GPU_INDEX_ARGUMENT)
    ) continue;
    args.push(argument);
  }
  if (deviceId) args.push(`${GPU_DEVICE_ARGUMENT}${deviceId}`);
  return args;
}

function restartForGpuDevice(deviceId) {
  app.relaunch({ args: relaunchArgumentsForGpuDevice(deviceId) });
  app.exit(0);
}

const STEP_EXTENSIONS = new Set(['.step', '.stp']);
const DEFAULT_OPTIONS = Object.freeze({
  meshSize: 0.020,
  rotationSpeed: 0.8,
  rotationSpeed2: 0.3,
  volumeFraction: 0.20,
  reposeAngle: 35,
  depositionRate: 0.0005,
  terminationCondition: 'particles',
  terminationCycles: 1,
});

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizedNumber(value, fallback, min, max) {
  const number = Number(value);
  return Number.isFinite(number) ? clamp(number, min, max) : fallback;
}

function normalizeOptions(options = {}) {
  const terminationCondition = options.terminationCondition === 'cycles' ? 'cycles' : 'particles';
  return {
    meshSize: normalizedNumber(options.meshSize, DEFAULT_OPTIONS.meshSize, 0.002, 0.03),
    rotationSpeed: normalizedNumber(options.rotationSpeed, DEFAULT_OPTIONS.rotationSpeed, 0.1, 5),
    rotationSpeed2: normalizedNumber(options.rotationSpeed2, DEFAULT_OPTIONS.rotationSpeed2, 0, 3),
    volumeFraction: normalizedNumber(options.volumeFraction, DEFAULT_OPTIONS.volumeFraction, 0.05, 0.95),
    reposeAngle: normalizedNumber(options.reposeAngle, DEFAULT_OPTIONS.reposeAngle, 15, 50),
    depositionRate: normalizedNumber(options.depositionRate, DEFAULT_OPTIONS.depositionRate, 0.00001, 0.02),
    terminationCondition,
    terminationCycles: Number.isFinite(Number(options.terminationCycles))
      ? Math.max(1, Math.floor(Number(options.terminationCycles)))
      : DEFAULT_OPTIONS.terminationCycles,
    maxSteps: options.maxSteps == null
      ? null
      : (Number.isFinite(Number(options.maxSteps)) ? Math.max(1, Math.floor(Number(options.maxSteps))) : null),
  };
}

function normalizeParallelism(value) {
  if (value === null || value === undefined || value === '') return null;
  const parallelism = Number(value);
  if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 16) {
    throw new Error('Parallelism must be an integer from 1 to 16.');
  }
  return parallelism;
}

function isStepFile(filePath) {
  return STEP_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function resultStem(filePath) {
  return path.basename(filePath, path.extname(filePath)).replace(/[^a-zA-Z0-9._-]+/g, '_') || 'model';
}

function safePathSegment(value, fallback = 'model') {
  const normalized = String(value || '')
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^\.+/, '')
    .replace(/[. ]+$/, '');
  return normalized || fallback;
}

async function writeTextAtomically(filePath, content) {
  const temporaryPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fsp.writeFile(temporaryPath, content, 'utf8');
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      await fsp.rename(temporaryPath, filePath);
      return;
    } catch (error) {
      const retryable = error?.code === 'EPERM' || error?.code === 'EBUSY';
      if (!retryable || attempt === 5) {
        try { await fsp.unlink(temporaryPath); } catch (_) {}
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 50 * (attempt + 1)));
    }
  }
}

class SystemLoadMonitor {
  constructor(onChange) {
    this.onChange = onChange;
    this.previousCpu = null;
    this.cpuPercent = null;
    this.gpuPercent = null;
    this.timer = null;
    this.gpuSampling = false;
    this.lastGpuSample = 0;
  }

  snapshot() {
    return {
      cpuPercent: this.cpuPercent,
      gpuPercent: this.gpuPercent,
      logicalCores: os.availableParallelism?.() || os.cpus().length || 1,
    };
  }

  start() {
    if (this.timer) return;
    this.sample();
    this.timer = setInterval(() => this.sample(), 2000);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  async sample() {
    this.cpuPercent = this._sampleCpu();
    if (process.platform === 'win32' && !this.gpuSampling && Date.now() - this.lastGpuSample >= 4000) {
      this.gpuSampling = true;
      this.lastGpuSample = Date.now();
      this._sampleWindowsGpu().then(value => {
        this.gpuPercent = value;
      }).catch(() => {
        this.gpuPercent = null;
      }).finally(() => {
        this.gpuSampling = false;
        this.onChange?.();
      });
    }
    this.onChange?.();
  }

  _sampleCpu() {
    const current = os.cpus().reduce((total, cpu) => {
      total.idle += cpu.times.idle;
      total.total += Object.values(cpu.times).reduce((sum, time) => sum + time, 0);
      return total;
    }, { idle: 0, total: 0 });
    const previous = this.previousCpu;
    this.previousCpu = current;
    if (!previous || current.total <= previous.total) return null;
    const idle = current.idle - previous.idle;
    const total = current.total - previous.total;
    return clamp((1 - idle / total) * 100, 0, 100);
  }

  _sampleWindowsGpu() {
    const command = "$ErrorActionPreference='Stop'; $samples=(Get-Counter -Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples | ForEach-Object {$_.CookedValue}; if ($samples) {[Math]::Min(100,($samples | Measure-Object -Sum).Sum).ToString('F2',[System.Globalization.CultureInfo]::InvariantCulture)}";
    return new Promise((resolve, reject) => {
      execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], {
        windowsHide: true,
        timeout: 3500,
      }, (error, stdout) => {
        if (error) return reject(error);
        const match = String(stdout).match(/\d+(?:\.\d+)?/);
        if (!match) return resolve(null);
        resolve(clamp(Number.parseFloat(match[0]), 0, 100));
      });
    });
  }
}

class BatchSimulationManager {
  constructor() {
    this.jobs = [];
    this.active = new Map();
    this.activeAffinity = new Map();
    this.jobByWebContents = new Map();
    this.paused = false;
    this.outputDirectory = null;
    this.options = DEFAULT_OPTIONS;
    this.lastParallelism = 1;
    this.parallelismOverride = null;
    this.onUpdate = null;
    this.onFinished = null;
    this.onJobStateChange = null;
    this.onGpuAdapter = null;
    this.stopRequested = false;
    this.monitor = new SystemLoadMonitor(() => this._drain());
  }

  hasWork() {
    return this.active.size > 0 || this.jobs.some(job => job.status === 'pending' || job.status === 'paused' || job.status === 'running');
  }

  async start(filePaths, options, outputDirectory, parallelismOverride = null) {
    if (!Array.isArray(filePaths) || filePaths.length === 0) throw new Error('At least one STEP/STP input is required.');
    return this.startCustomJobs(filePaths.map(inputPath => ({ inputPath })), options, outputDirectory, parallelismOverride);
  }

  async startCustomJobs(jobDefinitions, defaultOptions, outputDirectory, parallelismOverride = null) {
    if (this.hasWork()) throw new Error('A batch simulation is already running.');
    if (!Array.isArray(jobDefinitions) || jobDefinitions.length === 0) {
      throw new Error('At least one simulation job is required.');
    }

    this.options = normalizeOptions(defaultOptions);
    this.parallelismOverride = normalizeParallelism(parallelismOverride);
    this.outputDirectory = outputDirectory
      ? path.resolve(outputDirectory)
      : path.join(app.getPath('documents'), 'Rotomolding Simulation Results', timestamp());
    await fsp.mkdir(this.outputDirectory, { recursive: true });
    const usedNames = new Map();
    const verifiedJobs = [];
    for (const definition of jobDefinitions) {
      const inputPath = path.resolve(String(definition.inputPath || ''));
      if (!isStepFile(inputPath)) throw new Error(`Not a STEP/STP file: ${inputPath}`);
      const stats = await fsp.stat(inputPath);
      if (!stats.isFile()) throw new Error(`Not a file: ${inputPath}`);
      verifiedJobs.push({ ...definition, inputPath });
    }
    this.jobs = verifiedJobs.map((definition, index) => {
      const jobOutputDirectory = definition.outputDirectory
        ? path.resolve(definition.outputDirectory)
        : this.outputDirectory;
      const requestedStem = definition.outputStem ? safePathSegment(definition.outputStem) : resultStem(definition.inputPath);
      const nameKey = `${jobOutputDirectory}\u0000${requestedStem}`;
      const duplicate = usedNames.get(nameKey) || 0;
      usedNames.set(nameKey, duplicate + 1);
      const outputStem = duplicate === 0 ? requestedStem : `${requestedStem}_${duplicate + 1}`;
      const metadata = definition.metadata || null;
      const affinityKey = String(
        definition.affinityKey
          || (metadata?.dataset && metadata?.relativePath
            ? `${metadata.dataset}:${metadata.relativePath}`
            : definition.inputPath),
      );
      return {
        id: definition.id || `${Date.now()}-${index}`,
        inputPath: definition.inputPath,
        name: definition.name || path.basename(definition.inputPath),
        outputDirectory: jobOutputDirectory,
        outputStem,
        objFileName: definition.objFileName || `${outputStem}_thickness.obj`,
        writeSummary: definition.writeSummary !== false,
        options: normalizeOptions({ ...this.options, ...(definition.options || {}) }),
        metadata,
        affinityKey,
        status: 'pending',
        stage: 'Waiting for scheduling',
        progress: 0,
        error: null,
        startedAt: null,
        completedAt: null,
        output: null,
      };
    });
    this.paused = false;
    this.stopRequested = false;
    this.activeAffinity.clear();
    this.monitor.start();
    this._notifyJobStateChange();
    this._emit();
    this._drain();
    return this.snapshot();
  }

  pause() {
    if (!this.hasWork()) return this.snapshot();
    this.paused = true;
    for (const { job, window } of this.active.values()) {
      job.status = 'paused';
      window.webContents.send('batch:worker-control', { paused: true });
    }
    this._emit();
    return this.snapshot();
  }

  resume() {
    this.paused = false;
    for (const { job, window } of this.active.values()) {
      job.status = 'running';
      window.webContents.send('batch:worker-control', { paused: false });
    }
    this._emit();
    this._drain();
    return this.snapshot();
  }

  requestStop() {
    this.stopRequested = true;
    for (const job of this.jobs) {
      if (job.status === 'pending') {
        job.status = 'paused';
        job.stage = 'Stopped by user; resumable';
      }
    }
    this._notifyJobStateChange();
    this._emit();
    return this.snapshot();
  }

  snapshot() {
    const counts = { pending: 0, running: 0, paused: 0, completed: 0, failed: 0 };
    for (const job of this.jobs) counts[job.status] = (counts[job.status] || 0) + 1;
    return {
      paused: this.paused,
      stopRequested: this.stopRequested,
      total: this.jobs.length,
      active: this.active.size,
      parallelism: this.lastParallelism,
      parallelismOverride: this.parallelismOverride,
      counts,
      system: this.monitor.snapshot(),
      outputDirectory: this.outputDirectory,
      jobs: this.jobs.map(job => ({
        id: job.id,
        name: job.name,
        status: job.status,
        stage: job.stage,
        progress: job.progress,
        error: job.error,
        output: job.output,
        gpuAdapter: job.gpuAdapter || null,
        metadata: job.metadata,
      })),
    };
  }

  getJobForWebContents(webContentsId) {
    return this.jobByWebContents.get(webContentsId) || null;
  }

  updateProgress(webContentsId, payload = {}) {
    const job = this.getJobForWebContents(webContentsId);
    if (!job || job.status === 'completed' || job.status === 'failed') return;
    job.stage = String(payload.stage || job.stage);
    if (Number.isFinite(payload.progress)) job.progress = clamp(payload.progress, 0, 1);
    job.simTime = Number.isFinite(payload.simTime) ? payload.simTime : job.simTime;
    job.remainingRatio = Number.isFinite(payload.remainingRatio) ? payload.remainingRatio : job.remainingRatio;
    if (typeof payload.gpuAdapter === 'string' && payload.gpuAdapter && job.gpuAdapter !== payload.gpuAdapter) {
      job.gpuAdapter = payload.gpuAdapter;
      this.onGpuAdapter?.(job);
    }
    this._emit();
  }

  async writeResult(webContentsId, payload = {}) {
    const job = this.getJobForWebContents(webContentsId);
    if (!job) throw new Error('批处理作业不存在');
    if (typeof payload.obj !== 'string' || payload.obj.length === 0) throw new Error('批处理作业未提供 OBJ 结果');
    const outputDirectory = job.outputDirectory || this.outputDirectory;
    await fsp.mkdir(outputDirectory, { recursive: true });
    const objPath = path.join(outputDirectory, job.objFileName || `${job.outputStem}_thickness.obj`);
    const summaryPath = path.join(outputDirectory, `${job.outputStem}_summary.json`);
    const summary = {
      input: job.inputPath,
      output: objPath,
      options: job.options,
      ...payload.summary,
    };
    const writes = [writeTextAtomically(objPath, payload.obj)];
    if (job.writeSummary) writes.push(writeTextAtomically(summaryPath, `${JSON.stringify(summary, null, 2)}\n`));
    await Promise.all(writes);
    return { objPath, summaryPath: job.writeSummary ? summaryPath : null, statePath: null, summary };
  }

  complete(webContentsId, payload = {}) {
    const job = this.getJobForWebContents(webContentsId);
    if (!job || job.status === 'completed' || job.status === 'failed') return;
    const entry = this.active.get(job.id);
    this.active.delete(job.id);
    if (entry?.affinityKey && this.activeAffinity.get(entry.affinityKey) === job.id) {
      this.activeAffinity.delete(entry.affinityKey);
    }
    this.jobByWebContents.delete(webContentsId);
    if (payload.status === 'completed') {
      job.status = 'completed';
      job.stage = '仿真完成';
      job.progress = 1;
      job.output = payload.output || null;
    } else {
      job.status = 'failed';
      job.stage = '仿真失败';
      job.error = String(payload.error || '未知错误');
    }
    job.completedAt = Date.now();
    this._notifyJobStateChange();
    const canReuseWorker = payload.status === 'completed'
      && !this.paused
      && !this.stopRequested
      && entry?.window
      && !entry.window.isDestroyed();
    if (canReuseWorker) {
      const nextJob = this._nextJobForAffinity(entry.affinityKey) || this._nextJobForLaunch();
      if (nextJob) {
        nextJob.status = 'running';
        nextJob.stage = 'Reusing cached WebGPU worker';
        nextJob.progress = 0;
        nextJob.startedAt = Date.now();
        this.active.set(nextJob.id, { job: nextJob, window: entry.window, affinityKey: nextJob.affinityKey });
        this.activeAffinity.set(nextJob.affinityKey, nextJob.id);
        this.jobByWebContents.set(webContentsId, nextJob);
      } else {
        entry.window.destroy();
      }
    } else if (entry?.window && !entry.window.isDestroyed()) {
      entry.window.destroy();
    }
    this._notifyJobStateChange();
    this._emit();
    this._drain();
  }

  _parallelism() {
    if (this.parallelismOverride !== null) return this.parallelismOverride;
    const system = this.monitor.snapshot();
    const workerCeiling = Math.min(8, Math.max(1, Math.floor(system.logicalCores / 2)));
    const levels = [4, 6].filter(level => level <= workerCeiling);
    if (levels.length === 0) return workerCeiling;
    if (system.cpuPercent == null) return levels[0];
    if (system.cpuPercent >= 90 || (system.gpuPercent != null && system.gpuPercent >= 92)) return 0;
    const cpuHeadroom = clamp((90 - system.cpuPercent) / 40, 0, 1);
    const gpuHeadroom = system.gpuPercent == null
      ? 1
      : clamp((88 - system.gpuPercent) / 38, 0, 1);
    const headroom = Math.min(cpuHeadroom, gpuHeadroom);
    const levelIndex = headroom < 0.25 ? 0 : headroom < 0.5 ? 1 : headroom < 0.75 ? 2 : 3;
    return levels[Math.min(levelIndex, levels.length - 1)];
  }

  _drain() {
    this.lastParallelism = this._parallelism();
    if (!this.paused && !this.stopRequested) {
      while (this.active.size < this.lastParallelism) {
        const job = this._nextJobForLaunch();
        if (!job) break;
        this._launch(job);
      }
    }
    this._emit();
    if (!this.hasWork() || (this.stopRequested && this.active.size === 0)) {
      this.monitor.stop();
      this.onFinished?.(this.snapshot());
    }
  }

  _launch(job) {
    job.status = 'running';
    job.stage = '启动隐藏仿真器';
    job.startedAt = Date.now();
    this._notifyJobStateChange();
    const window = new BrowserWindow({
      show: false,
      width: 2,
      height: 2,
      webPreferences: {
        preload: path.join(__dirname, 'preload.cjs'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false,
        backgroundThrottling: false,
      },
    });
    this.active.set(job.id, { job, window, affinityKey: job.affinityKey });
    this.activeAffinity.set(job.affinityKey, job.id);
    this.jobByWebContents.set(window.webContents.id, job);
    window.webContents.on('render-process-gone', (_event, details) => {
      this.complete(window.webContents.id, { status: 'failed', error: `批处理渲染进程已退出: ${details.reason}` });
    });
    window.loadFile(path.join(app.getAppPath(), 'dist', 'batch-worker.html')).catch(error => {
      this.complete(window.webContents.id, { status: 'failed', error: `无法加载批处理仿真器: ${error.message}` });
    });
  }

  _nextJobForAffinity(affinityKey) {
    if (!affinityKey || this.activeAffinity.has(affinityKey)) return null;
    return this.jobs.find(candidate => candidate.status === 'pending' && candidate.affinityKey === affinityKey) || null;
  }

  _nextJobForLaunch() {
    return this.jobs.find(candidate => candidate.status === 'pending' && !this.activeAffinity.has(candidate.affinityKey)) || null;
  }

  _emit() {
    const snapshot = this.snapshot();
    this.onUpdate?.(snapshot);
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send('batch:status-update', snapshot);
    }
  }

  _notifyJobStateChange() {
    try {
      this.onJobStateChange?.(this.jobs);
    } catch (error) {
      console.error('Unable to persist batch job state:', error);
    }
  }
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

const manager = new BatchSimulationManager();
const sharedMeshCache = new Map();
const sharedCavityCache = new Map();
const pendingMeshCache = new Set();
const pendingCavityCache = new Set();
const SHARED_CACHE_MAX_ENTRIES = 12;

function cacheAcquire(cache, pending, key) {
  const entry = cache.get(String(key));
  if (entry) {
    entry.lastUsed = Date.now();
    return { status: 'hit', value: entry.value };
  }
  const normalizedKey = String(key);
  if (pending.has(normalizedKey)) return { status: 'wait' };
  pending.add(normalizedKey);
  return { status: 'owner' };
}

function cacheSet(cache, pending, key, value) {
  const normalizedKey = String(key);
  cache.set(normalizedKey, { value, lastUsed: Date.now() });
  pending.delete(normalizedKey);
  while (cache.size > SHARED_CACHE_MAX_ENTRIES) {
    const oldest = [...cache.entries()].sort((a, b) => a[1].lastUsed - b[1].lastUsed)[0];
    if (!oldest) break;
    cache.delete(oldest[0]);
  }
  return true;
}

function cacheRelease(pending, key) {
  pending.delete(String(key));
  return true;
}
let mainWindow = null;
let gpuRestartPending = false;

async function gpuStatus() {
  let info = {};
  try {
    info = await app.getGPUInfo('complete');
  } catch (error) {
    console.warn('Unable to query GPU information:', error);
  }
  const devices = Array.isArray(info.gpuDevice) ? info.gpuDevice : [];
  return {
    preference: gpuPreference,
    selection: gpuSelectionPayload(),
    selectableDevices: windowsGpuInventory.map(device => ({
      index: device.index,
      id: gpuKey(device.vendorId, device.deviceId),
      name: device.name,
      vendorId: device.vendorId,
      deviceId: device.deviceId,
    })),
    devices: devices.map((device, index) => ({
      id: index,
      active: Boolean(device.active),
      name: device.description || device.deviceString || device.vendorString || '',
      vendorId: device.vendorId || '',
      deviceId: device.deviceId || '',
      driverVersion: device.driverVersion || '',
    })),
    renderer: info.auxAttributes?.gl_renderer || '',
  };
}

function registerIpc() {
  ipcMain.handle('gpu:status', () => gpuStatus());
  ipcMain.handle('gpu:apply-device', (_event, deviceId) => {
    if (manager.hasWork()) throw new Error('Cannot change GPU while batch simulation is running');
    const requestedId = String(deviceId || '');
    if (!requestedId) {
      if (!gpuSelection?.id) return { restarting: false };
      gpuRestartPending = true;
      restartForGpuDevice(null);
      return { restarting: true };
    }
    const device = windowsGpuInventory.find(candidate => gpuKey(candidate.vendorId, candidate.deviceId) === requestedId);
    if (!device) throw new Error('Selected GPU is not available');
    if (gpuSelection?.id === requestedId) return { restarting: false };
    gpuRestartPending = true;
    restartForGpuDevice(requestedId);
    return { restarting: true };
  });
  ipcMain.handle('batch:start', (_event, payload = {}) => manager.start(payload.filePaths, payload.options));
  ipcMain.handle('batch:pause', () => manager.pause());
  ipcMain.handle('batch:resume', () => manager.resume());
  ipcMain.handle('batch:status', () => manager.snapshot());
  ipcMain.handle('batch:worker-job', event => {
    const job = manager.getJobForWebContents(event.sender.id);
    if (!job) return null;
    return {
      inputPath: job.inputPath,
      options: job.options,
      outputStem: job.outputStem,
      gpuPreference,
      gpuSelection: gpuSelectionPayload(),
    };
  });
  ipcMain.handle('batch:read-input', async (event, requestedPath) => {
    const job = manager.getJobForWebContents(event.sender.id);
    if (!job || path.resolve(String(requestedPath)) !== job.inputPath) throw new Error('无权读取此输入文件');
    const data = await fsp.readFile(job.inputPath);
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  });
  ipcMain.handle('batch:mesh-cache-acquire', (_event, key) => cacheAcquire(sharedMeshCache, pendingMeshCache, key));
  ipcMain.handle('batch:mesh-cache-get', (_event, key) => sharedMeshCache.get(String(key))?.value || null);
  ipcMain.handle('batch:mesh-cache-set', (_event, key, value) => cacheSet(sharedMeshCache, pendingMeshCache, key, value));
  ipcMain.handle('batch:mesh-cache-release', (_event, key) => cacheRelease(pendingMeshCache, key));
  ipcMain.handle('batch:cavity-cache-acquire', (_event, key) => cacheAcquire(sharedCavityCache, pendingCavityCache, key));
  ipcMain.handle('batch:cavity-cache-get', (_event, key) => sharedCavityCache.get(String(key))?.value || null);
  ipcMain.handle('batch:cavity-cache-set', (_event, key, value) => cacheSet(sharedCavityCache, pendingCavityCache, key, value));
  ipcMain.handle('batch:cavity-cache-release', (_event, key) => cacheRelease(pendingCavityCache, key));
  ipcMain.handle('batch:write-result', (event, payload) => manager.writeResult(event.sender.id, payload));
  ipcMain.on('batch:worker-progress', (event, payload) => manager.updateProgress(event.sender.id, payload));
  ipcMain.on('batch:worker-complete', (event, payload) => manager.complete(event.sender.id, payload));
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 920,
    minHeight: 620,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  mainWindow.loadFile(path.join(app.getAppPath(), 'dist', 'index.html')).catch(error => {
    console.error('Unable to load main window:', error);
  });
  mainWindow.on('closed', () => { mainWindow = null; });
}

function cliHelp() {
  return [
    'Rotomolding CLI',
    'batch: RotomoldingSimulation-cli.exe batch --input <STEP/STP or directory> [--input <...>] --output <directory> [options]',
    'sweep: RotomoldingSimulation-cli.exe sweep --training-input <directory> --validation-input <directory> --output <directory> [--gpu-index <N>]',
    'sweep creates 25 jobs per model from the Cartesian product of 0.2, 0.4, 0.6, 0.8, and 1.0 rad/s on the primary and secondary axes.',
    'sweep resumes automatically from <output>\\sweep-progress.json. Use --retry-failed to retry failed jobs and --prepare-only to create the workspace without starting simulations.',
    'Run --list-gpus to enumerate selectable Windows GPUs. Use --gpu-index <N> or --gpu-device <vendorId:deviceId> to require a specific GPU.',
    'options: --mesh-size 0.020 --rotation-speed 0.8 --rotation-speed2 0.3 --fill 0.20 --repose-angle 35 --deposition-rate 0.0005 --termination-condition particles|cycles --termination-cycles 1 --config <JSON> --gpu-index <N> --gpu-device <vendorId:deviceId> --parallelism <1-16>',
    'Automatic parallelism selects 4 or 6 workers from CPU/GPU headroom; --parallelism overrides it.',
    'Legacy --gpu <default|low-power|high-performance> remains supported but does not select a specific device.',
    'Progress is emitted as JSON Lines.',
  ].join('\n');
}

async function expandCliInputs(inputs) {
  const files = [];
  async function visitDirectory(directory) {
    const children = await fsp.readdir(directory, { withFileTypes: true });
    for (const child of children) {
      const candidate = path.join(directory, child.name);
      if (child.isDirectory()) await visitDirectory(candidate);
      else if (child.isFile() && isStepFile(candidate)) files.push(candidate);
    }
  }
  for (const input of inputs) {
    const resolved = path.resolve(input);
    const stats = await fsp.stat(resolved);
    if (stats.isDirectory()) {
      await visitDirectory(resolved);
    } else if (stats.isFile() && isStepFile(resolved)) {
      files.push(resolved);
    } else {
      throw new Error(`Not a STEP/STP file or directory: ${resolved}`);
    }
  }
  return [...new Set(files.map(filePath => path.resolve(filePath)))].sort((left, right) => left.localeCompare(right));
}

async function parseCliArguments() {
  const args = process.argv.slice(2).filter(argument => argument !== '--cli');
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) return { help: true };
  if (args.includes(GPU_LIST_OPTION)) {
    const statusFile = statusFileFromProcessArguments();
    const permittedArguments = new Set([GPU_LIST_OPTION, '--status-file', statusFile]);
    if (args.some(argument => !permittedArguments.has(argument))) {
      throw new Error('--list-gpus cannot be combined with simulation options.');
    }
    return { listGpus: true, statusFile };
  }
  const command = ['batch', 'simulate', 'sweep'].includes(args[0]) ? args.shift() : 'batch';
  const inputs = [];
  let outputDirectory = null;
  let configPath = null;
  let statusFile = null;
  let trainingInput = null;
  let validationInput = null;
  let prepareOnly = false;
  let retryFailed = false;
  let parallelism = null;
  const options = {};
  const optionNames = {
    '--mesh-size': 'meshSize',
    '--rotation-speed': 'rotationSpeed',
    '--rotation-speed2': 'rotationSpeed2',
    '--fill': 'volumeFraction',
    '--repose-angle': 'reposeAngle',
    '--deposition-rate': 'depositionRate',
    '--termination-condition': 'terminationCondition',
    '--termination-cycles': 'terminationCycles',
    '--max-steps': 'maxSteps',
  };
  const takeValue = (argument, index) => {
    const value = args[index + 1];
    if (value === undefined || value.startsWith('--')) throw new Error(`Option ${argument} requires a value.`);
    return value;
  };
  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === '--input') {
      inputs.push(takeValue(argument, index));
      index++;
    } else if (argument === '--output') {
      outputDirectory = takeValue(argument, index);
      index++;
    } else if (argument === '--config') {
      configPath = takeValue(argument, index);
      index++;
    } else if (argument === '--status-file') {
      statusFile = takeValue(argument, index);
      index++;
    } else if (argument === '--training-input') {
      trainingInput = takeValue(argument, index);
      index++;
    } else if (argument === '--validation-input') {
      validationInput = takeValue(argument, index);
      index++;
    } else if (argument === '--gpu') {
      const preference = takeValue(argument, index);
      if (!GPU_PREFERENCES.has(preference)) throw new Error(`Unsupported GPU preference: ${preference}`);
      index++;
    } else if (argument === GPU_INDEX_OPTION || argument === GPU_DEVICE_OPTION) {
      takeValue(argument, index);
      index++;
    } else if (argument === '--parallelism') {
      parallelism = takeValue(argument, index);
      index++;
    } else if (argument.startsWith(GPU_PREFERENCE_ARGUMENT)) {
      const preference = argument.slice(GPU_PREFERENCE_ARGUMENT.length);
      if (!GPU_PREFERENCES.has(preference)) throw new Error(`Unsupported GPU preference: ${preference}`);
    } else if (argument.startsWith(GPU_INDEX_ARGUMENT) || argument.startsWith(GPU_DEVICE_ARGUMENT)) {
    } else if (argument === '--prepare-only') {
      prepareOnly = true;
    } else if (argument === '--retry-failed') {
      retryFailed = true;
    } else if (optionNames[argument]) {
      options[optionNames[argument]] = takeValue(argument, index);
      index++;
    } else {
      throw new Error(`Unknown option: ${argument}`);
    }
  }
  const configOptions = configPath
    ? JSON.parse(await fsp.readFile(path.resolve(configPath), 'utf8'))
    : {};
  parallelism = normalizeParallelism(parallelism);
  if (command === 'sweep') {
    if (!trainingInput || !validationInput) throw new Error('sweep requires --training-input and --validation-input.');
    if (!outputDirectory) throw new Error('sweep requires --output.');
    return {
      command,
      trainingInput: path.resolve(trainingInput),
      validationInput: path.resolve(validationInput),
      outputDirectory: path.resolve(outputDirectory),
      options: { ...configOptions, ...options },
      statusFile: statusFile ? path.resolve(statusFile) : null,
      prepareOnly,
      retryFailed,
      parallelism,
    };
  }
  if (inputs.length === 0) throw new Error('At least one --input is required.');
  return {
    command,
    inputs: await expandCliInputs(inputs),
    outputDirectory,
    options: { ...configOptions, ...options },
    statusFile: statusFile ? path.resolve(statusFile) : null,
    parallelism,
  };
}

function emitCliEvent(payload, statusFile) {
  const line = `${JSON.stringify(payload)}\n`;
  try { process.stdout.write(line); } catch (_) {}
  if (statusFile) fs.appendFileSync(statusFile, line, 'utf8');
}

function statusFileFromProcessArguments() {
  const index = process.argv.indexOf('--status-file');
  return index >= 0 && process.argv[index + 1]
    ? path.resolve(process.argv[index + 1])
    : null;
}

function queryNvidiaGpus() {
  return new Promise(resolve => {
    execFile('nvidia-smi', ['--query-gpu=name,uuid', '--format=csv,noheader'], {
      windowsHide: true,
      timeout: 3500,
    }, (error, stdout) => {
      if (error) return resolve([]);
      const devices = String(stdout).split(/\r?\n/)
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => {
          const separator = line.lastIndexOf(',');
          return separator >= 0
            ? { name: line.slice(0, separator).trim(), uuid: line.slice(separator + 1).trim() }
            : { name: line, uuid: '' };
        });
      resolve(devices);
    });
  });
}

async function emitGpuSystemEvent(statusFile) {
  const status = await gpuStatus();
  const nvidiaDevices = await queryNvidiaGpus();
  emitCliEvent({
    type: 'gpu-system',
    preference: gpuPreference,
    selection: status.selection,
    selectableDevices: status.selectableDevices,
    renderer: status.renderer,
    devices: status.devices.map(device => ({ name: device.name, active: device.active })),
    nvidiaDevices,
  }, statusFile);
}

const SWEEP_SPEEDS = Object.freeze([0.2, 0.4, 0.6, 0.8, 1.0]);
const SWEEP_PROGRESS_FILE = 'sweep-progress.json';
const SWEEP_PARAMETERS_FILE = 'simulations.json';

function compactSnapshot(snapshot) {
  const { jobs, ...compact } = snapshot;
  return compact;
}

function sweepCounts(jobs) {
  const counts = { pending: 0, running: 0, paused: 0, completed: 0, failed: 0 };
  for (const job of jobs) counts[job.status] = (counts[job.status] || 0) + 1;
  return counts;
}

function sweepCombinations() {
  const combinations = [];
  let number = 1;
  for (const primaryAxisSpeed of SWEEP_SPEEDS) {
    for (const secondaryAxisSpeed of SWEEP_SPEEDS) {
      combinations.push({ number: number++, primaryAxisSpeed, secondaryAxisSpeed });
    }
  }
  return combinations;
}

async function copyFileAtomically(sourcePath, destinationPath) {
  try {
    const stats = await fsp.stat(destinationPath);
    if (stats.isFile() && stats.size > 0) return;
  } catch (_) {
  }
  const temporaryPath = `${destinationPath}.${process.pid}.${Date.now()}.tmp`;
  await fsp.copyFile(sourcePath, temporaryPath);
  await fsp.rename(temporaryPath, destinationPath);
}

async function outputFileIsComplete(filePath) {
  try {
    const stats = await fsp.stat(filePath);
    return stats.isFile() && stats.size > 0;
  } catch (_) {
    return false;
  }
}

function isInterruptedRendererFailure(job) {
  const error = String(job.error || '').toLowerCase();
  return error.includes('render process') || error.includes('renderer') || error.includes('crashed');
}

async function writeJsonAtomically(filePath, value) {
  await writeTextAtomically(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function modelOutputDirectory(outputDirectory, datasetName, inputRoot, inputPath) {
  const relativePath = path.relative(inputRoot, inputPath);
  const relativeDirectory = path.dirname(relativePath);
  const modelName = safePathSegment(resultStem(inputPath));
  return relativeDirectory === '.'
    ? path.join(outputDirectory, datasetName, modelName)
    : path.join(outputDirectory, datasetName, relativeDirectory, modelName);
}

function jobResultOutputDirectory(datasetName, relativePath) {
  const resultRoot = String(process.env.ROTOMOLDING_RESULTS_ROOT || '').trim();
  if (!resultRoot) return null;
  const relativeDirectory = path.dirname(relativePath);
  const modelName = safePathSegment(resultStem(relativePath));
  return path.join(resultRoot, datasetName, relativeDirectory, modelName, 'simulation_results');
}

function sweepParameters(datasetName, relativePath, modelFileName, options) {
  return {
    schemaVersion: 1,
    dataset: datasetName,
    sourceModel: relativePath,
    modelFile: modelFileName,
    defaults: {
      meshSize: options.meshSize,
      volumeFraction: options.volumeFraction,
      reposeAngle: options.reposeAngle,
      depositionRate: options.depositionRate,
      terminationCondition: options.terminationCondition,
      terminationCycles: options.terminationCycles,
    },
    simulations: sweepCombinations().map(({ number, primaryAxisSpeed, secondaryAxisSpeed }) => ({
      simulation: number,
      primaryAxisRotationSpeed: primaryAxisSpeed,
      secondaryAxisRotationSpeed: secondaryAxisSpeed,
      result: `simulation_${String(number).padStart(2, '0')}.obj`,
    })),
  };
}

async function createSweepProgress(parsed) {
  const progressPath = path.join(parsed.outputDirectory, SWEEP_PROGRESS_FILE);
  try {
    const existing = JSON.parse(await fsp.readFile(progressPath, 'utf8'));
    if (existing.schemaVersion !== 1 || !Array.isArray(existing.jobs)) throw new Error('Unsupported sweep-progress.json format.');
    if (JSON.stringify(existing.speedValues || []) !== JSON.stringify(SWEEP_SPEEDS)) {
      throw new Error('Existing sweep-progress.json uses a different speed grid. Use a new output directory for the 25-combination sweep.');
    }
    if (path.resolve(existing.trainingInput) !== parsed.trainingInput || path.resolve(existing.validationInput) !== parsed.validationInput) {
      throw new Error('The existing sweep-progress.json was created for different input datasets.');
    }
    for (const job of existing.jobs) {
      const resultOutputDirectory = jobResultOutputDirectory(job.dataset, job.relativePath);
      if (resultOutputDirectory) {
        job.outputDirectory = resultOutputDirectory;
        job.objPath = path.join(resultOutputDirectory, job.objFileName);
      }
      if (await outputFileIsComplete(job.objPath)) {
        job.status = 'completed';
        job.progress = 1;
        job.stage = 'Completed';
      } else if (
        job.status === 'running'
        || job.status === 'paused'
        || (job.status === 'failed' && (parsed.retryFailed || isInterruptedRendererFailure(job)))
      ) {
        job.status = 'pending';
        job.progress = 0;
        job.stage = 'Resuming';
        job.error = null;
      }
    }
    existing.updatedAt = new Date().toISOString();
    existing.gpuPreference = gpuPreference;
    existing.gpuSelection = gpuSelectionPayload();
    existing.counts = sweepCounts(existing.jobs);
    await writeJsonAtomically(progressPath, existing);
    return { progressPath, progress: existing, resumed: true };
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const baseOptions = normalizeOptions(parsed.options);
  const datasets = [
    { name: 'training_set', inputDirectory: parsed.trainingInput },
    { name: 'validation_set', inputDirectory: parsed.validationInput },
  ];
  const jobs = [];
  await fsp.mkdir(parsed.outputDirectory, { recursive: true });
  for (const dataset of datasets) {
    const stats = await fsp.stat(dataset.inputDirectory);
    if (!stats.isDirectory()) throw new Error(`Dataset input is not a directory: ${dataset.inputDirectory}`);
    const inputPaths = await expandCliInputs([dataset.inputDirectory]);
    if (inputPaths.length === 0) throw new Error(`No STEP/STP files were found in ${dataset.inputDirectory}`);
    dataset.modelCount = inputPaths.length;
    for (const inputPath of inputPaths) {
      const relativePath = path.relative(dataset.inputDirectory, inputPath);
      const modelDirectory = modelOutputDirectory(parsed.outputDirectory, dataset.name, dataset.inputDirectory, inputPath);
      const outputDirectory = path.join(modelDirectory, 'results');
      const resultOutputDirectory = jobResultOutputDirectory(dataset.name, relativePath) || outputDirectory;
      const modelFileName = path.basename(inputPath);
      await fsp.mkdir(outputDirectory, { recursive: true });
      await copyFileAtomically(inputPath, path.join(modelDirectory, modelFileName));
      await writeJsonAtomically(
        path.join(modelDirectory, SWEEP_PARAMETERS_FILE),
        sweepParameters(dataset.name, relativePath, modelFileName, baseOptions),
      );
      for (const { number, primaryAxisSpeed, secondaryAxisSpeed } of sweepCombinations()) {
        const resultFileName = `simulation_${String(number).padStart(2, '0')}.obj`;
        jobs.push({
          id: `${dataset.name}:${relativePath}:${number}`,
          dataset: dataset.name,
          relativePath,
          inputPath,
          outputDirectory: resultOutputDirectory,
          objPath: path.join(resultOutputDirectory, resultFileName),
          objFileName: resultFileName,
          simulation: number,
          affinityKey: `${dataset.name}:${relativePath}`,
          options: {
            ...baseOptions,
            rotationSpeed: primaryAxisSpeed,
            rotationSpeed2: secondaryAxisSpeed,
          },
          status: 'pending',
          stage: 'Waiting for scheduling',
          progress: 0,
          error: null,
          gpuAdapter: null,
          startedAt: null,
          completedAt: null,
        });
      }
    }
  }
  const progress = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputDirectory: parsed.outputDirectory,
    trainingInput: parsed.trainingInput,
    validationInput: parsed.validationInput,
    gpuPreference,
    gpuSelection: gpuSelectionPayload(),
    speedValues: SWEEP_SPEEDS,
    defaults: baseOptions,
    datasets,
    jobs,
    counts: sweepCounts(jobs),
  };
  await writeJsonAtomically(progressPath, progress);
  return { progressPath, progress, resumed: false };
}

function applyManagerJobState(progressJob, managerJob) {
  progressJob.status = managerJob.status;
  progressJob.stage = managerJob.stage;
  progressJob.progress = managerJob.progress;
  progressJob.error = managerJob.error || null;
  progressJob.gpuAdapter = managerJob.gpuAdapter || progressJob.gpuAdapter || null;
  progressJob.startedAt = managerJob.startedAt || progressJob.startedAt || null;
  progressJob.completedAt = managerJob.completedAt || progressJob.completedAt || null;
}

async function startSweep(parsed) {
  const { progressPath, progress, resumed } = await createSweepProgress(parsed);
  const stopPath = path.join(parsed.outputDirectory, 'STOP');
  try { await fsp.unlink(stopPath); } catch (_) {}
  const stopPoll = setInterval(() => {
    fsp.access(stopPath).then(() => manager.requestStop()).catch(() => {});
  }, 500);
  progress.parallelism = parsed.parallelism;
  const jobsById = new Map(progress.jobs.map(job => [job.id, job]));
  const queuedJobs = progress.jobs.filter(job => job.status === 'pending');
  const knownAdapters = new Set();
  let persistTimer = null;
  let persistChain = Promise.resolve();
  const persist = () => {
    progress.updatedAt = new Date().toISOString();
    progress.counts = sweepCounts(progress.jobs);
    persistChain = persistChain.catch(() => undefined).then(() => writeJsonAtomically(progressPath, progress));
    return persistChain;
  };
  const schedulePersist = () => {
    if (persistTimer) return;
    persistTimer = setTimeout(() => {
      persistTimer = null;
      persist();
    }, 500);
  };
  const flushProgress = async () => {
    if (persistTimer) {
      clearTimeout(persistTimer);
      persistTimer = null;
    }
    await persist();
  };

  emitCliEvent({
    type: 'sweep-prepared',
    resumed,
    gpuPreference,
    gpuSelection: gpuSelectionPayload(),
    outputDirectory: parsed.outputDirectory,
    models: progress.datasets.reduce((total, dataset) => total + dataset.modelCount, 0),
    jobs: progress.jobs.length,
    counts: progress.counts,
  }, parsed.statusFile);

  if (parsed.prepareOnly || queuedJobs.length === 0) {
    await flushProgress();
    emitCliEvent({
      type: parsed.prepareOnly ? 'sweep-prepared-only' : 'sweep-complete',
      outputDirectory: parsed.outputDirectory,
      counts: progress.counts,
    }, parsed.statusFile);
    app.exit(progress.counts.failed > 0 ? 1 : 0);
    return;
  }

  manager.onJobStateChange = managerJobs => {
    for (const managerJob of managerJobs) {
      const progressJob = jobsById.get(managerJob.id);
      if (progressJob) applyManagerJobState(progressJob, managerJob);
    }
    schedulePersist();
  };
  manager.onGpuAdapter = managerJob => {
    const progressJob = jobsById.get(managerJob.id);
    if (progressJob) progressJob.gpuAdapter = managerJob.gpuAdapter;
    schedulePersist();
    if (!knownAdapters.has(managerJob.gpuAdapter)) {
      knownAdapters.add(managerJob.gpuAdapter);
      emitCliEvent({
        type: 'gpu-adapter',
        preference: gpuPreference,
        selection: gpuSelectionPayload(),
        adapter: managerJob.gpuAdapter,
      }, parsed.statusFile);
    }
  };
  manager.onUpdate = snapshot => emitCliEvent({
    type: 'sweep-status',
    ...compactSnapshot(snapshot),
    overall: sweepCounts(progress.jobs),
  }, parsed.statusFile);
  manager.onFinished = snapshot => {
    void (async () => {
      clearInterval(stopPoll);
      await flushProgress();
      const counts = sweepCounts(progress.jobs);
      emitCliEvent({ type: 'sweep-complete', ...compactSnapshot(snapshot), overall: counts }, parsed.statusFile);
      app.exit(counts.failed > 0 ? 1 : 0);
    })();
  };

  const definitions = queuedJobs.map(job => ({
    id: job.id,
    inputPath: job.inputPath,
    name: `${job.dataset}/${job.relativePath} #${job.simulation}`,
    outputDirectory: job.outputDirectory,
    outputStem: path.basename(job.objFileName, '.obj'),
    objFileName: job.objFileName,
    writeSummary: false,
    options: job.options,
    metadata: { dataset: job.dataset, relativePath: job.relativePath, simulation: job.simulation },
  }));
  await manager.startCustomJobs(definitions, progress.defaults, parsed.outputDirectory, parsed.parallelism);
}

async function startCli() {
  let statusFile = statusFileFromProcessArguments();
  try {
    const parsed = await parseCliArguments();
    statusFile = parsed.statusFile;
    if (parsed.help) {
      process.stdout.write(`${cliHelp()}\n`);
      app.exit(0);
      return;
    }
    if (parsed.listGpus) {
      emitCliEvent({ type: 'gpu-list', devices: windowsGpuInventory.map(device => ({
        index: device.index,
        id: gpuKey(device.vendorId, device.deviceId),
        name: device.name,
        vendorId: device.vendorId,
        deviceId: device.deviceId,
      })) }, parsed.statusFile);
      app.exit(0);
      return;
    }
    assertGpuSelectionIsAvailable();
    if (parsed.statusFile) {
      await fsp.mkdir(path.dirname(parsed.statusFile), { recursive: true });
      await fsp.writeFile(parsed.statusFile, '', 'utf8');
    }
    await emitGpuSystemEvent(parsed.statusFile);
    if (parsed.command === 'sweep') {
      await startSweep(parsed);
      return;
    }
    if (parsed.inputs.length === 0) throw new Error('No STEP/STP files were found in the input path.');
    manager.onGpuAdapter = job => emitCliEvent({
      type: 'gpu-adapter',
      preference: gpuPreference,
      selection: gpuSelectionPayload(),
      adapter: job.gpuAdapter,
    }, parsed.statusFile);
    manager.onUpdate = snapshot => emitCliEvent({ type: 'batch-status', ...compactSnapshot(snapshot) }, parsed.statusFile);
    manager.onFinished = snapshot => {
      emitCliEvent({ type: 'batch-complete', ...compactSnapshot(snapshot) }, parsed.statusFile);
      app.exit(snapshot.counts.failed > 0 ? 1 : 0);
    };
    await manager.start(parsed.inputs, parsed.options, parsed.outputDirectory, parsed.parallelism);
  } catch (error) {
    emitCliEvent({ type: 'error', message: error.message || String(error) }, statusFile);
    app.exit(2);
  }
}

const cliMode = process.argv.includes('--cli');
app.whenReady().then(() => {
  registerIpc();
  if (cliMode) return startCli();
  createMainWindow();
});

app.on('window-all-closed', () => {
  if (!cliMode && !manager.hasWork() && !gpuRestartPending) app.quit();
});
