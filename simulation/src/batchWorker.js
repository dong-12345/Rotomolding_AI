import { Simulation } from './simulation.js';
import { Renderer } from './renderer.js';
import { initOpenCascade, loadSTEPFile } from './stepLoader.js';

const WALL_THICKNESS_DELAY_SECONDS = 5;
const THICKNESS_UPDATE_INTERVAL = 6;
const FAST_BATCH_STEPS = 30;

const desktop = window.rotomoldingDesktop;
let paused = false;
let resumeWaiters = [];

desktop?.worker.onControl(({ paused: nextPaused }) => {
  paused = Boolean(nextPaused);
  if (!paused) {
    for (const resume of resumeWaiters) resume();
    resumeWaiters = [];
  }
});

function waitForResume() {
  if (!paused) return Promise.resolve();
  return new Promise(resolve => resumeWaiters.push(resolve));
}

function report(stage, progress, extra = {}) {
  desktop?.worker.reportProgress({ stage, progress, ...extra });
}

function normalizedGpuText(value) {
  return String(value || '').trim().toLowerCase();
}

function verifySelectedGpu(selection, adapterInfo) {
  if (!selection) return;
  const expectedVendor = normalizedGpuText(selection.vendor);
  const adapterVendor = normalizedGpuText(adapterInfo.vendor);
  const adapterDescription = normalizedGpuText(adapterInfo.description);
  if (expectedVendor && adapterVendor && !adapterVendor.includes(expectedVendor)) {
    throw new Error(`Selected GPU ${selection.name} (${selection.id}) was not assigned. WebGPU selected ${adapterInfo.vendor || 'an unknown adapter'} instead.`);
  }
  if (selection.requiresAdapterDescription && !adapterDescription) {
    throw new Error(`WebGPU did not expose an adapter description, so GPU ${selection.name} (${selection.id}) cannot be distinguished from another ${selection.vendor} GPU.`);
  }
  if (adapterDescription && selection.name && !adapterDescription.includes(normalizedGpuText(selection.name))) {
    throw new Error(`Selected GPU ${selection.name} (${selection.id}) was not assigned. WebGPU selected ${adapterInfo.description || adapterInfo.device}.`);
  }
  if (!adapterVendor && !adapterDescription) {
    throw new Error(`WebGPU did not expose adapter identity, so GPU ${selection.name} (${selection.id}) cannot be verified.`);
  }
}

async function initializeRuntime(job) {
  report('Initializing GPU', 0);
  if (!navigator.gpu) throw new Error('WebGPU is not available in this renderer.');
  const adapter = await navigator.gpu.requestAdapter(
    job.gpuPreference === 'default' ? {} : { powerPreference: job.gpuPreference || 'high-performance' },
  );
  if (!adapter) throw new Error('Unable to acquire a WebGPU adapter.');
  const adapterInfo = adapter.info || {};
  verifySelectedGpu(job.gpuSelection, adapterInfo);
  const gpuAdapter = adapterInfo.description || adapterInfo.device || adapterInfo.vendor || 'WebGPU adapter name unavailable';
  report('Initializing GPU', 0, {
    gpuAdapter,
    gpuSelection: job.gpuSelection || null,
    gpuAdapterInfo: {
      vendor: adapterInfo.vendor || '',
      device: adapterInfo.device || '',
      description: adapterInfo.description || '',
    },
  });
  const device = await adapter.requestDevice();
  const renderer = new Renderer();
  await renderer.init(device, document.getElementById('batchCanvas'));
  report('Loading OpenCascade WASM', 0.01);
  await initOpenCascade(message => report(message, 0.02));
  return { renderer, meshKey: null, meshResult: null, cavityKey: null, cavityData: null };
}

async function getCachedMesh(runtime, job) {
  const meshKey = `${job.inputPath}|${job.options.meshSize}`;
  if (runtime.meshKey === meshKey && runtime.meshResult) {
    report('Using cached STEP mesh', 0.06);
    return runtime.meshResult;
  }

  const lease = await desktop.worker.acquireMeshCache(meshKey);
  let meshResult = lease?.status === 'hit' ? lease.value : null;
  if (!meshResult && lease?.status === 'wait') {
    for (let attempt = 0; attempt < 1200 && !meshResult; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      meshResult = await desktop.worker.getMeshCache(meshKey);
    }
  }
  if (!meshResult) {
    await waitForResume();
    report('Parsing STEP', 0.03);
    try {
      const input = await desktop.worker.readInput(job.inputPath);
      meshResult = await loadSTEPFile(
        input,
        0.008,
        message => report(message, 0.06),
        job.options.meshSize,
      );
      await desktop.worker.setMeshCache(meshKey, meshResult);
    } catch (error) {
      await desktop.worker.releaseMeshCache(meshKey);
      throw error;
    }
  }
  runtime.meshKey = meshKey;
  runtime.meshResult = meshResult;
  return meshResult;
}

async function getSharedCavity(runtime, meshKey, sim) {
  if (runtime.cavityKey === meshKey && runtime.cavityData) return runtime.cavityData;
  const lease = await desktop.worker.acquireCavityCache(meshKey);
  let cavityData = lease?.status === 'hit' ? lease.value : null;
  if (!cavityData && lease?.status === 'wait') {
    for (let attempt = 0; attempt < 1200 && !cavityData; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      cavityData = await desktop.worker.getCavityCache(meshKey);
    }
  }
  if (!cavityData) {
    sim.loadMesh(runtime.meshResult.analysisVertices || runtime.meshResult.vertices, runtime.meshResult.analysisIndices || runtime.meshResult.indices);
    cavityData = sim.getCavityData();
    await desktop.worker.setCavityCache(meshKey, cavityData);
  }
  runtime.cavityKey = meshKey;
  runtime.cavityData = cavityData;
  return cavityData;
}

async function runJob(runtime, job) {
  const { renderer } = runtime;
  const sim = new Simulation();
  await sim.init();
  const meshResult = await getCachedMesh(runtime, job);

  await waitForResume();
  report('Computing cavity voxels', 0.1);
  sim.setParams(job.options);
  const cavityData = await getSharedCavity(runtime, runtime.meshKey, sim);
  sim.loadMesh(
    meshResult.analysisVertices || meshResult.vertices,
    meshResult.analysisIndices || meshResult.indices,
    cavityData,
  );
  if (!cavityData) {
    runtime.cavityKey = runtime.meshKey;
    runtime.cavityData = sim.getCavityData();
  }
  if (sim.numCavityVoxels === 0) throw new Error('No closed cavity was found in the STEP model.');
  renderer.loadMoldMesh({ vertices: meshResult.vertices, indices: meshResult.indices });
  renderer.loadGranuleVoxels(sim.getGranularParams());

  let maxWallThickness = 0;
  let steps = 0;
  let wallThicknessElapsed = 0;
  let wallThicknessSteps = 0;
  let cycles = sim.getRotationCycles();
  report('Simulating', 0.1, { simTime: 0, remainingRatio: 1, maxWallThickness: 0 });
  while (!sim.isTerminationComplete() && (job.options.maxSteps == null || steps < job.options.maxSteps)) {
    await waitForResume();
    for (let batch = 0; batch < FAST_BATCH_STEPS; batch++) {
      await waitForResume();
      sim.step();
      steps++;
      const collectStats = sim.simTime >= WALL_THICKNESS_DELAY_SECONDS;
      renderer.stepGranules(sim.getGranularParams(), undefined, collectStats);
      await renderer.waitForGranuleReadback();

      if (sim.simTime >= WALL_THICKNESS_DELAY_SECONDS) {
        wallThicknessElapsed += sim.lastStepDuration;
        wallThicknessSteps++;
        if (wallThicknessSteps >= THICKNESS_UPDATE_INTERVAL || sim.isTerminationComplete()) {
          const deposition = await renderer.accumulateThickness(
            wallThicknessElapsed,
            sim.remainingGranuleVolume,
            job.options.depositionRate,
          );
          sim.consumeGranules(deposition.depositedVolume);
          renderer.setGranuleVolumeRatio(sim.getDepositionStats().remainingRatio);
          maxWallThickness = deposition.maxThickness;
          wallThicknessElapsed = 0;
          wallThicknessSteps = 0;
        }
      }
      if (sim.isTerminationComplete()) break;
      if (job.options.maxSteps != null && steps >= job.options.maxSteps) break;
    }

    const stats = sim.getDepositionStats();
    cycles = sim.getRotationCycles();
    report('Simulating', 0.1 + (1 - stats.remainingRatio) * 0.85, {
      simTime: sim.simTime,
      remainingRatio: stats.remainingRatio,
      maxWallThickness,
      jointCycles: cycles.joint,
    });
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  report('Exporting result', 0.97);
  await renderer.prepareExport();
  const obj = renderer.exportColoredOBJ();
  if (!obj) throw new Error('The completed simulation did not produce an OBJ.');
  const stats = sim.getDepositionStats();
  const summary = {
    cavityVolume: stats.cavityVolume,
    initialGranuleVolume: stats.initialGranuleVolume,
    depositedVolume: stats.depositedVolume,
    simulationTime: sim.simTime,
    terminationCondition: job.options.terminationCondition,
    terminationCycles: job.options.terminationCycles,
    completedJointCycles: cycles.joint,
    steps,
    maxWallThickness,
    triangleCount: meshResult.indices.length / 3,
    cavityVoxelCount: sim.numCavityVoxels,
  };
  await renderer.prepareGranuleState();
  const state = renderer.exportGranuleState();
  if (state) state.summary = summary;
  const output = await desktop.worker.writeResult({ obj, summary, state });
  return { output, summary: output.summary };
}

async function run() {
  if (!desktop?.worker) throw new Error('Batch simulation requires the desktop build.');
  const firstJob = await desktop.worker.getJob();
  if (!firstJob) return;
  const runtime = await initializeRuntime(firstJob);
  let job = firstJob;
  while (job) {
    try {
      const result = await runJob(runtime, job);
      desktop.worker.complete({ status: 'completed', progress: 1, output: result.output, summary: result.summary });
      await new Promise(resolve => setTimeout(resolve, 0));
      job = await desktop.worker.getJob();
    } catch (error) {
      console.error('[Batch worker] failed:', error);
      desktop.worker.complete({ status: 'failed', error: error.message || String(error) });
      break;
    }
  }
}

run().catch(error => {
  console.error('[Batch worker] failed:', error);
  desktop?.worker.complete({ status: 'failed', error: error.message || String(error) });
});
