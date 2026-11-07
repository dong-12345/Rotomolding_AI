const { contextBridge, ipcRenderer, webUtils } = require('electron');

function subscribe(channel, callback) {
  const listener = (_event, payload) => callback(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld('rotomoldingDesktop', {
  gpu: {
    status: () => ipcRenderer.invoke('gpu:status'),
    applyDevice: deviceId => ipcRenderer.invoke('gpu:apply-device', deviceId),
  },
  files: {
    getPaths(files) {
      return Array.from(files || [])
        .map(file => webUtils.getPathForFile(file))
        .filter(Boolean);
    },
  },
  batch: {
    start: payload => ipcRenderer.invoke('batch:start', payload),
    pause: () => ipcRenderer.invoke('batch:pause'),
    resume: () => ipcRenderer.invoke('batch:resume'),
    status: () => ipcRenderer.invoke('batch:status'),
    onStatus: callback => subscribe('batch:status-update', callback),
  },
  worker: {
    getJob: () => ipcRenderer.invoke('batch:worker-job'),
    readInput: inputPath => ipcRenderer.invoke('batch:read-input', inputPath),
    acquireMeshCache: key => ipcRenderer.invoke('batch:mesh-cache-acquire', key),
    getMeshCache: key => ipcRenderer.invoke('batch:mesh-cache-get', key),
    setMeshCache: (key, value) => ipcRenderer.invoke('batch:mesh-cache-set', key, value),
    releaseMeshCache: key => ipcRenderer.invoke('batch:mesh-cache-release', key),
    acquireCavityCache: key => ipcRenderer.invoke('batch:cavity-cache-acquire', key),
    getCavityCache: key => ipcRenderer.invoke('batch:cavity-cache-get', key),
    setCavityCache: (key, value) => ipcRenderer.invoke('batch:cavity-cache-set', key, value),
    releaseCavityCache: key => ipcRenderer.invoke('batch:cavity-cache-release', key),
    writeResult: payload => ipcRenderer.invoke('batch:write-result', payload),
    reportProgress: payload => ipcRenderer.send('batch:worker-progress', payload),
    complete: payload => ipcRenderer.send('batch:worker-complete', payload),
    onControl: callback => subscribe('batch:worker-control', callback),
  },
});
