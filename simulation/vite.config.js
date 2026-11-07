/**
 * Vite 构建配置
 *
 * 关键配置说明：
 *   1. COOP/COEP 头：WebGPU 需要跨源隔离（cross-origin isolation）才能使用 SharedArrayBuffer
 *   2. esnext 目标：使用最新 ES 语法，不做降级转换（WebGPU 本身就是现代 API）
 *   3. opencascade.js 通过 ?raw/?url 方式导入，不经过 Vite 预构建
 */
import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const OCCT_ASSET_PATTERN = /^\/opencascade\.full\.(?:js|wasm)$/;
const OCCT_CACHE_CONTROL = 'public, max-age=31536000, immutable';

function cacheOpenCascadeAssets() {
  const install = server => {
    server.middlewares.use((req, res, next) => {
      const pathname = new URL(req.url, 'http://localhost').pathname;
      if (OCCT_ASSET_PATTERN.test(pathname)) {
        res.setHeader('Cache-Control', OCCT_CACHE_CONTROL);
      }
      next();
    });
  };

  return {
    name: 'cache-local-opencascade',
    configureServer: install,
    configurePreviewServer: install,
  };
}

export default defineConfig({
  // Relative assets also work when Electron loads the Vite build from file://.
  base: './',
  plugins: [cacheOpenCascadeAssets()],
  server: {
    headers: {
      // 跨源开放者策略 — 允许跨源访问
      'Cross-Origin-Opener-Policy': 'same-origin',
      // 跨源嵌入者策略 — credentialless 模式允许跨源资源加载（不发送凭证）
      // 比 require-corp 更宽松：不要求远端服务器发送 Cross-Origin-Resource-Policy 头
      // 与 COOP: same-origin 组合仍可实现跨源隔离（启用 SharedArrayBuffer）
      'Cross-Origin-Embedder-Policy': 'credentialless'
    }
  },
  build: {
    target: 'esnext',  // 不做语法降级（WebGPU 需要现代浏览器）
    minify: false,      // 不压缩代码（方便调试）
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        batchWorker: resolve(__dirname, 'batch-worker.html'),
      },
    },
  }
});
