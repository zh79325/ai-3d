import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Vite 配置。
 *
 * - `base: '/app/'`：产物由 FastAPI 挂在 `/app`（见 `ai3d/viewer/server.py`），
 *   资源路径必须带该前缀，否则构建后 404。
 * - 开发代理：`/v2`（四阶段新 API）与 `/v1`（旧任务 API，过渡期对照用）都转发到
 *   后端。默认打 8766 —— `run_viewer_ide.py` 起的统一入口已挂载两套路由；
 *   想直连 retarget 独立服务（8790）时设 `VITE_API_TARGET=http://127.0.0.1:8790`。
 *
 * 不读 `process.env`：工程未装 `@types/node`，用 vite 自带的 `loadEnv` 取
 * `web/.env*` 与真实环境变量即可，避免引入 node 类型依赖。
 */
const DEFAULT_TARGET = 'http://127.0.0.1:8766'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const target = env.VITE_API_TARGET || DEFAULT_TARGET
  const proxy = { target, changeOrigin: true } as const
  return {
    base: '/app/',
    plugins: [react()],
    server: {
      port: Number(env.VITE_DEV_PORT || 5173),
      proxy: {
        '/v2': { ...proxy },
        '/v1': { ...proxy },
      },
    },
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      // GLB 与 three 体积较大，放宽警告阈值以免构建日志噪音淹没真问题
      chunkSizeWarningLimit: 1600,
    },
  }
})
