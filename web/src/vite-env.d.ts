/// <reference types="vite/client" />

/**
 * Vite 环境变量声明（与 `vite/client` 的全局 `ImportMetaEnv` 接口合并）。
 *
 * 只声明本工程真正读取的键；未列出的键取值为 `any`，容易把拼错的变量名带到线上，
 * 故一律显式声明为可选 string。
 */
interface ImportMetaEnv {
  /** 后端基址；留空即同源（开发走 vite proxy，生产由 FastAPI 挂在 `/app`）。 */
  readonly VITE_API_BASE?: string
  /** 开发代理目标，仅 `vite.config.ts` 读取。 */
  readonly VITE_API_TARGET?: string
  /** 开发服务器端口，仅 `vite.config.ts` 读取。 */
  readonly VITE_DEV_PORT?: string
}
