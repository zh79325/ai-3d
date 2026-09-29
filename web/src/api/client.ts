/**
 * `/v2` 接口封装 + 轮询 hook。
 *
 * 约定：
 * - **同源**。开发时 vite 把 `/v2`、`/v1` 代理到后端（`vite.config.ts`），生产时前端由
 *   FastAPI 挂在 `/app`、API 在同一进程，故 `API_BASE` 默认空串；跨域部署才需要
 *   `VITE_API_BASE`。
 * - **错误即抛**。所有非 2xx 都转成 `ApiError`（带 status 与后端 `detail`），由调用方
 *   决定是 toast 还是就地显示；这里不吞错误。
 * - **GLB 一律带 `?v=`**。后端已发 `Cache-Control: no-store`，但 three 的 loader 在
 *   同一页面会话内按 URL 缓存，S1 反复重算后不加版本键会看到旧几何。
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import type {
  AlignDoc,
  AlignPatch,
  AlignResponse,
  AssetInfo,
  AssetKind,
  AssetListResponse,
  BindMethod,
  BindResponse,
  Conventions,
  CreateAssetResponse,
  DeleteAssetResponse,
  HealthResponse,
  JobInfoV2,
  PatchRigResponse,
  ReskinResponse,
  RigDoc,
  RigPatch,
  SkinReport,
  UniRigLogResponse,
  UploadViewsResponse,
  ViewListResponse,
  ViewSpec,
} from './types'

export const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/+$/, '')

/** 后端返回的失败：`status` 用于区分 4xx（用户可修）与 5xx（服务端故障）。 */
export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, detail: unknown, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** 把 FastAPI 的 `detail` 压成一句人话（pydantic 校验错误是数组，逐条拼 `loc: msg`）。 */
function describe(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      const loc = Array.isArray(item?.loc) ? (item.loc as unknown[]).slice(1).join('.') : ''
      const msg = typeof item?.msg === 'string' ? item.msg : ''
      return loc ? `${loc}: ${msg}` : msg
    }).filter(Boolean)
    if (parts.length) return parts.join('；')
  }
  return `请求失败（HTTP ${status}）`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch (exc) {
    // 网络层失败（后端没起、CORS、断网）：与 HTTP 错误区分开，提示语更有指向性
    throw new ApiError(0, null, `无法连接后端服务：${(exc as Error).message}`)
  }
  if (!res.ok) {
    let detail: unknown = null
    try {
      detail = (await res.json()).detail
    } catch {
      // 响应体不是 JSON（网关 502 页面、空体等），detail 保持 null
    }
    throw new ApiError(res.status, detail, describe(detail, res.status))
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

function get<T>(path: string): Promise<T> {
  return request<T>(path)
}

function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: { 'Content-Type': 'application/json' } }
  if (body !== undefined) init.body = JSON.stringify(body)
  return request<T>(path, init)
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const usp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') usp.set(k, String(v))
  }
  const s = usp.toString()
  return s ? `?${s}` : ''
}

const enc = encodeURIComponent

// --------------------------------------------------------------------------- //
// 健康 / 约定
// --------------------------------------------------------------------------- //
export const getHealth = () => get<HealthResponse>('/v2/health')
export const getConventions = () => get<Conventions>('/v2/conventions')

// --------------------------------------------------------------------------- //
// 素材库
// --------------------------------------------------------------------------- //
export const createAsset = (kind: AssetKind, name?: string) =>
  send<CreateAssetResponse>('POST', '/v2/assets', { kind, name: name || null })

export const listAssets = (kind?: AssetKind, limit = 50, offset = 0) =>
  get<AssetListResponse>(`/v2/assets${query({ kind, limit, offset })}`)

export const getAsset = (assetId: string) => get<AssetInfo>(`/v2/assets/${enc(assetId)}`)

export const deleteAsset = (assetId: string, force = false) =>
  send<DeleteAssetResponse>('DELETE', `/v2/assets/${enc(assetId)}${query({ force })}`)

/** 上传原件；后端**同步跑完 S1**，直接返回对齐提案。 */
export function uploadAsset(assetId: string, file: File): Promise<AlignResponse> {
  const form = new FormData()
  form.append('file', file, file.name)
  // 不手动设 Content-Type：浏览器要自己带 boundary
  return request<AlignResponse>(`/v2/assets/${enc(assetId)}/upload`, { method: 'POST', body: form })
}

export const getAlign = (assetId: string) =>
  get<AlignDoc>(`/v2/assets/${enc(assetId)}/align`)

/** 只改 `manual` 段，后端重算 `final` 并重写 canon 节点。 */
export const patchAlign = (assetId: string, patch: AlignPatch) =>
  send<AlignResponse>('PATCH', `/v2/assets/${enc(assetId)}/align`, patch)

/** 清空全部人工修正，回到自动探测结果（等价于 `patchAlign(id, { reset: true })`）。 */
export const resetAlign = (assetId: string) => patchAlign(assetId, { reset: true })

/** 从上传原件重跑归一 + S1（保留 manual）；asset.glb 被外部改坏时用。 */
export const realignAsset = (assetId: string) =>
  send<AlignResponse>('POST', `/v2/assets/${enc(assetId)}/realign`)

/**
 * 确认 S1：model → 置 READY 并**立即起 S2**，animation → READY。
 *
 * `poseAi=false` 跳过 DWPose，直接按人体比例生骨架（非人形/道具，或 DWPose 检不到
 * 人时的兜底）；`autoBind=false` 只确认 S1 不起绑定，留给调用方自己 POST /bind。
 */
export const confirmAsset = (
  assetId: string,
  { poseAi = true, autoBind = true }: { poseAi?: boolean; autoBind?: boolean } = {},
) =>
  send<AssetInfo>(
    'POST',
    `/v2/assets/${enc(assetId)}/confirm${query({ pose_ai: poseAi, auto_bind: autoBind })}`,
  )

/**
 * 起 S2 绑定（后台线程跑，立即返回；用 `usePollAsset` 跟进度）。
 *
 * `method`：`ai`（多视角，默认）/ `bbox`（比例骨架，等价 poseAi=false）/
 * `unirig`（UniRig CPU 新链路，不需视角图）。默认 `ai`，旧行为不变。
 */
export const bindAsset = (assetId: string, poseAi = true, method: BindMethod = 'ai') =>
  send<BindResponse>(
    'POST', `/v2/assets/${enc(assetId)}/bind${query({ pose_ai: poseAi, method })}`)

/** 重跑 S2。与 `/bind` 同一实现，差别只在语义：它会**覆写** rig.json（人工 revision 丢）。 */
export const rebindAsset = (assetId: string, poseAi = true, method: BindMethod = 'ai') =>
  send<BindResponse>(
    'POST', `/v2/assets/${enc(assetId)}/rebind${query({ pose_ai: poseAi, method })}`)

// --------------------------------------------------------------------------- //
// S2 绑定：多视角回传 / 关节微调 / 蒙皮报告
// --------------------------------------------------------------------------- //
/**
 * 取多视角渲染规格。**调用即持久化**：后端会把这份 spec 写进 `view_spec.json`，
 * SOLVE_RIG 用它重建相机。所以必须先拉 spec 再渲染回传，不能自己编相机参数。
 */
export const getViewSpec = (assetId: string) =>
  get<ViewSpec>(`/v2/assets/${enc(assetId)}/view_spec`)

/**
 * 回传多视角图。`name` 必须以 `view_id` 为文件名主干（`front.png` → view_id `front`），
 * 对不上时那个视角只是不参与三角化，**不报错**，骨架会静默退化到比例先验。
 *
 * 后端先 rmtree 清空 `views/` 再写，因此每次都要把**全部**视角一起传。
 */
export function uploadViews(
  assetId: string,
  files: { name: string; blob: Blob }[],
): Promise<UploadViewsResponse> {
  const form = new FormData()
  for (const f of files) form.append('files', f.blob, f.name)
  // 不手动设 Content-Type：浏览器要自己带 boundary
  return request<UploadViewsResponse>(`/v2/assets/${enc(assetId)}/views`, {
    method: 'POST',
    body: form,
  })
}

export const listViews = (assetId: string) =>
  get<ViewListResponse>(`/v2/assets/${enc(assetId)}/views`)

export const getBinding = (assetId: string) =>
  get<AssetInfo['binding']>(`/v2/assets/${enc(assetId)}/binding`)

export const getRig = (assetId: string) => get<RigDoc>(`/v2/assets/${enc(assetId)}/rig`)

export const getSkinReport = (assetId: string) =>
  get<SkinReport>(`/v2/assets/${enc(assetId)}/skin_report`)

/**
 * 人工微调关节（revision 乐观锁）。`reskin=false` 只写 rig.json 不重算蒙皮 ——
 * 连着调好几根关节时用 false（秒回），调完再调 `reskinAsset` 一次性重算。
 */
export const patchRig = (assetId: string, patch: RigPatch, reskin = true) =>
  send<PatchRigResponse>(
    'PATCH',
    `/v2/assets/${enc(assetId)}/rig${query({ reskin })}`,
    patch,
  )

/** 只重算蒙皮（骨架 revision 与 `source="manual"` 标记都不变）。 */
export const reskinAsset = (assetId: string) =>
  send<ReskinResponse>('POST', `/v2/assets/${enc(assetId)}/reskin`)

/**
 * 增量拉取 UniRig 执行日志（`<asset_dir>/unirig.log`）。
 *
 * `offset` 传**上一次响应里的同名字段**（首次传 0）；后端按字节分片，一次最多
 * `maxBytes`（下限 1KB），剩下的下一轮接着拉。每行自带 `[+ 12.3s]` 相对时间戳，
 * 直接就能看出哪一步吃了多少时间。
 */
export const getUniRigLog = (assetId: string, offset = 0, maxBytes = 256 * 1024) =>
  get<UniRigLogResponse>(
    `/v2/assets/${enc(assetId)}/unirig/log${query({ offset, max_bytes: maxBytes })}`)

// --------------------------------------------------------------------------- //
// 产物 URL（供 <img> / useGLTF / <a download> 直接使用）
// --------------------------------------------------------------------------- //
/**
 * 归一化 GLB 的预览地址。`version` 传素材的 `updated_at`（或任何随重算变化的值），
 * 保证 S1 重算后 URL 变化、绕过 three loader 的按 URL 缓存。
 */
export function assetGlbUrl(assetId: string, version?: string | number | null): string {
  return `${API_BASE}/v2/assets/${enc(assetId)}/glb${query({ v: version })}`
}

/** 上传原件地址（排查归一化问题时对照用）。 */
export function assetRawUrl(assetId: string): string {
  return `${API_BASE}/v2/assets/${enc(assetId)}/raw`
}

/**
 * 已回传的视角图地址。后端对图片发了 `Cache-Control: no-store`，但浏览器对 `<img>`
 * 仍可能走内存缓存，故同样带 `?v=`。
 */
export function viewImageUrl(assetId: string, name: string, version?: string | number | null): string {
  return `${API_BASE}/v2/assets/${enc(assetId)}/views/${enc(name)}${query({ v: version })}`
}

// --------------------------------------------------------------------------- //
// 轮询
// --------------------------------------------------------------------------- //
export interface PollOptions<T> {
  /** 轮询间隔（毫秒）；0 表示不轮询（只在 refresh 时拉一次）。 */
  intervalMs?: number
  /** `id` 为空时不发请求（列表页/未选中态）。 */
  enabled?: boolean
  /** 命中即停止轮询（进入终态）；返回 true 后不再自动刷新。 */
  stopWhen?: (data: T) => boolean
}

export interface PollResult<T> {
  data: T | null
  error: string | null
  /** 首次加载中（不含轮询中的静默刷新，避免界面闪烁）。 */
  loading: boolean
  /** 正在发请求（含轮询）。 */
  fetching: boolean
  refresh: () => void
}

/**
 * 通用轮询：`fetcher` 返回 `null` 表示跳过本轮（依赖未就绪）。
 *
 * 组件卸载或 `fetcher` 变化后到达的响应会被丢弃（`alive` 标记），避免 setState on
 * unmounted 与旧响应覆盖新响应。轮询失败不清空 `data`，只更新 `error`——后端重启的
 * 一两秒里界面不该白掉。
 */
export function usePoll<T>(
  fetcher: () => Promise<T> | null,
  { intervalMs = 1200, enabled = true, stopWhen }: PollOptions<T> = {},
): PollResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(enabled)
  const [fetching, setFetching] = useState(false)
  const [tick, setTick] = useState(0)
  const stopped = useRef(false)

  const refresh = useCallback(() => {
    stopped.current = false
    setTick((n) => n + 1)
  }, [])

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }
    let alive = true
    const run = async (): Promise<void> => {
      const task = fetcher()
      if (task === null) {
        if (alive) setLoading(false)
        return
      }
      if (alive) setFetching(true)
      try {
        const value = await task
        if (!alive) return
        setData(value)
        setError(null)
        if (stopWhen?.(value)) stopped.current = true
      } catch (exc) {
        if (!alive) return
        setError(exc instanceof ApiError ? exc.message : (exc as Error).message)
      } finally {
        if (alive) {
          setLoading(false)
          setFetching(false)
        }
      }
    }
    void run()
    if (intervalMs <= 0) return () => { alive = false }
    const timer = window.setInterval(() => {
      if (!stopped.current) void run()
    }, intervalMs)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
    // fetcher 由调用方用 useCallback 稳定；stopWhen 同为纯函数，一并作为依赖
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, tick, fetcher, stopWhen])

  return { data, error, loading, fetching, refresh }
}

/**
 * 轮询素材详情。过渡态只有两个：S1 的 `NORMALIZING`、S2 的 `PENDING`/`RUNNING`。
 *
 * `WAIT_VIEWS` 算**停驻态**（等用户回传视角图，不是等后端）—— 不停会永远轮询下去；
 * 回传后组件调 `refresh()` 重新起轮。
 */
export function usePollAsset(assetId: string | null | undefined, opts: PollOptions<AssetInfo> = {}) {
  const fetcher = useCallback(
    () => (assetId ? getAsset(assetId) : null),
    [assetId],
  )
  const stopWhen = useCallback(
    (info: AssetInfo) =>
      info.state !== 'NORMALIZING'
      && info.binding.state !== 'RUNNING'
      && info.binding.state !== 'PENDING',
    [],
  )
  return usePoll<AssetInfo>(fetcher, { stopWhen, ...opts })
}

/** 轮询作业详情（`/v2/jobs/{id}` 路由在 P4 接入；契约已定，此处先按契约实现）。 */
export function usePollJob(jobId: string | null | undefined, opts: PollOptions<JobInfoV2> = {}) {
  const fetcher = useCallback(
    () => (jobId ? get<JobInfoV2>(`/v2/jobs/${enc(jobId)}`) : null),
    [jobId],
  )
  const stopWhen = useCallback(
    (info: JobInfoV2) => ['DONE', 'FAILED', 'NEEDS_REVIEW', 'WAITING'].includes(info.state),
    [],
  )
  return usePoll<JobInfoV2>(fetcher, { stopWhen, ...opts })
}
