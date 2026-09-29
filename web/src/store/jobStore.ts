/**
 * 作业状态（zustand）。
 *
 * 作业 = 模型素材 × 动画素材，承载 S2~S4。本期（P2）只落**列表与选中**这层壳，
 * `GET /v2/jobs`、`POST /v2/jobs` 等路由在 P4 接入；详情与阶段进度由各页面用
 * `usePollJob` 就地轮询，不进全局 store（理由同 `assetStore`）。
 */
import { create } from 'zustand'

import { API_BASE } from '../api/client'
import type { JobInfoV2, StageGroup } from '../api/types'

/** 作业列表项（`JobSummaryV2`）。 */
export interface JobSummaryV2 {
  job_id: string
  name: string | null
  state: string
  stage_group: StageGroup | null
  model_asset_id: string
  anim_asset_id: string | null
  created_at: string | null
  updated_at: string | null
}

export interface JobStoreState {
  jobs: JobSummaryV2[]
  total: number
  loading: boolean
  error: string | null
  selectedId: string | null
  /** 详情面板展开的阶段组；S1 属于素材，故这里只有 S2~S4。 */
  stageGroup: StageGroup

  select: (jobId: string | null) => void
  setStageGroup: (group: StageGroup) => void
  loadJobs: (modelAssetId?: string) => Promise<void>
  createJob: (modelAssetId: string, name?: string) => Promise<string | null>
  selectAnimation: (jobId: string, animAssetId: string | null) => Promise<JobInfoV2 | null>
  exportJob: (jobId: string, force?: boolean) => Promise<boolean>
}

const enc = encodeURIComponent

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: { 'Content-Type': 'application/json' } }
  if (body !== undefined) init.body = JSON.stringify(body)
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    const text = typeof detail?.detail === 'string' ? detail.detail : `HTTP ${res.status}`
    throw new Error(text)
  }
  return (await res.json()) as T
}

export const useJobStore = create<JobStoreState>((set, get) => ({
  jobs: [],
  total: 0,
  loading: false,
  error: null,
  selectedId: null,
  stageGroup: 'S2',

  select: (jobId) => set({ selectedId: jobId }),
  setStageGroup: (stageGroup) => set({ stageGroup }),

  loadJobs: async (modelAssetId) => {
    set({ loading: true })
    try {
      const qs = modelAssetId ? `?model_asset_id=${enc(modelAssetId)}` : ''
      const res = await call<{ jobs: JobSummaryV2[]; total: number }>(
        'GET', `/v2/jobs${qs}`)
      set({ jobs: res.jobs, total: res.total, loading: false, error: null })
    } catch (exc) {
      set({ loading: false, error: (exc as Error).message })
    }
  },

  createJob: async (modelAssetId, name) => {
    set({ error: null })
    try {
      const res = await call<{ job_id: string }>('POST', '/v2/jobs',
        { model_asset_id: modelAssetId, name: name || null })
      set({ selectedId: res.job_id })
      await get().loadJobs(modelAssetId)
      return res.job_id
    } catch (exc) {
      set({ error: (exc as Error).message })
      return null
    }
  },

  /** 选素材库动画并触发 S3 重定向。 */
  selectAnimation: async (jobId, animAssetId) => {
    set({ error: null })
    try {
      return await call<JobInfoV2>('PATCH', `/v2/jobs/${enc(jobId)}/animation`,
        { anim_asset_id: animAssetId })
    } catch (exc) {
      set({ error: (exc as Error).message })
      return null
    }
  },

  /** 触发 S4 导出；`force` 越过门控（P5 接入）。 */
  exportJob: async (jobId, force = false) => {
    set({ error: null })
    try {
      await call<unknown>('POST', `/v2/jobs/${enc(jobId)}/export`, { force })
      return true
    } catch (exc) {
      set({ error: (exc as Error).message })
      return false
    }
  },
}))
