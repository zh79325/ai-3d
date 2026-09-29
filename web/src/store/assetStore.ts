/**
 * 素材库状态（zustand）。
 *
 * 只放**跨组件共享**的东西：约定常量、列表、当前选中、变更请求的飞行状态与错误提示。
 * 单个素材的详情/align 由各页面用 `usePollAsset` 就地拉取 —— 详情是高频轮询数据，
 * 塞进全局 store 会让无关组件跟着重渲染。
 *
 * 变更类 action 统一「捕获异常 → 写 `error` → 返回 null」，调用方据返回值分支，
 * 不再各自 try/catch。
 */
import { create } from 'zustand'

import {
  ApiError,
  bindAsset,
  confirmAsset,
  createAsset,
  deleteAsset,
  getConventions,
  listAssets,
  patchAlign,
  realignAsset,
  rebindAsset,
  resetAlign,
  reskinAsset,
  uploadAsset,
} from '../api/client'
import type {
  AlignPatch,
  AlignResponse,
  AssetInfo,
  AssetKind,
  AssetSummary,
  BindResponse,
  Conventions,
  ReskinResponse,
} from '../api/types'

export interface AssetStoreState {
  /** 前后端共享的固定约定（面编号标签、语义轴取值域等），启动时拉一次。 */
  conventions: Conventions | null
  kind: AssetKind
  assets: AssetSummary[]
  total: number
  loading: boolean
  /** 变更请求在飞（上传 / PATCH / 确认 / 删除），用于禁用按钮防重复提交。 */
  busy: boolean
  error: string | null
  /** 一次性成功提示，展示后由调用方 `setNotice(null)` 清掉。 */
  notice: string | null
  selectedId: string | null

  setKind: (kind: AssetKind) => void
  select: (assetId: string | null) => void
  setNotice: (message: string | null) => void
  setError: (message: string | null) => void

  loadConventions: () => Promise<void>
  loadAssets: () => Promise<void>
  create: (name?: string) => Promise<string | null>
  upload: (assetId: string, file: File) => Promise<AlignResponse | null>
  patch: (assetId: string, patch: AlignPatch) => Promise<AlignResponse | null>
  reset: (assetId: string) => Promise<AlignResponse | null>
  realign: (assetId: string) => Promise<AlignResponse | null>
  confirm: (assetId: string, opts?: ConfirmOptions) => Promise<AssetInfo | null>
  /** 起 S2；后台线程跑，返回只代表「排上了」，进度靠轮询素材详情。 */
  bind: (assetId: string, poseAi?: boolean) => Promise<BindResponse | null>
  /** 重跑 S2（会覆写 rig.json，人工微调丢）。 */
  rebind: (assetId: string, poseAi?: boolean) => Promise<BindResponse | null>
  /** 只重算蒙皮（关节微调后 skin.npz 与 rig.json 不同步时用）。 */
  reskin: (assetId: string) => Promise<ReskinResponse | null>
  remove: (assetId: string, force?: boolean) => Promise<boolean>
}

/** 确认 S1 时的选项，直接透传给 `confirmAsset`。 */
export interface ConfirmOptions {
  poseAi?: boolean
  autoBind?: boolean
}

function messageOf(exc: unknown): string {
  if (exc instanceof ApiError) return exc.message
  return exc instanceof Error ? exc.message : String(exc)
}

export const useAssetStore = create<AssetStoreState>((set, get) => {
  /** 变更类请求的公共包装：置 busy、失败写 error、成功后刷新列表。 */
  async function mutate<T>(
    run: () => Promise<T>,
    after?: (value: T) => void,
  ): Promise<T | null> {
    set({ busy: true, error: null })
    try {
      const value = await run()
      after?.(value)
      set({ busy: false })
      await get().loadAssets()
      return value
    } catch (exc) {
      set({ busy: false, error: messageOf(exc) })
      await get().loadAssets()
      return null
    }
  }

  return {
    conventions: null,
    kind: 'model',
    assets: [],
    total: 0,
    loading: false,
    busy: false,
    error: null,
    notice: null,
    selectedId: null,

    setKind: (kind) => set({ kind, selectedId: null }),
    select: (assetId) => set({ selectedId: assetId }),
    setNotice: (message) => set({ notice: message }),
    setError: (message) => set({ error: message }),

    loadConventions: async () => {
      // 约定拉不到不阻断界面：组件会回落到 FACE_SEMANTICS_FALLBACK
      try {
        set({ conventions: await getConventions() })
      } catch (exc) {
        set({ error: messageOf(exc) })
      }
    },

    loadAssets: async () => {
      set({ loading: true })
      try {
        const res = await listAssets(get().kind, 200, 0)
        set({ assets: res.assets, total: res.total, loading: false, error: null })
      } catch (exc) {
        set({ loading: false, error: messageOf(exc) })
      }
    },

    create: (name) => mutate(async () => (await createAsset(get().kind, name)).asset_id),

    upload: (assetId, file) => mutate(
      () => uploadAsset(assetId, file),
      (res) => set({ notice: res.message || 'S1 导入矫正完成', selectedId: assetId }),
    ),

    patch: (assetId, patch) => mutate(
      () => patchAlign(assetId, patch),
      (res) => set({ notice: res.message || '已重算对齐' }),
    ),

    reset: (assetId) => mutate(
      () => resetAlign(assetId),
      (res) => set({ notice: res.message || '已恢复自动探测' }),
    ),

    realign: (assetId) => mutate(
      () => realignAsset(assetId),
      (res) => set({ notice: res.message || '已重新归一并对齐' }),
    ),

    confirm: (assetId, opts) => mutate(
      () => confirmAsset(assetId, opts),
      (info) => set({
        notice: info.kind === 'animation'
          ? '动画已入库，可被任意模型复用'
          : (opts?.autoBind === false
            ? 'S1 已确认'
            : 'S1 已确认，S2 绑定已开始'),
      }),
    ),

    bind: (assetId, poseAi = true) => mutate(
      () => bindAsset(assetId, poseAi),
      (res) => set({ notice: res.message || 'S2 绑定已开始' }),
    ),

    rebind: (assetId, poseAi = true) => mutate(
      () => rebindAsset(assetId, poseAi),
      (res) => set({ notice: res.message || 'S2 已重跑' }),
    ),

    reskin: (assetId) => mutate(
      () => reskinAsset(assetId),
      (res) => set({ notice: res.message || '已排队重算蒙皮' }),
    ),

    remove: async (assetId, force = false) => {
      const ok = await mutate(() => deleteAsset(assetId, force))
      if (ok && get().selectedId === assetId) set({ selectedId: null })
      return ok !== null
    },
  }
})
