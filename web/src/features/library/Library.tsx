/**
 * 列表页：`模型管理` / `动画管理` 两个一级菜单各自的素材列表，按 `kind` 参数化。
 *
 * 交互定稿（一气呵成的入口）：
 * - 顶部按路由 kind 固定，无内部 model/animation tab 切换。
 * - `新建`：只输名称 → `create(kind, name)` → 直接跳工作区 `/models/:id`（或 `/animations/:id`）。
 * - 点列表项 → 跳工作区续作；保留删除。
 * - 列表页**不做**文件上传、不做右栏 3D 预览 —— 上传与预览都在工作区完成。
 *
 * 与旧 `AssetLibrary` 的差别：上传/预览搬到工作区，本页只负责「列表 + 新建 + 删除」。
 */
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import type { AssetKind, AssetSummary } from '../../api/types'
import { useAssetStore } from '../../store/assetStore'

const KIND_TITLE: Record<AssetKind, string> = {
  model: '🧍 模型管理',
  animation: '🎬 动画管理',
}

const KIND_HINT: Record<AssetKind, string> = {
  model: '目标模型（无骨骼网格 FBX/GLB）：新建后进工作区上传，自动完成轴校准与单位归一。',
  animation: '源动画（已绑定骨骼 FBX/GLB）：新建后进工作区上传，归一化入库后可被任意模型复用。',
}

export interface LibraryProps {
  kind: AssetKind
}

export function Library({ kind }: LibraryProps) {
  const navigate = useNavigate()
  const storeKind = useAssetStore((s) => s.kind)
  const setKind = useAssetStore((s) => s.setKind)
  const assets = useAssetStore((s) => s.assets)
  const loading = useAssetStore((s) => s.loading)
  const busy = useAssetStore((s) => s.busy)
  const error = useAssetStore((s) => s.error)
  const notice = useAssetStore((s) => s.notice)
  const setNotice = useAssetStore((s) => s.setNotice)
  const loadAssets = useAssetStore((s) => s.loadAssets)
  const create = useAssetStore((s) => s.create)
  const remove = useAssetStore((s) => s.remove)

  const [name, setName] = useState('')

  // 路由 kind 决定列表拉哪一类；setKind 会同步 store.kind，供 loadAssets / create 使用
  useEffect(() => {
    if (storeKind !== kind) setKind(kind)
  }, [kind, storeKind, setKind])

  useEffect(() => {
    void loadAssets()
  }, [loadAssets, kind])

  /** 新建：只输名称 → 建空素材 → 跳工作区（文件在工作区内上传）。 */
  async function onCreate() {
    const assetId = await create(name.trim() || undefined)
    if (!assetId) return
    setName('')
    navigate(`/${kind === 'model' ? 'models' : 'animations'}/${assetId}`)
  }

  async function onDelete(item: AssetSummary) {
    const label = item.name || item.filename || item.asset_id
    const refs = item.kind === 'model' ? '（连带删除引用它的作业）' : ''
    if (!window.confirm(`确认删除素材「${label}」？${refs}`)) return
    await remove(item.asset_id, true)
  }

  return (
    <>
      <aside className="side">
        <h1 style={{ marginBottom: 2 }}>{KIND_TITLE[kind]}</h1>
        <div className="muted" style={{ marginBottom: 12 }}>{KIND_HINT[kind]}</div>

        <div className="card">
          <label>新建（只需名称）</label>
          <input
            type="text"
            value={name}
            placeholder={kind === 'model' ? '例如：主角模型' : '例如：走路循环'}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') void onCreate() }}
          />
          <div className="row" style={{ marginTop: 8 }}>
            <button disabled={busy} onClick={() => void onCreate()}>
              {busy ? '创建中…' : '＋ 新建'}
            </button>
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            新建后进入工作区，选文件即自动上传并完成 X / Y 轴校准。
          </div>
        </div>

        {error ? <div className="note err">{error}</div> : null}
        {notice ? (
          <div className="note ok" onClick={() => setNotice(null)} role="presentation">
            {notice} <span className="muted">（点击关闭）</span>
          </div>
        ) : null}

        <h2>列表（{assets.length}）</h2>
        <div className="card" style={{ padding: 6 }}>
          {loading && !assets.length ? <div className="muted"><span className="spin" /> 加载中…</div> : null}
          {!loading && !assets.length ? <div className="muted">暂无{kind === 'model' ? '模型' : '动画'}，点上方「新建」开始</div> : null}
          {assets.map((item) => (
            <div
              key={item.asset_id}
              className="item"
              onClick={() => navigate(`/${kind === 'model' ? 'models' : 'animations'}/${item.asset_id}`)}
              role="presentation"
            >
              <span className="grow">
                <div>{item.name || item.filename || item.asset_id.slice(0, 8)}</div>
                <div className="muted">
                  {item.height_m ? `${item.height_m.toFixed(3)} m · ` : ''}
                  {item.unit ? `原始单位 ${item.unit}` : '未矫正'}
                </div>
              </span>
              <span className={`badge b-${item.state}`}>{item.state}</span>
              <button
                className="del"
                title="删除素材"
                onClick={(e) => { e.stopPropagation(); void onDelete(item) }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="main">
        <div className="empty">
          <div style={{ fontSize: 40, marginBottom: 12 }}>{kind === 'model' ? '🧍' : '🎬'}</div>
          <div>从左侧「新建」创建，或点开一个已有{kind === 'model' ? '模型' : '动画'}继续工作。</div>
          <div className="muted" style={{ marginTop: 8 }}>
            进入工作区后选择文件即自动上传、校准并展示结果，无需再点「进入 S1」。
          </div>
        </div>
      </section>
    </>
  )
}

export default Library
