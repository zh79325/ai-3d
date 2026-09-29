/**
 * 素材库（模型 / 动画两个 tab）。
 *
 * 交互：左栏建素材 + 上传（后端**同步跑完 S1**，回来就是可修正的对齐提案），
 * 右栏直接预览归一化后的 GLB 与外切盒；点「进入 S1」跳到矫正页做人工修正。
 *
 * 动画素材同样走 S1（文件归一 + 轴系 + 单位），入库后可被任意模型复用，
 * 不需要每个模型单独导入动画 —— 这是本次重构相对旧「一对一任务」的核心差别。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { assetGlbUrl, usePollAsset } from '../../api/client'
import type { AssetKind, AssetSummary } from '../../api/types'
import { AxisGizmo } from '../../components/three/AxisGizmo'
import { BoundingBoxFaces } from '../../components/three/BoundingBoxFaces'
import { GlbModel, type ModelBounds } from '../../components/three/GlbModel'
import { obbRadius, toSceneObb } from '../../components/three/obb'
import { ViewerCanvas, type ViewFocus } from '../../components/three/ViewerCanvas'
import { useAssetStore } from '../../store/assetStore'

const KINDS: ReadonlyArray<{ key: AssetKind; label: string }> = [
  { key: 'model', label: '🧍 模型' },
  { key: 'animation', label: '🎬 动画' },
]

const KIND_HINT: Record<AssetKind, string> = {
  model: '目标模型（无骨骼网格 FBX/GLB）：走 S1 导入矫正 → S2 绑定 → S3 重定向 → S4 导出。',
  animation: '源动画（已绑定骨骼 FBX/GLB）：只走 S1 归一化，入库后可被任意模型复用。',
}

/** 上传白名单，与后端 `server_v2.ALLOWED_EXT` 一致。 */
const ACCEPT = '.fbx,.glb,.gltf'

export function AssetLibrary() {
  const navigate = useNavigate()
  const kind = useAssetStore((s) => s.kind)
  const setKind = useAssetStore((s) => s.setKind)
  const assets = useAssetStore((s) => s.assets)
  const loading = useAssetStore((s) => s.loading)
  const busy = useAssetStore((s) => s.busy)
  const error = useAssetStore((s) => s.error)
  const notice = useAssetStore((s) => s.notice)
  const setNotice = useAssetStore((s) => s.setNotice)
  const selectedId = useAssetStore((s) => s.selectedId)
  const select = useAssetStore((s) => s.select)
  const loadAssets = useAssetStore((s) => s.loadAssets)
  const create = useAssetStore((s) => s.create)
  const upload = useAssetStore((s) => s.upload)
  const remove = useAssetStore((s) => s.remove)

  const [name, setName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  /** 模型实际包围球：`align.obb` 为 null（凸包退化）时的取景兜底。 */
  const [bounds, setBounds] = useState<ModelBounds | null>(null)
  const fileInput = useRef<HTMLInputElement | null>(null)
  const onBounds = useCallback((next: ModelBounds) => setBounds(next), [])

  useEffect(() => {
    void loadAssets()
  }, [loadAssets, kind])

  // 切素材时清掉上一个模型的包围球，否则新模型会先用旧取景参数
  useEffect(() => setBounds(null), [selectedId])

  const { data: detail } = usePollAsset(selectedId, { intervalMs: 1500 })

  /** 建素材 → 上传 → 后端同步跑完 S1；任一步失败都把 store.error 显出来。 */
  async function onUpload() {
    if (!file) {
      setNotice('请先选择文件')
      return
    }
    const assetId = await create(name.trim() || file.name.replace(/\.[^.]+$/, ''))
    if (!assetId) return
    const res = await upload(assetId, file)
    if (res) {
      setFile(null)
      setName('')
      if (fileInput.current) fileInput.current.value = ''
      select(assetId)
    }
  }

  async function onDelete(item: AssetSummary) {
    const label = item.name || item.filename || item.asset_id
    const refs = item.kind === 'model' ? '（连带删除引用它的作业）' : ''
    if (!window.confirm(`确认删除素材「${label}」？${refs}`)) return
    await remove(item.asset_id, true)
  }

  const glbUrl = useMemo(() => {
    // asset.glb 是 S1 的产物：有 align 就说明归一化跑成功过（FAILED 但保留了上一次
    // 结果的情形也成立）；CREATED 状态还没文件，请求会 404。
    if (!detail || !detail.align) return null
    return assetGlbUrl(detail.asset_id, detail.updated_at)
  }, [detail])

  const focus: ViewFocus | null = useMemo(() => {
    const obb = detail?.align?.obb
    if (obb) {
      const sceneObb = toSceneObb(obb, detail?.align?.final.rotation, detail?.align?.final.scale)
      return {
        center: [sceneObb.center.x, sceneObb.center.y, sceneObb.center.z],
        radius: Math.max(obbRadius(sceneObb), 0.05),
      }
    }
    if (bounds) return { center: bounds.center, radius: Math.max(bounds.radius, 0.05) }
    return null
  }, [detail, bounds])

  return (
    <>
      <aside className="side">
        <div className="row tight" style={{ gap: 6, marginBottom: 10 }}>
          {KINDS.map((k) => (
            <button
              key={k.key}
              className={kind === k.key ? 'on' : 'ghost'}
              style={{ flex: '1 1 auto' }}
              onClick={() => setKind(k.key)}
            >
              {k.label}
            </button>
          ))}
        </div>
        <div className="muted" style={{ marginBottom: 10 }}>{KIND_HINT[kind]}</div>

        <div className="card">
          <label>素材名称（可选，留空用文件名）</label>
          <input
            type="text"
            value={name}
            placeholder={kind === 'model' ? '例如：主角模型' : '例如：走路循环'}
            onChange={(e) => setName(e.target.value)}
          />
          <label>文件（{ACCEPT}）</label>
          <input
            ref={fileInput}
            type="file"
            accept={ACCEPT}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div className="row" style={{ marginTop: 8 }}>
            <button disabled={busy || !file} onClick={() => void onUpload()}>
              {busy ? '处理中…' : '⬆ 上传并跑 S1'}
            </button>
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            上传后自动完成格式归一、外切盒求解、轴系指派与单位推断，回来即可人工修正。
          </div>
        </div>

        {error ? <div className="note err">{error}</div> : null}
        {notice ? (
          <div className="note ok" onClick={() => setNotice(null)} role="presentation">
            {notice} <span className="muted">（点击关闭）</span>
          </div>
        ) : null}

        <h2>素材列表（{assets.length}）</h2>
        <div className="card" style={{ padding: 6 }}>
          {loading && !assets.length ? <div className="muted"><span className="spin" /> 加载中…</div> : null}
          {!loading && !assets.length ? <div className="muted">暂无{kind === 'model' ? '模型' : '动画'}素材</div> : null}
          {assets.map((item) => (
            <div
              key={item.asset_id}
              className={`item${item.asset_id === selectedId ? ' on' : ''}`}
              onClick={() => select(item.asset_id)}
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
        {detail ? (
          <>
            <ViewerCanvas focus={focus} cameraPosition={[1.8, 1.5, 2.4]}>
              {glbUrl ? <GlbModel url={glbUrl} onBounds={onBounds} /> : null}
              {detail.align?.obb ? (
                <BoundingBoxFaces
                  obb={detail.align.obb}
                  rotation={detail.align.final.rotation}
                  scale={detail.align.final.scale}
                  faceMap={detail.align.manual.face_map}
                  auto={detail.align.auto}
                />
              ) : null}
              <AxisGizmo length={focus ? focus.radius * 0.7 : 0.4} />
            </ViewerCanvas>
            <div className="overlay">
              <div style={{ fontWeight: 700 }}>
                {detail.name || detail.filename || detail.asset_id}
                {' '}
                <span className={`badge b-${detail.state}`}>{detail.state}</span>
              </div>
              <div className="muted" style={{ marginTop: 4 }}>
                {detail.filename || '尚未上传文件'}
                {detail.align ? ` · 身高 ${detail.align.unit.height_m.toFixed(3)} m` : ''}
                {bounds ? ` · ${bounds.meshes} 网格 / ${bounds.vertices} 顶点` : ''}
              </div>
              {detail.error ? <div className="note err">{detail.error}</div> : null}
              {detail.align?.unit.hint ? (
                <div className="note warn">{detail.align.unit.hint}</div>
              ) : null}
            </div>
            <div className="toolbar">
              <button onClick={() => navigate(`/asset/${detail.asset_id}/s1`)}>
                🔧 进入 S1 导入矫正
              </button>
              {detail.kind === 'model' ? (
                <button
                  className="ghost"
                  title="绑定是素材级的：这里绑一次，之后所有引用该模型的作业都复用"
                  onClick={() => navigate(`/asset/${detail.asset_id}/s2`)}
                >
                  🦴 S2 绑定（{detail.binding.state}）
                </button>
              ) : null}
            </div>
          </>
        ) : (
          <div className="empty">
            左栏选择或上传一个素材，这里会预览归一化后的模型与最小体积外切盒。
          </div>
        )}
      </section>
    </>
  )
}

export default AssetLibrary
