/**
 * S1 导入矫正页。
 *
 * 布局：左栏是「人工修正表单」（朝向二选确认 / 单位身高）+ 动作按钮，右栏是
 * 3D 视口（模型 + 规范系外切盒 + 六面语义标签 + 坐标轴箭头）。V2 下校准基由
 * 后端语义先验自动组装（旋转恒为 90° 置换），人工只需二选确认「朝向正确 / 前后相反」。
 *
 * 表单是**草稿式**的：改动只落在本地 state，点「应用修正」才发一次 PATCH。后端每次
 * PATCH 都会在原始模型坐标下重测并重写 canon 节点（`_reset_canon` 保证反复修正
 * 不累积），所以频繁提交既慢又会让相机反复重取景。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { assetGlbUrl, usePollAsset } from '../../api/client'
import { AxisGizmo } from '../../components/three/AxisGizmo'
import { BoundingBoxFaces, FACE_LABELS_FALLBACK } from '../../components/three/BoundingBoxFaces'
import { GlbModel, type ModelBounds } from '../../components/three/GlbModel'
import { GHOST_LEVELS, type GhostLevel } from '../../components/three/GhostMaterial'
import { obbRadius, toSceneObb } from '../../components/three/obb'
import { ViewerCanvas, type ViewFocus } from '../../components/three/ViewerCanvas'
import { useAssetStore } from '../../store/assetStore'
import { OrientationPanel, type OrientationDraft } from './OrientationPanel'
import { UnitPanel, type UnitDraft } from './UnitPanel'

/** S1 表单草稿：朝向二选 + 单位覆盖。 */
interface Draft extends OrientationDraft, UnitDraft {}

const EMPTY_DRAFT: Draft = {
  frontFlipped: false,
  unitOverride: null,
  heightOverride: null,
}

const ACCEPT = '.fbx,.glb,.gltf'

// 错误/加载态只渲染一个节点，要横跨 .shell 网格的两列，否则会挤在左栏宽度里
const FULL_ROW = { gridColumn: '1 / -1' } as const

export function AlignStage() {
  const { assetId = '' } = useParams()
  const navigate = useNavigate()
  const { data: detail, loading, error: pollError, refresh } = usePollAsset(
    assetId || null, { intervalMs: 1500 })

  const busy = useAssetStore((s) => s.busy)
  const storeError = useAssetStore((s) => s.error)
  const notice = useAssetStore((s) => s.notice)
  const setNotice = useAssetStore((s) => s.setNotice)
  const conventions = useAssetStore((s) => s.conventions)
  const patch = useAssetStore((s) => s.patch)
  const resetAlign = useAssetStore((s) => s.reset)
  const realign = useAssetStore((s) => s.realign)
  const confirm = useAssetStore((s) => s.confirm)
  const upload = useAssetStore((s) => s.upload)

  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [problem, setProblem] = useState<string | null>(null)
  const [showBox, setShowBox] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [showFaces, setShowFaces] = useState(true)
  const [showAxes, setShowAxes] = useState(true)
  const [ghost, setGhost] = useState<GhostLevel | null>(null)
  /** 模型实际包围球：`align.bbox` 为 null 时的取景兜底。 */
  const [bounds, setBounds] = useState<ModelBounds | null>(null)
  const fileInput = useRef<HTMLInputElement | null>(null)
  const onBounds = useCallback((next: ModelBounds) => setBounds(next), [])

  // 切素材时清掉上一个模型的包围球，否则新模型会先用旧取景参数
  useEffect(() => setBounds(null), [assetId])

  const align = detail?.align ?? null
  // 草稿只在素材**真的换了版本**时重置。用 updated_at 而不是 align 对象本身做键：
  // 轮询每轮都会得到新的 align 引用，拿它当依赖会在用户打字时把输入清空。
  const version = detail ? `${detail.asset_id}:${detail.updated_at}` : ''
  useEffect(() => {
    if (!align) {
      setDraft(EMPTY_DRAFT)
      return
    }
    setDraft({
      frontFlipped: Boolean(align.manual.front_flipped),
      unitOverride: align.manual.unit_override,
      heightOverride: align.manual.height_override,
    })
    setProblem(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version])

  const glbUrl = useMemo(
    () => (detail && align ? assetGlbUrl(detail.asset_id, detail.updated_at) : null),
    [detail, align],
  )

  const focus: ViewFocus | null = useMemo(() => {
    if (align?.bbox) {
      const sceneObb = toSceneObb(align.bbox)
      return {
        center: [sceneObb.center.x, sceneObb.center.y, sceneObb.center.z],
        radius: Math.max(obbRadius(sceneObb), 0.05),
      }
    }
    if (bounds) return { center: bounds.center, radius: Math.max(bounds.radius, 0.05) }
    return null
  }, [align, bounds])

  async function onApply() {
    setProblem(null)
    const res = await patch(assetId, {
      front_flipped: draft.frontFlipped,
      // 清除哨兵：null 在 PATCH 里表示「不改该项」，故用 ''/0 表达「取消覆盖」
      unit_override: draft.unitOverride ?? '',
      height_override: draft.heightOverride ?? 0,
    })
    if (res) refresh()
  }

  async function onResetToAuto() {
    const res = await resetAlign(assetId)
    if (res) refresh()
  }

  async function onRealign() {
    const res = await realign(assetId)
    if (res) refresh()
  }

  async function onConfirm() {
    // 模型确认后后端会立即起 S2（默认走 AI 路径，没视角图就停在 WAIT_VIEWS），
    // 所以直接跳到 S2 页看进度；动画素材确认后即入库，留在本页
    const info = await confirm(assetId)
    if (!info) return
    if (info.kind === 'model') navigate(`/asset/${assetId}/s2`)
    else refresh()
  }

  async function onUpload(file: File) {
    const res = await upload(assetId, file)
    if (res) refresh()
  }

  if (!assetId) return <div className="empty" style={FULL_ROW}>缺少素材 ID</div>
  if (loading && !detail) {
    return <div className="empty" style={FULL_ROW}><span className="spin" /> 加载素材…</div>
  }
  if (!detail) {
    return (
      <div className="empty" style={FULL_ROW}>
        素材不存在或已被删除。
        <div style={{ marginTop: 10 }}><Link to="/library">← 返回素材库</Link></div>
        {pollError ? <div className="note err">{pollError}</div> : null}
      </div>
    )
  }

  const canEdit = Boolean(align) && !busy
  const faceLabels = conventions?.face_labels ?? FACE_LABELS_FALLBACK

  return (
    <>
      <aside className="side">
        <div className="row tight" style={{ gap: 6, marginBottom: 8 }}>
          <button className="ghost sm" onClick={() => navigate('/library')}>← 素材库</button>
          <span className={`badge b-${detail.state}`}>S1 {detail.state}</span>
          {detail.kind === 'model' ? (
            <Link to={`/asset/${assetId}/s2`} className={`badge b-${detail.binding.state}`}>
              S2 {detail.binding.state}
            </Link>
          ) : (
            <span className={`badge b-${detail.binding.state}`}>S2 {detail.binding.state}</span>
          )}
        </div>
        <h1 style={{ marginBottom: 2 }}>S1 导入矫正</h1>
        <div className="muted" style={{ marginBottom: 10 }}>
          {detail.name || detail.filename || detail.asset_id}
          {detail.kind === 'animation' ? '（动画素材：确认后即入库复用）' : ''}
        </div>

        {storeError ? <div className="note err">{storeError}</div> : null}
        {pollError && pollError !== storeError ? <div className="note err">{pollError}</div> : null}
        {notice ? (
          <div className="note ok" onClick={() => setNotice(null)} role="presentation">
            {notice} <span className="muted">（点击关闭）</span>
          </div>
        ) : null}
        {problem ? <div className="note err">{problem}</div> : null}
        {detail.error ? <div className="note err">后端报错：{detail.error}</div> : null}

        {!align ? (
          <div className="card">
            <h3>上传文件</h3>
            <div className="muted" style={{ marginBottom: 6 }}>
              该素材还没有 S1 产物。上传 FBX / GLB / glTF 后会自动跑完格式归一、
              外切盒求解、轴系指派与单位推断。
            </div>
            <input
              ref={fileInput}
              type="file"
              accept={ACCEPT}
              onChange={(e) => {
                const picked = e.target.files?.[0]
                if (picked) void onUpload(picked)
              }}
            />
          </div>
        ) : (
          <>
            <OrientationPanel
              auto={align.auto}
              pca={align.pca}
              draft={draft}
              disabled={!canEdit}
              onDraftChange={(next) => setDraft((prev) => ({ ...prev, ...next }))}
            />
            <UnitPanel
              unit={align.unit}
              scale={align.final.scale}
              draft={draft}
              disabled={!canEdit}
              onDraftChange={(next) => setDraft((prev) => ({ ...prev, ...next }))}
            />

            <div className="card">
              <div className="row">
                <button disabled={!canEdit} onClick={() => void onApply()}>
                  {busy ? '重算中…' : '✅ 应用修正'}
                </button>
                <button className="ghost" disabled={!canEdit} onClick={() => void onResetToAuto()}>
                  ↺ 恢复自动
                </button>
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="ghost" disabled={!canEdit} onClick={() => void onRealign()}>
                  🔁 从原件重跑 S1
                </button>
                <button
                  className="ghost"
                  disabled={!canEdit}
                  title="重新上传原件：覆盖后自动重跑格式归一 + S1"
                  onClick={() => fileInput.current?.click()}
                >
                  ⬆ 换文件
                </button>
                <input
                  ref={fileInput}
                  type="file"
                  accept={ACCEPT}
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const picked = e.target.files?.[0]
                    if (picked) void onUpload(picked)
                    e.target.value = ''
                  }}
                />
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <button
                  disabled={busy || detail.state !== 'ALIGN_READY'}
                  title={detail.kind === 'model'
                    ? '确认后进入 S2 绑定（骨骼 + 蒙皮）'
                    : '确认后动画入库，可被任意模型复用'}
                  onClick={() => void onConfirm()}
                >
                  🎯 确认 S1{detail.kind === 'model' ? '，进入 S2' : '，入库'}
                </button>
              </div>
              {detail.state === 'READY' ? (
                <div className="note ok">
                  S1 已确认。{detail.kind === 'model'
                    ? (
                      <>
                        S2 绑定：{detail.binding.state}
                        {detail.binding.stage ? `（${detail.binding.stage}）` : ''} ·
                        {' '}<Link to={`/asset/${assetId}/s2`}>去 S2 页面</Link>
                      </>
                    )
                    : '动画已入库，可在模型的 S3 页面选用。'}
                </div>
              ) : null}
            </div>

            <div className="card">
              <h3>自动探测</h3>
              <div className="muted">
                方法：{align.auto.method}
                {align.bbox ? ` · 外切盒 ${align.bbox.extents.map((v) => v.toFixed(3)).join(' × ')} m` : ' · 外切盒求解失败'}
                {` · canon ${align.final.changed ? '已改写' : '未变化'}`}
              </div>
              {align.auto.notes.length ? (
                <ul style={{ margin: '6px 0 0', paddingLeft: 18 }} className="muted">
                  {align.auto.notes.map((note) => <li key={note}>{note}</li>)}
                </ul>
              ) : null}
              <details style={{ marginTop: 8 }}>
                <summary className="muted">final.rotation（模型 → 规范系）</summary>
                <pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', margin: '6px 0 0' }}>
                  {align.final.rotation.map((row) => row.map((v) => v.toFixed(6).padStart(10)).join('')).join('\n')}
                </pre>
              </details>
            </div>
          </>
        )}
      </aside>

      <section className="main">
        <ViewerCanvas focus={focus} cameraPosition={[1.9, 1.5, 2.5]}>
          {glbUrl ? <GlbModel url={glbUrl} ghost={ghost} onBounds={onBounds} /> : null}
          {showBox && align?.bbox ? (
            <BoundingBoxFaces
              bbox={align.bbox}
              faceLabels={faceLabels}
              showFaces={showFaces}
              showLabels={showLabels}
            />
          ) : null}
          {showAxes ? <AxisGizmo length={focus ? focus.radius * 0.8 : 0.4} /> : null}
        </ViewerCanvas>

        <div className="overlay">
          <div style={{ fontWeight: 700 }}>外切盒六面语义</div>
          <div className="muted" style={{ marginTop: 4, lineHeight: 1.6 }}>
            规范系米制 AABB，与模型同坐标系，底面贴模型最低点。
            <br />
            <span style={{ color: 'var(--ok)' }}>+Y up</span>
            {' · '}
            <span style={{ color: 'var(--acc)' }}>+Z front</span>
            {' · '}
            <span style={{ color: 'var(--warn)' }}>+X left</span>
            <br />
            拖动可自由旋转查看；确认角色面部朝向 +Z。
          </div>
          {align ? (
            <div className="muted" style={{ marginTop: 6 }}>
              身高 {align.unit.height_m.toFixed(3)} m · 缩放 {align.final.scale.toExponential(3)}
              {' · '}
              {draft.frontFlipped ? '前后已翻转' : '自动朝向'}
            </div>
          ) : null}
        </div>

        <div className="toolbar">
          <button className={showBox ? 'on' : 'ghost'} onClick={() => setShowBox((v) => !v)}>外切盒</button>
          <button className={showFaces ? 'on' : 'ghost'} onClick={() => setShowFaces((v) => !v)}>面</button>
          <button className={showLabels ? 'on' : 'ghost'} onClick={() => setShowLabels((v) => !v)}>编号</button>
          <button className={showAxes ? 'on' : 'ghost'} onClick={() => setShowAxes((v) => !v)}>坐标轴</button>
          <button
            className={ghost ? 'on' : 'ghost'}
            title={`模型半透明（${GHOST_LEVELS.faint}），便于看清盒子的后半段`}
            onClick={() => setGhost((v) => (v ? null : 'faint'))}
          >
            幽灵
          </button>
        </div>
      </section>
    </>
  )
}

export default AlignStage
