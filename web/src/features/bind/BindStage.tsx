/**
 * S2 绑定页。
 *
 * 布局与 S1 一致：左栏是控制面板（绑定 / 多视角 / 关节表 / 蒙皮报告），右栏是
 * 3D 视口（模型幽灵 + 语义骨架 + 可拖拽关节）。
 *
 * 三条要点：
 *
 * 1. **绑定是素材级的**：产物写 `output/assets/<id>/`，之后任何作业引用这个模型都
 *    直接复用，换动画不重算蒙皮。所以这一页挂在 `/asset/:id/s2` 而不是作业下。
 * 2. **`WAIT_VIEWS` 不是失败**：服务端没有 GPU，多视角图必须由浏览器渲染回传。
 *    S2 停在这个状态等 `POST /views`，回传后自动续跑。
 * 3. **关节微调走乐观锁**：`PATCH /rig` 必须带上当前 `revision`，409 说明别处已经
 *    改过（例如后台重跑刚完成），此时重新拉取而不是硬提交。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as THREE from 'three'

import {
  ApiError,
  assetGlbUrl,
  getRig,
  getSkinReport,
  getViewSpec,
  patchRig,
  uploadViews,
  usePollAsset,
} from '../../api/client'
import type { JointPatch, RigDoc, SkinReport, ViewSpec } from '../../api/types'
import { RigBones } from '../../components/three/BoneLines'
import { GlbModel, type ModelBounds } from '../../components/three/GlbModel'
import { GHOST_LEVELS, type GhostLevel } from '../../components/three/GhostMaterial'
import { RigPoints, SOURCE_COLORS, type Head3 } from '../../components/three/RigPoints'
import { ViewerCanvas, type ViewFocus } from '../../components/three/ViewerCanvas'
import { useAssetStore } from '../../store/assetStore'
import { captureViews } from './captureViews'
import { JointTable } from './JointTable'
import { SkinReportPanel } from './SkinReportPanel'
import { ViewsPanel } from './ViewsPanel'

// 错误/加载态只渲染一个节点，要横跨 .shell 网格的两列，否则会挤在左栏宽度里
const FULL_ROW = { gridColumn: '1 / -1' } as const

function messageOf(exc: unknown): string {
  if (exc instanceof ApiError) return exc.message
  return exc instanceof Error ? exc.message : String(exc)
}

/** 把本地草稿覆盖到 rig 上，只替换改过的关节（3D 里的骨架要跟着手走）。 */
function applyDraft(rig: RigDoc | null, draft: Record<string, Head3>): RigDoc | null {
  if (!rig) return null
  const ids = Object.keys(draft)
  if (!ids.length) return rig
  const joints = { ...rig.joints }
  for (const id of ids) {
    const joint = joints[id]
    if (joint) joints[id] = { ...joint, head: draft[id] }
  }
  return { ...rig, joints }
}

export function BindStage() {
  const { assetId = '' } = useParams()
  const navigate = useNavigate()
  const { data: detail, loading, error: pollError, refresh } = usePollAsset(
    assetId || null, { intervalMs: 1200 })

  const storeBusy = useAssetStore((s) => s.busy)
  const storeError = useAssetStore((s) => s.error)
  const notice = useAssetStore((s) => s.notice)
  const setNotice = useAssetStore((s) => s.setNotice)
  const bind = useAssetStore((s) => s.bind)
  const rebind = useAssetStore((s) => s.rebind)
  const reskin = useAssetStore((s) => s.reskin)

  const binding = detail?.binding ?? null
  const [spec, setSpec] = useState<ViewSpec | null>(null)
  const [rig, setRig] = useState<RigDoc | null>(null)
  const [report, setReport] = useState<SkinReport | null>(null)
  const [draft, setDraft] = useState<Record<string, Head3>>({})
  const [locks, setLocks] = useState<Record<string, boolean>>({})
  const [selected, setSelected] = useState<string | null>(null)
  const [progress, setProgress] = useState<string | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [skinStale, setSkinStale] = useState(false)
  const [bounds, setBounds] = useState<ModelBounds | null>(null)
  const [ghost, setGhost] = useState<GhostLevel | null>('faint')
  const [showMesh, setShowMesh] = useState(true)
  const [showLabels, setShowLabels] = useState(false)
  const [reloadToken, setReloadToken] = useState(0)
  const loader = useMemo(() => new GLTFLoader(), [])
  const onBounds = useCallback((next: ModelBounds) => setBounds(next), [])

  // 切素材时清掉上一个模型的一切本地态
  useEffect(() => {
    setBounds(null)
    setRig(null)
    setReport(null)
    setDraft({})
    setLocks({})
    setSelected(null)
    setProblem(null)
    setSkinStale(false)
  }, [assetId])

  // 渲染规格：GET 即持久化（后端 SOLVE_RIG 要读同一份相机参数），进页面就拉一次
  useEffect(() => {
    if (!assetId) return undefined
    let alive = true
    getViewSpec(assetId)
      .then((s) => { if (alive) setSpec(s) })
      .catch(() => { /* 拉不到不阻断：onCapture 里会再试一次 */ })
    return () => { alive = false }
  }, [assetId])

  const state = binding?.state ?? 'NONE'
  const running = state === 'RUNNING' || state === 'PENDING'
  const busy = storeBusy || progress !== null

  // 绑定产物（rig.json / skin_report.json）：只在 has_rig 时读，否则一路 404
  const rigKey = detail && binding?.has_rig
    ? `${detail.asset_id}:${binding.revision}:${state}:${reloadToken}`
    : ''
  useEffect(() => {
    if (!rigKey || !detail) {
      setRig(null)
      setReport(null)
      return undefined
    }
    let alive = true
    void (async () => {
      try {
        const nextRig = await getRig(detail.asset_id)
        // 报告缺失不算错：reskin 只写 rig.json 时报告仍是上一版，读不到就留空
        const nextReport = await getSkinReport(detail.asset_id).catch(() => null)
        if (!alive) return
        setRig(nextRig)
        setReport(nextReport)
        setSkinStale(false)
      } catch (exc) {
        if (alive) setProblem(`读取绑定产物失败：${messageOf(exc)}`)
      }
    })()
    return () => { alive = false }
    // rigKey 已含 asset_id/revision/state/reloadToken，不需要再列 detail
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rigKey])

  const glbUrl = useMemo(
    () => (detail ? assetGlbUrl(detail.asset_id, detail.updated_at) : null),
    [detail],
  )

  // 取景只看**服务端**骨架：拖拽时 rigView 每帧都变，拿它取景会让相机跟着手乱跑
  const focus: ViewFocus | null = useMemo(() => {
    if (rig) {
      const box = new THREE.Box3()
      for (const joint of Object.values(rig.joints)) {
        const head = joint.head
        if (head?.length === 3) box.expandByPoint(new THREE.Vector3(head[0], head[1], head[2]))
      }
      if (!box.isEmpty()) {
        const sphere = box.getBoundingSphere(new THREE.Sphere())
        return {
          center: [sphere.center.x, sphere.center.y, sphere.center.z],
          radius: Math.max(sphere.radius, 0.05),
        }
      }
    }
    if (bounds) return { center: bounds.center, radius: Math.max(bounds.radius, 0.05) }
    return null
  }, [rig, bounds])

  const rigView = useMemo(() => applyDraft(rig, draft), [rig, draft])
  const jointColors = useMemo(() => {
    if (!rigView) return null
    const out: Record<string, string> = {}
    for (const [id, joint] of Object.entries(rigView.joints)) {
      out[id] = draft[id] ? '#ffffff' : (SOURCE_COLORS[joint.source] ?? SOURCE_COLORS.unknown)
    }
    return out
  }, [rigView, draft])

  /** 提交一批关节改动；成功返回 true（调用方据此清草稿）。 */
  async function commit(joints: Record<string, JointPatch>, reskin: boolean): Promise<boolean> {
    if (!detail || !rig || !Object.keys(joints).length) return false
    setProblem(null)
    setProgress(reskin ? '提交并重算蒙皮…' : '提交骨架微调…')
    try {
      const res = await patchRig(detail.asset_id, { revision: rig.revision, joints }, reskin)
      // reskin 被跳过（多半是后端正忙）时，skin.npz 与 rig.json 就不同步了
      setSkinStale(res.reskin !== 'started')
      setNotice(res.reskin === 'started'
        ? `骨架已更新（revision ${res.rig_revision}），蒙皮重算中…`
        : `骨架已更新（revision ${res.rig_revision}），蒙皮未重算`)
      refresh()
      setReloadToken((n) => n + 1)
      return true
    } catch (exc) {
      const api = exc instanceof ApiError ? exc : null
      if (api?.status === 409) {
        setProblem(`${api.message}。已重新拉取最新骨架，请重做这次微调。`)
        setDraft({})
        setLocks({})
        refresh()
        setReloadToken((n) => n + 1)
      } else {
        setProblem(`提交失败：${messageOf(exc)}`)
      }
      return false
    } finally {
      setProgress(null)
    }
  }

  async function onCommitDraft(reskin: boolean) {
    if (!rig) return
    const joints: Record<string, JointPatch> = {}
    for (const [id, head] of Object.entries(draft)) joints[id] = { head }
    for (const [id, locked] of Object.entries(locks)) {
      if (Boolean(rig.joints[id]?.locked) !== locked) {
        joints[id] = { ...(joints[id] ?? {}), locked }
      }
    }
    if (await commit(joints, reskin)) {
      setDraft({})
      setLocks({})
    }
  }

  async function onDragEnd(id: string, head: Head3) {
    // 拖拽只改骨架不重算蒙皮：BBW 一次要数秒，边拖边等没法调；
    // 骨架满意后点「重算蒙皮」一次性提交。
    if (await commit({ [id]: { head } }, false)) {
      setDraft((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
    }
  }

  async function onCapture() {
    if (!detail) return
    setProblem(null)
    setProgress('拉取渲染规格…')
    try {
      const viewSpec = spec ?? await getViewSpec(detail.asset_id)
      setSpec(viewSpec)
      setProgress('加载模型…')
      // 单独用 GLTFLoader 载入一份干净的：画布里那份挂着幽灵材质与显示开关，
      // 截给 DWPose 会得到半透明轮廓，关键点检测直接失效
      const gltf = await loader.loadAsync(assetGlbUrl(detail.asset_id, detail.updated_at))
      const views = await captureViews(viewSpec, gltf.scene, (done, total, viewId) => {
        setProgress(`渲染 ${viewId}（${done}/${total}）…`)
      })
      setProgress(`回传 ${views.length} 张视角图…`)
      const res = await uploadViews(
        detail.asset_id,
        views.map((v) => ({ name: `${v.view_id}.png`, blob: v.blob })),
      )
      setNotice(res.message)
      refresh()
    } catch (exc) {
      setProblem(`多视角渲染失败：${messageOf(exc)}`)
    } finally {
      setProgress(null)
    }
  }

  async function onBind(poseAi: boolean) {
    if (!detail) return
    setProblem(null)
    const hasArtifacts = Boolean(binding?.has_rig)
    if (hasArtifacts
      && !window.confirm('重跑 S2 会覆写 rig.json，人工微调过的关节将丢失。继续？')) return
    const res = hasArtifacts
      ? await rebind(detail.asset_id, poseAi)
      : await bind(detail.asset_id, poseAi)
    if (res) {
      setSkinStale(false)
      setReloadToken((n) => n + 1)
      refresh()
    }
  }

  async function onReskin() {
    if (!detail) return
    // 还有待提交的微调就跟着 PATCH 一起走（reskin=true 一趟完成），
    // 否则单独打 /reskin：它不改骨架，revision 与 source 标记都留着
    if (Object.keys(draft).length || Object.keys(locks).length) {
      await onCommitDraft(true)
      return
    }
    const res = await reskin(detail.asset_id)
    if (res) {
      setSkinStale(false)
      setReloadToken((n) => n + 1)
      refresh()
    }
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
  if (detail.kind !== 'model') {
    return (
      <div className="empty" style={FULL_ROW}>
        这是<b>动画素材</b>，不走 S2 绑定：它只需 S1 归一化，入库后可被任意模型复用。
        <div style={{ marginTop: 10 }}>
          <Link to={`/asset/${assetId}/s1`}>← 回到 S1</Link>
        </div>
      </div>
    )
  }
  if (!detail.align) {
    return (
      <div className="empty" style={FULL_ROW}>
        该素材还没有 S1 产物，先在 S1 页上传文件。
        <div style={{ marginTop: 10 }}>
          <Link to={`/asset/${assetId}/s1`}>← 去 S1 导入矫正</Link>
        </div>
      </div>
    )
  }

  // S1 没落库（CREATED/NORMALIZING/FAILED）时后端会 409：单位没矫正就绑定，
  // weld_epsilon / foot_contact_height 这些米制绝对阈值会全错
  const aligned = detail.state === 'READY' || detail.state === 'ALIGN_READY'
  const canBind = aligned && !running && !busy
  const canEdit = Boolean(rig) && !running && !busy
  const dirty = Object.keys(draft).length > 0 || Object.keys(locks).length > 0

  return (
    <>
      <aside className="side">
        <div className="row tight" style={{ gap: 6, marginBottom: 8 }}>
          <button className="ghost sm" onClick={() => navigate('/library')}>← 素材库</button>
          <Link to={`/asset/${assetId}/s1`} className="badge b-CREATED" title="回到 S1 导入矫正">S1</Link>
          <span className="badge b-RUNNING" title="当前阶段">S2</span>
          <span className={`badge b-${state}`}>{state}</span>
        </div>
        <h1 style={{ marginBottom: 2 }}>S2 绑定</h1>
        <div className="muted" style={{ marginBottom: 10 }}>
          {detail.name || detail.filename || detail.asset_id}
        </div>

        {storeError ? <div className="note err">{storeError}</div> : null}
        {pollError && pollError !== storeError ? <div className="note err">{pollError}</div> : null}
        {problem ? <div className="note err">{problem}</div> : null}
        {notice ? (
          <div className="note ok" onClick={() => setNotice(null)} role="presentation">
            {notice} <span className="muted">（点击关闭）</span>
          </div>
        ) : null}

        {running ? (
          <div className="note info">
            <span className="spin" />
            {' '}<b>{binding?.stage ?? '排队中'}</b>：{binding?.message ?? '执行中…'}
          </div>
        ) : null}
        {state === 'WAIT_VIEWS' ? (
          <div className="note warn">
            {binding?.message ?? '等待前端回传多视角图'}
            <br />
            服务端没有 GPU，视角图必须由本页渲染回传；回传后 S2 自动续跑。
          </div>
        ) : null}
        {state === 'FAILED' && binding?.error ? (
          <div className="note err">S2 失败：{binding.error}</div>
        ) : null}
        {state === 'READY' && binding?.error ? (
          <div className="note warn">上一次重跑失败（已保留可用绑定）：{binding.error}</div>
        ) : null}
        {!aligned ? (
          <div className="note warn">
            S1 还没确认（当前 {detail.state}）。单位没矫正就绑定会让蒙皮的米制阈值全错，
            请先回 <Link to={`/asset/${assetId}/s1`}>S1</Link> 确认。
          </div>
        ) : null}

        <div className="card">
          <h3>绑定</h3>
          <div className="muted" style={{ marginBottom: 6 }}>
            AI 路径：回传多视角 → DWPose 出 2D 关键点 → 三角化 → 蒙皮（人形最准）。
            比例路径：直接按包围盒的人体比例生骨架 → 蒙皮（非人形/道具，或检不到人时兜底）。
          </div>
          <div className="row">
            <button disabled={!canBind} onClick={() => void onBind(true)}>
              🦴 AI 绑定（多视角）
            </button>
            <button className="ghost" disabled={!canBind} onClick={() => void onBind(false)}>
              📐 比例骨架绑定
            </button>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <button className="ghost" disabled={!canEdit} onClick={() => void onReskin()}>
              🎨 重算蒙皮
            </button>
            <span className="muted">
              {binding?.confidence != null ? `置信度 ${binding.confidence.toFixed(3)}` : '未绑定'}
              {rig ? ` · revision ${rig.revision}` : ''}
            </span>
          </div>
          {binding?.message && !running ? (
            <div className="muted" style={{ marginTop: 6 }}>{binding.message}</div>
          ) : null}
        </div>

        <ViewsPanel
          assetId={detail.asset_id}
          views={binding?.views ?? []}
          cameras={spec?.cameras ?? null}
          version={detail.updated_at}
          busy={progress !== null}
          disabled={!aligned || busy}
          progress={progress}
          onCapture={() => void onCapture()}
        />

        {rig ? (
          <JointTable
            rig={rig}
            draft={draft}
            locks={locks}
            selected={selected}
            disabled={!canEdit}
            dirty={dirty}
            onSelect={setSelected}
            onDraftChange={(id, head) => setDraft((prev) => ({ ...prev, [id]: head }))}
            onLockChange={(id, locked) => setLocks((prev) => ({ ...prev, [id]: locked }))}
            onCommit={(reskin) => void onCommitDraft(reskin)}
            onReset={() => { setDraft({}); setLocks({}) }}
          />
        ) : null}

        <SkinReportPanel
          report={report}
          confidence={binding?.confidence ?? null}
          stale={skinStale}
        />
      </aside>

      <section className="main">
        <ViewerCanvas focus={focus} cameraPosition={[1.9, 1.5, 2.5]}>
          {glbUrl && showMesh ? (
            <GlbModel url={glbUrl} ghost={ghost} onBounds={onBounds} />
          ) : null}
          <RigBones rig={rigView} />
          <RigPoints
            rig={rigView}
            colors={jointColors}
            selected={selected}
            editable={canEdit}
            showLabels={showLabels}
            onSelect={setSelected}
            onDrag={(id, head) => setDraft((prev) => ({ ...prev, [id]: head }))}
            onDragEnd={(id, head) => void onDragEnd(id, head)}
          />
        </ViewerCanvas>

        <div className="overlay">
          <div style={{ fontWeight: 700 }}>S2 绑定</div>
          <div className="muted" style={{ marginTop: 4, lineHeight: 1.6 }}>
            骨架与关节点是 <b>素材级</b>产物：这里绑一次，之后所有引用该模型的作业都复用。
            <br />
            拖动小球微调关节（松手即提交，不重算蒙皮）；左栏表格可精确输入坐标。
            {rig ? (
              <>
                <br />
                身高 {rig.height.toFixed(3)} m · {Object.keys(rig.joints).length} 关节
                {selected ? <> · 选中 <b>{selected}</b></> : null}
              </>
            ) : null}
          </div>
        </div>

        <div className="toolbar">
          <button className={showMesh ? 'on' : 'ghost'} onClick={() => setShowMesh((v) => !v)}>模型</button>
          <button
            className={ghost ? 'on' : 'ghost'}
            title={`模型半透明（${GHOST_LEVELS.faint}），便于看清内部骨架`}
            onClick={() => setGhost((v) => (v ? null : 'faint'))}
          >
            幽灵
          </button>
          <button className={showLabels ? 'on' : 'ghost'} onClick={() => setShowLabels((v) => !v)}>关节名</button>
        </div>
      </section>
    </>
  )
}

export default BindStage
