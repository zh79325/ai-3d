/**
 * UniRig 执行输出面板：实时看「已经跑到哪了」。
 *
 * UniRig 一次推理好几分钟（实测骨架 ~220s、蒙皮 ~25s），而 `binding.stage` 只有
 * `UNIRIG_EXTRACT/SKELETON/SKIN/ADAPT` 四个粗粒度值 —— 中间那几分钟里根本分不清
 * 是在加载 4.4GB 权重、在跑 batch、还是已经卡死。后端把 Lightning 进度条、print、
 * logging 与失败 traceback 全量落到 `<asset_dir>/unirig.log`，本面板按字节 offset
 * 每秒增量拉一次追加显示，每行自带 `[+ 12.3s]` 相对时间戳。
 *
 * 三条要点：
 *
 * 1. **只留尾部**。几分钟能刷出几 MB 输出，全塞进 DOM 会卡死页面，故超过
 *    `MAX_CHARS` 就从整行边界丢弃头部（后端文件仍是完整的，可去目录里看全文）。
 * 2. **起跑即重置**。重跑会覆盖日志文件，后端此时返回 `reset=true`；本地也在
 *    `running` 变 true 时清空并从 offset 0 重新拉，避免只剩上一次的尾巴。
 * 3. **结束就停轮询**。`running=false` 时只补拉一次收尾，不做无谓的定时请求；
 *    折叠状态下同样不轮询（展开时会立刻补齐）。
 */
import { useEffect, useRef, useState } from 'react'

import { ApiError, getUniRigLog } from '../../api/client'

/** 面板里最多保留的字符数（后端单份日志上限 8MB，这里只渲染尾部）。 */
const MAX_CHARS = 200_000
/** 增量轮询间隔：后端每行都 flush，1s 的粒度足够「看着它在动」。 */
const POLL_MS = 1000

export interface UniRigLogPanelProps {
  assetId: string
  /** S2 是否在跑（`binding.state` 为 RUNNING/PENDING）。 */
  running: boolean
  /** `binding.stage`：以 `UNIRIG` 开头说明这一轮走的是 UniRig 链路。 */
  stage: string | null
  defaultOpen?: boolean
}

function messageOf(exc: unknown): string {
  if (exc instanceof ApiError) return exc.message
  return exc instanceof Error ? exc.message : String(exc)
}

/** 只留尾部，且从整行边界切（别让第一行是半截，看着像丢了字）。 */
function clip(text: string): string {
  if (text.length <= MAX_CHARS) return text
  const tail = text.slice(text.length - MAX_CHARS)
  const nl = tail.indexOf('\n')
  return nl >= 0 ? tail.slice(nl + 1) : tail
}

export function UniRigLogPanel({
  assetId, running, stage, defaultOpen = false,
}: UniRigLogPanelProps) {
  const unirigStage = Boolean(stage && stage.startsWith('UNIRIG'))
  const [open, setOpen] = useState(defaultOpen || running)
  const [text, setText] = useState('')
  const [exists, setExists] = useState(false)
  const [size, setSize] = useState(0)
  const [follow, setFollow] = useState(true)
  const [problem, setProblem] = useState<string | null>(null)
  const offset = useRef(0)
  const boxRef = useRef<HTMLPreElement>(null)

  // 换素材：offset 是「那个文件」的字节位置，跨素材没有意义，整块重置
  useEffect(() => {
    offset.current = 0
    setText('')
    setExists(false)
    setSize(0)
    setProblem(null)
  }, [assetId])

  // 起跑瞬间从头看并自动展开：重跑会覆盖日志，接着上一次的 offset 只会拉到尾巴
  useEffect(() => {
    if (!running) return
    offset.current = 0
    setText('')
    setOpen(true)
  }, [running])

  useEffect(() => {
    if (!open) return undefined
    let alive = true
    const pull = async (): Promise<void> => {
      try {
        const res = await getUniRigLog(assetId, offset.current)
        if (!alive) return
        offset.current = res.offset
        setExists(res.exists)
        setSize(res.size)
        setProblem(null)
        if (res.reset) setText('')           // 后端把文件重写了，本地旧输出作废
        if (res.text) setText((prev) => clip(prev + res.text))
      } catch (exc) {
        // 轮询失败不清空已有输出：后端重启的一两秒里不该把日志抹掉
        if (alive) setProblem(messageOf(exc))
      }
    }
    void pull()
    if (!running) return () => { alive = false }      // 已结束：补拉一次即停
    const timer = window.setInterval(() => { void pull() }, POLL_MS)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [assetId, open, running])

  // 跟随最新输出：只在用户没手动上翻时自动滚底，否则看历史会被一直拽走
  useEffect(() => {
    const el = boxRef.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [text, follow])

  // 没跑过 UniRig（旧链路 / 还没绑定）就不占位；跑起来或日志存在才显示
  if (!running && !exists && !unirigStage) return null

  const clipped = text.length >= MAX_CHARS - 8192
  const lines = text ? text.split('\n').length : 0

  return (
    <div className="card">
      <div className="loghead">
        <h3 style={{ margin: 0 }}>🧠 UniRig 执行输出</h3>
        <span className={`badge ${running ? 'b-RUNNING' : 'b-NONE'}`}>
          {running ? '实时' : '已结束'}
        </span>
        <span className="spacer" />
        <button
          className={follow ? 'ghost sm on' : 'ghost sm'}
          title="自动滚到最新输出；想往上翻历史就先关掉"
          onClick={() => setFollow((v) => !v)}
        >
          跟随
        </button>
        <button className="ghost sm" title="清空显示（不影响后端文件）" onClick={() => setText('')}>
          清屏
        </button>
        <button className="ghost sm" onClick={() => setOpen((v) => !v)}>
          {open ? '收起' : '展开'}
        </button>
      </div>

      {open ? (
        <>
          <div className="muted" style={{ margin: '6px 0 4px' }}>
            {running ? '每 1s 增量拉取' : '本次运行已结束'}
            {' · '}
            {size} 字节 / {lines} 行
            {clipped ? ` · 仅显示尾部 ${Math.round(MAX_CHARS / 1000)}K 字符` : ''}
            {' · '}
            全文在 output/assets/{assetId.slice(0, 8)}…/unirig.log
          </div>
          {problem ? <div className="note err">{problem}</div> : null}
          <pre
            ref={boxRef}
            className="logbox"
            onScroll={(e) => {
              // 用户手动往上翻即停跟随；滚回底部再自动跟上
              const el = e.currentTarget
              const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 12
              if (atBottom !== follow) setFollow(atBottom)
            }}
          >
            {text || (running ? '等待后端输出…' : '（本次没有输出）')}
          </pre>
        </>
      ) : null}
    </div>
  )
}

export default UniRigLogPanel
