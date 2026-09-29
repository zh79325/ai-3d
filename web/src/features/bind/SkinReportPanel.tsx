/**
 * 蒙皮报告面板。
 *
 * 报告是 `skinning.compute_skin_weights` 的自检产物，字段名与后端 `_skin_report`
 * 一一对应（**没有** `vertices` 这种直觉上该有的键，别自己加）。这里只挑出「会影响
 * 最终效果」的几个数并给出结论，完整 JSON 折叠在下面备查。
 *
 * 判读要点：
 * - `zero_rows > 0`：有顶点没拿到任何权重，导出后这些顶点会钉在原地 —— 必须修；
 * - `components.fallback > 0`：BBW 在该连通分量上失败，退化到「最近骨」硬绑定，
 *   关节处会出现折角；
 * - `island_stats.tiny_lt16`：小于 16 顶点的碎岛，通常是建模阶段没拆好部件或网格
 *   有破面（BBW 需要封闭四面体化域）；
 * - `strain.max`：单条边最大相对拉伸，越大说明权重在局部突变。
 */
import type { SkinReport } from '../../api/types'

export interface SkinReportPanelProps {
  report: SkinReport | null
  /** 骨架整体置信度（`bindings.confidence`）。 */
  confidence: number | null
  /** 本地标记：改过骨架但还没重算蒙皮（此时 skin.npz 与 rig.json 不同步）。 */
  stale?: boolean
}

function num(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function pick(report: SkinReport, ...path: string[]): unknown {
  let cur: unknown = report
  for (const key of path) {
    if (cur === null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[key]
  }
  return cur
}

interface Issue {
  level: 'err' | 'warn' | 'info'
  text: string
}

function assess(report: SkinReport): Issue[] {
  const out: Issue[] = []
  const zero = num(report.zero_rows)
  if (zero > 0) out.push({ level: 'err', text: `${zero} 个顶点没有任何骨骼权重，导出后会钉在原地` })

  const fallback = num(pick(report, 'components', 'fallback'))
  if (fallback > 0) {
    out.push({ level: 'warn', text: `${fallback} 个顶点 BBW 求解失败，落到「最近骨」兜底（关节处会出现折角）` })
  }
  const proxy = num(pick(report, 'components', 'proxy'))
  if (proxy > 0) out.push({ level: 'info', text: `${proxy} 个顶点走代理网格求解` })

  const tiny = num(pick(report, 'island_stats', 'tiny_lt16'))
  if (tiny > 0) {
    out.push({ level: 'warn', text: `${tiny} 个小于 16 顶点的碎岛：建模阶段未拆好部件，或网格有破面` })
  }
  const islands = num(pick(report, 'island_stats', 'components'))
  if (islands > 1) out.push({ level: 'info', text: `网格有 ${islands} 个连通分量（按部件拆分是正常的）` })

  const strainMax = num(pick(report, 'strain', 'max'))
  const gt20 = num(pick(report, 'strain', 'gt20pct'))
  if (strainMax > 0.5 || gt20 > 0) {
    out.push({
      level: 'warn',
      text: `${gt20} 条边形变超 20%（最大 ${(strainMax * 100).toFixed(0)}%），权重在局部突变`,
    })
  }
  if (!out.length) out.push({ level: 'info', text: '没有发现异常指标' })
  return out
}

export function SkinReportPanel({ report, confidence, stale = false }: SkinReportPanelProps) {
  if (!report) {
    return (
      <div className="card">
        <h3>蒙皮报告</h3>
        <div className="muted">还没有蒙皮产物。</div>
      </div>
    )
  }

  const rows: [string, string][] = [
    ['骨架置信度', confidence === null ? '—' : confidence.toFixed(3)],
    ['零权重顶点', String(num(report.zero_rows))],
    ['焊接簇 / 合并顶点', `${num(pick(report, 'weld', 'clusters'))} / ${num(pick(report, 'weld', 'merged_verts'))}`],
    ['连通分量', String(num(pick(report, 'components', 'count')))],
    ['最大岛占比', `${(num(pick(report, 'island_stats', 'largest_share')) * 100).toFixed(1)}%`],
    ['权重跳变 均值 / 最大', `${num(pick(report, 'weight_jump', 'mean')).toFixed(3)} / ${num(pick(report, 'weight_jump', 'max')).toFixed(3)}`],
    ['形变 中位 / 最大', `${num(pick(report, 'strain', 'med')).toFixed(3)} / ${num(pick(report, 'strain', 'max')).toFixed(3)}`],
    ['耗时', `${num(report.seconds).toFixed(1)} s`],
  ]

  return (
    <div className="card">
      <h3>蒙皮报告</h3>
      {stale ? (
        <div className="note warn">
          骨架已改动但蒙皮还没重算，下面的指标对应上一版骨架。
        </div>
      ) : null}
      {assess(report).map((issue) => (
        <div key={issue.text} className={`note ${issue.level}`}>{issue.text}</div>
      ))}
      <table className="grid">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td className="muted">{label}</td>
              <td className="num">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details style={{ marginTop: 8 }}>
        <summary className="muted">skin_report.json 原文</summary>
        <pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', margin: '6px 0 0' }}>
          {JSON.stringify(report, null, 2)}
        </pre>
      </details>
    </div>
  )
}

export default SkinReportPanel
