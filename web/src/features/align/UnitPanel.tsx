/**
 * 单位面板：展示自动推断结果，允许强制单位或直接填真实身高。
 *
 * 后端 `estimate_unit` 的优先级是 **真实身高 > 强制单位 > 自动推断**，面板的控件
 * 顺序与之一致。自动推断取「换算后身高与成人中值 1.75m 的对数距离最小」的单位，
 * 比固定数量级区间鲁棒（inch 与 cm 在 50~120 重叠）；`hint` 非空即表示换算后身高
 * 落在成人区间 [1.4, 2.1] 之外，必须人工确认 —— 归一化资产（0.98m 之类）很常见。
 *
 * 单位没矫正就进 S2 是硬伤：`skin.weld_epsilon=2e-3`、`foot_contact_height=0.12`
 * 都是**米制绝对阈值**，所以这里的数据要一眼能核对。
 */
import type { AlignUnit } from '../../api/types'

export interface UnitDraft {
  /** 强制单位（`m`/`cm`/`mm`/`inch`/`ft`）；null = 交给自动推断。 */
  unitOverride: string | null
  /** 真实身高（米）；null = 不直填。优先级高于强制单位。 */
  heightOverride: number | null
}

export interface UnitPanelProps {
  unit: AlignUnit
  /** 最终写入 canon 节点的统一缩放（`align.final.scale`）。 */
  scale: number
  draft: UnitDraft
  disabled?: boolean
  onDraftChange: (next: UnitDraft) => void
}

/** 身高输入框用字符串承载，避免「清空」被 Number('') = 0 误当成填了 0 米。 */
function heightText(value: number | null): string {
  return value === null || !Number.isFinite(value) ? '' : String(value)
}

export function UnitPanel({ unit, scale, draft, disabled = false, onDraftChange }: UnitPanelProps) {
  const units = unit.candidates.map((c) => c.unit)
  const heightStr = heightText(draft.heightOverride)

  return (
    <div className="card">
      <h3>单位与身高</h3>
      <table className="grid">
        <tbody>
          <tr>
            <td className="muted">判定单位</td>
            <td><b>{unit.detected}</b></td>
            <td className="muted">统一缩放</td>
            <td className="num">{scale.toExponential(4)}</td>
          </tr>
          <tr>
            <td className="muted">矫正后身高</td>
            <td className="num"><b>{unit.height_m.toFixed(4)} m</b></td>
            <td className="muted">成人区间</td>
            <td className="num">1.4 ~ 2.1 m</td>
          </tr>
        </tbody>
      </table>

      {unit.hint ? <div className="note warn">{unit.hint}</div> : null}

      <label>真实身高（米，优先级最高；留空即不直填）</label>
      <input
        type="number"
        step={0.01}
        min={0}
        placeholder="例如 1.78"
        value={heightStr}
        disabled={disabled}
        onChange={(e) => {
          const raw = e.target.value.trim()
          const parsed = raw === '' ? null : Number(raw)
          onDraftChange({
            ...draft,
            // 非正数一律当「清除」：后端用 height_override <= 0 作为清除哨兵
            heightOverride: parsed !== null && Number.isFinite(parsed) && parsed > 0 ? parsed : null,
          })
        }}
      />

      <label>强制单位（真实身高留空时生效）</label>
      <select
        value={draft.unitOverride ?? ''}
        disabled={disabled}
        onChange={(e) => onDraftChange({ ...draft, unitOverride: e.target.value || null })}
      >
        <option value="">自动推断（当前判为 {unit.detected}）</option>
        {units.map((name) => <option key={name} value={name}>{name}</option>)}
      </select>

      {unit.candidates.length ? (
        <>
          <label>候选（按与 1.75m 的对数距离排序）</label>
          <table className="grid">
            <thead>
              <tr><th>单位</th><th>系数</th><th>身高 m</th><th>log 距离</th></tr>
            </thead>
            <tbody>
              {unit.candidates.map((c) => (
                <tr key={c.unit} style={c.unit === unit.detected ? { color: 'var(--ok)' } : undefined}>
                  <td>{c.unit}</td>
                  <td className="num">{c.scale}</td>
                  <td className="num">{c.height_m.toFixed(4)}</td>
                  <td className="num">{c.log_dist.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}
    </div>
  )
}

export default UnitPanel
