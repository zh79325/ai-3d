/**
 * 关节表 + 单关节微调编辑器（S2 的人工修正入口）。
 *
 * 骨架来自多视角三角化，DWPose 检不到的关节会落到比例先验（`source="proportion"`、
 * 置信度 0.2）—— 表里把来源与置信度直接列出来，用户才知道该重点核对哪几根。
 *
 * 改动是**草稿式**的：拖拽与数值输入都只落本地 `draft`，点「提交」才发一次
 * `PATCH /rig`。提交分两档：`reskin=false` 只写 rig.json（秒回，适合连着调好几根），
 * `reskin=true` 顺带重算蒙皮（BBW 要几秒到几十秒，且期间后端占用素材锁）。
 */
import { useState } from 'react'

import type { RigDoc } from '../../api/types'
import { SOURCE_COLORS, type Head3 } from '../../components/three/RigPoints'

export interface JointTableProps {
  rig: RigDoc
  /** 待提交的 head 覆盖（semantic_id → 坐标）。 */
  draft: Record<string, Head3>
  /** 待提交的锁定状态覆盖。 */
  locks: Record<string, boolean>
  selected: string | null
  disabled?: boolean
  dirty: boolean
  onSelect: (id: string | null) => void
  onDraftChange: (id: string, head: Head3) => void
  onLockChange: (id: string, locked: boolean) => void
  onCommit: (reskin: boolean) => void
  onReset: () => void
}

/** 数值输入用字符串承载，避免「清空」被 Number('') = 0 误当成填了 0。 */
function numText(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : ''
}

const AXES = ['x', 'y', 'z'] as const

export function JointTable({
  rig,
  draft,
  locks,
  selected,
  disabled = false,
  dirty,
  onSelect,
  onDraftChange,
  onLockChange,
  onCommit,
  onReset,
}: JointTableProps) {
  const [reskin, setReskin] = useState(true)
  const ids = Object.keys(rig.joints)
  const joint = selected ? rig.joints[selected] : null
  const head: Head3 | null = selected
    ? (draft[selected] ?? (joint?.head as Head3 | undefined) ?? null)
    : null
  const locked = selected ? (locks[selected] ?? Boolean(joint?.locked)) : false

  return (
    <div className="card">
      <h3>骨架（{ids.length} 关节 · revision {rig.revision}）</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        点行选中后可编辑；也可在右侧 3D 视口直接拖拽小球。
        <span style={{ color: SOURCE_COLORS.solved }}> 绿=三角化</span>
        <span style={{ color: SOURCE_COLORS.proportion }}> 橙=比例先验</span>
        <span style={{ color: SOURCE_COLORS.manual }}> 蓝=人工</span>
      </div>
      <div style={{ maxHeight: 236, overflowY: 'auto' }}>
        <table className="grid">
          <thead>
            <tr><th>关节</th><th>来源</th><th>置信</th><th>高度 m</th></tr>
          </thead>
          <tbody>
            {ids.map((id) => {
              const item = rig.joints[id]
              const overridden = Boolean(draft[id]) || locks[id] !== undefined
              return (
                <tr
                  key={id}
                  onClick={() => onSelect(id)}
                  style={{
                    cursor: 'pointer',
                    background: selected === id ? 'var(--input)' : undefined,
                  }}
                >
                  <td>
                    <span
                      style={{
                        display: 'inline-block', width: 7, height: 7, borderRadius: 4,
                        marginRight: 6, background: SOURCE_COLORS[item.source] ?? SOURCE_COLORS.unknown,
                      }}
                    />
                    {id}
                    {item.locked || locks[id] ? ' 🔒' : ''}
                    {overridden ? ' ●' : ''}
                  </td>
                  <td className="muted">{item.source}</td>
                  <td className="num">{item.confidence.toFixed(2)}</td>
                  <td className="num">
                    {(draft[id]?.[1] ?? item.head?.[1] ?? 0).toFixed(3)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {selected && joint && head ? (
        <>
          <label>
            微调 <b>{selected}</b>
            <span className="muted">（父：{joint.parent ?? '无'}）</span>
          </label>
          <div className="row">
            {AXES.map((axis, i) => (
              <input
                key={axis}
                type="number"
                step={0.005}
                disabled={disabled || locked}
                value={numText(head[i])}
                title={`${axis}（米）`}
                onChange={(e) => {
                  const parsed = Number(e.target.value)
                  if (!Number.isFinite(parsed)) return
                  const next: Head3 = [...head] as Head3
                  next[i] = parsed
                  onDraftChange(selected, next)
                }}
              />
            ))}
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              style={{ width: 'auto' }}
              checked={locked}
              disabled={disabled}
              onChange={(e) => onLockChange(selected, e.target.checked)}
            />
            锁定（重跑 S2 时不被自动求解覆盖）
          </label>
        </>
      ) : null}

      <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <input
          type="checkbox"
          style={{ width: 'auto' }}
          checked={reskin}
          disabled={disabled}
          onChange={(e) => setReskin(e.target.checked)}
        />
        提交后重算蒙皮（关掉则只改骨架，可连着调多根）
      </label>

      <div className="row" style={{ marginTop: 6 }}>
        <button disabled={disabled || !dirty} onClick={() => onCommit(reskin)}>
          {dirty ? '✅ 提交微调' : '没有待提交的改动'}
        </button>
        <button className="ghost" disabled={disabled || !dirty} onClick={onReset}>
          ↺ 丢弃
        </button>
      </div>
    </div>
  )
}

export default JointTable
