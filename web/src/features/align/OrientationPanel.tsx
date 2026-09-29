/**
 * 朝向确认面板：人工只需二选确认角色是否朝向 `+Z`。
 *
 * V2 校准基由后端语义先验自动组装（骨架 pelvis→head / 网格脚端双簇），旋转恒为
 * 90° 置换。PCA 只能定轴线不能定前后符号，故正面朝向交给人工二选：
 * 「当前朝向正确」/「前后方向相反」。选相反时后端左乘 `diag(-1,1,-1)`（绕规范系
 * Y 轴转 180°，同时反转 X 与 Z，仍是右手系、无镜像），用户不需要手动调欧拉角。
 *
 * 本组件是**受控**的：草稿状态在 `Workspace`，改完点「应用修正」才发 PATCH。
 */
import type { AlignAuto, PcaInfo } from '../../api/types'

export interface OrientationDraft {
  /** 前后方向是否相反（true = 绕规范系 Y 轴转 180°）。 */
  frontFlipped: boolean
}

export interface OrientationPanelProps {
  auto: AlignAuto
  pca: PcaInfo | null
  draft: OrientationDraft
  disabled?: boolean
  onDraftChange: (next: OrientationDraft) => void
}

/** 把单位向量格式化成简短的轴向串（吸附后每行恰一个 ±1）。 */
function axisText(v: number[]): string {
  const name = ['X', 'Y', 'Z']
  const i = v.reduce((best, cur, idx, arr) => (Math.abs(cur) > Math.abs(arr[best]) ? idx : best), 0)
  const sign = v[i] >= 0 ? '+' : '−'
  return `${sign}${name[i]}`
}

export function OrientationPanel({ auto, pca, draft, disabled = false, onDraftChange }: OrientationPanelProps) {
  return (
    <div className="card">
      <h3>朝向确认</h3>
      <div className="muted" style={{ marginBottom: 8 }}>
        自动校准：up={axisText(auto.up)}、front={axisText(auto.forward)}（{auto.method}）。
        请核对预览里角色是否<b>面部朝向 +Z（front）</b>；若背对则选「前后方向相反」。
      </div>
      <div className="row" style={{ gap: 8 }}>
        <label className="choice" style={{ flex: 1 }}>
          <input
            type="radio"
            name="orient"
            checked={!draft.frontFlipped}
            disabled={disabled}
            onChange={() => onDraftChange({ ...draft, frontFlipped: false })}
          />
          <span>当前朝向正确</span>
        </label>
        <label className="choice" style={{ flex: 1 }}>
          <input
            type="radio"
            name="orient"
            checked={draft.frontFlipped}
            disabled={disabled}
            onChange={() => onDraftChange({ ...draft, frontFlipped: true })}
          />
          <span>前后方向相反</span>
        </label>
      </div>
      <div className="muted" style={{ marginTop: 6 }}>
        选「相反」会绕 Y 轴整体转 180°（同时反转 X 与 Z），仍是右手系、不产生镜像。
      </div>

      {pca ? (
        <details style={{ marginTop: 8 }}>
          <summary className="muted">PCA 三主轴（调试：特征值降序，模型坐标）</summary>
          <pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', margin: '6px 0 0' }}>
            {pca.axes.map((ax, i) => (
              `λ${i + 1}=${pca.eigenvalues[i].toFixed(4)}  [${ax.map((v) => v.toFixed(3).padStart(7)).join('')}]`
            )).join('\n')}
          </pre>
        </details>
      ) : null}
    </div>
  )
}

export default OrientationPanel
