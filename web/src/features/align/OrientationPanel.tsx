/**
 * 朝向确认面板：人工按**外切盒面号**指派语义轴 —— 正面/左边/上面 各是几号面。
 *
 * 编号 1..6 按模型原坐标轴固定（1=+X 2=-X 3=+Y 4=-Y 5=+Z 6=-Z），贴在模型自身的
 * 面上、不随校准旋转改变（右图盒面上直接标着数字）。三个下拉的默认值 = 后端自动
 * 探测结果（`auto.axis_faces`）：**不改即表示自动朝向正确**；改了某个下拉，另外两个
 * 会按右手系自动跟着调（`cross(上, 正) = 左`），保证组合永远可旋转实现、不产生镜像。
 *
 * 本组件是**受控**的：草稿状态在 `Workspace`，改完点「应用修正」才发 PATCH。
 */
import type { AlignAuto, AxisFaces, PcaInfo, Vec3 } from '../../api/types'

export interface OrientationDraft {
  /** 当前生效的面号指派（默认 = 自动探测结果）。 */
  axisFaces: AxisFaces
}

export interface OrientationPanelProps {
  auto: AlignAuto
  pca: PcaInfo | null
  draft: OrientationDraft
  disabled?: boolean
  onDraftChange: (next: OrientationDraft) => void
}

/** 编号 1..6 在模型原坐标轴下的外法向（与后端 `_RAW_FACE_NORMALS` 同序）。 */
const FACE_NORMALS: readonly Vec3[] = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
]

const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]

/** 单位 ±轴向量 → 面号 1..6。 */
const idOf = (v: Vec3): number =>
  FACE_NORMALS.findIndex((n) => n[0] === v[0] && n[1] === v[1] && n[2] === v[2]) + 1

/** 三个下拉：改一个，其余按右手系 `cross(上, 正) = 左` 自动补齐。 */
const ROWS: { key: keyof AxisFaces; label: string }[] = [
  { key: 'front', label: '正面' },
  { key: 'left', label: '左边' },
  { key: 'up', label: '上面' },
]

function withConsistency(cur: AxisFaces, key: keyof AxisFaces, value: number): AxisFaces {
  const next: AxisFaces = { ...cur, [key]: value }
  const n = (id: number) => FACE_NORMALS[id - 1]
  if (key === 'front') next.left = idOf(cross(n(next.up), n(value)))
  if (key === 'up') next.left = idOf(cross(n(value), n(next.front)))
  if (key === 'left') next.up = idOf(cross(n(next.front), n(value)))
  return next
}

export function OrientationPanel({ auto, pca, draft, disabled = false, onDraftChange }: OrientationPanelProps) {
  const faces = draft.axisFaces
  const autoFaces = auto.axis_faces
  const isAuto = autoFaces
    && faces.front === autoFaces.front && faces.left === autoFaces.left && faces.up === autoFaces.up
  return (
    <div className="card">
      <h3>朝向确认</h3>
      <div className="muted" style={{ marginBottom: 8 }}>
        右图盒面上标着编号 1~6（贴在模型自身、不随旋转改变）。选择
        <b> 正面 / 左边 / 上面 </b>各是几号面；不选即表示自动朝向正确。
      </div>
      {ROWS.map(({ key, label }) => (
        <div className="row" key={key} style={{ gap: 8, marginBottom: 6 }}>
          <span style={{ width: 44 }}>{label}</span>
          <select
            value={faces[key]}
            disabled={disabled}
            onChange={(e) => onDraftChange({
              ...draft,
              axisFaces: withConsistency(faces, key, Number(e.target.value)),
            })}
          >
            {[1, 2, 3, 4, 5, 6].map((id) => <option key={id} value={id}>{id} 号面</option>)}
          </select>
          <span className="muted">
            {key === 'front' ? '角色面部朝向' : key === 'left' ? '角色左手侧' : '头顶方向'}
          </span>
        </div>
      ))}
      <div className="muted" style={{ marginTop: 6 }}>
        {isAuto
          ? `当前为自动朝向（${auto.method}）；改任一下拉即按所选面号重新定向。`
          : '已人工指定面号：其余两轴按右手系自动补齐，不会产生镜像。'}
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
