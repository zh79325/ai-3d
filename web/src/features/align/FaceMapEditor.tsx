/**
 * 面映射编辑器：六面序号 → 语义轴（6 行下拉）+ 绕规范系的欧拉角微调。
 *
 * 交互按用户要求做成「直接编辑序号和对应面的映射关系」：立方体上标 1..6，这里六行
 * 下拉逐面指派 up/front/left 的正负方向，第三轴留空由后端按右手系
 * `left = cross(up, forward)` 导出。`trim_euler` 是在面映射之后叠加的后置微调
 * （度，绕规范系 X→Y→Z 外旋，即 `R = Rz·Ry·Rx · R_face`），用于模型本身略歪的情况。
 *
 * 本组件是**受控**的：草稿状态在 `AlignStage`，改完点「应用修正」才发 PATCH，
 * 避免每动一下下拉就重算一次 OBB。
 */
import type { AlignAuto, AlignFace, FaceSemantic } from '../../api/types'
import { FACE_OPPOSITE } from '../../components/three/obb'

/** 语义轴 → `align.final.rotation` 的行索引（0=left/+X、1=up/+Y、2=front/+Z）。 */
export function semanticAxis(token: FaceSemantic): 0 | 1 | 2 {
  if (token.startsWith('up')) return 1
  if (token.startsWith('front')) return 2
  return 0
}

const AXIS_NAMES = ['left(+X)', 'up(+Y)', 'front(+Z)'] as const

/** 后端 `obb.FACE_LABELS` 的兜底值，conventions 拉不到时用。 */
export const FACE_LABELS_FALLBACK = ['u+', 'u-', 'v+', 'v-', 'w+', 'w-']

/** 后端 `axis_norm.FACE_SEMANTICS` 的兜底值。 */
export const FACE_SEMANTICS_FALLBACK: FaceSemantic[] = [
  'up+', 'up-', 'front+', 'front-', 'left+', 'left-',
]

export interface FaceMapDraft {
  /** 面序号（字符串键，与后端 `manual.face_map` 一致）→ 语义轴。 */
  faceMap: Record<string, FaceSemantic>
  /** 绕规范系 XYZ 的外旋微调（度）。 */
  trim: number[]
}

export interface FaceMapEditorProps {
  faces: AlignFace[]
  auto: AlignAuto
  draft: FaceMapDraft
  semantics?: FaceSemantic[]
  faceLabels?: string[]
  disabled?: boolean
  onDraftChange: (next: FaceMapDraft) => void
}

/** 前置校验：后端会拒的两类错误（同一语义轴被两面指派、对偶面互指）先在前端拦住。 */
export function validateFaceMap(faceMap: Record<string, FaceSemantic>): string | null {
  const entries = Object.entries(faceMap)
  if (!entries.length) return null          // 空 = 采用自动探测，合法
  const byAxis = new Map<number, number>()
  for (const [key, token] of entries) {
    const axis = semanticAxis(token)
    const faceId = Number(key)
    const prev = byAxis.get(axis)
    if (prev !== undefined) {
      return `${AXIS_NAMES[axis]} 被面 ${prev} 与面 ${faceId} 重复指派（含正负号，一个轴只能占一次）`
    }
    byAxis.set(axis, faceId)
  }
  if (byAxis.size < 2) return '至少要在六面中指派两个不同的语义轴（第三个由右手系导出）'
  for (const [key] of entries) {
    const faceId = Number(key)
    const opposite = FACE_OPPOSITE[faceId - 1]
    if (faceMap[String(opposite)] && semanticAxis(faceMap[String(opposite)]) !== semanticAxis(faceMap[key])) {
      return `面 ${faceId} 与它的对偶面 ${opposite} 被指派成了两个不同语义轴，得到的三轴会构成左手系`
    }
  }
  return null
}

export function FaceMapEditor({
  faces,
  auto,
  draft,
  semantics = FACE_SEMANTICS_FALLBACK,
  faceLabels = FACE_LABELS_FALLBACK,
  disabled = false,
  onDraftChange,
}: FaceMapEditorProps) {
  const problem = validateFaceMap(draft.faceMap)

  const setFace = (faceId: number, token: string) => {
    const next = { ...draft.faceMap }
    if (token) next[String(faceId)] = token as FaceSemantic
    else delete next[String(faceId)]
    onDraftChange({ ...draft, faceMap: next })
  }

  const setTrim = (index: number, value: number) => {
    const next = [...draft.trim, 0, 0, 0].slice(0, 3)
    next[index] = Number.isFinite(value) ? value : 0
    onDraftChange({ ...draft, trim: next })
  }

  /** 自动指派的面序号（1..6）；用于「这一面后端原本认为是什么」。 */
  const autoOf = (faceId: number): string => {
    if (auto.up_face === faceId) return 'up+'
    if (auto.forward_face === faceId) return 'front+'
    if (auto.left_face === faceId) return 'left+'
    return ''
  }

  return (
    <div className="card">
      <h3>面序号 → 语义轴</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        立方体上标了 1..6，与下表逐面对应。留空表示不指派，第三个轴由右手系导出。
        自动探测：up=面{auto.up_face ?? '-'}、front=面{auto.forward_face ?? '-'}、
        left=面{auto.left_face ?? '-'}（{auto.method}）
      </div>
      <table className="grid">
        <thead>
          <tr>
            <th style={{ width: 34 }}>面</th>
            <th style={{ width: 40 }}>轴</th>
            <th style={{ width: 62 }}>厚度 m</th>
            <th>指派为</th>
          </tr>
        </thead>
        <tbody>
          {faces.map((face) => (
            <tr key={face.id}>
              <td><b>{face.id}</b></td>
              <td className="muted">{faceLabels[face.id - 1] ?? face.extent_face}</td>
              <td className="num">{face.extent.toFixed(3)}</td>
              <td>
                <select
                  value={draft.faceMap[String(face.id)] ?? ''}
                  disabled={disabled}
                  onChange={(e) => setFace(face.id, e.target.value)}
                >
                  <option value="">
                    {autoOf(face.id) ? `（自动 ${autoOf(face.id)}）` : '— 不指派 —'}
                  </option>
                  {semantics.map((token) => (
                    <option key={token} value={token}>{token}</option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {problem ? <div className="note err">{problem}</div> : null}

      <label>盒内微调（度，绕规范系 X / Y / Z 外旋，面映射之后叠加）</label>
      <div className="row">
        {(['X', 'Y', 'Z'] as const).map((axisName, index) => (
          <span key={axisName} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <span className="muted">{axisName}</span>
            <input
              type="number"
              step={0.5}
              value={draft.trim[index] ?? 0}
              disabled={disabled}
              onChange={(e) => setTrim(index, Number(e.target.value))}
            />
          </span>
        ))}
      </div>
      <div className="muted" style={{ marginTop: 4 }}>
        模型本身略歪（例如肩膀不平）时用它微调；只做轴系翻转请用上面的面映射。
      </div>
    </div>
  )
}

export default FaceMapEditor
