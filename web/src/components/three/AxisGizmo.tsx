/**
 * 规范系坐标轴指示器：+Y up（绿）、+Z front（蓝，角色朝向）、+X left（橙，角色左侧）。
 *
 * 颜色与 `BoundingBoxFaces` 的语义轴着色、`styles.css` 的 `.face-tag.up/.front/.left`
 * 一致，看箭头就能判断模型是否已经转正 —— S1 人工修正时这是最直接的目视参照。
 *
 * 轴向语义来自后端 `GET /v2/conventions` 的 `canonical_axes`，可由 `labels` 覆盖；
 * 拉不到时用这里的默认值（右手系 +Y up / +Z front / +X left，与
 * `axis_norm._SEMANTIC_ROWS` 的行序一致）。
 */
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'

export interface AxisLabels {
  up: string
  front: string
  left: string
}

const DEFAULT_LABELS: AxisLabels = { up: '+Y', front: '+Z', left: '+X' }

interface AxisSpec {
  key: keyof AxisLabels
  name: string
  dir: THREE.Vector3
  color: string
}

const AXES: AxisSpec[] = [
  { key: 'up', name: 'up', dir: new THREE.Vector3(0, 1, 0), color: '#39d98a' },
  { key: 'front', name: 'front', dir: new THREE.Vector3(0, 0, 1), color: '#4da3ff' },
  { key: 'left', name: 'left', dir: new THREE.Vector3(1, 0, 0), color: '#ffb020' },
]

// 默认值必须是模块常量：写成内联数组字面量会每次渲染都产生新引用，
// 把下面的 useMemo 打穿成「每帧重建箭头」。
const ORIGIN: [number, number, number] = [0, 0, 0]

export interface AxisGizmoProps {
  /** 箭头长度（米）；一般取外切盒最长边的一半左右。 */
  length?: number
  /** 箭尾起点；调用方传内联数组时要自己 useMemo，否则会每帧重建。 */
  origin?: [number, number, number]
  labels?: AxisLabels | null
  showLabels?: boolean
}

export function AxisGizmo({
  length = 0.5,
  origin = ORIGIN,
  labels = null,
  showLabels = true,
}: AxisGizmoProps) {
  const text = { ...DEFAULT_LABELS, ...(labels ?? {}) }
  const from = useMemo(() => new THREE.Vector3(...origin), [origin])
  const arrows = useMemo(
    () => AXES.map((axis) => new THREE.ArrowHelper(
      axis.dir, from, length, axis.color, length * 0.22, length * 0.12)),
    [from, length],
  )

  useEffect(() => () => {
    for (const arrow of arrows) arrow.dispose?.()
  }, [arrows])

  return (
    <group>
      {arrows.map((arrow) => <primitive key={arrow.uuid} object={arrow} />)}
      {showLabels ? AXES.map((axis) => {
        const at = from.clone().addScaledVector(axis.dir, length * 1.18)
        return (
          <Html
            key={axis.key}
            position={[at.x, at.y, at.z]}
            center
            zIndexRange={[30, 10]}
            style={{ pointerEvents: 'none' }}
          >
            <div className="face-tag" style={{ background: axis.color, color: '#08111f' }}>
              {text[axis.key]} {axis.name}
            </div>
          </Html>
        )
      }) : null}
    </group>
  )
}

export default AxisGizmo
