/**
 * 最小体积外切盒：线框 + 六面编号标签 + 按语义轴着色的半透明面。
 *
 * 面编号是**前后端共享的固定约定**（`obb.FACE_AXES`）：
 * `1=+axis0, 2=-axis0, 3=+axis1, 4=-axis1, 5=+axis2, 6=-axis2`。
 * 立方体上标的数字必须与 `FaceMapEditor` 下拉里的面序号、以及后端 `align.json`
 * 的 `faces[].id` 完全一致，否则人工指派会指到别的面上去。
 *
 * 传入的 `obb` 是**原始模型坐标**下的（`build_align` 会先复位 canon 节点再测），
 * 而场景里的 `asset.glb` 已带 canon 的 `final.rotation` + `final.scale`，
 * 故必须用同两个参数把盒子搬到场景坐标，见 `three/obb.toSceneObb`。
 */
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'

import type { FaceSemantic, Mat3, ObbDict } from '../../api/types'
import {
  faceCenter,
  faceCorners,
  faceNormal,
  obbCorners,
  obbEdges,
  obbRadius,
  toSceneObb,
} from './obb'

/** 语义轴 → 颜色，与 `styles.css` 里 `.face-tag.up/.front/.left` 同色。 */
export const SEMANTIC_COLORS: Record<FaceSemantic, string> = {
  'up+': '#39d98a',
  'up-': '#39d98a',
  'front+': '#4da3ff',
  'front-': '#4da3ff',
  'left+': '#ffb020',
  'left-': '#ffb020',
}

const UNASSIGNED_COLOR = '#4a5262'

/** 自动探测结果（`align.auto`）里用到的三个面序号。 */
export interface AutoFaces {
  up_face?: number | null
  forward_face?: number | null
  left_face?: number | null
}

export interface BoundingBoxFacesProps {
  obb: ObbDict
  /** canon 旋转（`align.final.rotation`）；null 表示盒子已在场景坐标。 */
  rotation?: Mat3 | null
  /** canon 统一缩放（`align.final.scale`）。 */
  scale?: number
  /** 人工面映射（`align.manual.face_map`）；为空时回落到 `auto`。 */
  faceMap?: Record<string, FaceSemantic> | null
  auto?: AutoFaces | null
  /** 半透明面色块（关掉只看线框）。 */
  showFaces?: boolean
  showLabels?: boolean
  lineColor?: string
}

/** 把人工映射与自动指派合成「面序号 → 语义轴」，人工优先。 */
export function effectiveFaceMap(
  faceMap?: Record<string, FaceSemantic> | null,
  auto?: AutoFaces | null,
): Record<number, FaceSemantic> {
  const out: Record<number, FaceSemantic> = {}
  const manual = Object.entries(faceMap ?? {})
  if (manual.length) {
    for (const [k, v] of manual) {
      const id = Number(k)
      if (Number.isInteger(id) && id >= 1 && id <= 6) out[id] = v
    }
    return out
  }
  if (auto?.up_face) out[auto.up_face] = 'up+'
  if (auto?.forward_face) out[auto.forward_face] = 'front+'
  if (auto?.left_face) out[auto.left_face] = 'left+'
  return out
}

export function BoundingBoxFaces({
  obb,
  rotation = null,
  scale = 1,
  faceMap = null,
  auto = null,
  showFaces = true,
  showLabels = true,
  lineColor = '#7f8b9d',
}: BoundingBoxFacesProps) {
  const scene = useMemo(() => toSceneObb(obb, rotation, scale), [obb, rotation, scale])
  const assigned = useMemo(() => effectiveFaceMap(faceMap, auto), [faceMap, auto])

  const lineGeometry = useMemo(() => {
    const corners = obbCorners(scene)
    const positions: number[] = []
    for (const [a, b] of obbEdges()) {
      positions.push(corners[a].x, corners[a].y, corners[a].z)
      positions.push(corners[b].x, corners[b].y, corners[b].z)
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return geo
  }, [scene])

  const faceGeometry = useMemo(() => {
    const positions: number[] = []
    const colors: number[] = []
    const color = new THREE.Color()
    for (let id = 1; id <= 6; id += 1) {
      const quad = faceCorners(scene, id)
      const semantic = assigned[id]
      color.set(semantic ? SEMANTIC_COLORS[semantic] : UNASSIGNED_COLOR)
      // 两个三角形：0-1-2 与 0-2-3（quad 已按环排好序）
      for (const idx of [0, 1, 2, 0, 2, 3]) {
        const p = quad[idx]
        positions.push(p.x, p.y, p.z)
        colors.push(color.r, color.g, color.b)
      }
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    return geo
  }, [scene, assigned])

  // 两份 geometry 都是 useMemo 里新建的，卸载/重算时必须手动释放，否则显存泄漏
  useEffect(() => () => {
    lineGeometry.dispose()
    faceGeometry.dispose()
  }, [lineGeometry, faceGeometry])

  // 标签沿外法向抬一点，避免与面片共面闪烁
  const labelOffset = Math.max(obbRadius(scene) * 0.02, 0.005)
  const labels = showLabels
    ? [1, 2, 3, 4, 5, 6].map((id) => {
      const at = faceCenter(scene, id).addScaledVector(faceNormal(scene, id), labelOffset)
      const semantic = assigned[id]
      const kind = semantic ? semantic.slice(0, semantic.length - 1) : ''
      return { id, at, semantic, kind }
    })
    : []

  return (
    <group>
      <lineSegments geometry={lineGeometry} renderOrder={2}>
        <lineBasicMaterial color={lineColor} transparent opacity={0.95} depthTest={false} />
      </lineSegments>
      {showFaces ? (
        <mesh geometry={faceGeometry} renderOrder={1}>
          <meshBasicMaterial
            vertexColors
            transparent
            opacity={0.16}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      ) : null}
      {labels.map(({ id, at, semantic, kind }) => (
        <Html
          key={id}
          position={[at.x, at.y, at.z]}
          center
          zIndexRange={[40, 20]}
          style={{ pointerEvents: 'none' }}
        >
          <div className={`face-tag${kind ? ` ${kind}` : ''}`}>
            {id}
            {semantic ? ` ${semantic}` : ''}
          </div>
        </Html>
      ))}
    </group>
  )
}

export default BoundingBoxFaces
