/**
 * 规范系外切盒：线框 + 六面编号标签 + 按语义轴着色的半透明面。
 *
 * V2 下外切盒是**规范系米制 AABB**（`align.bbox`），已在场景坐标下；六面**编号**
 * 按模型原坐标轴固定（1=+X 2=-X 3=+Y 4=-Y 5=+Z 6=-Z），贴在模型自身的面上、不随
 * 校准旋转改变，人工才能稳定地「按号指认正面」。每个编号面在当前规范系下的外法向
 * 由 `align.face_axes` 下发（`faceAxes`），据此把编号画到对应的盒面上。
 *
 * 标签只显示编号数字；语义方向改由着色表达：上面（绿）、正面（蓝）、左边（橙），
 * 与 `AxisGizmo` 的中文轴向箭头、`styles.css` 的 `.face-tag.up/.front/.left` 同色。
 */
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'

import type { BboxDict, Vec3 } from '../../api/types'
import {
  faceCenter,
  faceCorners,
  faceNormal,
  obbCorners,
  obbEdges,
  obbRadius,
  toSceneObb,
} from './obb'

/** 面序号 1..6 → 语义类别（用于着色与标签样式）。 */
type FaceKind = 'left' | 'up' | 'front'
const FACE_KIND: readonly FaceKind[] = ['left', 'left', 'up', 'up', 'front', 'front']

/** 规范系六面外法向（面序 1..6 = +X -X +Y -Y +Z -Z），与编号同序。 */
const CANON_NORMALS: readonly Vec3[] = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
]

/** 语义类别 → 颜色，与 `styles.css` 里 `.face-tag.up/.front/.left` 同色。 */
const KIND_COLORS: Record<FaceKind, string> = {
  up: '#39d98a',
  front: '#4da3ff',
  left: '#ffb020',
}

export interface BoundingBoxFacesProps {
  bbox: BboxDict
  /** 每个编号面（1..6）在当前规范系下的外法向（`align.face_axes`）。 */
  faceAxes?: Vec3[]
  /** 半透明面色块（关掉只看线框）。 */
  showFaces?: boolean
  showLabels?: boolean
  lineColor?: string
}

export function BoundingBoxFaces({
  bbox,
  faceAxes,
  showFaces = true,
  showLabels = true,
  lineColor = '#7f8b9d',
}: BoundingBoxFacesProps) {
  const scene = useMemo(() => toSceneObb(bbox), [bbox])

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
      color.set(KIND_COLORS[FACE_KIND[id - 1]])
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
  }, [scene])

  // 两份 geometry 都是 useMemo 里新建的，卸载/重算时必须手动释放，否则显存泄漏
  useEffect(() => () => {
    lineGeometry.dispose()
    faceGeometry.dispose()
  }, [lineGeometry, faceGeometry])

  // 标签沿外法向抬一点，避免与面片共面闪烁；编号 = 该规范盒面对应的模型原坐标轴面号
  const labelOffset = Math.max(obbRadius(scene) * 0.02, 0.005)
  const labels = showLabels
    ? [1, 2, 3, 4, 5, 6].map((c) => {
      const n = CANON_NORMALS[c - 1]
      let id = c
      if (faceAxes && faceAxes.length === 6) {
        let bestDot = -Infinity
        faceAxes.forEach((ax, i) => {
          const dot = ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]
          if (dot > bestDot) { bestDot = dot; id = i + 1 }
        })
      }
      const at = faceCenter(scene, c).addScaledVector(faceNormal(scene, c), labelOffset)
      return { id, at, kind: FACE_KIND[c - 1] }
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
      {labels.map(({ id, at, kind }) => (
        <Html
          key={id}
          position={[at.x, at.y, at.z]}
          center
          zIndexRange={[40, 20]}
          style={{ pointerEvents: 'none' }}
        >
          <div className={`face-tag ${kind}`}>{id}</div>
        </Html>
      ))}
    </group>
  )
}

export default BoundingBoxFaces
