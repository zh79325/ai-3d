/**
 * 规范系外切盒：线框 + 六面语义标签 + 按语义轴着色的半透明面。
 *
 * V2 下外切盒是**规范系米制 AABB**（`align.bbox`），已在场景坐标下，六面语义固定：
 * `1=+X left, 2=-X right, 3=+Y up, 4=-Y down, 5=+Z front, 6=-Z back`（与后端
 * `obb.FACE_LABELS` 行序一致）。人工不再逐面指派，颜色与标签直接由面序号决定，
 * 用户只需在左栏二选确认「朝向正确 / 前后相反」。
 *
 * 每个面的标签为「面编号 · 对应语义轴」（如 `3 · +Y up`）：编号让人能按号指认
 * 面，语义轴说明该面朝哪条规范轴。编号 1..6 按规范系固定序，与着色一致。
 */
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'

import type { BboxDict } from '../../api/types'
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

/** 语义类别 → 颜色，与 `styles.css` 里 `.face-tag.up/.front/.left` 同色。 */
const KIND_COLORS: Record<FaceKind, string> = {
  up: '#39d98a',
  front: '#4da3ff',
  left: '#ffb020',
}

/** 后端 `obb.FACE_LABELS` 的兜底值，conventions 拉不到时用。 */
export const FACE_LABELS_FALLBACK = ['+X left', '-X right', '+Y up', '-Y down', '+Z front', '-Z back']

export interface BoundingBoxFacesProps {
  bbox: BboxDict
  /** 六面标签文本（`conventions.face_labels`）；缺省用 `FACE_LABELS_FALLBACK`。 */
  faceLabels?: string[]
  /** 半透明面色块（关掉只看线框）。 */
  showFaces?: boolean
  showLabels?: boolean
  lineColor?: string
}

export function BoundingBoxFaces({
  bbox,
  faceLabels = FACE_LABELS_FALLBACK,
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

  // 标签沿外法向抬一点，避免与面片共面闪烁
  const labelOffset = Math.max(obbRadius(scene) * 0.02, 0.005)
  const labels = showLabels
    ? [1, 2, 3, 4, 5, 6].map((id) => {
      const at = faceCenter(scene, id).addScaledVector(faceNormal(scene, id), labelOffset)
      return { id, at, kind: FACE_KIND[id - 1], text: faceLabels[id - 1] ?? FACE_LABELS_FALLBACK[id - 1] }
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
      {labels.map(({ id, at, kind, text }) => (
        <Html
          key={id}
          position={[at.x, at.y, at.z]}
          center
          zIndexRange={[40, 20]}
          style={{ pointerEvents: 'none' }}
        >
          <div className={`face-tag ${kind}`}><b>{id}</b> · {text}</div>
        </Html>
      ))}
    </group>
  )
}

export default BoundingBoxFaces
