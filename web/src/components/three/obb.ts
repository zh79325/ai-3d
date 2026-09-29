/**
 * OBB 几何helper（three 侧）。
 *
 * 后端 `align.json` 里的 `obb` 是在**原始模型坐标**下测的（`axis_norm.build_align`
 * 开头会 `_reset_canon` 复位旧的 canon 节点），而前端加载的 `asset.glb` 已经带上了
 * canon 节点的 `final.rotation` + `final.scale`。要把外切盒画在归一化后的模型上，
 * 必须先做同一个变换：`center' = s·R·center`、`axes' = R·axes`（单位向量，不受 s 影响）、
 * `extents' = s·extents`。
 *
 * 面编号与棱的生成顺序严格对齐后端 `obb.OBB.corners()/edges()/FACE_AXES`，
 * 这样角点索引与面序号在两侧语义一致。
 */
import * as THREE from 'three'

import type { Mat3, ObbDict } from '../../api/types'

/** 场景坐标系下的外切盒（已应用 canon 变换，可直接画）。 */
export interface SceneObb {
  center: THREE.Vector3
  /** 三轴单位向量，按行；轴 i 与 `extents[i]` 对应。 */
  axes: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  extents: THREE.Vector3
}

/** 面序号 1..6 → (轴索引, 符号)；与后端 `obb.FACE_AXES` 一致，不得重排。 */
export const FACE_AXES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
]

/** 面序号 1..6 的对偶面（1↔2 / 3↔4 / 5↔6）。 */
export const FACE_OPPOSITE: readonly number[] = [2, 1, 4, 3, 6, 5]

function mat3ToMatrix4(rows: Mat3): THREE.Matrix4 {
  // align.final.rotation 的行序是 [left(+X), up(+Y), front(+Z)]，即规范系基向量在
  // 模型坐标下的表示；three 的 Matrix4.set 按行填，直接对应。
  const m = new THREE.Matrix4()
  m.set(
    rows[0][0], rows[0][1], rows[0][2], 0,
    rows[1][0], rows[1][1], rows[1][2], 0,
    rows[2][0], rows[2][1], rows[2][2], 0,
    0, 0, 0, 1,
  )
  return m
}

/** 把 `align.json` 的 obb 变换到场景坐标（canon 的旋转 + 统一缩放）。 */
export function toSceneObb(obb: ObbDict, rotation?: Mat3 | null, scale = 1): SceneObb {
  const axes = obb.axes.map((row) => new THREE.Vector3(row[0], row[1], row[2]).normalize()) as
    [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  let center = new THREE.Vector3(obb.center[0], obb.center[1], obb.center[2])
  const extents = new THREE.Vector3(obb.extents[0], obb.extents[1], obb.extents[2])
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1
  if (rotation) {
    const m = mat3ToMatrix4(rotation)
    center = center.applyMatrix4(m)
    for (const axis of axes) axis.applyMatrix4(m).normalize()
  }
  return { center: center.multiplyScalar(s), axes, extents: extents.multiplyScalar(s) }
}

/**
 * 八个角点，顺序与后端 `OBB.corners()` 一致：符号按 (±x, ±y, ±z) 三重循环，
 * `-1` 在前。
 */
export function obbCorners(obb: SceneObb): THREE.Vector3[] {
  const half = obb.extents.clone().multiplyScalar(0.5)
  const out: THREE.Vector3[] = []
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        out.push(obb.center.clone()
          .addScaledVector(obb.axes[0], sx * half.x)
          .addScaledVector(obb.axes[1], sy * half.y)
          .addScaledVector(obb.axes[2], sz * half.z))
      }
    }
  }
  return out
}

/** 十二条棱的角点索引对（恰差一位符号即共棱），配合 `obbCorners` 画线框。 */
export function obbEdges(): Array<[number, number]> {
  const out: Array<[number, number]> = []
  for (let a = 0; a < 8; a += 1) {
    for (let b = a + 1; b < 8; b += 1) {
      let xor = a ^ b
      let bits = 0
      while (xor) {
        bits += xor & 1
        xor >>= 1
      }
      if (bits === 1) out.push([a, b])
    }
  }
  return out
}

/** 面序号（1..6）的外法向。 */
export function faceNormal(obb: SceneObb, faceId: number): THREE.Vector3 {
  const [axis, sign] = FACE_AXES[faceId - 1]
  return obb.axes[axis].clone().multiplyScalar(sign)
}

/** 面序号（1..6）的中心点，用于贴编号标签。 */
export function faceCenter(obb: SceneObb, faceId: number): THREE.Vector3 {
  const [axis, sign] = FACE_AXES[faceId - 1]
  const half = obb.extents.getComponent(axis) * 0.5
  return obb.center.clone().addScaledVector(obb.axes[axis], sign * half)
}

/** 面的四个角点（绕外法向成环），用于画半透明色块。 */
export function faceCorners(obb: SceneObb, faceId: number): THREE.Vector3[] {
  const [axis, sign] = FACE_AXES[faceId - 1]
  const u = (axis + 1) % 3
  const v = (axis + 2) % 3
  const half = obb.extents.clone().multiplyScalar(0.5)
  const base = obb.center.clone()
    .addScaledVector(obb.axes[axis], sign * half.getComponent(axis))
  const corners: THREE.Vector3[] = []
  for (const su of [-1, 1]) {
    for (const sv of [-1, 1]) {
      corners.push(base.clone()
        .addScaledVector(obb.axes[u], su * half.getComponent(u))
        .addScaledVector(obb.axes[v], sv * half.getComponent(v)))
    }
  }
  // 生成顺序是 (-,-) (+,-) (-,+) (+,+)，画四边形要按环走：0 → 1 → 3 → 2
  return [corners[0], corners[1], corners[3], corners[2]]
}

/** 外切盒的包围半径，用于相机取景。 */
export function obbRadius(obb: SceneObb): number {
  return obb.extents.length() * 0.5
}
