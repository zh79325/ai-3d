/**
 * 外切盒几何 helper（three 侧）。
 *
 * 后端 `align.json` 里的 `bbox` 是**规范系米制 AABB**，且已在场景坐标下
 * （`axis_norm.build_align` 先把顶点旋到规范朝向再乘 scale，与带 canon 节点的
 * `asset.glb` 同坐标系），三轴恒为单位阵。故前端直接画，不再需要 V1 的
 * `toSceneObb(rotation, scale)` 二次变换。
 *
 * 面编号与棱的生成顺序严格对齐后端 `obb.OBB.corners()/edges()/FACE_LABELS`：
 * `1=+X left, 2=-X right, 3=+Y up, 4=-Y down, 5=+Z front, 6=-Z back`。
 */
import * as THREE from 'three'

import type { BboxDict } from '../../api/types'

/** 场景坐标系下的外切盒（三轴恒为单位阵，可直接画）。 */
export interface SceneObb {
  center: THREE.Vector3
  /** 三轴单位向量，按行；轴 i 与 `extents[i]` 对应（规范系下恒为 XYZ）。 */
  axes: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  extents: THREE.Vector3
}

/** 面序号 1..6 → (轴索引, 符号)；与后端 `obb.FACE_LABELS` 行序一致，不得重排。 */
export const FACE_AXES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
]

const IDENTITY_AXES: [THREE.Vector3, THREE.Vector3, THREE.Vector3] = [
  new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1),
]

/** 把 `align.json` 的 bbox 包成可直接绘制的 SceneObb（轴恒为单位阵）。 */
export function toSceneObb(bbox: BboxDict): SceneObb {
  return {
    center: new THREE.Vector3(bbox.center[0], bbox.center[1], bbox.center[2]),
    axes: IDENTITY_AXES.map((a) => a.clone()) as [THREE.Vector3, THREE.Vector3, THREE.Vector3],
    extents: new THREE.Vector3(bbox.extents[0], bbox.extents[1], bbox.extents[2]),
  }
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
