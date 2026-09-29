/**
 * 骨骼线。两种来源，都关 `depthTest` —— 骨骼套在网格内部，开深度测试只能看到
 * 露出的一小截，判断不出骨架是否居中。
 *
 * - `BoneLines`：GLB 自带骨骼（动画素材 / 源模型），移植旧页 `buildBoneLines`。
 *   按名过滤 `ik_/twist/socket/virtual/dummy` 辅助骨：`SkeletonHelper` 会把
 *   `ik_foot_root→脚`、`ik_hand_gun→手` 画成横穿身体的怪线。位置每帧从骨骼世界
 *   矩阵重取，故播放动画时线会跟着动。
 * - `RigBones`：S2 产出的 `rig.json`（22 语义关节），按 `parent → 自己` 连线。
 *   head 已是规范系米制坐标，与 `asset.glb` 同空间，不需要额外变换。
 */
import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import type { RigDoc } from '../../api/types'

/** 辅助骨名匹配（与旧页 `isHelperBone` 同一套正则）。 */
const HELPER_BONE = /ik_|twist|socket|virtual|dummy/i

export function isHelperBone(name?: string | null): boolean {
  return HELPER_BONE.test(name || '')
}

/** 收集「子骨 → 父骨」配对；父或子任一是辅助骨就整条跳过。 */
export function collectBonePairs(root: THREE.Object3D): [THREE.Object3D, THREE.Object3D][] {
  const bones: THREE.Object3D[] = []
  root.traverse((o) => {
    if ((o as THREE.Bone).isBone && !isHelperBone(o.name)) bones.push(o)
  })
  const pairs: [THREE.Object3D, THREE.Object3D][] = []
  for (const bone of bones) {
    const parent = bone.parent
    if (parent && (parent as THREE.Bone).isBone && !isHelperBone(parent.name)) {
      pairs.push([bone, parent])
    }
  }
  return pairs
}

/** rig.json → 线段端点（父在前、子在后），供 `RigBones` 与取景计算共用。 */
export function rigSegments(rig: RigDoc | null): [THREE.Vector3, THREE.Vector3][] {
  if (!rig) return []
  const joints = rig.joints || {}
  const at = (id: string): THREE.Vector3 | null => {
    const head = joints[id]?.head
    return head && head.length === 3
      ? new THREE.Vector3(head[0], head[1], head[2])
      : null
  }
  const out: [THREE.Vector3, THREE.Vector3][] = []
  for (const [id, joint] of Object.entries(joints)) {
    if (!joint.parent) continue
    const parent = at(joint.parent)
    const self = at(id)
    if (parent && self) out.push([parent, self])
  }
  return out
}

export interface BoneLinesProps {
  /** 含骨骼的场景（`GlbModel` 克隆出来的那份）；null 时不渲染。 */
  root: THREE.Object3D | null
  color?: number | string
}

/** GLB 自带骨骼的连线（跟随动画每帧更新）。 */
export function BoneLines({ root, color = 0x39d98a }: BoneLinesProps) {
  const pairs = useMemo(() => (root ? collectBonePairs(root) : []), [root])
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 6), 3))
    return geo
  }, [pairs])
  const material = useMemo(
    () => new THREE.LineBasicMaterial({ color, depthTest: false, depthWrite: false }),
    [color],
  )

  // 两份都是 useMemo 里新建的，卸载时必须手动释放
  useEffect(() => () => {
    geometry.dispose()
    material.dispose()
  }, [geometry, material])

  const cursor = useMemo(() => new THREE.Vector3(), [])
  useFrame(() => {
    if (!pairs.length) return
    const attr = geometry.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < pairs.length; i += 1) {
      pairs[i][0].getWorldPosition(cursor)
      attr.setXYZ(i * 2, cursor.x, cursor.y, cursor.z)
      pairs[i][1].getWorldPosition(cursor)
      attr.setXYZ(i * 2 + 1, cursor.x, cursor.y, cursor.z)
    }
    attr.needsUpdate = true
    // 骨骼会动，包围球必须跟着重算，否则会被视锥误剔除
    geometry.computeBoundingSphere()
  })

  if (!pairs.length) return null
  return <lineSegments geometry={geometry} material={material} renderOrder={999} />
}

export interface RigBonesProps {
  rig: RigDoc | null
  color?: number | string
}

/** S2 语义骨架的连线（静态：只在 rig 变化时重建）。 */
export function RigBones({ rig, color = 0x00e5ff }: RigBonesProps) {
  const geometry = useMemo(() => {
    const points: THREE.Vector3[] = []
    for (const [a, b] of rigSegments(rig)) points.push(a, b)
    return new THREE.BufferGeometry().setFromPoints(points)
  }, [rig])
  const material = useMemo(
    () => new THREE.LineBasicMaterial({ color, depthTest: false, depthWrite: false }),
    [color],
  )

  useEffect(() => () => {
    geometry.dispose()
    material.dispose()
  }, [geometry, material])

  if (!rig) return null
  return (
    <lineSegments
      geometry={geometry}
      material={material}
      renderOrder={999}
      frustumCulled={false}
    />
  )
}

export default BoneLines
