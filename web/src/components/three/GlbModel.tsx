/**
 * GLB 模型加载与显示。
 *
 * - `useGLTF` 按 URL 缓存，故 URL 必须带 `?v=<updated_at>`（见 `api/client.assetGlbUrl`）；
 *   后端虽然发了 `Cache-Control: no-store`，但 S1 反复重算后若 URL 不变，three 的
 *   loader 缓存仍会返回旧几何。
 * - 场景树**先克隆再挂**：`gltf.scene` 是跨组件共享的缓存对象，直接 `<primitive>`
 *   挂两处会互相抢父节点（three 的 add 会把它从原父节点摘走）。克隆用
 *   `SkeletonUtils.clone`，它同时正确复制骨骼与 `SkinnedMesh` 的绑定，
 *   `Object3D.clone` 做不到这点。
 */
import { useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { clone as cloneWithSkeleton } from 'three/examples/jsm/utils/SkeletonUtils.js'

import { GhostMaterial, type GhostLevel } from './GhostMaterial'

/** 加载完成后回报的场景概要（取景与界面提示用）。 */
export interface ModelBounds {
  box: THREE.Box3
  center: [number, number, number]
  radius: number
  meshes: number
  vertices: number
}

export interface GlbModelProps {
  /** 带版本键的 GLB 地址，用 `assetGlbUrl(id, updated_at)` 生成。 */
  url: string
  /** null = 不透明；幽灵档用于看清内部外切盒与骨骼。 */
  ghost?: GhostLevel | null
  visible?: boolean
  /** 场景包围盒变化时回调（调用方需用 useCallback 稳定引用）。 */
  onBounds?: (bounds: ModelBounds) => void
}

export function GlbModel({ url, ghost = null, visible = true, onBounds }: GlbModelProps) {
  const gltf = useGLTF(url)
  const scene = useMemo(
    () => cloneWithSkeleton(gltf.scene) as THREE.Group,
    [gltf.scene],
  )

  useLayoutEffect(() => {
    if (!onBounds) return
    const box = new THREE.Box3().setFromObject(scene)
    if (box.isEmpty()) return
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    let meshes = 0
    let vertices = 0
    scene.traverse((node) => {
      const mesh = node as THREE.Mesh
      if (!mesh.isMesh) return
      meshes += 1
      const pos = mesh.geometry?.getAttribute?.('position')
      if (pos) vertices += pos.count
    })
    onBounds({
      box,
      center: [sphere.center.x, sphere.center.y, sphere.center.z],
      radius: sphere.radius,
      meshes,
      vertices,
    })
    // onBounds 由调用方 useCallback 稳定；scene 变化时才需要重算
  }, [scene, onBounds])

  return (
    <>
      <primitive object={scene} visible={visible} />
      <GhostMaterial object={scene} level={ghost} />
    </>
  )
}

export default GlbModel
