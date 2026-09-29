/**
 * 幽灵材质：把整棵场景树半透明化，好看清内部的外切盒、骨骼与关节点。
 *
 * 两档透明度沿用旧单页 HTML 的取值（0.25 / 0.45）：淡档用于叠加对照，中档用于
 * 「模型还在但让位给骨骼」。材质必须**克隆**后再改 —— `useGLTF` 的缓存是跨实例共享的，
 * 直接改原材质会让别的视口跟着变透明。
 */
import { useLayoutEffect } from 'react'
import * as THREE from 'three'

/** 幽灵档位的透明度取值（与旧 HTML 一致）。 */
export const GHOST_LEVELS = { faint: 0.25, mid: 0.45 } as const

export type GhostLevel = keyof typeof GHOST_LEVELS

interface Touched {
  mesh: THREE.Mesh
  original: THREE.Material | THREE.Material[]
}

/** 把 `root` 下所有网格换成半透明克隆材质；`level` 为 null 时不做任何事。 */
function ghostify(root: THREE.Object3D, level: GhostLevel): Touched[] {
  const opacity = GHOST_LEVELS[level]
  const touched: Touched[] = []
  root.traverse((node) => {
    const mesh = node as THREE.Mesh
    if (!mesh.isMesh) return
    const original = mesh.material
    touched.push({ mesh, original })
    const make = (m: THREE.Material): THREE.Material => {
      const clone = m.clone() as THREE.MeshStandardMaterial
      clone.transparent = true
      clone.opacity = opacity
      // 半透明时关掉深度写入，否则盒子的后半段线框会被模型挡住
      clone.depthWrite = false
      return clone
    }
    mesh.material = Array.isArray(original) ? original.map(make) : make(original as THREE.Material)
  })
  return touched
}

function restore(touched: Touched[]): void {
  for (const { mesh, original } of touched) {
    const current = mesh.material
    for (const m of Array.isArray(current) ? current : [current]) m.dispose()
    mesh.material = original
  }
}

export interface GhostMaterialProps {
  /** 目标场景树（通常是克隆后的 `gltf.scene`）；null 表示模型还没加载。 */
  object: THREE.Object3D | null
  /** null = 不透明。 */
  level: GhostLevel | null
}

/**
 * 挂载即应用、卸载或换档即还原的幽灵材质包装组件。
 *
 * 还原是必须的：`GlbModel` 的克隆体可能被 `useGLTF` 缓存共享，残留的透明材质会
 * 影响下一次加载。
 */
export function GhostMaterial({ object, level }: GhostMaterialProps) {
  useLayoutEffect(() => {
    if (!object || !level) return undefined
    const touched = ghostify(object, level)
    return () => restore(touched)
  }, [object, level])
  return null
}

export default GhostMaterial
