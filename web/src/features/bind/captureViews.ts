/**
 * 多视角离屏渲染（移植旧页 `captureViews`）。
 *
 * **相机参数必须与后端 `solve.build_camera` 逐字对齐**：正交、`half = maxDim*0.62`、
 * `dist = maxDim*3`、`dir = (sinθ·cosφ, sinφ, cosθ·cosφ)`、`up = +Y`。后端三角化按
 * `view_spec.json` 重建同一套相机，任何一处偏差都会让重建出来的关节整体错位 ——
 * 在预览里只表现为「骨架略微偏移」，极难排查，所以这里不做任何"优化"。
 *
 * 包围盒取 `Box3.setFromObject(克隆场景)`，对应后端 `glb_io.mesh_bbox(asset.glb)`：
 * 两者都是规范系米制空间下的整体包围盒。
 *
 * 单独开一个 `WebGLRenderer` 而不是复用交互画布：交互画布带地面网格、轨道相机与
 * UI 覆盖物，截下来会把这些一起喂给 DWPose。
 */
import * as THREE from 'three'

import type { ViewSpec } from '../../api/types'

export interface CapturedView {
  view_id: string
  blob: Blob
}

/** 正交取景常量（与 `solve.build_camera` 一致，改一处必须改两处）。 */
const HALF_RATIO = 0.62
const DIST_RATIO = 3.0

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('画布导出 PNG 失败（浏览器拒绝了 toBlob）'))
    }, 'image/png')
  })
}

/**
 * 按 `spec.cameras` 逐个方位渲染，返回 `{view_id, blob}` 列表。
 *
 * `model` 只读不写：内部先 `clone(true)` 再挂到临时场景，调用方传进来的那份
 * （通常是 `useGLTF` 的缓存场景）不会被改动 —— 直接挂到第二个 Scene 会把它
 * 从原父节点摘走。
 */
export async function captureViews(
  spec: ViewSpec,
  model: THREE.Object3D,
  onProgress?: (done: number, total: number, viewId: string) => void,
): Promise<CapturedView[]> {
  const width = spec.width || 512
  const height = spec.height || 512
  const canvas = document.createElement('canvas')
  const renderer = new THREE.WebGLRenderer({
    canvas,
    preserveDrawingBuffer: true,
    antialias: true,
    alpha: false,
  })
  try {
    // 第三参 false：不要把 CSS 尺寸写进 style，离屏画布不在文档流里
    renderer.setSize(width, height, false)
    renderer.setClearColor(new THREE.Color(spec.background || '#ffffff'), 1)

    const scene = new THREE.Scene()
    const obj = model.clone(true)
    obj.updateMatrixWorld(true)
    scene.add(obj)
    // 无阴影的平光：DWPose 只需要轮廓清晰，打侧光反而会在身上画出深色分界线
    scene.add(new THREE.AmbientLight(0xffffff, 1.0))
    const key = new THREE.DirectionalLight(0xffffff, 0.5)
    key.position.set(2, 4, 3)
    scene.add(key)

    const box = new THREE.Box3().setFromObject(obj)
    if (box.isEmpty()) throw new Error('模型包围盒为空，无法取景')
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const half = maxDim * HALF_RATIO
    const dist = maxDim * DIST_RATIO

    const camera = spec.ortho
      ? new THREE.OrthographicCamera(-half, half, half, -half, 0.01, dist * 3)
      : null
    if (!camera) throw new Error('当前只支持正交视角（view_spec.ortho=false 未实现）')
    camera.up.set(0, 1, 0)

    const out: CapturedView[] = []
    const total = spec.cameras.length
    for (let i = 0; i < total; i += 1) {
      const cam = spec.cameras[i]
      const theta = THREE.MathUtils.degToRad(cam.azimuth)
      const phi = THREE.MathUtils.degToRad(cam.elevation || 0)
      const dir = new THREE.Vector3(
        Math.sin(theta) * Math.cos(phi),
        Math.sin(phi),
        Math.cos(theta) * Math.cos(phi),
      )
      camera.position.copy(center).addScaledVector(dir, dist)
      camera.lookAt(center)
      camera.updateProjectionMatrix()
      renderer.render(scene, camera)
      // render 与 toBlob 之间不能插 await：没有 preserveDrawingBuffer 时缓冲会被清掉
      const blob = await toBlob(canvas)
      out.push({ view_id: cam.view_id, blob })
      onProgress?.(i + 1, total, cam.view_id)
    }
    return out
  } finally {
    renderer.dispose()
  }
}

export default captureViews
