/**
 * 通用 3D 视口：R3F Canvas + 灯光 + 地面网格 + 轨道控制 + 自动取景。
 *
 * 所有阶段页与骨骼查看器共用。**尺寸与 DPR 全权交给 R3F** —— 旧单页 HTML 里手写的
 * `renderer.setSize(w, h, false)` 会在 HiDPI 屏上把画布缩成 1/2（CSS 尺寸与缓冲区尺寸
 * 脱钩），R3F 内部按容器 ResizeObserver + `dpr` 处理，不要再插手。
 *
 * `gl.preserveDrawingBuffer` 打开：S2 要离屏渲染多视角图回传后端（P3），
 * 关闭时 `readPixels`/`toDataURL` 会拿到空帧。
 */
import { Suspense, useEffect, useRef, type ReactNode } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { Grid, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

/** 取景目标：把包围球放到视野中央。 */
export interface ViewFocus {
  center: [number, number, number]
  /** 包围半径（米）。 */
  radius: number
}

export interface ViewerCanvasProps {
  children?: ReactNode
  /** 给定时自动取景；值实质变化（量化后不同）才重取景，避免打断用户手动旋转。 */
  focus?: ViewFocus | null
  background?: string
  /** 地面参考网格。 */
  grid?: boolean
  className?: string
  /** 首次取景前的默认相机位置。 */
  cameraPosition?: [number, number, number]
  fov?: number
}

/** 重取景：保留当前观察方向，只改目标点与距离，这样用户转好的角度不会被重置。 */
function FrameCamera({ focus }: { focus: ViewFocus }) {
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls) as { target: THREE.Vector3; update: () => void } | null
  const applied = useRef('')

  useEffect(() => {
    // 量化后比较：align 重算会带来 1e-9 级的浮点抖动，不该触发重取景
    const key = `${focus.center.map((v) => v.toFixed(4)).join(',')}/${focus.radius.toFixed(4)}`
    if (key === applied.current) return
    applied.current = key

    const center = new THREE.Vector3(...focus.center)
    const radius = Math.max(focus.radius, 1e-3)
    const dir = camera.position.clone().sub(center)
    if (dir.lengthSq() < 1e-9) dir.set(0.85, 0.45, 1.2)
    dir.normalize()
    const dist = radius * 2.8
    camera.position.copy(center).addScaledVector(dir, dist)
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = Math.max(dist / 200, 1e-4)
      camera.far = dist * 40
      camera.updateProjectionMatrix()
    }
    camera.lookAt(center)
    if (controls) {
      controls.target.copy(center)
      controls.update()
    }
  }, [focus, camera, controls])

  return null
}

export function ViewerCanvas({
  children,
  focus = null,
  background = '#14161a',
  grid = true,
  className,
  cameraPosition = [1.6, 1.4, 2.2],
  fov = 42,
}: ViewerCanvasProps) {
  return (
    <div className={className ?? 'viewport'}>
      <Canvas
        dpr={[1, 2]}
        camera={{ position: cameraPosition, fov, near: 0.01, far: 200 }}
        gl={{ antialias: true, preserveDrawingBuffer: true, alpha: false }}
      >
        <color attach="background" args={[background]} />
        <hemisphereLight args={[0xffffff, 0x35404f, 0.55]} />
        <directionalLight position={[3, 6, 4]} intensity={1.7} />
        <directionalLight position={[-4, 2, -3]} intensity={0.5} />
        {grid ? (
          <Grid
            position={[0, 0, 0]}
            args={[10, 10]}
            cellSize={0.1}
            cellThickness={0.6}
            cellColor="#2b3240"
            sectionSize={1}
            sectionThickness={1}
            sectionColor="#3b4759"
            fadeDistance={26}
            fadeStrength={1.2}
            infiniteGrid
          />
        ) : null}
        {focus ? <FrameCamera focus={focus} /> : null}
        <Suspense fallback={null}>{children}</Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
      </Canvas>
    </div>
  )
}

export default ViewerCanvas
