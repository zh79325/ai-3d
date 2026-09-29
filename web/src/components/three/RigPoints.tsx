/**
 * 语义关节点（移植旧页 `buildJointDots`）：小球 + 可选拖拽微调。
 *
 * `depthTest:false` + `renderOrder:999` 是必须的 —— 关节在网格内部，开深度测试
 * 只看得到露出的一小截，判断不出骨架是否套在模型中间。
 *
 * 拖拽用「相机正对平面」而不是世界轴平面：正交侧视图下沿 X 拖动会退化成零向量，
 * 而垂直于视线的平面在任何视角下都有解，符合「看着哪个方向就在那个方向上挪」的直觉。
 * 指针捕获挂在 canvas 上（不是 R3F 的事件对象），移动时自己算射线 —— 指针一旦离开
 * 小球，R3F 的 raycast 就不再命中该 mesh，`onPointerMove` 会断流。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useThree, type ThreeEvent } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'

import type { RigDoc } from '../../api/types'

export type Head3 = [number, number, number]

/** 关节来源配色：与 `rig.json` 的 `source` 对应，一眼看出哪些是人工改过的。 */
export const SOURCE_COLORS: Record<string, string> = {
  proportion: '#ffb020',
  solved: '#39d98a',
  manual: '#4da3ff',
  unknown: '#9aa3b2',
}

export interface RigPointsProps {
  rig: RigDoc | null
  /** 小球半径（米）。缺省按 `rig.height` 推，保证大小模型都看得清。 */
  size?: number | null
  /** 统一颜色；给了 `colors` 时以 `colors` 为准。 */
  color?: number | string
  /** 逐关节覆盖色（如按 `source` 上色）。 */
  colors?: Record<string, number | string> | null
  selected?: string | null
  /** 允许拖拽（`locked` 的关节仍不可拖）。 */
  editable?: boolean
  showLabels?: boolean
  /** 拖拽中的实时位置覆盖，提交前只改本地 state。 */
  overrides?: Record<string, Head3> | null
  onSelect?: (id: string | null) => void
  onDrag?: (id: string, head: Head3) => void
  /** 松手时回调：调用方在这里发 `PATCH /rig`。 */
  onDragEnd?: (id: string, head: Head3) => void
}

export function RigPoints({
  rig,
  size = null,
  color = 0xffb300,
  colors = null,
  selected = null,
  editable = false,
  showLabels = false,
  overrides = null,
  onSelect,
  onDrag,
  onDragEnd,
}: RigPointsProps) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const controls = useThree((s) => s.controls) as { enabled: boolean } | null

  const [dragId, setDragId] = useState<string | null>(null)
  const plane = useMemo(() => new THREE.Plane(), [])
  const hit = useMemo(() => new THREE.Vector3(), [])
  const latest = useRef<Head3 | null>(null)

  // 回调存 ref：拖拽监听只在 dragId 变化时挂/摘，别让父组件每次渲染都重挂
  const dragCb = useRef(onDrag)
  const endCb = useRef(onDragEnd)
  useEffect(() => {
    dragCb.current = onDrag
    endCb.current = onDragEnd
  }, [onDrag, onDragEnd])

  const joints = rig?.joints ?? {}
  const radius = size ?? Math.max((rig?.height ?? 1.7) * 0.012, 0.006)
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 14, 14), [])
  useEffect(() => () => geometry.dispose(), [geometry])

  const headOf = (id: string): Head3 | null => {
    const override = overrides?.[id]
    if (override) return override
    const head = joints[id]?.head
    return head && head.length === 3 ? [head[0], head[1], head[2]] : null
  }

  function beginDrag(event: ThreeEvent<PointerEvent>, id: string) {
    event.stopPropagation()
    onSelect?.(id)
    const at = headOf(id)
    if (!editable || !at || joints[id]?.locked) return
    const normal = new THREE.Vector3()
    camera.getWorldDirection(normal)
    plane.setFromNormalAndCoplanarPoint(normal, new THREE.Vector3(...at))
    latest.current = at
    setDragId(id)
    gl.domElement.setPointerCapture(event.pointerId)
  }

  useEffect(() => {
    if (!dragId) return undefined
    const el = gl.domElement
    const raycaster = new THREE.Raycaster()
    const ndc = new THREE.Vector2()

    const project = (ev: PointerEvent): Head3 | null => {
      const rect = el.getBoundingClientRect()
      if (!rect.width || !rect.height) return null
      ndc.set(
        ((ev.clientX - rect.left) / rect.width) * 2 - 1,
        -((ev.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(ndc, camera)
      if (!raycaster.ray.intersectPlane(plane, hit)) return null
      return [hit.x, hit.y, hit.z]
    }
    const move = (ev: PointerEvent) => {
      const at = project(ev)
      if (!at) return
      latest.current = at
      dragCb.current?.(dragId, at)
    }
    const up = (ev: PointerEvent) => {
      const at = project(ev) ?? latest.current
      setDragId(null)
      if (el.hasPointerCapture(ev.pointerId)) el.releasePointerCapture(ev.pointerId)
      if (at) endCb.current?.(dragId, at)
    }

    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
    return () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
    }
  }, [dragId, gl, camera, plane, hit])

  // 拖拽期间必须关掉 OrbitControls，否则挪关节会连带转视角
  useEffect(() => {
    if (!dragId || !controls) return undefined
    const prev = controls.enabled
    controls.enabled = false
    return () => {
      controls.enabled = prev
    }
  }, [dragId, controls])

  if (!rig) return null

  return (
    <group>
      {Object.keys(joints).map((id) => {
        const at = headOf(id)
        if (!at) return null
        const joint = joints[id]
        const on = selected === id || dragId === id
        const scale = radius * (on ? 1.7 : 1)
        return (
          <mesh
            key={id}
            position={at}
            scale={scale}
            geometry={geometry}
            renderOrder={999}
            onPointerDown={(e) => beginDrag(e, id)}
            onPointerOver={(e) => {
              e.stopPropagation()
              if (editable) gl.domElement.style.cursor = joint.locked ? 'not-allowed' : 'grab'
            }}
            onPointerOut={() => {
              gl.domElement.style.cursor = ''
            }}
          >
            <meshBasicMaterial
              color={colors?.[id] ?? (on ? '#ffffff' : color)}
              depthTest={false}
              depthWrite={false}
            />
          </mesh>
        )
      })}
      {showLabels
        ? Object.keys(joints).map((id) => {
          const at = headOf(id)
          if (!at) return null
          return (
            <Html
              key={`label-${id}`}
              position={[at[0], at[1] + radius * 2.2, at[2]]}
              center
              zIndexRange={[30, 10]}
              style={{ pointerEvents: 'none' }}
            >
              <div className="face-tag">{id}</div>
            </Html>
          )
        })
        : null}
    </group>
  )
}

export default RigPoints
