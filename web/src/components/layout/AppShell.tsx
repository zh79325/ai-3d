/**
 * 应用外壳：48px 固定顶栏 + 两列主区（左栏 / 右主区）。
 *
 * 顶栏高度沿用 `viewer/server.py` 注入 `#__appnav` 的 48px 约定，React 页与旧页
 * 来回跳转时布局不跳动。顶栏同时承担**全局导航**（素材库 / 作业）与旧页外链，
 * 阶段导航属于单个素材，放在页面左栏里。
 *
 * 子路由自己返回 `<aside className="side">` + `<section className="main">` 两个兄弟
 * 节点铺满 `.shell` 网格 —— 素材库是「列表 + 预览」，S1 页是「矫正面板 + 3D 视口」，
 * 两列语义一致，故不在外壳里写死侧栏内容。
 */
import { NavLink, Outlet } from 'react-router-dom'

import { useAssetStore } from '../../store/assetStore'

/** 顶栏右侧的旧页外链：过渡期新旧并存，从这里回得去旧界面。 */
const LEGACY_LINKS: ReadonlyArray<{ href: string; label: string }> = [
  { href: '/retarget', label: '🦴 旧版动画迁移' },
  { href: '/api/skeleton-viewer/viewer', label: '🎬 骨骼查看器' },
  { href: '/', label: '🧊 功能菜单' },
]

export function AppShell() {
  const busy = useAssetStore((s) => s.busy)
  const error = useAssetStore((s) => s.error)
  return (
    <>
      <nav className="nav">
        <span className="brand">🧊 AI 3D 工具集</span>
        <NavLink to="/library" className={({ isActive }) => `tab${isActive ? ' on' : ''}`}>
          📦 素材库
        </NavLink>
        {/* 作业（模型 × 动画，S2~S4）路由在 P4 接入，先占位不可点 */}
        <span className="tab" title="S3 重定向作业列表在 P4 接入" style={{ opacity: .45 }}>
          🎬 作业
        </span>
        <span className="spacer" />
        <span className="hint">
          {busy ? <><span className="spin" /> 处理中</> : error ? `⚠ ${error}` : ''}
        </span>
        {LEGACY_LINKS.map((link) => (
          <a key={link.href} className="tab" href={link.href}>{link.label}</a>
        ))}
      </nav>
      <div className="shell">
        <Outlet />
      </div>
    </>
  )
}
