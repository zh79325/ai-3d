/**
 * 路由表。
 *
 * 顶栏两个独立一级菜单：`模型管理`（`/models`）与 `动画管理`（`/animations`），
 * 共用同一套「列表页 `Library` + 工作区 `Workspace`」组件，按 `kind` 参数化。
 *
 * 工作区一气呵成：选文件即自动上传 + 后端同步跑完轴校准 → 同页直接展示 3D 结果与
 * 人工调整面板，无「进入 S1 / 确认 S1」阶段跳转，流程停在工作区。S2 绑定仍是素材级
 * 的独立页 `/asset/:assetId/s2`，由工作区的低调「去 S2 绑定」入口进入。
 *
 * 旧路径 `/library`、`/asset/:assetId(/s1)` 一律重定向到 `/models`（无法从旧 URL
 * 反推 kind，且新流程已合并，统一回模型列表即可）。
 */
import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { BindStage } from './features/bind/BindStage'
import { Library } from './features/library/Library'
import { Workspace } from './features/workspace/Workspace'
import { useAssetStore } from './store/assetStore'

export function App() {
  const loadConventions = useAssetStore((s) => s.loadConventions)
  useEffect(() => {
    // 面编号标签等固定约定由后端单点下发，启动时拉一次
    void loadConventions()
  }, [loadConventions])

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/models" element={<Library kind="model" />} />
        <Route path="/models/:assetId" element={<Workspace kind="model" />} />
        <Route path="/animations" element={<Library kind="animation" />} />
        <Route path="/animations/:assetId" element={<Workspace kind="animation" />} />
        <Route path="/asset/:assetId/s2" element={<BindStage />} />
        {/* 旧路径兼容 */}
        <Route path="/library" element={<Navigate to="/models" replace />} />
        <Route path="/asset/:assetId" element={<Navigate to="/models" replace />} />
        <Route path="/asset/:assetId/s1" element={<Navigate to="/models" replace />} />
        <Route path="/" element={<Navigate to="/models" replace />} />
        <Route path="*" element={<Navigate to="/models" replace />} />
      </Route>
    </Routes>
  )
}

export default App
