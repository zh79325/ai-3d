/**
 * 路由表。
 *
 * S1（导入矫正）与 S2（绑定）都是**素材级**的：产物写 `output/assets/<id>/`，
 * 绑定结果被所有引用该模型的作业复用，所以挂在 `/asset/:assetId/*`。
 * S3（重定向）与 S4（导出）是作业级的（模型 × 动画），在 P4/P5 挂到 `/job/:jobId/*`。
 * `/asset/:assetId` 裸路径重定向到 S1 —— 流程总是从导入矫正开始。
 */
import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { AlignStage } from './features/align/AlignStage'
import { BindStage } from './features/bind/BindStage'
import { AssetLibrary } from './features/library/AssetLibrary'
import { useAssetStore } from './store/assetStore'

export function App() {
  const loadConventions = useAssetStore((s) => s.loadConventions)
  useEffect(() => {
    // 面编号标签、语义轴取值域等固定约定由后端单点下发，启动时拉一次
    void loadConventions()
  }, [loadConventions])

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/library" element={<AssetLibrary />} />
        <Route path="/asset/:assetId" element={<Navigate to="s1" replace />} />
        <Route path="/asset/:assetId/s1" element={<AlignStage />} />
        <Route path="/asset/:assetId/s2" element={<BindStage />} />
        <Route path="*" element={<Navigate to="/library" replace />} />
      </Route>
    </Routes>
  )
}

export default App
