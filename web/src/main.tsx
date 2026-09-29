/**
 * 入口。
 *
 * 用 **HashRouter**：产物由 FastAPI 的 `StaticFiles(html=True)` 挂在 `/app`，
 * 它只对目录请求回落 `index.html`，`/app/asset/xxx/s1` 这种真实路径刷新会 404。
 * hash 路由把层级放在 `#` 之后，服务端永远只看到 `/app/`，无需额外回落配置。
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import { App } from './App'
import './styles.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('找不到 #root 挂载点：请检查 web/index.html')
}

createRoot(container).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
