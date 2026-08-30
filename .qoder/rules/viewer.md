---
trigger: glob
glob: "src/ai3d/viewer/**"
---

# Viewer 开发约束

## 后端
- 所有接口挂在 `router = APIRouter(prefix="/api/skeleton-viewer")` 上，不在 `server.py` 里建 `FastAPI()` 实例（独立运行入口除外）
- 新接口必须能在被 `app.include_router(router, prefix=...)` 二次挂载的情况下正常工作：前端请求路径用相对路径，不写死 `/api/skeleton-viewer`
- 请求/响应体用 pydantic `BaseModel` 定义，不用裸 dict
- 长耗时处理走任务机制（`create_task` / `update_task` / `get_task`），接口立即返回 `task_id`，前端轮询进度，不阻塞请求
- 上传文件存 `tempfile` 目录，任务结束或失败都要清理

## 前端
- `index.html` 单文件，不引入构建工具、不拆包
- 三方库（three.js 等）走本地或已在用的 CDN，不新增外部依赖源
- 骨架数据结构变更时，同步改 `src/ai3d/models/skeleton.py` 和 `index.html` 的解析逻辑，两边字段名保持一致

## 调试
- 只验证前端 UI 用 `python scripts/test_server.py`，不要为此跑完整 pipeline
- 改完接口在 README 或 `docs/VIEWER_INTEGRATION.md` 更新接口列表
