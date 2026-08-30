---
trigger: always_on
---

# 项目基本约束

## 环境
- Python 解释器固定用 `/Users/eleme/Desktop/dudu/server/.venv/bin/python`，不要新建虚拟环境
- 新依赖必须同时写入 `pyproject.toml` 的 `dependencies` 和 `requirements.txt`
- 包源码在 `src/ai3d/`，import 一律用 `from ai3d.xxx import ...`，不写相对上跳路径

## 启动
- viewer 端口固定 `8766`（8765 被占用），改端口要同步 `run_viewer_ide.py`、`scripts/test_server.py`、`scripts/launch_viewer.py`
- 启动前先查端口占用：`lsof -ti:8766 | xargs kill`
- 三个入口用途不要混：`run_viewer_ide.py`（IDE 跑完整流程）、`scripts/test_server.py`（只测前端 UI）、`python -m ai3d.viewer.server`（独立服务）

## 离线要求
- 模型只能从 `./models/` 本地加载，禁止任何联网下载（ultralytics 自动下载也要禁掉）
- 模型路径统一从 `ai3d.config.ModelConfig` 取，不在业务代码里硬编码
- 新增模型文件放 `models/`，并在 `scripts/download_models.py` 登记

## 输出目录
- 产物写 `./output/`（来自 `ExportConfig.output_dir`），临时文件用 `tempfile` 且处理完删除
- 不提交 `output/`、`*.mp4`、`*.pt` 到 git
