#!/usr/bin/env python3
"""IDE-friendly launcher —— 3D 骨骼动画迁移工具 (retarget)。

在 IDE 中直接运行此文件即可启动服务（前端页面 + /v1 接口）。

前置条件:
- Python 解释器: ~/Desktop/ai-game/ai-3d/.venv/bin/python
- 端口默认 8790（可在 src/ai3d/config/retarget.yaml 或 RETARGET_SERVER_PORT 修改）
"""

import os
import sys

# 确保项目根与 src 在路径中（未安装时也能 import ai3d）
project_root = os.path.dirname(os.path.abspath(__file__))
for p in (project_root, os.path.join(project_root, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai3d.retarget.server import main  # noqa: E402

if __name__ == "__main__":
    main()
