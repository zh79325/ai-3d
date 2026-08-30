---
trigger: model_decision
description: 写测试、跑测试、验证改动是否可用时遵循的规则
---

# 测试与验证

## 位置与命名
- 测试全部放 `tests/`，文件名 `test_*.py`，函数名 `test_<被测行为>`
- 用 pytest，不用 unittest 的 class 写法

## 依赖隔离
- 不在测试里做真实模型推理：mock `core/pose_estimator.py`、`core/depth_estimator.py` 的推理调用
- 不依赖 `examples/videoplayback.mp4` 等大文件；需要视频时用 numpy 造几帧假数据
- 测试不能联网、不能写 `output/` 之外的目录，用 `tmp_path` fixture

## 运行
```bash
/Users/eleme/Desktop/dudu/server/.venv/bin/python -m pytest tests/ -v
```

## 验证标准
- 改完代码必须实际跑一次相关测试或启动脚本，把真实输出贴出来；不允许只说"应该可以"
- 失败就报失败并附错误输出，不掩盖、不跳过
- 涉及 viewer 的改动，用 `scripts/test_server.py` 起服务确认接口返回正常
