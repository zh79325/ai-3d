---
trigger: glob
glob: "src/ai3d/**/*.py"
---

# Python 编码规范

## 格式
- black，`line-length = 100`，`target-version = py39`
- 语法不能超过 Python 3.9（`requires-python = ">=3.9"`）：不用 `match`、不用 `X | Y` 类型写法，用 `Optional[X]` / `Union[X, Y]`

## 结构
- 新增模块必须在同级 `__init__.py` 里显式导出，并加进 `__all__`
- 配置项一律加到 `src/ai3d/config.py` 的对应 dataclass，同步更新 `Config.save_to_yaml` 的字典
- 分层不要串味：`core/` 只做模型推理，`pipeline/` 做流程编排，`models/` 只放数据结构，`utils/` 无业务依赖，`viewer/` 不直接调 YOLO

## 代码风格
- 所有公开函数写类型注解 + 中文 docstring（一句话说清用途）
- 日志用 `logging`，禁止 `print`（`scripts/` 下的启动脚本除外）
- import 写在文件顶部，不要函数内 `import`（循环依赖场景才允许，且加注释说明）
- numpy 数组运算优先向量化，禁止逐帧 Python 循环做逐点计算

## 异常
- 捕获异常要具体类型，禁止裸 `except:` 和 `except Exception: pass`
- 文件/模型加载失败要抛带路径信息的异常，不静默返回 None
