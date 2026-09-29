# UniRig CPU/离线 patch 数据（由 scripts/setup_unirig.py 幂等应用）。
#
# 每条 patch：对 vendor 仓库（third_party/UniRig）内某个文件做精确字符串替换。
# 幂等规则：文件已含 new → 跳过；含 old → 替换；两者皆无 → 报错（上游变更）。
#
# 说明：flash_attn / spconv 的 import 行不在此 patch——它们由
# third_party/patches/shims/ 下的纯 torch shim 顶替（运行时 sys.path 注入），
# UniRig 源码的 import 语句零改动。

PATCHES = [
    {
        # bpy 仅 FBX/VRM 导入导出需要；本链路输入 glb、输出 npz，不装 bpy。
        # run.py 顶层 `from src.data.extract import get_files` 会连带 import 本模块，
        # 故 bpy 必须容忍缺失（get_files 为纯文件枚举，不触 bpy）。
        "path": "src/data/extract.py",
        "old": "import bpy, os\n",
        "new": (
            "import os\n"
            "try:\n"
            "    import bpy\n"
            "except ImportError:  # CPU 离线链路不装 bpy，仅 get_files 等纯枚举函数可用\n"
            "    bpy = None\n"
        ),
    },
    {
        # OPT-350m 骨架自回归模型：flash_attention_2 仅 CUDA，
        # transformers 原生 sdpa 在 CPU 等价可用（配置级替换，免改代码）。
        "path": "configs/model/unirig_ar_350m_1024_81920_float32.yaml",
        "old": "_attn_implementation: flash_attention_2",
        "new": "_attn_implementation: sdpa",
    },
    {
        # 蒙皮推理的 voxel_skin 顶点组默认 pyrender backend（需 OpenGL 离屏渲染）；
        # open3d 分支是纯几何体素化，无头 CPU 可用（上游 README 亦推荐此替代）。
        # 必须**替换**原 backend 行而非新增：YAML 重复键后者生效，新增会被原 pyrender 覆盖。
        "path": "configs/transform/inference_skin_transform.yaml",
        "old": "        backend: pyrender # switch to 'open3d' if pyrender does not work",
        "new": "        backend: open3d # CPU 离线链路：pyrender 需 OpenGL，open3d 体素化纯 CPU",
    },
]
