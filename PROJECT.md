# Dex Retargeting — 项目配置文件

## 项目概述

**Dex Retargeting** 是一个将人手运动重定向（retargeting）到机器人灵巧手的优化库，源自 [AnyTeleop](https://yzqin.github.io/anyteleop/) 项目。支持多种机器人手型号和多种重定向算法，可用于遥操作（teleoperation）和离线数据处理（offline imitation learning）。

- **仓库**: https://github.com/dexsuite/dex-retargeting
- **许可证**: MIT
- **当前版本**: v0.5.0
- **Python 版本**: >=3.7, <3.13
- **作者**: Yuzhe Qin (y1qin@ucsd.edu)

---

## 目录结构

```
dex-retargeting/
├── pyproject.toml              # 构建配置 (hatchling)、依赖、工具配置
├── README.md                   # 项目文档
├── CITATION.cff                # 引用信息
├── LICENSE                     # MIT 许可证
├── .editorconfig               # 编辑器格式规范
├── .gitmodules                 # Git 子模块 (已注释，原指向 dex-urdf)
│
├── src/dex_retargeting/        # 核心源码包
│   ├── __init__.py             # 入口，检查 torch 依赖
│   ├── constants.py            # 枚举定义 (RobotName, RetargetingType, HandType)、坐标变换常量
│   ├── retargeting_config.py   # RetargetingConfig 数据类，YAML 配置加载与构建
│   ├── optimizer.py            # 优化器实现 (PositionOptimizer, VectorOptimizer, DexPilotOptimizer)
│   ├── optimizer_utils.py      # 低通滤波器 LPFilter
│   ├── seq_retarget.py         # SeqRetargeting 序列重定向封装
│   ├── robot_wrapper.py        # RobotWrapper (pinocchio 运动学封装)
│   ├── kinematics_adaptor.py   # 运动学适配器 (MimicJointKinematicAdaptor)
│   ├── yourdfpy.py             # URDF 解析工具 (自定义 yourdfpy)
│   └── configs/                # 预置 YAML 配置文件
│       ├── teleop/             # 遥操作配置 (vector / dexpilot)
│       └── offline/            # 离线处理配置 (position)
│
├── tests/                      # 测试
│   ├── test_retargeting_config.py
│   └── test_optimizer.py
│
├── example/                    # 示例代码
│   ├── vector_retargeting/     # 基于视频的向量重定向示例
│   ├── position_retargeting/   # 基于手物体姿态数据集的位置重定向示例
│   └── profiling/              # 性能分析
│
├── assets/                     # 资源文件 (来自 dex-urdf)
│   └── robots/hands/           # 机器人手 URDF 模型
│       ├── ability_hand/
│       ├── inspire_hand/
│       ├── linker_l20_hand/
│       ├── linker_l20a_hand/
│       ├── schunk_hand/
│       └── shadow_hand/
│
└── .github/workflows/          # CI/CD
    ├── test.yml                # 测试工作流
    ├── build_stable.yml        # 稳定版构建
    └── build_nightly.yml       # 每日构建
```

---

## 构建系统与依赖

### 构建后端

使用 **hatchling** 作为构建后端 (`pyproject.toml`)。

### 核心依赖

| 包 | 最低版本 | 用途 |
|---|---|---|
| `numpy` | >=2.0.0 | 数值计算 |
| `pytransform3d` | >=3.5.0 | 3D 变换 (四元数、旋转矩阵) |
| `pin` (pinocchio) | >=3.3.1 | 机器人运动学 |
| `nlopt` | >=2.8.0 | 非线性优化求解器 |
| `anytree` | >=2.12.0 | 树结构 (URDF 解析) |
| `pyyaml` | >=6.0.0 | YAML 配置解析 |
| `lxml` | >=5.2.2 | XML/URDF 解析 |
| `torch` | (隐式) | 损失函数计算与自动微分 (CPU 即可) |

### 可选依赖

- **dev**: `pytest`, `ruff`, `isort`
- **example**: `tyro`, `tqdm`, `opencv-python`, `mediapipe`, `sapien==3.0.0b0`, `loguru`

### 安装

```bash
# 基础安装
pip install dex_retargeting

# 开发安装
pip install -e ".[dev]"

# 含示例依赖
pip install -e ".[example]"
```

---

## 核心架构

### 重定向类型 (RetargetingType)

| 类型 | 场景 | 配置目录 | 说明 |
|---|---|---|---|
| `vector` | 遥操作 | `configs/teleop/` | 基于向量匹配，将人手关键点间的向量映射到机器人手 |
| `dexpilot` | 遥操作 | `configs/teleop/` | 基于 DexPilot 论文，带指尖接近投影先验 |
| `position` | 离线处理 | `configs/offline/` | 基于 3D 位置匹配，支持 6D 自由关节 |

### 支持的机器人手 (RobotName)

| 枚举值 | 配置前缀 | 说明 |
|---|---|---|
| `allegro` | `allegro_hand` | Allegro Hand (4 指) |
| `shadow` | `shadow_hand` | Shadow Hand (5 指) |
| `svh` | `schunk_svh_hand` | Schunk SVH Hand |
| `leap` | `leap_hand` | LEAP Hand |
| `ability` | `ability_hand` | Ability Hand |
| `inspire` | `inspire_hand` | Inspire Hand |
| `panda` | `panda_gripper` | Panda Gripper (2 指夹爪) |
| `linker_l20` | `linker_l20_hand` | Linker L20 Hand (5 指，含 mimic 关节) |
| `linker_l20a` | `linker_l20a_hand` | Linker L20A Hand |

### 手型 (HandType)

- `right` — 右手
- `left` — 左手
- 夹爪类型 (如 `panda_gripper`) 不区分左右手

### 核心类关系

```
RetargetingConfig          # 配置数据类，从 YAML 加载
    ├── .load_from_file()  # 从 YAML 文件加载配置
    ├── .from_dict()       # 从字典创建配置
    └── .build()           # 构建 SeqRetargeting 实例
         │
         ├── RobotWrapper          # pinocchio 运动学封装
         ├── Optimizer (抽象基类)
         │   ├── VectorOptimizer   # 向量重定向优化器
         │   ├── PositionOptimizer # 位置重定向优化器
         │   └── DexPilotOptimizer # DexPilot 优化器
         ├── MimicJointKinematicAdaptor  # mimic 关节适配
         ├── LPFilter              # 低通滤波器
         └── SeqRetargeting        # 序列重定向封装 (最终用户接口)
```

---

## YAML 配置文件格式

所有配置文件位于 `src/dex_retargeting/configs/` 下，顶层键为 `retargeting`。

### Vector 类型配置 (遥操作)

```yaml
retargeting:
  type: vector
  urdf_path: allegro_hand/allegro_hand_right.urdf  # 相对于 URDF 默认目录

  # 机器人手上的向量定义：origin -> task
  target_origin_link_names: ["wrist", "wrist", "wrist", "wrist"]
  target_task_link_names: ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]

  # 可选：显式指定优化的关节名称 (默认使用所有 DOF 关节)
  target_joint_names: null

  # 缩放因子 (机器人手与人手的尺寸比)
  scaling_factor: 1.6
  # 可选：每个向量独立的缩放因子 (覆盖 scaling_factor)
  # vector_scaling_factors: [1.2, 1.2, 1.2, 1.2]

  # 人手关键点索引映射 (2×N 数组: [origin_indices, task_indices])
  target_link_human_indices: [[0, 0, 0, 0], [4, 8, 12, 16]]

  # 低通滤波 alpha (0=不动, 1=无滤波, 越小越平滑但延迟越大)
  low_pass_alpha: 0.2
```

### DexPilot 类型配置 (遥操作)

```yaml
retargeting:
  type: DexPilot
  urdf_path: allegro_hand/allegro_hand_right.urdf

  wrist_link_name: "wrist"
  finger_tip_link_names: ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
  scaling_factor: 1.6

  # DexPilot 特有参数
  # project_dist: 0.03   # 投影距离阈值
  # escape_dist: 0.05    # 逃逸距离阈值

  low_pass_alpha: 0.2
```

### Position 类型配置 (离线处理)

```yaml
retargeting:
  type: position
  urdf_path: allegro_hand/allegro_hand_right.urdf

  target_joint_names: null
  target_link_names: ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip",
                      "link_14.0", "link_2.0", "link_6.0", "link_10.0"]

  # 人手关键点索引 (1×N 数组)
  target_link_human_indices: [4, 8, 12, 16, 2, 6, 10, 14]

  # 启用 6D 自由关节 (手可在空间中自由移动)
  add_dummy_free_joint: true

  low_pass_alpha: 1  # 离线处理通常不需要滤波
```

### 全部可配置参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | str | (必填) | `vector` / `position` / `dexpilot` |
| `urdf_path` | str | (必填) | URDF 文件路径 (相对或绝对) |
| `add_dummy_free_joint` | bool | `false` | 是否添加 6D 自由关节 |
| `target_link_human_indices` | array | `null` | 人手关键点索引映射 |
| `wrist_link_name` | str | `null` | 腕部链接名 (DexPilot 必填) |
| `target_link_names` | list[str] | `null` | 目标链接名 (Position 必填) |
| `target_joint_names` | list[str] | `null` | 优化关节名 (默认全部 DOF) |
| `target_origin_link_names` | list[str] | `null` | 向量起点链接名 (Vector 必填) |
| `target_task_link_names` | list[str] | `null` | 向量终点链接名 (Vector 必填) |
| `finger_tip_link_names` | list[str] | `null` | 指尖链接名 (DexPilot 必填) |
| `scaling_factor` | float | `1.0` | 全局缩放因子 |
| `vector_scaling_factors` | array | `null` | 每向量独立缩放 |
| `normal_delta` | float | `4e-3` | 正则化权重 |
| `huber_delta` | float | `2e-2` | Huber 损失阈值 |
| `project_dist` | float | `0.03` | DexPilot 投影距离 |
| `escape_dist` | float | `0.05` | DexPilot 逃逸距离 |
| `has_joint_limits` | bool | `true` | 是否启用关节限位 |
| `ignore_mimic_joint` | bool | `false` | 是否忽略 mimic 关节约束 |
| `low_pass_alpha` | float | `0.1` | 低通滤波系数 |

---

## 人手关键点索引约定

人手关键点遵循 MediaPipe Hand Landmarks 的 21 点约定：

```
 0: Wrist
 1: Thumb CMC       5: Index MCP       9: Middle MCP     13: Ring MCP      17: Pinky MCP
 2: Thumb MCP       6: Index PIP      10: Middle PIP     14: Ring PIP      18: Pinky PIP
 3: Thumb IP        7: Index DIP      11: Middle DIP     15: Ring DIP      19: Pinky DIP
 4: Thumb Tip       8: Index Tip      12: Middle Tip     16: Ring Tip      20: Pinky Tip
```

- **Vector 类型**: `target_link_human_indices` 为 `2×N` 数组，第一行为向量起点索引，第二行为终点索引
- **Position 类型**: `target_link_human_indices` 为 `1×N` 数组，直接对应目标链接
- **DexPilot 类型**: 默认自动生成索引 (基于指尖数量)，通常无需手动指定

---

## Mimic 关节

部分机器人手 (如 Linker L20) 具有 mimic 关节约束，即某些关节的运动由其他关节驱动：

```
# Linker L20 mimic 关节
thumb_joint4  mimic thumb_joint3
index_joint3  mimic index_joint2
middle_joint3 mimic middle_joint2
ring_joint3   mimic ring_joint2
little_joint3 mimic little_joint2
```

- 在 URDF 中通过 `<mimic>` 标签定义
- `retargeting_config.py` 中的 `parse_mimic_joint()` 自动解析
- `MimicJointKinematicAdaptor` 在前向运动学和反向雅可比中处理 mimic 约束
- 配置中需通过 `target_joint_names` 显式排除 mimic 关节
- 设置 `ignore_mimic_joint: true` 可跳过 mimic 约束处理

---

## 开发指南

### 代码风格

- **格式化**: [Ruff](https://docs.astral.sh/ruff/) (行宽 88)
- **导入排序**: Ruff isort (combine-as-imports, known-first-party: `dex_retargeting`)
- **缩进**: Python 4 空格, YAML/XML 2 空格 (`.editorconfig`)
- **换行**: Unix LF

### 常用命令

```bash
# 运行测试
pytest tests/

# 代码检查
ruff check .

# 格式化
ruff format .

# 导入排序
ruff check --select I --fix .
```

### 测试

- 框架: **pytest**
- 配置: `pyproject.toml` → `[tool.pytest.ini_options]`
- 测试目录: `tests/`
- 选项: `--disable-warnings`

### CI/CD

- `.github/workflows/test.yml` — 运行测试
- `.github/workflows/build_stable.yml` — 稳定版构建发布
- `.github/workflows/build_nightly.yml` — 每日构建

---

## 典型使用流程

### 1. 遥操作 (Teleoperation)

```python
from dex_retargeting.retargeting_config import RetargetingConfig

# 加载配置并构建重定向器
config = RetargetingConfig.load_from_file("path/to/teleop/allegro_hand_right.yml")
RetargetingConfig.set_default_urdf_dir("path/to/assets/robots/hands")
retargeting = config.build()

# 每帧调用
robot_qpos = retargeting.retarget(hand_keypoints_3d)
```

### 2. 离线数据处理

```python
config = RetargetingConfig.load_from_file("path/to/offline/allegro_hand_right.yml")
RetargetingConfig.set_default_urdf_dir("path/to/assets/robots/hands")
retargeting = config.build()

# 可选：初始化腕部姿态
retargeting.warm_start(wrist_pos, wrist_quat)

# 逐帧处理
for frame in dataset:
    robot_qpos = retargeting.retarget(frame.hand_positions)
```

### 3. 关节顺序映射

不同库 (ROS, SAPIEN, 物理仿真器) 解析 URDF 的关节顺序可能不同，需通过关节名称显式映射：

```python
retargeting_joint_names = retargeting.joint_names
target_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
index_map = np.array([retargeting_joint_names.index(name) for name in target_joint_names])
robot.set_qpos(retarget_qpos[index_map])
```

---

## 添加新机器人手

1. 准备 URDF 模型，放入 `assets/robots/hands/<robot_name>/`
2. 在 `constants.py` 中添加 `RobotName` 枚举和 `ROBOT_NAME_MAP` 映射
3. 创建 YAML 配置文件：
   - `configs/teleop/<robot_name>_<hand_type>.yml` (vector)
   - `configs/teleop/<robot_name>_<hand_type>_dexpilot.yml` (dexpilot)
   - `configs/offline/<robot_name>_<hand_type>.yml` (position)
4. 确定关键参数：
   - 链接名称 (从 URDF 中获取)
   - `scaling_factor` (机器人手与人手的尺寸比)
   - `target_link_human_indices` (人手关键点到机器人链接的映射)
5. 如有 mimic 关节，在 URDF 中定义 `<mimic>` 标签，并在配置中显式指定 `target_joint_names`

---

## 注意事项

- `torch` 是隐式依赖 (在 `__init__.py` 中检查)，CPU 版本即可满足需求
- `numpy >= 2.0.0` 从 v0.5.0 开始要求；`mediapipe` 虽声明依赖 numpy 1.x 但兼容 2.x
- URDF 路径可以是相对路径，会相对于 `RetargetingConfig._DEFAULT_URDF_DIR` 解析
- `assets/` 目录原为 `dex-urdf` 子模块，现已内联；`.gitignore` 中仅保留 `linker_l20_hand` 和 `linker_l20a_hand`
- 优化器使用 **NLopt SLSQP** 算法，梯度通过 pinocchio 雅可比 + torch 自动微分计算
