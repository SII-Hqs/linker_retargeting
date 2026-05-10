# visualize_hand_object.py 如何使用 DexYCB 数据集

## 1. 概述

`visualize_hand_object.py` 的功能是：从 DexYCB 数据集中读取人手抓取物体的轨迹数据，
在 SAPIEN 物理仿真引擎中进行 3D 可视化渲染，同时可选地将人手动作 retarget 到机器人灵巧手上。

**核心数据流：**

```
dexYCB 数据集文件
       │
       ▼
 DexYCBVideoDataset (dataset.py)    ← 数据加载与组织
       │
       ▼
 visualize_hand_object.py           ← 入口脚本，选择数据、创建 viewer
       │
       ├──► HandDatasetSAPIENViewer (hand_viewer.py)         ← 纯人手可视化
       │        └── MANOLayer (mano_layer.py → manopth)      ← MANO 手部模型前向推理
       │
       └──► RobotHandDatasetSAPIENViewer (hand_robot_viewer.py) ← 人手 + 机器人手可视化
                ├── MANOLayer                                    ← 同上
                └── SeqRetargeting (dex_retargeting)             ← 人手→机器人 retarget
```

---

## 2. DexYCB 数据集目录结构

```
dexYCB/
├── calibration/                          ← 标定数据
│   ├── intrinsics/                       ← 相机内参
│   │   ├── 836212060125_640x480.yml      ← 每个相机序列号一个文件
│   │   ├── 839512060362_640x480.yml
│   │   └── ...
│   ├── extrinsics_20200702_151821/       ← 相机外参（按标定时间命名）
│   │   └── extrinsics.yml                ← 包含各相机的外参 + apriltag 变换
│   ├── extrinsics_20200813_100608/
│   │   └── extrinsics.yml
│   ├── mano_20200709_140042_subject-01_right/  ← MANO 手部形状参数
│   │   └── mano.yml                            ← 包含 betas (10维形状系数)
│   └── ...
│
├── models/                               ← YCB 物体 3D 模型
│   ├── 002_master_chef_can/
│   │   ├── textured_simple.obj           ← ★ 脚本实际加载的简化网格
│   │   ├── texture_map.png               ← 纹理贴图
│   │   └── ...
│   ├── 003_cracker_box/
│   └── ... (共 21 类 YCB 物体)
│
└── 20200709-subject-01/                  ← 受试者数据（按日期-编号命名）
    ├── 20200709_141754/                  ← 一次抓取录制（capture）
    │   ├── meta.yml                      ← ★ 元数据（关联标定、物体、手型等）
    │   ├── pose.npz                      ← ★ 逐帧位姿数据
    │   ├── 836212060125/                 ← 各相机的图像数据（本脚本未使用）
    │   └── ...
    ├── 20200709_141841/
    └── ...
```

---

## 3. 各数据文件的内容与格式

### 3.1 `meta.yml` — 每次抓取的元数据

```yaml
serials:                          # 参与录制的相机序列号列表
  - '836212060125'
  - '839512060362'
  - ...
num_frames: 72                    # 总帧数
extrinsics: '20200702_151821'     # 引用的外参标定名称 → 对应 calibration/extrinsics_xxx/
ycb_ids: [3, 5, 16]              # 场景中的 YCB 物体 ID 列表（对应 YCB_CLASSES 字典）
ycb_grasp_ind: 0                  # 被抓取的物体在 ycb_ids 中的索引
mano_sides:                       # 手的类型
  - 'right'                       #   （'right' 或 'left'，决定是否被加载）
mano_calib:                       # 引用的 MANO 标定名称
  - '20200709_140042_subject-01_right'  # → 对应 calibration/mano_xxx/mano.yml
```

**脚本如何使用：**
- `mano_sides` → 过滤：只加载包含目标手型（默认 "right"）的 capture
- `extrinsics` → 查找对应的外参标定文件，获取 apriltag 变换矩阵
- `ycb_ids` → 确定场景中有哪些物体，加载对应的 3D 模型
- `mano_calib` → 查找对应的 MANO 形状参数

### 3.2 `pose.npz` — 逐帧位姿数据

| 键名 | 形状 | 含义 |
|------|------|------|
| `pose_m` | `(N, 1, 51)` | MANO 手部参数：前 48 维为姿态 PCA 系数，后 3 维为平移 |
| `pose_y` | `(N, K, 7)` | YCB 物体位姿：前 4 维为旋转四元数 `[qx,qy,qz,qw]`，后 3 维为平移 |

- `N` = 帧数（与 `meta.yml` 中的 `num_frames` 一致）
- `K` = 场景中物体数量（与 `meta.yml` 中 `ycb_ids` 的长度一致）

### 3.3 `calibration/extrinsics_xxx/extrinsics.yml` — 相机外参

```yaml
extrinsics:
  '836212060125': !!python/tuple    # 各相机的 3x4 外参矩阵（按行展开为 12 个浮点数）
    - -0.950...
    - ...
  apriltag: !!python/tuple          # ★ AprilTag 标定板定义的世界坐标系变换
    - -0.888...                     #   3x4 矩阵，按行展开为 12 个浮点数
    - ...                           #   reshape 为 [3,4] 后补 [0,0,0,1] 行 → 4x4 齐次矩阵
master: '840412060917'              # 主相机序列号
```

**脚本如何使用：** 读取 `apriltag` 键的 12 个数值 → reshape 为 `(3,4)` → 补齐为 `(4,4)` 外参矩阵。
这个矩阵定义了从世界坐标系（AprilTag）到相机坐标系的变换，用于将物体和手的位姿从相机坐标系转换到世界坐标系进行渲染。

### 3.4 `calibration/mano_xxx/mano.yml` — MANO 手部形状参数

```yaml
betas:
- -0.8566880226135254
- 0.12083425372838974
- ...                    # 共 10 个浮点数，MANO 模型的形状 PCA 系数
```

**脚本如何使用：** 加载为 `(10,)` numpy 数组 → 传入 `MANOLayer` 构造函数 → 控制手部网格的个体形状差异（手指粗细、手掌大小等）。

### 3.5 `models/xxx/textured_simple.obj` — YCB 物体 3D 模型

标准 OBJ 格式的简化网格文件，配合 `texture_map.png` 纹理贴图。
脚本通过 SAPIEN 的 `add_visual_from_file()` 直接加载到场景中。

---

## 4. 完整数据流（逐步追踪）

### 第一步：入口 — `main()` (visualize_hand_object.py)

```python
def main(dexycb_dir, robots=None, fps=10):
    data_root = Path(dexycb_dir).absolute()       # "dexYCB" 目录路径
    RetargetingConfig.set_default_urdf_dir(...)    # 设置机器人 URDF 搜索路径
    viz_hand_object(robots, data_root, fps)
```

### 第二步：数据加载 — `DexYCBVideoDataset.__init__()` (dataset.py)

```
输入: data_root = "dexYCB/", hand_type = "right"
```

**加载标定数据：**
```
calibration/intrinsics/*.yml          →  self._intrinsics  (dict: 相机序列号 → 3x3 内参矩阵)
calibration/extrinsics_*/extrinsics.yml →  self._extrinsics (dict: 标定名 → 外参数据)
calibration/mano_*/mano.yml           →  self._mano_parameters (dict: 标定名 → betas[10])
```

**扫描所有 capture：**
```
遍历: dexYCB/20200709-subject-01/20200709_141754/
                                  20200709_141841/
                                  ...
对每个 capture:
  1. 读取 meta.yml
  2. 检查 mano_sides 是否包含 "right" → 不包含则跳过
  3. 加载 pose.npz
  4. 记录到 self._captures 列表（capture 名称）
     以及 self._capture_meta / self._capture_pose 字典
```

### 第三步：取样一条轨迹 — `dataset[data_id]`

```python
data_id = 4                          # 硬编码的索引，可修改
sampled_data = dataset[data_id]      # 返回一个 dict
```

返回的 `sampled_data` 字典结构：

| 键 | 类型 | 来源 | 含义 |
|----|------|------|------|
| `hand_pose` | `np.array (N,1,51)` | `pose.npz["pose_m"]` | 每帧的 MANO 手部参数 |
| `object_pose` | `np.array (N,K,7)` | `pose.npz["pose_y"]` | 每帧各物体的位姿 |
| `extrinsics` | `np.array (4,4)` | `extrinsics.yml → apriltag` | 世界→相机变换矩阵 |
| `ycb_ids` | `list[int]` | `meta.yml → ycb_ids` | 场景中物体的 YCB ID |
| `hand_shape` | `np.array (10,)` | `mano.yml → betas` | MANO 形状系数 |
| `object_mesh_file` | `list[str]` | 由 `ycb_ids` 拼接路径 | 各物体的 OBJ 文件绝对路径 |
| `capture_name` | `str` | capture 目录名 | 如 `"20200709_152843"` |

### 第四步：加载场景 — `viewer.load_object_hand(sampled_data)`

```
hand_viewer.py → load_object_hand():

1. 遍历 ycb_ids + object_mesh_file:
   对每个物体调用 SAPIEN 的 add_visual_from_file(obj_path)
   → 在 3D 场景中创建物体 actor

2. 创建 MANOLayer("right", hand_shape):
   hand_shape (10维 betas) → manopth ManoLayer
   → 初始化一个参数化的手部网格模型

3. 计算相机位姿:
   extrinsics (4x4) → sapien.Pose → self.camera_pose
   → 用于将相机坐标系下的数据转换到世界坐标系
```

### 第五步：逐帧渲染 — `viewer.render_dexycb_data(sampled_data, fps)`

```
对每一帧 i (共 N 帧):

  ┌─ 手部渲染 ─────────────────────────────────────────────┐
  │ hand_pose[i] → (1, 51)                                 │
  │   ├── [:48] → MANO 姿态参数 (关节角度 PCA 系数)         │
  │   └── [48:51] → 手部平移 (相机坐标系)                   │
  │                                                         │
  │ MANOLayer.forward(pose, trans)                          │
  │   → vertex (778, 3): 手部网格顶点 (相机坐标系)          │
  │   → joint  (21, 3):  手部关节位置 (相机坐标系)          │
  │                                                         │
  │ vertex × camera_pose → 世界坐标系下的顶点               │
  │ → 计算法线 → 创建 SAPIEN mesh → 渲染手部网格            │
  └─────────────────────────────────────────────────────────┘

  ┌─ 物体渲染 ─────────────────────────────────────────────┐
  │ object_pose[i][k] → (7,)                               │
  │   ├── [:3]  → 旋转四元数 qx, qy, qz                    │
  │   ├── [3]   → 旋转四元数 qw                             │
  │   └── [4:7] → 平移 tx, ty, tz (相机坐标系)             │
  │                                                         │
  │ camera_pose × object_pose → 世界坐标系下的物体位姿      │
  │ → 设置 SAPIEN actor 的 pose → 渲染物体                  │
  └─────────────────────────────────────────────────────────┘

  scene.update_render() → 更新画面
```

### 第六步（可选）：机器人 Retarget — `RobotHandDatasetSAPIENViewer`

当命令行指定 `--robots` 参数时（如 `--robots allegro`）：

```
对每一帧:
  1. MANOLayer 计算人手 21 个关节 3D 位置 (同上)
  2. 提取腕部位姿 (位置 + 旋转)
  3. 构造 retarget 参考值 (关节位置或向量)
  4. SeqRetargeting.retarget(ref_value) → 求解机器人关节角 qpos
  5. 设置 SAPIEN 机器人关节角 → 渲染机器人手
```

---

## 5. 坐标系变换总结

```
MANO 输出 (相机坐标系)
    │
    │  × camera_pose.inv()  (extrinsics.yml → apriltag)
    ▼
世界坐标系 (AprilTag 定义)
    │
    │  SAPIEN 场景直接使用世界坐标系渲染
    ▼
屏幕显示
```

- **相机坐标系**：MANO 的 `pose_m` 和物体的 `pose_y` 都在相机坐标系下
- **世界坐标系**：由 AprilTag 标定板定义，`extrinsics.yml` 中的 `apriltag` 矩阵提供变换
- `camera_pose = Pose(extrinsic_mat).inv()`：将相机坐标系下的点变换到世界坐标系

---

## 6. 关键数据对应关系速查

| 代码中的变量 | 数据集文件 | 文件中的键/字段 |
|-------------|-----------|----------------|
| `hand_pose` | `pose.npz` | `pose_m` → `(N, 1, 51)` |
| `object_pose` | `pose.npz` | `pose_y` → `(N, K, 7)` |
| `extrinsic_mat` | `extrinsics.yml` | `extrinsics.apriltag` → 12 floats → `(4,4)` |
| `hand_shape` / `betas` | `mano.yml` | `betas` → `(10,)` |
| `ycb_ids` | `meta.yml` | `ycb_ids` → `list[int]` |
| `object_mesh_file` | `models/xxx/` | `textured_simple.obj` |
| `detected_hand_type` | `meta.yml` | `mano_sides` → `"right"` / `"left"` |
| `mano_name` | `meta.yml` | `mano_calib[0]` → 引用 `calibration/mano_xxx/` |
| `extrinsic_name` | `meta.yml` | `extrinsics` → 引用 `calibration/extrinsics_xxx/` |
