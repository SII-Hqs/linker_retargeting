# `detect_from_rgbd.py` vs `detect_from_rgbd_table_fixed.py` 对比

## 1. 一句话概括

| 脚本 | 定位 |
|------|------|
| `detect_from_rgbd.py` | **原版**：调用 `RGBDHandDetector.detect()` 一步到位，输出 MANO 约定的关节 |
| `detect_from_rgbd_table_fixed.py` | **改进版**：自定义 `TableFrameHandDetector`，显式区分桌面坐标系与 MANO 坐标系，并修正深度缩放 |

---

## 2. 核心差异总览

| 维度 | `detect_from_rgbd.py` (原版) | `detect_from_rgbd_table_fixed.py` (改进版) |
|------|-----|------|
| **检测器类** | 直接使用 `RGBDHandDetector` | 继承并扩展为 `TableFrameHandDetector` |
| **检测入口** | `detector.detect(rgb, depth)` → 返回 6 个散列值 | `detector.detect_table_frame(rgb, depth)` → 返回一个 `dict` |
| **深度缩放** | 硬编码：`.npy` 假设米，`.png` uint16 除以 1000 | 新增 `--depth-scale` 参数（默认 `0.001`），统一 `raw × scale` |
| **坐标系变换命名** | `T_cam2table` 变量名有歧义（实际存的是 `T_table2cam`） | 显式区分 `T_table2cam` 和 `T_cam2table = inv(T_table2cam)` |
| **POSITION ref_value** | `joint_pos + wrist_pos`（MANO 约定的相对位置 + 腕部绝对位置） | `joint_pos_table[indices]`（桌面坐标系下的原始绝对 3D 点） |
| **输出 pickle 字段** | 5 个字段 | 8 个字段，额外保存 `joint_pos_table_relative`、`joint_pos_mano`、`T_cam2table`、`depth_scale` |
| **meta_data** | 无坐标约定说明 | 新增 `coordinate_contract` 和 `depth_scale` 字段 |
| **代码复用** | 独立实现所有函数 | 从原版 import `_create_visualization_writer`、`_render_detection_overlay`、`load_camera_params` |

---

## 3. 逐项详解

### 3.1 深度图加载

**原版** — 深度缩放逻辑分散在主循环中，`.npy` 假设已经是米：

```python
# detect_from_rgbd.py
if depth_file.suffix == ".npy":
    depth = np.load(str(depth_file)).astype(np.float32)       # 假设单位: 米
else:
    if depth_raw.dtype == np.uint16:
        depth = depth_raw.astype(np.float32) / 1000.0          # 硬编码 mm → m
```

**改进版** — 抽取为独立函数 `_load_depth()`，通过参数控制缩放：

```python
# detect_from_rgbd_table_fixed.py
def _load_depth(path: Path, depth_scale: float) -> np.ndarray:
    ...
    return depth_raw * float(depth_scale)   # 统一: raw × scale → 米
```

命令行新增 `--depth-scale 0.001`（毫米→米）或 `--depth-scale 1.0`（已是米）。

**影响**：原版对 `.npy` 文件假设单位是米，但如果实际数据是毫米（如本项目），会导致深度值偏大 1000 倍。改进版通过显式参数避免了这个隐患。

---

### 3.2 坐标系变换命名修正

**原版** `RGBDHandDetector.detect_marker()` 中：

```python
# 变量名叫 T_cam2table，但实际语义是 T_table2cam
# （ArUco estimatePoseSingleMarkers 返回的是 marker 在相机坐标系中的位姿）
T_cam2table = np.eye(4)
T_cam2table[:3, :3] = R      # R_table2cam
T_cam2table[:3, 3] = tvec    # t_table2cam
self._T_cam2table = T_cam2table
```

然后在 `detect()` 中取逆：

```python
T_table2cam = np.linalg.inv(T_cam2table)   # 变量名也是反的
kp_3d_table = (T_table2cam @ kp_3d_cam_homo.T).T[:, :3]
```

**改进版** `TableFrameHandDetector.detect_table_transform()` 显式命名：

```python
T_table2cam = np.eye(4)          # ArUco 直接输出: 桌面→相机
T_table2cam[:3, :3] = R_table2cam
T_table2cam[:3, 3] = tvec

T_cam2table = np.linalg.inv(T_table2cam)   # 求逆: 相机→桌面
return T_table2cam, T_cam2table             # 两个都返回，语义清晰
```

**影响**：原版变量名与实际语义相反，容易造成理解混乱。改进版修正了命名，并同时返回两个方向的变换矩阵。

---

### 3.3 检测器返回值结构

**原版** — 返回 6 个散列值（位置参数）：

```python
num_box, joint_pos, keypoint_2d, wrist_pos, wrist_rot, T_cam2table = detector.detect(rgb, depth)
# joint_pos: MANO 约定，以腕部为原点的相对位置
```

**改进版** — 返回一个字典，包含多种坐标表示：

```python
result = detector.detect_table_frame(rgb, depth)
# result 包含:
#   joint_pos_table:          桌面坐标系下的绝对 3D 位置 (21, 3)
#   joint_pos_table_relative: 桌面坐标系下以腕部为原点的相对位置 (21, 3)
#   joint_pos_mano:           MANO 约定的相对位置 (21, 3)
#   wrist_pos_table:          腕部绝对位置 (3,)
#   wrist_rot_table_mano:     腕部旋转矩阵 (3, 3)
#   T_table2cam / T_cam2table: 两个方向的变换矩阵
```

**影响**：改进版保留了中间结果，方便调试和下游使用不同坐标表示。

---

### 3.4 POSITION retargeting 的 ref_value 构造

这是**最关键的差异**。

**原版**：

```python
if retargeting_type == "POSITION":
    joint_pos_absolute = joint_pos + wrist_pos[None, :]
    ref_value = joint_pos_absolute[indices, :]
```

这里 `joint_pos` 是经过 `wrist_rot_frame @ operator2mano` 旋转后的 MANO 约定坐标，加上 `wrist_pos`（桌面坐标系下的腕部位置）。**混合了两个不同旋转空间的数据**：腕部位置在桌面坐标系，而关节相对位置在 MANO 旋转后的局部坐标系。

**改进版**：

```python
if retargeting_type == "POSITION":
    ref_value = result["joint_pos_table"][indices, :]
```

直接使用 `joint_pos_table`——桌面坐标系下的原始绝对 3D 位置，**没有经过任何 MANO 旋转变换**。这与 POSITION 优化器的期望完全一致：FK 输出在哪个坐标系，ref_value 就在哪个坐标系。

**影响**：原版的 ref_value 在 MANO 旋转空间中，与机器人 FK 输出的桌面坐标系不完全对齐，可能导致优化结果偏差。改进版确保了坐标系一致性。

---

### 3.5 输出 pickle 结构

**原版** 每帧数据：

```python
frame_data = dict(
    qpos=qpos,
    wrist_pos=wrist_pos,
    wrist_rot=wrist_rot,
    joint_pos_table=joint_pos + wrist_pos[None, :],  # MANO 约定 + 腕部偏移
    T_cam2table=T_cam2table,                          # 实际是 T_table2cam
)
```

**改进版** 每帧数据：

```python
frame_data = dict(
    qpos=qpos,
    wrist_pos=result["wrist_pos_table"],
    wrist_rot=result["wrist_rot_table_mano"],
    joint_pos_table=result["joint_pos_table"],              # 真正的桌面坐标系绝对位置
    joint_pos_table_relative=result["joint_pos_table_relative"],
    joint_pos_mano=result["joint_pos_mano"],                # MANO 约定（供 VECTOR 模式使用）
    T_table2cam=result["T_table2cam"],                      # 命名正确
    T_cam2table=result["T_cam2table"],                      # 两个方向都保存
    depth_scale=depth_scale,                                # 记录深度缩放因子
)
```

**改进版 meta_data** 额外包含：

```python
depth_scale=depth_scale,
coordinate_contract="POSITION ref_value uses joint_pos_table: absolute 3D points in the ArUco table frame, meters."
```

---

## 4. 数据流对比图

### 原版

```
MediaPipe 2D → 深度反投影 → kp_3d_cam (相机坐标系)
    → T_table2cam (名为 T_cam2table) 变换 → kp_3d_table (桌面坐标系)
    → 减去腕部 → 估计 wrist_rot_frame
    → × wrist_rot_frame × operator2mano → joint_pos (MANO 约定, 相对)
    → + wrist_pos (桌面坐标系) → ref_value
                                  ↑ 混合了 MANO 旋转空间和桌面坐标系
```

### 改进版

```
MediaPipe 2D → 深度反投影 → kp_3d_cam (相机坐标系)
    → T_cam2table 变换 → joint_pos_table (桌面坐标系, 绝对)
    │                         ↓
    │                    ref_value = joint_pos_table[indices]
    │                    ↑ 纯桌面坐标系，与 FK 输出一致
    │
    → 减去腕部 → 估计 wrist_rot_frame
    → × wrist_rot_frame × operator2mano → joint_pos_mano (MANO 约定)
                                           ↑ 仅供 VECTOR 模式和 warm_start 使用
```

---

## 5. 命令行参数差异

| 参数 | 原版 | 改进版 |
|------|------|--------|
| `--depth-scale` | ❌ 不存在 | ✅ 默认 `0.001`（毫米→米） |
| 其余参数 | 相同 | 相同 |

---

## 6. 总结

改进版做了三件事：

1. **修正深度缩放**：用显式 `--depth-scale` 参数替代硬编码假设，避免 `.npy` 单位不一致的 bug
2. **修正坐标系命名**：`T_table2cam` / `T_cam2table` 语义清晰，不再混淆
3. **修正 POSITION ref_value**：直接使用桌面坐标系下的绝对 3D 点，确保与优化器 FK 输出在同一坐标系下
