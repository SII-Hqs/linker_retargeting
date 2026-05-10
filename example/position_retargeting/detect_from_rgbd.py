"""
从 RGB-D 图像序列中检测人手姿态，并 retarget 到机器人灵巧手。

整体流程:
  1. 读取相机内参 (fx, fy, cx, cy) 和畸变系数
  2. 加载 retargeting 配置，构建优化器（包含机器人 URDF 模型）
  3. 逐帧处理:
     a. 读取 RGB + Depth 图像
     b. RGBDHandDetector 检测:
        - MediaPipe 检测 2D 手部关键点
        - 深度图反投影得到相机坐标系下的 3D 关键点
        - ArUco marker 估计 T_cam2table（相机→桌面变换）
        - 用 T_cam2table 的逆将 3D 点从相机坐标系变换到桌面坐标系
        - 计算腕部位姿，关节位置转换到 MANO 约定
     c. 根据 retargeting 类型（POSITION / VECTOR）构造参考值
     d. 调用 retargeting.retarget() 求解机器人关节角 qpos
  4. 将所有帧的 qpos 和元数据保存为 pickle 文件

输出 pickle 结构:
  {
    "data": [  # 每帧一个 dict
      {
        "qpos":           np.array,  # 机器人关节角（前6维=腕部free joint位姿，后面=手指关节角）
        "wrist_pos":      np.array,  # (3,) 腕部在桌面坐标系下的位置 (米)
        "wrist_rot":      np.array,  # (3,3) 腕部旋转矩阵 (MANO约定, 桌面坐标系)
        "joint_pos_table": np.array, # (21,3) 所有关节在桌面坐标系下的绝对位置
        "T_cam2table":    np.array,  # (4,4) 相机→桌面的变换矩阵
      },
      ...
    ],
    "meta_data": {
      "config_path":   str,    # retargeting 配置文件路径
      "dof":           int,    # 机器人自由度数
      "joint_names":   list,   # 各关节名称
      "camera_matrix": list,   # 相机内参 (3x3)
      "dist_coeffs":   list,   # 畸变系数 (5,)
      "marker_length": float,  # ArUco marker 边长 (米)
    }
  }

坐标系说明:
  - 相机坐标系: 原点在相机光心, z 朝前, x 朝右, y 朝下
  - 桌面坐标系: 由 ArUco marker 定义, 原点在 marker 中心, z 垂直于 marker 平面
  - MANO 约定:  手部关节的标准表示约定, 通过 operator2mano 旋转矩阵转换
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tqdm
import tyro
from pytransform3d import rotations

from dex_retargeting.constants import RobotName, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from rgbd_hand_detector import RGBDHandDetector


def _create_visualization_writer(output_path: str, frame_size) -> cv2.VideoWriter:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output), fourcc, 20.0, frame_size)


def _render_detection_overlay(
    bgr: np.ndarray,
    detector: RGBDHandDetector,
    frame_index: int,
    rgb_name: str,
    detection_ok: bool,
    marker_ok: bool,
    skipped_reason: Optional[str] = None,
) -> np.ndarray:
    vis = bgr.copy()
    debug = detector.last_debug_info
    keypoint_2d = debug.get("keypoint_2d")
    pixel_coords = debug.get("pixel_coords")
    invalid_mask = debug.get("invalid_depth_mask")

    if keypoint_2d is not None:
        detector.draw_skeleton_on_image(vis, keypoint_2d, style="default")

    if pixel_coords is not None:
        pixel_coords_int = np.round(pixel_coords).astype(int)
        for idx, (u, v) in enumerate(pixel_coords_int):
            if invalid_mask is not None and idx < len(invalid_mask) and invalid_mask[idx]:
                color = (0, 0, 255)
                radius = 6
            else:
                color = (0, 255, 0)
                radius = 4
            cv2.circle(vis, (u, v), radius, color, 2)
            cv2.putText(
                vis,
                str(idx),
                (u + 4, v - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    lines = [
        f"frame: {frame_index:05d}  file: {rgb_name}",
        f"hand_detected: {'yes' if detection_ok else 'no'}",
        (
            f"aruco: {'detected' if debug.get('marker_detected') else ('cached' if debug.get('used_cached_marker') else 'missing')}"
            f"  retarget_ready: {'yes' if marker_ok and detection_ok else 'no'}"
        ),
        "green: valid depth  red: invalid depth",
    ]
    if skipped_reason is not None:
        lines.append(f"skipped_reason: {skipped_reason}")

    y = 24
    for text in lines:
        cv2.putText(
            vis,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 220, 255) if "skipped_reason" in text else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return vis


def load_camera_params(json_path: str):
    """从 JSON 文件加载相机内参和畸变系数。

    JSON 格式示例 (与 camera_params/color_intrinsics.json 一致):
        {
            "fx": 615.0, "fy": 615.0,    # 焦距 (像素)
            "cx": 320.0, "cy": 240.0,    # 主点坐标 (像素)
            "distortion": {               # 畸变系数 (可选)
                "k1": 0.0, "k2": 0.0,    # 径向畸变
                "p1": 0.0, "p2": 0.0,    # 切向畸变
                "k3": 0.0                 # 高阶径向畸变
            }
        }

    Returns:
        camera_matrix: (3, 3) 相机内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
        dist_coeffs:   (5,)  畸变系数 [k1, k2, p1, p2, k3]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # 构造 3x3 相机内参矩阵
    camera_matrix = np.array(
        [
            [data["fx"], 0, data["cx"]],
            [0, data["fy"], data["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # 提取畸变系数，缺失时默认为 0
    d = data.get("distortion", {})
    dist_coeffs = np.array(
        [d.get("k1", 0), d.get("k2", 0), d.get("p1", 0), d.get("p2", 0), d.get("k3", 0)],
        dtype=np.float32,
    )

    return camera_matrix, dist_coeffs


def retarget_rgbd(
    retargeting: SeqRetargeting,
    rgb_dir: str,
    depth_dir: str,
    output_path: str,
    config_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    hand_type: str,
    visualize: bool = False,
    visualization_output: Optional[str] = None,
    save_visualization_frames: bool = False,
):
    """处理 RGB-D 图像序列，检测人手并 retarget 到机器人手。

    Args:
        retargeting:   已构建的 SeqRetargeting 对象（包含优化器和机器人模型）
        rgb_dir:       RGB 图像目录，文件按名称排序
        depth_dir:     深度图目录，文件名需与 RGB 一一对应
        output_path:   输出 pickle 文件路径
        config_path:   retargeting 配置文件路径（记录到元数据中）
        camera_matrix: (3,3) 相机内参矩阵
        dist_coeffs:   (5,) 畸变系数
        marker_length: ArUco marker 物理边长 (米)
        hand_type:     "Right" 或 "Left"
        visualize:     是否弹出窗口实时显示检测关键点
        visualization_output: 可选，保存可视化视频路径(.mp4)
        save_visualization_frames: 是否额外逐帧保存叠加后的图像
    """
    rgb_path = Path(rgb_dir)
    depth_path = Path(depth_dir)

    rgb_extensions = {".png", ".jpg", ".jpeg"}
    depth_extensions = {".png", ".npy"}

    # 按文件名排序，确保 RGB 和 Depth 帧一一对应
    rgb_files = sorted(
        [f for f in rgb_path.iterdir() if f.suffix.lower() in rgb_extensions]
    )
    depth_files = sorted(
        [f for f in depth_path.iterdir() if f.suffix.lower() in depth_extensions]
    )

    # ---------- 基本校验 ----------
    if len(rgb_files) == 0:
        print(f"Error: No RGB images found in {rgb_dir}")
        return
    if len(depth_files) == 0:
        print(f"Error: No depth images found in {depth_dir}")
        return
    if len(rgb_files) != len(depth_files):
        print(
            f"Warning: RGB ({len(rgb_files)}) and depth ({len(depth_files)}) "
            f"frame counts do not match. Using minimum."
        )

    num_frames = min(len(rgb_files), len(depth_files))

    # ---------- 初始化手部检测器 ----------
    # RGBDHandDetector 内部集成了:
    #   - MediaPipe Hands: 2D 手部关键点检测 (21个关键点)
    #   - ArUco marker 检测: 估计 T_cam2table 变换矩阵
    #   - 深度反投影: 2D关键点 + 深度 → 3D相机坐标 → 3D桌面坐标
    detector = RGBDHandDetector(
        hand_type=hand_type,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length=marker_length,
        selfie=False,  # 非自拍模式（相机朝向桌面）
    )

    vis_writer = None
    vis_frame_dir = None
    if visualization_output is not None:
        sample_bgr = cv2.imread(str(rgb_files[0]))
        if sample_bgr is not None:
            h, w = sample_bgr.shape[:2]
            vis_writer = _create_visualization_writer(visualization_output, (w, h))
    if save_visualization_frames:
        if visualization_output is not None:
            vis_frame_dir = Path(visualization_output).with_suffix("")
        else:
            vis_frame_dir = Path(output_path).with_suffix("")
        vis_frame_dir = vis_frame_dir.parent / f"{vis_frame_dir.name}_vis_frames"
        vis_frame_dir.mkdir(parents=True, exist_ok=True)

    data = []      # 存储每帧的 retarget 结果
    skipped = 0    # 跳过的帧数（检测失败等）

    with tqdm.tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            skipped_reason = None
            # ========== 1. 读取 RGB 图像 ==========
            bgr = cv2.imread(str(rgb_files[i]))
            if bgr is None:
                pbar.update(1)
                skipped += 1
                skipped_reason = "rgb_read_failed"
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # MediaPipe 需要 RGB 格式

            # ========== 2. 读取深度图 ==========
            # 支持两种格式:
            #   - .npy: 直接加载，假设单位为米
            #   - .png: 16-bit unsigned int，单位为毫米，需除以 1000 转为米
            depth_file = depth_files[i]
            if depth_file.suffix == ".npy":
                depth = np.load(str(depth_file)).astype(np.float32)
            else:
                depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    pbar.update(1)
                    skipped += 1
                    skipped_reason = "depth_read_failed"
                    continue
                if depth_raw.dtype == np.uint16:
                    depth = depth_raw.astype(np.float32) / 1000.0  # mm → m
                else:
                    depth = depth_raw.astype(np.float32)

            # ========== 3. 手部检测 ==========
            # detector.detect() 返回:
            #   num_box:      检测到的手的数量 (0 表示未检测到)
            #   joint_pos:    (21,3) 以腕部为原点的关节位置 (桌面坐标系, MANO约定)
            #   keypoint_2d:  MediaPipe 原始 2D 关键点
            #   wrist_pos:    (3,) 腕部在桌面坐标系下的绝对位置
            #   wrist_rot:    (3,3) 腕部旋转矩阵 (MANO约定)
            #   T_cam2table:  (4,4) 相机→桌面变换矩阵 (由 ArUco marker 估计)
            num_box, joint_pos, keypoint_2d, wrist_pos, wrist_rot, T_cam2table = (
                detector.detect(rgb, depth)
            )

            # 未检测到手 → 跳过
            if num_box == 0 or joint_pos is None:
                skipped_reason = "hand_not_detected"

            # 未检测到 ArUco marker（且无缓存）→ 无法确定桌面坐标系 → 跳过
            if T_cam2table is None and skipped_reason is None:
                skipped_reason = "aruco_not_detected"

            detection_ok = num_box > 0 and joint_pos is not None
            marker_ok = T_cam2table is not None

            if visualize or vis_writer is not None or vis_frame_dir is not None:
                overlay = _render_detection_overlay(
                    bgr=bgr,
                    detector=detector,
                    frame_index=i,
                    rgb_name=rgb_files[i].name,
                    detection_ok=detection_ok,
                    marker_ok=marker_ok,
                    skipped_reason=skipped_reason,
                )
                if visualize:
                    cv2.imshow("RGB-D Hand Detection Debug", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("Visualization interrupted by user.")
                        break
                if vis_writer is not None:
                    vis_writer.write(overlay)
                if vis_frame_dir is not None:
                    cv2.imwrite(str(vis_frame_dir / f"{i:05d}.png"), overlay)

            if skipped_reason is not None:
                pbar.update(1)
                skipped += 1
                continue

            # ========== 4. 构造 retarget 参考值 ==========
            # retargeting_type 决定了如何将人手关键点映射到机器人:
            #   - "POSITION": 使用桌面坐标系下的绝对关节位置
            #   - 其他 (如 "VECTOR"): 使用关节间的相对向量
            retargeting_type = retargeting.optimizer.retargeting_type
            # indices: 指定哪些人手关键点对应机器人的哪些目标链接
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "POSITION":
                # 绝对位置模式: joint_pos 是以腕部为原点的相对位置，
                # 加上 wrist_pos 恢复为桌面坐标系下的绝对位置
                joint_pos_absolute = joint_pos + wrist_pos[None, :]
                # 只取优化器需要的关键点子集
                ref_value = joint_pos_absolute[indices, :]
            else:
                # 向量模式: 计算指定关节对之间的相对向量
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # ========== 5. Warm start (仅首帧, 仅 POSITION + free joint) ==========
            # 当机器人模型包含 6D free joint（腕部浮动关节）时，
            # 需要用检测到的腕部位姿初始化优化器，避免从零开始搜索导致收敛到局部最优
            if (
                retargeting_type == "POSITION"
                and retargeting.optimizer.has_free_joint
                and not retargeting.is_warm_started
            ):
                # 将腕部旋转矩阵转为四元数 (pytransform3d 格式: [w, x, y, z])
                wrist_quat = rotations.quaternion_from_matrix(wrist_rot)
                retargeting.warm_start(
                    wrist_pos,
                    wrist_quat,
                    hand_type=HandType.right if hand_type == "Right" else HandType.left,
                    is_mano_convention=True,  # 输入的位姿遵循 MANO 约定
                )

            # ========== 6. 执行 retarget 求解 ==========
            # retarget() 通过优化求解机器人关节角 qpos，使机器人手的目标链接
            # 尽可能匹配 ref_value 指定的位置/向量
            # qpos 结构 (以 POSITION + free joint 为例):
            #   [tx, ty, tz, qw, qx, qy, qz, j1, j2, ..., jN]
            #    ↑ 腕部平移(桌面系)  ↑ 腕部旋转(四元数)  ↑ 手指关节角度
            qpos = retargeting.retarget(ref_value)

            # ========== 7. 保存当前帧数据 ==========
            frame_data = dict(
                qpos=qpos,                                    # 机器人关节角
                wrist_pos=wrist_pos,                          # 腕部位置 (桌面坐标系)
                wrist_rot=wrist_rot,                          # 腕部旋转 (桌面坐标系, MANO约定)
                joint_pos_table=joint_pos + wrist_pos[None, :],  # 所有关节绝对位置 (桌面坐标系)
                T_cam2table=T_cam2table,                      # 相机→桌面变换矩阵
            )
            data.append(frame_data)
            pbar.update(1)

    if vis_writer is not None:
        vis_writer.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"Processed {len(data)} frames, skipped {skipped} frames.")

    # ========== 8. 保存结果到 pickle 文件 ==========
    meta_data = dict(
        config_path=config_path,
        dof=len(retargeting.optimizer.robot.dof_joint_names),       # 机器人自由度数
        joint_names=retargeting.optimizer.robot.dof_joint_names,    # 各关节名称列表
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.tolist(),
        marker_length=marker_length,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)
    print(f"Saved {len(data)} frames to {output_path}")

    # 打印 retargeting 优化器的统计信息（耗时、误差等）
    retargeting.verbose()


def main(
    robot_name: RobotName,
    rgb_dir: str,
    depth_dir: str,
    output_path: str,
    camera_params_json: str = "camera_params/color_intrinsics.json",
    hand_type: HandType = HandType.right,
    marker_length: float = 0.04,
    config_path: Optional[str] = None,
    visualize: bool = False,
    visualization_output: Optional[str] = None,
    save_visualization_frames: bool = False,
):
    """从 RGB-D 图像中检测人手姿态，通过 ArUco marker 定位桌面坐标系，retarget 到机器人手。

    输出的 pickle 文件包含每帧的 qpos（关节角，含 6D 腕部 free joint）和腕部位姿，
    可直接用于 Isaac Gym 中的回放。

    用法示例:
        python detect_from_rgbd.py \\
            --robot-name allegro \\
            --rgb-dir ./data/rgb \\
            --depth-dir ./data/depth \\
            --output-path ./output/retarget.pickle

    Args:
        robot_name:        机器人名称 (如 allegro, shadow, leap, inspire 等)
        rgb_dir:           RGB 图像目录 (.png/.jpg)，按文件名排序
        depth_dir:         深度图目录 (.png 16-bit 毫米 或 .npy 米)
        output_path:       输出 pickle 文件路径
        camera_params_json: 相机内参 JSON 文件路径
        hand_type:         跟踪哪只手 (right 或 left)
        marker_length:     ArUco marker 物理边长 (米)，默认 0.04m = 4cm
        config_path:       自定义 retargeting 配置路径；不指定则使用该机器人的默认 position 配置
        visualize:         是否弹窗显示每一帧检测到的关键点
        visualization_output: 保存可视化视频的路径，例如 ./output/debug.mp4
        save_visualization_frames: 是否把每一帧叠加图保存成 png
    """
    # 加载相机参数
    camera_matrix, dist_coeffs = load_camera_params(camera_params_json)

    # 如果未指定配置文件，使用该机器人的默认 position retargeting 配置
    if config_path is None:
        from dex_retargeting.constants import RetargetingType

        config_path = str(
            get_default_config_path(robot_name, RetargetingType.position, hand_type)
        )

    # 定位机器人 URDF 模型目录: <项目根>/assets/robots/hands/
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    # 设置 URDF 搜索路径，使配置文件中的相对路径能正确解析
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    # 从 YAML 配置文件加载并构建 retargeting 优化器
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type_str = "Right" if hand_type == HandType.right else "Left"

    # 执行主处理流程
    retarget_rgbd(
        retargeting,
        rgb_dir,
        depth_dir,
        output_path,
        config_path,
        camera_matrix,
        dist_coeffs,
        marker_length,
        hand_type_str,
        visualize,
        visualization_output,
        save_visualization_frames,
    )


if __name__ == "__main__":
    # tyro.cli 自动将 main() 的参数解析为命令行参数
    # 运行 python detect_from_rgbd.py --help 查看所有可用参数
    tyro.cli(main)
