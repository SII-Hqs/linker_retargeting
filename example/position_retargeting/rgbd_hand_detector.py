"""
RGB-D 手部检测器模块。

功能概述:
  从 RGB-D（彩色 + 深度）图像中检测人手的 21 个关键点 3D 位置，
  并通过 ArUco marker 将结果从相机坐标系变换到桌面坐标系，
  最终输出符合 MANO 手部模型约定的关节位置和腕部位姿。

核心流程 (detect 方法):
  1. 深度图中值滤波去噪
  2. ArUco marker 检测 → 估计 T_cam2table（相机→桌面 4x4 变换矩阵）
  3. MediaPipe Hands 检测 → 21 个 2D 归一化关键点
  4. 2D 关键点 → 像素坐标 → 采样深度 → 反投影为 3D 相机坐标
  5. 用 T_cam2table 的逆将 3D 点从相机坐标系变换到桌面坐标系
  6. 以腕部为原点做中心化，估计腕部旋转帧
  7. 通过 operator2mano 矩阵转换到 MANO 约定

坐标系说明:
  - 相机坐标系: 原点在光心, z 朝前(深度方向), x 朝右, y 朝下
  - 桌面坐标系: 由 ArUco marker 定义, 原点在 marker 中心
  - MANO 约定:   手部关节的标准表示, 通过 OPERATOR2MANO 旋转矩阵转换

依赖:
  - OpenCV (cv2): 图像处理、ArUco marker 检测、位姿估计
  - MediaPipe:    2D 手部关键点检测 (21 个关键点)
  - NumPy:        矩阵运算、坐标变换
"""

import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

# ============================================================================
# 操作者坐标系 → MANO 手部模型坐标系的旋转矩阵 (3x3)
#
# 背景: MediaPipe / 深度相机得到的手部关键点处于"操作者坐标系"（由桌面 ArUco
#        marker 和腕部旋转帧定义），而下游的 retargeting 优化器期望输入遵循
#        MANO 手部模型的坐标约定。这两个旋转矩阵完成轴的重映射。
#
# 右手映射:
#   MANO_x = -Operator_z   (MANO 的 x 轴 = 操作者的 -z 轴)
#   MANO_y = -Operator_x   (MANO 的 y 轴 = 操作者的 -x 轴)
#   MANO_z =  Operator_y   (MANO 的 z 轴 = 操作者的  y 轴)
#
# 左手映射 (y, z 轴取反以适应镜像):
#   MANO_x = -Operator_z
#   MANO_y =  Operator_x
#   MANO_z = -Operator_y
# ============================================================================
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class RGBDHandDetector:
    """从 RGB-D 图像中检测手部关节 3D 位置，基于 ArUco marker 做坐标变换。

    集成了三个核心组件:
      1. MediaPipe Hands — 从 RGB 图像检测 21 个 2D 手部关键点
      2. 深度图反投影   — 结合相机内参，将 2D 关键点 + 深度值还原为 3D 坐标
      3. ArUco marker   — 估计相机到桌面的刚体变换 T_cam2table

    关键点索引 (MediaPipe 21 点):
      0: 腕部 (WRIST)
      1-4: 拇指 (THUMB_CMC → THUMB_TIP)
      5-8: 食指 (INDEX_FINGER_MCP → INDEX_FINGER_TIP)
      9-12: 中指 (MIDDLE_FINGER_MCP → MIDDLE_FINGER_TIP)
      13-16: 无名指 (RING_FINGER_MCP → RING_FINGER_TIP)
      17-20: 小指 (PINKY_MCP → PINKY_TIP)
    """

    def __init__(
        self,
        hand_type="Right",
        camera_matrix=None,
        dist_coeffs=None,
        marker_length=0.04,
        aruco_dict_type=aruco.DICT_4X4_50,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
        depth_median_kernel=5,
        depth_search_radius=3,
        depth_unit="m",
    ):
        """初始化手部检测器。

        Args:
            hand_type: "Right" 或 "Left"，指定要检测哪只手。
            camera_matrix: 3x3 相机内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]。
                           fx, fy 为焦距(像素), cx, cy 为主点坐标(像素)。
            dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3]，None 时默认全零(无畸变)。
            marker_length: ArUco marker 的物理边长(米)，用于位姿估计的尺度恢复。
            aruco_dict_type: ArUco 字典类型，默认 DICT_4X4_50 (4x4 bit, 50 个 marker)。
            min_detection_confidence: MediaPipe 首次检测的置信度阈值 (0~1)。
                                      值越高越严格，漏检率上升但误检率下降。
            min_tracking_confidence: MediaPipe 跟踪阶段的置信度阈值 (0~1)。
                                     低于此值时会重新触发检测而非继续跟踪。
            selfie: 是否为自拍模式(图像左右镜像)。
                    影响 MediaPipe 返回的手部标签(Left/Right)的解释方式。
            depth_median_kernel: 深度图中值滤波核大小(必须为奇数)，0 表示不滤波。
                                 用于去除深度图中的椒盐噪声。
            depth_search_radius: 当关键点处深度无效时，在周围多大像素半径内搜索有效深度。
            depth_unit: 深度图的单位，"m" 表示米，"mm" 表示毫米。
        """
        # ---------- MediaPipe 手部检测器 ----------
        # static_image_mode=False: 启用跟踪模式，连续帧间复用检测结果以提高速度
        # max_num_hands=1: 只检测一只手，减少计算量
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        # 根据手的类型选择对应的 operator→MANO 旋转矩阵
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        # ---------- 手部标签映射 ----------
        # MediaPipe 假设输入图像是自拍(镜像)的，所以:
        #   - 自拍模式: MediaPipe 标签与实际手一致
        #   - 非自拍模式(如桌面相机): MediaPipe 标签与实际手相反
        # 因此非自拍时需要反转标签来匹配目标手
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

        # ---------- 相机内参 ----------
        if camera_matrix is None:
            raise ValueError("camera_matrix must be provided")
        self.camera_matrix = camera_matrix.astype(np.float32)
        self.dist_coeffs = (
            dist_coeffs.astype(np.float32)
            if dist_coeffs is not None
            else np.zeros(5, dtype=np.float32)
        )
        # 缓存常用内参值，反投影时频繁使用
        self.fx = self.camera_matrix[0, 0]  # x 方向焦距 (像素)
        self.fy = self.camera_matrix[1, 1]  # y 方向焦距 (像素)
        self.cx = self.camera_matrix[0, 2]  # 主点 x 坐标 (像素)
        self.cy = self.camera_matrix[1, 2]  # 主点 y 坐标 (像素)

        # ---------- ArUco marker 检测器 ----------
        # 用于估计相机相对于桌面(marker)的位姿
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = marker_length  # marker 物理边长(米)

        # ---------- 深度处理参数 ----------
        self.depth_median_kernel = depth_median_kernel
        self.depth_search_radius = depth_search_radius
        self.depth_unit = depth_unit

        # ---------- 缓存的 marker 变换 ----------
        # 当 marker 被手遮挡时，复用上一帧的变换矩阵，避免丢失桌面坐标系
        self._T_cam2table = None
        self._last_marker_detected = False
        self.last_debug_info = {}

    def detect_marker(self, frame_bgr):
        """检测 ArUco marker 并计算相机→桌面的变换矩阵。

        原理:
          1. 在图像中检测 ArUco marker 的四个角点
          2. 利用 marker 的已知物理尺寸 + 相机内参，通过 PnP 求解
             marker 相对于相机的旋转向量 rvec 和平移向量 tvec
          3. 将 rvec 转为旋转矩阵 R，组合成 4x4 齐次变换矩阵 T_cam2table
             T_cam2table 表示: 桌面坐标系原点在相机坐标系中的位姿

        注意: 当 marker 未被检测到时(如被手遮挡)，返回上一次缓存的变换矩阵。

        Args:
            frame_bgr: BGR 格式图像 (H, W, 3)，OpenCV 默认格式。

        Returns:
            T_cam2table: 4x4 齐次变换矩阵 (相机坐标系 → 桌面坐标系)，
                         即桌面坐标系在相机坐标系中的表示。
                         如果从未检测到 marker 则返回 None。
        """
        # detectMarkers 返回: corners(角点列表), ids(marker ID), rejected(被拒绝的候选)
        corners, ids, _ = self.aruco_detector.detectMarkers(frame_bgr)
        if ids is None or len(ids) == 0:
            # marker 未检测到，返回缓存值(可能为 None)
            self._last_marker_detected = False
            return self._T_cam2table

        # estimatePoseSingleMarkers: 对每个检测到的 marker 做 PnP 位姿估计
        # rvecs: 旋转向量(Rodrigues 表示), tvecs: 平移向量(米)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )

        # 只使用第一个检测到的 marker
        rvec = rvecs[0].flatten()  # (3,) 旋转向量
        tvec = tvecs[0].flatten()  # (3,) 平移向量(米)

        # Rodrigues 旋转向量 → 3x3 旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 组装 4x4 齐次变换矩阵
        T_cam2table = np.eye(4)
        T_cam2table[:3, :3] = R     # 旋转部分
        T_cam2table[:3, 3] = tvec   # 平移部分
        self._T_cam2table = T_cam2table  # 缓存，供 marker 被遮挡时复用
        self._last_marker_detected = True
        return T_cam2table

    def _sample_depth(self, depth_image, pixel_coords):
        """在像素坐标处采样深度值，支持邻域搜索兜底。

        深度图中经常存在无效像素(值为 0 或 NaN)，尤其在物体边缘和反光区域。
        本方法的策略:
          1. 首先尝试精确像素位置的深度值
          2. 如果无效，在 depth_search_radius 范围内搜索有效深度，取中值
          3. 如果邻域内也全部无效，返回 0.0 (后续由 detect() 方法处理)

        Args:
            depth_image: 深度图 (H, W)，值为米。
            pixel_coords: (N, 2) 像素坐标数组，每行为 (u, v)。
                          u = 水平方向(列), v = 垂直方向(行)。

        Returns:
            depths: (N,) 深度值数组(米)。无效位置为 0.0。
        """
        h, w = depth_image.shape[:2]
        r = self.depth_search_radius
        depths = np.zeros(len(pixel_coords), dtype=np.float64)

        for i, (u, v) in enumerate(pixel_coords):
            # 浮点像素坐标 → 整数像素坐标，并裁剪到图像边界内
            u_int, v_int = int(round(u)), int(round(v))
            u_int = np.clip(u_int, 0, w - 1)
            v_int = np.clip(v_int, 0, h - 1)

            # 尝试精确位置的深度
            d = depth_image[v_int, u_int]  # 注意: OpenCV 图像索引为 [行, 列] = [v, u]
            if d > 0 and np.isfinite(d):
                depths[i] = d
                continue

            # 精确位置无效 → 在邻域 patch 中搜索有效深度
            u_min = max(0, u_int - r)
            u_max = min(w, u_int + r + 1)
            v_min = max(0, v_int - r)
            v_max = min(h, v_int + r + 1)
            patch = depth_image[v_min:v_max, u_min:u_max]
            valid = patch[(patch > 0) & np.isfinite(patch)]
            if len(valid) > 0:
                depths[i] = np.median(valid)  # 取中值，对异常值更鲁棒
            else:
                depths[i] = 0.0  # 邻域内也无有效深度

        return depths

    def _backproject(self, pixel_coords, depths):
        """将 2D 像素坐标 + 深度值反投影为 3D 相机坐标。

        针孔相机模型的反投影公式:
          X = (u - cx) * Z / fx
          Y = (v - cy) * Z / fy
          Z = depth

        其中 (fx, fy) 为焦距, (cx, cy) 为主点, (u, v) 为像素坐标。

        Args:
            pixel_coords: (N, 2) 像素坐标数组 (u, v)。
            depths: (N,) 深度值数组(米)。

        Returns:
            points_3d: (N, 3) 相机坐标系下的 3D 点 [X, Y, Z]。
        """
        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]
        x = (u - self.cx) * depths / self.fx
        y = (v - self.cy) * depths / self.fy
        z = depths
        return np.stack([x, y, z], axis=1)

    def detect(self, rgb, depth):
        """从一对 RGB-D 图像中检测手部关节的 3D 位置。

        这是本类的核心方法，串联了整个检测流水线:
          深度滤波 → ArUco 检测 → MediaPipe 2D 检测 → 深度采样 →
          反投影 → 坐标变换 → 腕部位姿估计 → MANO 约定转换

        Args:
            rgb: RGB 图像 (H, W, 3), uint8, 通道顺序为 R-G-B。
            depth: 深度图 (H, W), float, 值为米。

        Returns:
            num_box: 检测到的手的数量 (0 表示未检测到)。
            joint_pos: (21, 3) 以腕部为原点的关节位置(桌面坐标系, MANO 约定, 米)。
                       未检测到时为 None。
            keypoint_2d: MediaPipe 原始 2D 归一化关键点。未检测到时为 None。
            wrist_pos: (3,) 腕部在桌面坐标系下的绝对位置(米)。未检测到时为 None。
            wrist_rot: (3, 3) 腕部旋转矩阵(MANO 约定)。未检测到时为 None。
            T_cam2table: 4x4 相机→桌面变换矩阵。marker 从未检测到时为 None。
        """
        self.last_debug_info = dict(
            marker_detected=False,
            used_cached_marker=False,
            pixel_coords=None,
            depths=None,
            invalid_depth_mask=None,
            keypoint_2d=None,
            num_box=0,
        )

        # ===== 步骤 1: 深度图预处理 =====
        depth_m = depth.astype(np.float32)
        if self.depth_unit == "mm":
            depth_m = depth_m / 1000.0
        # 中值滤波去除深度图中的椒盐噪声，保留边缘
        if self.depth_median_kernel > 0:
            depth_filtered = cv2.medianBlur(depth_m, self.depth_median_kernel)
        else:
            depth_filtered = depth_m

        # ===== 步骤 2: ArUco marker 检测 =====
        # detect_marker 需要 BGR 格式 (OpenCV 惯例)
        bgr = rgb[..., ::-1]  # RGB → BGR
        T_cam2table = self.detect_marker(bgr)
        self.last_debug_info["marker_detected"] = self._last_marker_detected
        self.last_debug_info["used_cached_marker"] = (
            (not self._last_marker_detected) and (T_cam2table is not None)
        )

        # ===== 步骤 3: MediaPipe 手部 2D 关键点检测 =====
        # process() 接受 RGB 格式图像，返回最多 max_num_hands 只手的关键点
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None, None, T_cam2table

        # 在检测到的手中找到目标手(Left/Right)
        # MediaPipe 的 multi_handedness 包含每只手的分类标签
        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            # 检测到了手，但不是目标手(左/右不匹配)
            return 0, None, None, None, None, T_cam2table

        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)
        self.last_debug_info["keypoint_2d"] = keypoint_2d
        self.last_debug_info["num_box"] = num_box

        # ===== 步骤 4: 2D 归一化坐标 → 像素坐标 =====
        # MediaPipe 输出的关键点坐标是 [0,1] 归一化的，需要乘以图像尺寸
        img_h, img_w = rgb.shape[:2]
        pixel_coords = self.parse_keypoint_2d(keypoint_2d, (img_h, img_w))
        self.last_debug_info["pixel_coords"] = pixel_coords.copy()

        # ===== 步骤 5: 深度采样 + 无效深度处理 =====
        depths = self._sample_depth(depth_filtered, pixel_coords)
        invalid_mask = depths <= 0
        self.last_debug_info["depths"] = depths.copy()
        self.last_debug_info["invalid_depth_mask"] = invalid_mask.copy()
        if np.all(invalid_mask):
            # 所有关键点的深度都无效，无法重建 3D
            return 0, None, None, None, None, T_cam2table

        # 部分关键点深度无效时，用有效关键点的平均深度填充
        # 这是一个简单的兜底策略，假设手部各点深度相近
        if np.any(invalid_mask) and not np.all(invalid_mask):
            valid_mean = np.mean(depths[~invalid_mask])
            depths[invalid_mask] = valid_mean

        # ===== 步骤 6: 反投影为 3D 相机坐标 =====
        kp_3d_cam = self._backproject(pixel_coords, depths)

        # ===== 步骤 7: 相机坐标系 → 桌面坐标系 =====
        # T_cam2table: 桌面坐标系在相机坐标系中的表示
        # T_table2cam = T_cam2table^{-1}: 相机坐标系在桌面坐标系中的表示
        # P_table = T_table2cam @ P_cam: 将相机坐标系下的点变换到桌面坐标系
        if T_cam2table is not None:
            T_table2cam = np.linalg.inv(T_cam2table)
            # 转为齐次坐标 (N, 4): [x, y, z, 1]
            kp_3d_cam_homo = np.hstack(
                [kp_3d_cam, np.ones((kp_3d_cam.shape[0], 1))]
            )
            # 矩阵乘法变换坐标系，取前 3 列 (去掉齐次分量)
            kp_3d_table = (T_table2cam @ kp_3d_cam_homo.T).T[:, :3]
        else:
            # 无 marker 变换时，直接使用相机坐标(降级模式)
            kp_3d_table = kp_3d_cam

        # ===== 步骤 8: 腕部位置提取 + 关键点中心化 =====
        # 腕部(索引 0)的绝对位置，用于 retargeting 中的 free joint 初始化
        wrist_pos = kp_3d_table[0].copy()

        # 以腕部为原点做中心化，得到相对关节位置
        # 这样 joint_pos 只包含手指的形状信息，与腕部的绝对位置解耦
        kp_3d_centered = kp_3d_table - kp_3d_table[0:1, :]

        # ===== 步骤 9: 腕部旋转估计 + MANO 约定转换 =====
        # estimate_frame_from_hand_points: 从手部关键点拟合一个 3D 坐标帧
        # 返回 (3, 3) 旋转矩阵，列向量为 [x, normal, z] 三个正交轴
        wrist_rot_frame = self.estimate_frame_from_hand_points(kp_3d_centered)

        # 将关节位置和腕部旋转从操作者坐标系转换到 MANO 约定
        # joint_pos = kp_3d_centered @ wrist_rot_frame @ operator2mano
        #   - kp_3d_centered @ wrist_rot_frame: 将关节位置旋转到腕部局部坐标系
        #   - @ operator2mano: 再从操作者约定转到 MANO 约定
        joint_pos = kp_3d_centered @ wrist_rot_frame @ self.operator2mano
        wrist_rot = wrist_rot_frame @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, wrist_pos, wrist_rot, T_cam2table

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
    ):
        """在图像上绘制手部骨架(关键点 + 连接线)。

        用于可视化和 debug，支持两种绘制风格:
          - "default": MediaPipe 默认彩色风格
          - "white":   红色关键点 + 白色连接线，在深色背景上更清晰

        Args:
            image: 待绘制的图像 (H, W, 3)，会被原地修改。
            keypoint_2d: MediaPipe 归一化关键点 (21 个点, 坐标范围 [0,1])。
            style: 绘制风格，"default" 或 "white"。

        Returns:
            image: 绘制了骨架的图像(与输入为同一对象)。
        """
        if style == "default":
            # 使用 MediaPipe 内置的默认绘制样式
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            # 自定义样式: 红色实心圆点 + 白色细线
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1  # -1 = 填充圆
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        return image

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        """将 MediaPipe 归一化关键点转换为像素坐标。

        MediaPipe 输出的关键点坐标是 [0, 1] 范围的归一化值:
          landmark.x = 水平位置 / 图像宽度
          landmark.y = 垂直位置 / 图像高度

        本方法将其乘以图像实际尺寸，得到像素坐标 (u, v)。

        Args:
            keypoint_2d: MediaPipe 归一化关键点 (21 个点)。
            img_size: (H, W) 图像尺寸，注意顺序是 (高, 宽)。

        Returns:
            keypoint: (21, 2) 像素坐标数组，每行为 (u, v)。
                      u = 水平像素坐标, v = 垂直像素坐标。
        """
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x  # 归一化 x (水平)
            keypoint[i][1] = keypoint_2d.landmark[i].y  # 归一化 y (垂直)
        # 归一化坐标 × [宽, 高] → 像素坐标
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """从手部 3D 关键点估计腕部的局部坐标帧(旋转矩阵)。

        使用 3 个关键点来定义手掌平面:
          - 索引 0: 腕部 (WRIST) — 中心化后为原点 [0,0,0]
          - 索引 5: 食指根部 (INDEX_FINGER_MCP)
          - 索引 9: 中指根部 (MIDDLE_FINGER_MCP)

        坐标帧构建步骤:
          1. x 轴: 从中指根部指向腕部的方向 (手掌纵向)
          2. normal: 通过 SVD 拟合 3 点平面的法向量 (手掌法线方向)
          3. z 轴: x × normal 的叉积 (手掌横向, 从小指侧指向食指侧)
          4. Gram-Schmidt 正交化确保三轴正交归一
          5. 方向校正: 确保 z 轴与"小指→食指"方向一致

        Args:
            keypoint_3d_array: (21, 3) 以腕部为原点的 3D 关键点。

        Returns:
            frame: (3, 3) 旋转矩阵，列向量为 [x, normal, z]。
                   表示腕部局部坐标帧在桌面坐标系中的朝向。
        """
        assert keypoint_3d_array.shape == (21, 3)
        # 取腕部(0)、食指根部(5)、中指根部(9) 三个点定义手掌平面
        points = keypoint_3d_array[[0, 5, 9], :]

        # x 轴方向: 从中指根部(points[2])指向腕部(points[0])
        x_vector = points[0] - points[2]

        # SVD 拟合平面法向量:
        # 将 3 个点去均值后做 SVD，最小奇异值对应的右奇异向量即为平面法向量
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)
        normal = v[2, :]  # 最小奇异值对应的方向 = 平面法向量

        # Gram-Schmidt 正交化: 将 x_vector 投影到法平面上，确保与 normal 正交
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)  # 归一化
        z = np.cross(x, normal)     # z = x × normal，构成右手坐标系

        # 方向校正: 确保 z 轴与"中指根部→食指根部"方向一致
        # (即从小指侧指向食指侧)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1  # 翻转法向量
            z *= -1       # 相应翻转 z 轴

        # 组装旋转矩阵: 列向量为 [x, normal, z]
        frame = np.stack([x, normal, z], axis=1)
        return frame
