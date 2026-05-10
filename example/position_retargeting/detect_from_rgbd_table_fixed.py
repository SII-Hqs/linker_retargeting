"""
RGB-D hand detection and retargeting with explicit table-frame POSITION targets.

This script is a fixed-coordinate variant of detect_from_rgbd.py. It keeps the
original script untouched, but changes two important assumptions:

1. Depth images are scaled explicitly. The default depth_scale is 0.001 because
   the .npy depth data for this project is in millimeters.
2. POSITION retargeting receives absolute 3D points in the table frame, matching
   the coordinate frame of the robot FK output.
"""

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
from rgbd_hand_detector import (
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    RGBDHandDetector,
)

from detect_from_rgbd import (
    _create_visualization_writer,
    _render_detection_overlay,
    load_camera_params,
)


class TableFrameHandDetector(RGBDHandDetector):
    """RGBDHandDetector variant that exposes table-frame and MANO-frame points."""

    def detect_table_transform(self, frame_bgr):
        """Return explicit marker/table transforms.

        OpenCV ArUco pose estimation returns the marker/table coordinate frame
        expressed in the camera frame:

            P_cam = T_table2cam @ P_table

        For depth points reconstructed from RGB-D images, the input point is
        already in camera coordinates. To move it into the desired table/world
        frame, use the inverse:

            P_table = T_cam2table @ P_cam
        """
        corners, ids, _ = self.aruco_detector.detectMarkers(frame_bgr)
        if ids is None or len(ids) == 0:
            self._last_marker_detected = False
            T_table2cam = self._T_cam2table
        else:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            rvec = rvecs[0].flatten()
            tvec = tvecs[0].flatten()
            R_table2cam, _ = cv2.Rodrigues(rvec)

            T_table2cam = np.eye(4)
            T_table2cam[:3, :3] = R_table2cam
            T_table2cam[:3, 3] = tvec

            # Keep using the parent cache slot, but store the correctly named
            # semantic value: marker/table frame to camera frame.
            self._T_cam2table = T_table2cam
            self._last_marker_detected = True

        if T_table2cam is None:
            return None, None

        T_cam2table = np.linalg.inv(T_table2cam)
        return T_table2cam, T_cam2table

    def detect_table_frame(self, rgb, depth):
        self.last_debug_info = dict(
            marker_detected=False,
            used_cached_marker=False,
            pixel_coords=None,
            depths=None,
            invalid_depth_mask=None,
            keypoint_2d=None,
            num_box=0,
        )

        if self.depth_median_kernel > 0:
            depth_filtered = cv2.medianBlur(
                depth.astype(np.float32), self.depth_median_kernel
            )
        else:
            depth_filtered = depth.astype(np.float32)

        bgr = rgb[..., ::-1]
        T_table2cam, T_cam2table = self.detect_table_transform(bgr)
        self.last_debug_info["marker_detected"] = self._last_marker_detected
        self.last_debug_info["used_cached_marker"] = (
            (not self._last_marker_detected) and (T_table2cam is not None)
        )

        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return dict(num_box=0, T_table2cam=T_table2cam)

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return dict(num_box=0, T_table2cam=T_table2cam)

        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)
        self.last_debug_info["keypoint_2d"] = keypoint_2d
        self.last_debug_info["num_box"] = num_box

        img_h, img_w = rgb.shape[:2]
        pixel_coords = self.parse_keypoint_2d(keypoint_2d, (img_h, img_w))
        self.last_debug_info["pixel_coords"] = pixel_coords.copy()

        depths = self._sample_depth(depth_filtered, pixel_coords)
        invalid_mask = depths <= 0
        self.last_debug_info["depths"] = depths.copy()
        self.last_debug_info["invalid_depth_mask"] = invalid_mask.copy()
        if np.all(invalid_mask):
            return dict(num_box=0, T_table2cam=T_table2cam)

        if np.any(invalid_mask):
            depths[invalid_mask] = np.mean(depths[~invalid_mask])

        kp_3d_cam = self._backproject(pixel_coords, depths)

        if T_cam2table is not None:
            kp_3d_cam_homo = np.hstack(
                [kp_3d_cam, np.ones((kp_3d_cam.shape[0], 1))]
            )
            # RGB-D gives P_cam. The desired optimization target is P_table.
            joint_pos_table = (T_cam2table @ kp_3d_cam_homo.T).T[:, :3]
        else:
            joint_pos_table = kp_3d_cam

        wrist_pos_table = joint_pos_table[0].copy()
        joint_pos_table_relative = joint_pos_table - wrist_pos_table[None, :]

        wrist_rot_frame = self.estimate_frame_from_hand_points(
            joint_pos_table_relative
        )
        operator2mano = (
            OPERATOR2MANO_RIGHT
            if self.operator2mano is OPERATOR2MANO_RIGHT
            else OPERATOR2MANO_LEFT
        )

        joint_pos_mano = joint_pos_table_relative @ wrist_rot_frame @ operator2mano
        wrist_rot_table_mano = wrist_rot_frame @ operator2mano

        return dict(
            num_box=num_box,
            keypoint_2d=keypoint_2d,
            joint_pos_table=joint_pos_table,
            joint_pos_table_relative=joint_pos_table_relative,
            joint_pos_mano=joint_pos_mano,
            wrist_pos_table=wrist_pos_table,
            wrist_rot_table_mano=wrist_rot_table_mano,
            T_table2cam=T_table2cam,
            T_cam2table=T_cam2table,
        )


def _load_depth(path: Path, depth_scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth_raw = np.load(str(path)).astype(np.float32)
    else:
        depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError(f"failed to read depth image: {path}")
        depth_raw = depth_raw.astype(np.float32)
    return depth_raw * float(depth_scale)


def retarget_rgbd_table_fixed(
    retargeting: SeqRetargeting,
    rgb_dir: str,
    depth_dir: str,
    output_path: str,
    config_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    hand_type: str,
    depth_scale: float,
    visualize: bool = False,
    visualization_output: Optional[str] = None,
    save_visualization_frames: bool = False,
):
    rgb_path = Path(rgb_dir)
    depth_path = Path(depth_dir)

    rgb_extensions = {".png", ".jpg", ".jpeg"}
    depth_extensions = {".png", ".npy"}
    rgb_files = sorted(
        [f for f in rgb_path.iterdir() if f.suffix.lower() in rgb_extensions]
    )
    depth_files = sorted(
        [f for f in depth_path.iterdir() if f.suffix.lower() in depth_extensions]
    )

    if not rgb_files:
        print(f"Error: No RGB images found in {rgb_dir}")
        return
    if not depth_files:
        print(f"Error: No depth images found in {depth_dir}")
        return
    if len(rgb_files) != len(depth_files):
        print(
            f"Warning: RGB ({len(rgb_files)}) and depth ({len(depth_files)}) "
            "frame counts do not match. Using minimum."
        )

    detector = TableFrameHandDetector(
        hand_type=hand_type,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length=marker_length,
        selfie=False,
    )

    vis_writer = None
    vis_frame_dir = None
    if visualization_output is not None:
        sample_bgr = cv2.imread(str(rgb_files[0]))
        if sample_bgr is not None:
            h, w = sample_bgr.shape[:2]
            vis_writer = _create_visualization_writer(visualization_output, (w, h))
    if save_visualization_frames:
        vis_base = (
            Path(visualization_output).with_suffix("")
            if visualization_output is not None
            else Path(output_path).with_suffix("")
        )
        vis_frame_dir = vis_base.parent / f"{vis_base.name}_vis_frames"
        vis_frame_dir.mkdir(parents=True, exist_ok=True)

    data = []
    skipped = 0
    num_frames = min(len(rgb_files), len(depth_files))

    with tqdm.tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            skipped_reason = None

            bgr = cv2.imread(str(rgb_files[i]))
            if bgr is None:
                skipped_reason = "rgb_read_failed"
                skipped += 1
                pbar.update(1)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            try:
                depth = _load_depth(depth_files[i], depth_scale)
            except ValueError:
                skipped_reason = "depth_read_failed"

            result = None
            if skipped_reason is None:
                result = detector.detect_table_frame(rgb, depth)
                if result.get("num_box", 0) == 0:
                    skipped_reason = "hand_not_detected"
                elif result.get("T_table2cam") is None:
                    skipped_reason = "aruco_not_detected"

            detection_ok = (
                result is not None
                and result.get("num_box", 0) > 0
                and result.get("joint_pos_table") is not None
            )
            marker_ok = result is not None and result.get("T_table2cam") is not None

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
                skipped += 1
                pbar.update(1)
                continue

            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "POSITION":
                # POSITION expects absolute 3D targets in the same frame as robot FK.
                ref_value = result["joint_pos_table"][indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = (
                    result["joint_pos_mano"][task_indices, :]
                    - result["joint_pos_mano"][origin_indices, :]
                )

            if (
                retargeting_type == "POSITION"
                and retargeting.optimizer.has_free_joint
                and not retargeting.is_warm_started
            ):
                wrist_quat = rotations.quaternion_from_matrix(
                    result["wrist_rot_table_mano"]
                )
                retargeting.warm_start(
                    result["wrist_pos_table"],
                    wrist_quat,
                    hand_type=HandType.right if hand_type == "Right" else HandType.left,
                    is_mano_convention=True,
                )

            qpos = retargeting.retarget(ref_value)

            frame_data = dict(
                qpos=qpos,
                wrist_pos=result["wrist_pos_table"],
                wrist_rot=result["wrist_rot_table_mano"],
                joint_pos_table=result["joint_pos_table"],
                joint_pos_table_relative=result["joint_pos_table_relative"],
                joint_pos_mano=result["joint_pos_mano"],
                T_table2cam=result["T_table2cam"],
                T_cam2table=result["T_cam2table"],
                depth_scale=depth_scale,
            )
            data.append(frame_data)
            pbar.update(1)

    if vis_writer is not None:
        vis_writer.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"Processed {len(data)} frames, skipped {skipped} frames.")

    meta_data = dict(
        config_path=config_path,
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.tolist(),
        marker_length=marker_length,
        depth_scale=depth_scale,
        coordinate_contract=(
            "POSITION ref_value uses joint_pos_table: absolute 3D points in the "
            "ArUco table frame, meters."
        ),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)
    print(f"Saved {len(data)} frames to {output_path}")
    retargeting.verbose()


def main(
    robot_name: RobotName,
    rgb_dir: str,
    depth_dir: str,
    output_path: str,
    camera_params_json: str = "camera_params/color_intrinsics.json",
    hand_type: HandType = HandType.right,
    marker_length: float = 0.04,
    depth_scale: float = 0.001,
    config_path: Optional[str] = None,
    visualize: bool = False,
    visualization_output: Optional[str] = None,
    save_visualization_frames: bool = False,
):
    """Retarget RGB-D hand data using table-frame absolute POSITION targets.

    Args:
        depth_scale: multiplier applied to raw depth values. Use 0.001 for
            millimeters to meters, or 1.0 if depth is already in meters.
    """
    camera_matrix, dist_coeffs = load_camera_params(camera_params_json)

    if config_path is None:
        from dex_retargeting.constants import RetargetingType

        config_path = str(
            get_default_config_path(robot_name, RetargetingType.position, hand_type)
        )

    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type_str = "Right" if hand_type == HandType.right else "Left"
    retarget_rgbd_table_fixed(
        retargeting=retargeting,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        output_path=output_path,
        config_path=config_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length=marker_length,
        hand_type=hand_type_str,
        depth_scale=depth_scale,
        visualize=visualize,
        visualization_output=visualization_output,
        save_visualization_frames=save_visualization_frames,
    )


if __name__ == "__main__":
    tyro.cli(main)
