"""Export DexYCB position-retargeting trajectories to a pickle file.

This is the non-rendering counterpart of visualize_hand_object.py. It reads a
DexYCB human hand-object sequence, computes MANO joints, retargets them to a
robot hand with the offline position config, and saves qpos frames for replay.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import torch
import tqdm
import tyro
from pytransform3d import rotations
from pytransform3d import transformations as pt

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from mano_layer import MANOLayer

# For numpy version compatibility with older manopth/chumpy code paths.
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_


def _compute_mano_joint_world(
    mano_layer: MANOLayer,
    camera_pose,
    hand_pose_frame: np.ndarray,
):
    """Return MANO joints in the same scene/world frame used by the viewer."""
    if np.abs(hand_pose_frame).sum() < 1e-5:
        return None

    pose = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    translation = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    _, joint = mano_layer(pose, translation)
    joint = joint.cpu().numpy()[0]

    camera_mat = camera_pose.to_transformation_matrix()
    joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
    return np.ascontiguousarray(joint)


def _compute_wrist_rot_world(camera_pose, hand_pose_frame: np.ndarray) -> np.ndarray:
    """Return wrist rotation from MANO local axes to the viewer/world frame."""
    wrist_rot_camera = rotations.matrix_from_compact_axis_angle(
        hand_pose_frame[0, 0:3]
    )
    camera_mat = camera_pose.to_transformation_matrix()
    return camera_mat[:3, :3] @ wrist_rot_camera


def _first_valid_frame(hand_pose: np.ndarray) -> int:
    for frame_id in range(hand_pose.shape[0]):
        if np.abs(hand_pose[frame_id]).sum() >= 1e-5:
            return frame_id
    raise ValueError("No valid hand pose frame found in this DexYCB sequence.")


def export_dexycb_retarget_pickle(
    dexycb_dir: str,
    output_path: str,
    robot_name: RobotName = RobotName.linker_l20,
    hand_type: HandType = HandType.right,
    data_id: int = 4,
    config_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: int = 10,
):
    """Generate an Isaac-Gym-friendly qpos pickle from a DexYCB sequence.

    Args:
        dexycb_dir: DexYCB root containing subject, calibration, and models dirs.
        output_path: Destination .pkl/.pickle file.
        robot_name: Target robot hand.
        hand_type: MANO/robot hand side. The original example uses right hand.
        data_id: DexYCB trajectory index, matching visualize_hand_object.py.
        config_path: Optional custom position-retargeting config.
        start_frame: Optional inclusive frame index. Defaults to first valid hand.
        end_frame: Optional exclusive frame index. Defaults to sequence end.
        fps: Playback frame rate recorded in metadata.
    """
    data_root = Path(dexycb_dir).absolute()
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir does not exist: {data_root}")

    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    if config_path is None:
        config_path = str(
            get_default_config_path(robot_name, RetargetingType.position, hand_type)
        )

    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type.name)
    if data_id < 0 or data_id >= len(dataset):
        raise ValueError(
            f"data_id {data_id} out of range; dataset has {len(dataset)} sequences."
        )

    sampled_data = dataset[data_id]
    hand_pose = sampled_data["hand_pose"]
    object_pose = sampled_data["object_pose"]
    num_frames = hand_pose.shape[0]

    first_valid = _first_valid_frame(hand_pose)
    frame_start = first_valid if start_frame is None else max(start_frame, first_valid)
    frame_end = num_frames if end_frame is None else min(end_frame, num_frames)
    if frame_start >= frame_end:
        raise ValueError(
            f"Invalid frame range [{frame_start}, {frame_end}) for sequence with {num_frames} frames."
        )

    retargeting = RetargetingConfig.load_from_file(config_path).build()
    indices = retargeting.optimizer.target_link_human_indices

    mano_layer = MANOLayer(hand_type.name, sampled_data["hand_shape"].astype(np.float32))
    pose_vec = pt.pq_from_transform(sampled_data["extrinsics"])
    camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

    warm_joint = _compute_mano_joint_world(
        mano_layer, camera_pose, hand_pose[frame_start]
    )
    if warm_joint is None:
        raise ValueError(f"Frame {frame_start} does not contain a valid hand pose.")
    wrist_quat = rotations.quaternion_from_compact_axis_angle(
        hand_pose[frame_start][0, 0:3]
    )
    retargeting.warm_start(
        warm_joint[0, :],
        wrist_quat,
        hand_type=hand_type,
        is_mano_convention=True,
    )

    frames = []
    skipped = 0
    for frame_id in tqdm.trange(frame_start, frame_end):
        joint = _compute_mano_joint_world(mano_layer, camera_pose, hand_pose[frame_id])
        if joint is None:
            skipped += 1
            continue
        wrist_rot = _compute_wrist_rot_world(camera_pose, hand_pose[frame_id])

        ref_value = joint[indices, :]
        qpos = retargeting.retarget(ref_value)

        frames.append(
            dict(
                frame_id=frame_id,
                qpos=qpos,
                wrist_pos=joint[0, :],
                wrist_rot=wrist_rot,
                joint_pos=joint,
                object_pose=object_pose[frame_id],
            )
        )

    meta_data = dict(
        source="DexYCB",
        capture_name=sampled_data["capture_name"],
        dexycb_dir=str(data_root),
        data_id=data_id,
        config_path=str(config_path),
        robot_name=robot_name.name,
        hand_type=hand_type.name,
        retargeting_type=str(retargeting.optimizer.retargeting_type),
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
        target_link_human_indices=np.asarray(indices).tolist(),
        fps=fps,
        frame_start=frame_start,
        frame_end=frame_end,
        skipped_frames=skipped,
        ycb_ids=sampled_data["ycb_ids"],
        object_mesh_file=sampled_data["object_mesh_file"],
        coordinate_contract=(
            "qpos is produced by offline position retargeting from MANO joints "
            "transformed into the same world frame used by visualize_hand_object.py. "
            "The first dummy free joints, when present, are xyz translation and "
            "intrinsic XYZ Euler rotation."
        ),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(dict(data=frames, meta_data=meta_data), f)

    print(f"Saved {len(frames)} frames to {output}")
    print(f"Skipped {skipped} invalid frames.")
    retargeting.verbose()


def main(
    dexycb_dir: str,
    output_path: str,
    robot_name: RobotName = RobotName.linker_l20,
    hand_type: HandType = HandType.right,
    data_id: int = 4,
    config_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: int = 10,
):
    export_dexycb_retarget_pickle(
        dexycb_dir=dexycb_dir,
        output_path=output_path,
        robot_name=robot_name,
        hand_type=hand_type,
        data_id=data_id,
        config_path=config_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
    )


if __name__ == "__main__":
    tyro.cli(main)
