import pickle
from pathlib import Path
from typing import Optional

import cv2
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def _retarget_frame(retargeting: SeqRetargeting, joint_pos):
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting_type == "POSITION":
        ref_value = joint_pos[indices, :]
    else:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    return retargeting.retarget(ref_value)


def _save_pickle(
    retargeting: SeqRetargeting,
    data,
    output_path: str,
    config_path: str,
    **extra_meta_data,
):
    meta_data = dict(
        config_path=config_path,
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
        **extra_meta_data,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)

    print(f"Saved {len(data)} frames to {output_path}")


def retarget_video(
    retargeting: SeqRetargeting,
    video_path: str,
    output_path: str,
    config_path: str,
    hand_type: HandType,
):
    cap = cv2.VideoCapture(video_path)

    data = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        detector = SingleHandDetector(hand_type=hand_type.name.capitalize(), selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(
                    rgb
                )
                if num_box == 0:
                    pbar.update(1)
                    continue

                qpos = _retarget_frame(retargeting, joint_pos)
                data.append(qpos)
                pbar.update(1)

        _save_pickle(
            retargeting,
            data,
            output_path,
            config_path,
            source_type="video",
            source_path=str(video_path),
        )

        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()


def retarget_image_sequence(
    retargeting: SeqRetargeting,
    rgb_dir: str,
    output_path: str,
    config_path: str,
    hand_type: HandType,
):
    rgb_path = Path(rgb_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = sorted(
        [path for path in rgb_path.iterdir() if path.suffix.lower() in image_extensions]
    )
    if len(image_files) == 0:
        print(f"Error: No RGB images found in {rgb_dir}")
        return

    detector = SingleHandDetector(hand_type=hand_type.name.capitalize(), selfie=False)
    data = []
    skipped = 0

    with tqdm.tqdm(total=len(image_files)) as pbar:
        for image_file in image_files:
            frame = cv2.imread(str(image_file))
            if frame is None:
                skipped += 1
                pbar.update(1)
                continue

            rgb = frame[..., ::-1]
            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
            if num_box == 0:
                skipped += 1
                pbar.update(1)
                continue

            qpos = _retarget_frame(retargeting, joint_pos)
            data.append(qpos)
            pbar.update(1)

    _save_pickle(
        retargeting,
        data,
        output_path,
        config_path,
        source_type="rgb_image_sequence",
        source_path=str(rgb_path),
        input_frame_count=len(image_files),
        skipped_frames=skipped,
        hand_type=hand_type.name,
        retargeting_type=str(retargeting.optimizer.retargeting_type),
    )
    print(f"Processed {len(data)} frames, skipped {skipped} frames.")
    retargeting.verbose()


def main(
    robot_name: RobotName,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    video_path: Optional[str] = None,
    rgb_dir: Optional[str] = None,
):
    """
    Detects human hand pose from a video or RGB image sequence and translates it into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        video_path: The file path for the input video in .mp4 format.
        rgb_dir: A directory containing RGB image frames, sorted by filename.
    """
    if (video_path is None) == (rgb_dir is None):
        raise ValueError("Please specify exactly one input: --video-path or --rgb-dir.")

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    # import numpy as np
    # # retargeting = RetargetingConfig.load_from_file(config_path).build()
    # retargeting.set_qpos(np.zeros(len(retargeting.joint_names), dtype=np.float32))

    if rgb_dir is not None:
        retarget_image_sequence(
            retargeting, rgb_dir, output_path, str(config_path), hand_type
        )
    else:
        retarget_video(
            retargeting, video_path, output_path, str(config_path), hand_type
        )


if __name__ == "__main__":
    tyro.cli(main)
