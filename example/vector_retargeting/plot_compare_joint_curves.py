"""Compare joint trajectories from two vector retargeting pickle files.

The default inputs compare the original linker_l20 trajectory against the
linker_l20a trajectory generated from the same video.
"""

import argparse
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_L20_PICKLE = Path(__file__).parent / "data" / "finger_l20_vec.pkl"
DEFAULT_L20A_PICKLE = Path(__file__).parent / "data" / "finger_l20a_vec.pkl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "finger_l20_vs_l20a_joint_curves.png"

FINGER_ORDER: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("thumb", ("thumb",)),
    ("index", ("index",)),
    ("middle", ("middle",)),
    ("ring", ("ring",)),
    ("little", ("little", "pinky")),
)


def _extract_qpos(frame) -> np.ndarray:
    if isinstance(frame, dict):
        if "qpos" not in frame:
            raise ValueError("Frame dict does not contain a 'qpos' field")
        frame = frame["qpos"]
    return np.asarray(frame, dtype=np.float64)


def load_trajectory(pickle_path: Path) -> Tuple[np.ndarray, List[str]]:
    with pickle_path.open("rb") as f:
        result = pickle.load(f)

    frames = result.get("data", result) if isinstance(result, dict) else result
    if len(frames) == 0:
        raise ValueError(f"No frame data found in {pickle_path}")

    qpos = np.asarray([_extract_qpos(frame) for frame in frames], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Expected a 2D qpos array, got shape {qpos.shape}")

    meta_data = result.get("meta_data", {}) if isinstance(result, dict) else {}
    joint_names = list(meta_data.get("joint_names", []))
    dof = qpos.shape[1]
    if len(joint_names) < dof:
        joint_names.extend(f"joint_{i}" for i in range(len(joint_names), dof))
    elif len(joint_names) > dof:
        joint_names = joint_names[:dof]

    return qpos, joint_names


def _joint_groups(joint_names: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = OrderedDict((name, []) for name, _ in FINGER_ORDER)
    other: List[str] = []

    for joint_name in joint_names:
        lower_name = joint_name.lower()
        matched_group = None
        for group_name, keywords in FINGER_ORDER:
            if any(keyword in lower_name for keyword in keywords):
                matched_group = group_name
                break
        if matched_group is None:
            other.append(joint_name)
        else:
            groups[matched_group].append(joint_name)

    groups = OrderedDict((name, values) for name, values in groups.items() if values)
    if other:
        groups["other"] = other
    return groups


def _plot_joint(
    ax: plt.Axes,
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    qpos_a: np.ndarray,
    qpos_b: np.ndarray,
    names_a: Sequence[str],
    names_b: Sequence[str],
    joint_name: str,
    label_a: str,
    label_b: str,
    unit: str,
) -> None:
    name_to_index_a = {name: i for i, name in enumerate(names_a)}
    name_to_index_b = {name: i for i, name in enumerate(names_b)}

    if joint_name in name_to_index_a:
        values = qpos_a[:, name_to_index_a[joint_name]]
        if unit == "deg":
            values = np.rad2deg(values)
        ax.plot(frames_a, values, linewidth=1.3, label=label_a)

    if joint_name in name_to_index_b:
        values = qpos_b[:, name_to_index_b[joint_name]]
        if unit == "deg":
            values = np.rad2deg(values)
        ax.plot(frames_b, values, linewidth=1.3, linestyle="--", label=label_b)

    ax.set_title(joint_name, fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Angle ({unit})")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7)


def plot_compare_joint_curves(
    pickle_a: Path,
    pickle_b: Path,
    output_path: Path,
    label_a: str = "linker_l20",
    label_b: str = "linker_l20a",
    unit: str = "rad",
    dpi: int = 160,
) -> None:
    if unit not in {"rad", "deg"}:
        raise ValueError("unit must be 'rad' or 'deg'")

    qpos_a, names_a = load_trajectory(pickle_a)
    qpos_b, names_b = load_trajectory(pickle_b)

    all_joint_names = list(OrderedDict.fromkeys([*names_a, *names_b]))
    groups = _joint_groups(all_joint_names)
    ordered_joint_names = [joint for joints in groups.values() for joint in joints]

    n_cols = 4
    n_rows = int(np.ceil(len(ordered_joint_names) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    frames_a = np.arange(qpos_a.shape[0])
    frames_b = np.arange(qpos_b.shape[0])
    axes_flat = axes.flatten()

    for ax, joint_name in zip(axes_flat, ordered_joint_names):
        _plot_joint(
            ax=ax,
            frames_a=frames_a,
            frames_b=frames_b,
            qpos_a=qpos_a,
            qpos_b=qpos_b,
            names_a=names_a,
            names_b=names_b,
            joint_name=joint_name,
            label_a=label_a,
            label_b=label_b,
            unit=unit,
        )

    for ax in axes_flat[len(ordered_joint_names) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Joint trajectory comparison: {pickle_a.name} vs {pickle_b.name}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"{label_a}: {qpos_a.shape[0]} frames, {qpos_a.shape[1]} joints")
    print(f"{label_b}: {qpos_b.shape[0]} frames, {qpos_b.shape[1]} joints")
    print(f"Saved joint comparison plot to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and compare every joint trajectory in two retargeting pickle files."
    )
    parser.add_argument("--pickle-a", type=Path, default=DEFAULT_L20_PICKLE)
    parser.add_argument("--pickle-b", type=Path, default=DEFAULT_L20A_PICKLE)
    parser.add_argument("--label-a", type=str, default="linker_l20")
    parser.add_argument("--label-b", type=str, default="linker_l20a")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unit", choices=("rad", "deg"), default="rad")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_compare_joint_curves(
        pickle_a=args.pickle_a,
        pickle_b=args.pickle_b,
        output_path=args.output,
        label_a=args.label_a,
        label_b=args.label_b,
        unit=args.unit,
        dpi=args.dpi,
    )
