import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_qpos_curves(pickle_path: str, joint_indices=None, save_fig=None):
    """
    绘制 pickle 文件中存储的 qpos 随时间变化的曲线。

    Args:
        pickle_path: 输出 pickle 文件路径（由 detect_from_rgbd.py 生成）。
        joint_indices: 要绘制的关节索引列表，例如 [0,1,2]；若为 None 则绘制所有关节。
        save_fig: 若指定，则保存图片到该路径；否则显示窗口。
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # 提取数据
    qpos_list = [frame['qpos'] for frame in data['data']]  # 每个元素是一维数组
    qpos_array = np.array(qpos_list)  # shape: (num_frames, dof)
    num_frames, dof = qpos_array.shape

    # 获取关节名称（若元数据中有）
    joint_names = data['meta_data'].get('joint_names', [f'joint_{i}' for i in range(dof)])

    # 选择要绘制的关节
    if joint_indices is None:
        joint_indices = list(range(dof))
    else:
        joint_indices = [i for i in joint_indices if 0 <= i < dof]

    # 创建子图
    n_joints = len(joint_indices)
    if n_joints == 0:
        print("没有有效的关节索引")
        return

    # 动态计算子图布局（每行最多4个）
    n_cols = min(4, n_joints)
    n_rows = (n_joints + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
    if n_joints == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, joint_idx in enumerate(joint_indices):
        ax = axes[idx]
        ax.plot(qpos_array[:, joint_idx], linewidth=1)
        ax.set_title(f"{joint_names[joint_idx]} (index {joint_idx})")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Joint angle (rad)")
        ax.grid(True, linestyle='--', alpha=0.6)

    # 隐藏多余的子图（如果关节数不是整数倍）
    for idx in range(n_joints, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Joint angle trajectories\n{Path(pickle_path).name}", fontsize=14)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150)
        print(f"图片已保存至: {save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize qpos curves from detect_from_rgbd output")
    parser.add_argument("pickle_file", type=str, help="Path to output pickle file")
    parser.add_argument("--joints", type=int, nargs='+', help="Joint indices to plot (e.g., 0 1 2 3)")
    parser.add_argument("--save", type=str, help="Save figure to this path")
    args = parser.parse_args()

    plot_qpos_curves(args.pickle_file, args.joints, args.save)