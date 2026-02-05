#!/usr/bin/env python3
"""
Demo script for the parametric cat skeleton model.

This script demonstrates the cat wireframe model by displaying
various predefined poses and allowing interactive exploration.

Usage:
    python demo_cat_skeleton.py                    # Show all poses
    python demo_cat_skeleton.py --pose sitting     # Show specific pose
    python demo_cat_skeleton.py --interactive      # Interactive pose viewer
    python demo_cat_skeleton.py --multiview        # Show multiple viewpoints
    python demo_cat_skeleton.py --compare sitting sleeping  # Compare two poses
"""

import argparse
import sys

# Add parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from animal_models import CatSkeleton, SkeletonVisualizer


def show_single_pose(pose_name: str, show_labels: bool = False) -> None:
    """Display a single pose."""
    cat = CatSkeleton(scale=1.0)

    if pose_name not in cat.get_available_poses():
        print(f"Unknown pose: {pose_name}")
        print(f"Available poses: {', '.join(cat.get_available_poses())}")
        return

    cat.set_pose(pose_name)

    viz = SkeletonVisualizer(cat)
    viz.draw_skeleton(
        show_labels=show_labels,
        title=f"Cat Skeleton - {pose_name.replace('_', ' ').title()} Pose"
    )
    viz.set_view(elevation=15, azimuth=-60)
    viz.show()


def show_all_poses() -> None:
    """Display all available poses in a grid."""
    import matplotlib.pyplot as plt

    cat = CatSkeleton(scale=1.0)
    poses = cat.get_available_poses()

    # Calculate grid dimensions
    n_poses = len(poses)
    n_cols = 3
    n_rows = (n_poses + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(15, 5 * n_rows))
    fig.suptitle("Cat Skeleton - All Predefined Poses", fontsize=16)

    for idx, pose_name in enumerate(poses):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        cat.set_pose(pose_name)

        # Draw bones
        for start_type, end_type in cat.BONES:
            if start_type in cat.joints and end_type in cat.joints:
                start_pos = cat.joints[start_type].world_position
                end_pos = cat.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#4488CC',
                    linewidth=2
                )

        # Draw joints
        for joint in cat.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#FF6644', s=30)

        # Set equal aspect
        min_corner, max_corner = cat.get_bounding_box()
        range_vals = max_corner - min_corner
        max_range = max(range_vals) * 1.3
        center = (min_corner + max_corner) / 2

        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(pose_name.replace('_', ' ').title())
        ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.show()


def show_multiview(pose_name: str) -> None:
    """Show a pose from multiple viewpoints."""
    cat = CatSkeleton(scale=1.0)

    if pose_name not in cat.get_available_poses():
        print(f"Unknown pose: {pose_name}")
        return

    cat.set_pose(pose_name)

    viz = SkeletonVisualizer(cat)
    viz.draw_multi_view(
        title=f"Cat Skeleton - {pose_name.replace('_', ' ').title()} (Multiple Views)"
    )
    viz.show()


def compare_poses(pose_a: str, pose_b: str) -> None:
    """Compare two poses side by side."""
    cat_a = CatSkeleton(scale=1.0)
    cat_b = CatSkeleton(scale=1.0)

    if pose_a not in cat_a.get_available_poses():
        print(f"Unknown pose: {pose_a}")
        return
    if pose_b not in cat_b.get_available_poses():
        print(f"Unknown pose: {pose_b}")
        return

    cat_a.set_pose(pose_a)
    cat_b.set_pose(pose_b)

    viz = SkeletonVisualizer(cat_a)
    viz.draw_pose_comparison(
        cat_b,
        title_a=pose_a.replace('_', ' ').title(),
        title_b=pose_b.replace('_', ' ').title()
    )
    viz.show()


def interactive_viewer() -> None:
    """Interactive pose viewer with keyboard controls."""
    import matplotlib.pyplot as plt

    cat = CatSkeleton(scale=1.0)
    poses = cat.get_available_poses()
    current_pose_idx = 0

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    def draw_current_pose():
        ax.clear()

        pose_name = poses[current_pose_idx]
        cat.set_pose(pose_name)

        # Draw bones
        for start_type, end_type in cat.BONES:
            if start_type in cat.joints and end_type in cat.joints:
                start_pos = cat.joints[start_type].world_position
                end_pos = cat.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#4488CC',
                    linewidth=2.5
                )

        # Draw joints
        for joint in cat.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#FF6644', s=50, edgecolors='black')

        # Set equal aspect
        min_corner, max_corner = cat.get_bounding_box()
        range_vals = max_corner - min_corner
        max_range = max(range_vals) * 1.3
        center = (min_corner + max_corner) / 2

        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Forward/Back)')
        ax.set_zlabel('Z (Up/Down)')
        ax.set_title(
            f"Cat Skeleton - {pose_name.replace('_', ' ').title()}\n"
            f"[{current_pose_idx + 1}/{len(poses)}] "
            f"(Left/Right arrows to change pose, Q to quit)"
        )
        ax.view_init(elev=15, azim=-60)

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current_pose_idx

        if event.key == 'right':
            current_pose_idx = (current_pose_idx + 1) % len(poses)
            draw_current_pose()
        elif event.key == 'left':
            current_pose_idx = (current_pose_idx - 1) % len(poses)
            draw_current_pose()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    draw_current_pose()
    plt.tight_layout()
    plt.show()


def show_interpolation(pose_a: str, pose_b: str, steps: int = 5) -> None:
    """Show interpolation between two poses."""
    import matplotlib.pyplot as plt

    cat = CatSkeleton(scale=1.0)

    if pose_a not in cat.get_available_poses() or pose_b not in cat.get_available_poses():
        print(f"Invalid poses. Available: {', '.join(cat.get_available_poses())}")
        return

    fig = plt.figure(figsize=(16, 4))
    fig.suptitle(
        f"Pose Interpolation: {pose_a.replace('_', ' ').title()} → {pose_b.replace('_', ' ').title()}",
        fontsize=14
    )

    for idx, t in enumerate([i / (steps - 1) for i in range(steps)]):
        ax = fig.add_subplot(1, steps, idx + 1, projection='3d')

        cat.interpolate_pose(pose_a, pose_b, t)

        # Draw bones
        for start_type, end_type in cat.BONES:
            if start_type in cat.joints and end_type in cat.joints:
                start_pos = cat.joints[start_type].world_position
                end_pos = cat.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#4488CC',
                    linewidth=2
                )

        # Draw joints
        for joint in cat.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#FF6644', s=25)

        # Set equal aspect
        min_corner, max_corner = cat.get_bounding_box()
        range_vals = max_corner - min_corner
        max_range = max(range_vals) * 1.3
        center = (min_corner + max_corner) / 2

        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

        ax.set_title(f"t = {t:.1f}")
        ax.view_init(elev=15, azim=-60)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()


def print_skeleton_info() -> None:
    """Print information about the skeleton structure."""
    cat = CatSkeleton(scale=1.0)

    print("\n" + "=" * 60)
    print("CAT SKELETON MODEL")
    print("=" * 60)

    print(f"\nTotal joints: {len(cat.joints)}")
    print(f"Total bones: {len(cat.BONES)}")
    print(f"\nAvailable poses: {', '.join(cat.get_available_poses())}")

    print("\n" + "-" * 60)
    print("JOINT HIERARCHY")
    print("-" * 60)

    def print_joint_tree(joint, indent=0):
        pos = joint.world_position
        print(f"{'  ' * indent}├─ {joint.name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        for child in joint.children:
            print_joint_tree(child, indent + 1)

    print_joint_tree(cat.root)

    print("\n" + "-" * 60)
    print("SHAPE PARAMETERS")
    print("-" * 60)
    shape = cat.shape
    print(f"  Body length: {shape.body_length}")
    print(f"  Body width: {shape.body_width}")
    print(f"  Neck length: {shape.neck_length}")
    print(f"  Head length: {shape.head_length}")
    print(f"  Tail length: {shape.tail_length}")
    print(f"  Front leg (upper/lower/paw): {shape.front_upper_leg_length}/{shape.front_lower_leg_length}/{shape.front_paw_length}")
    print(f"  Back leg (upper/lower/paw): {shape.back_upper_leg_length}/{shape.back_lower_leg_length}/{shape.back_paw_length}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cat Skeleton Wireframe Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_cat_skeleton.py                          Show all poses in a grid
  python demo_cat_skeleton.py --pose sitting           Show the sitting pose
  python demo_cat_skeleton.py --pose standing --labels Show pose with joint labels
  python demo_cat_skeleton.py --multiview sitting      Show pose from multiple angles
  python demo_cat_skeleton.py --compare sitting sleeping
  python demo_cat_skeleton.py --interpolate standing sitting
  python demo_cat_skeleton.py --interactive            Interactive pose viewer
  python demo_cat_skeleton.py --info                   Print skeleton structure info
        """
    )

    parser.add_argument(
        '--pose',
        type=str,
        help='Show a specific pose (standing, sitting, lying_sphinx, sleeping, walking, alert, stretching, arched)'
    )

    parser.add_argument(
        '--labels',
        action='store_true',
        help='Show joint labels'
    )

    parser.add_argument(
        '--multiview',
        type=str,
        metavar='POSE',
        help='Show a pose from multiple viewpoints'
    )

    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('POSE_A', 'POSE_B'),
        help='Compare two poses side by side'
    )

    parser.add_argument(
        '--interpolate',
        nargs=2,
        metavar=('POSE_A', 'POSE_B'),
        help='Show interpolation between two poses'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive pose viewer (use arrow keys)'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Print skeleton structure information'
    )

    args = parser.parse_args()

    # Handle different modes
    if args.info:
        print_skeleton_info()
    elif args.pose:
        show_single_pose(args.pose, show_labels=args.labels)
    elif args.multiview:
        show_multiview(args.multiview)
    elif args.compare:
        compare_poses(args.compare[0], args.compare[1])
    elif args.interpolate:
        show_interpolation(args.interpolate[0], args.interpolate[1])
    elif args.interactive:
        interactive_viewer()
    else:
        # Default: show all poses
        show_all_poses()


if __name__ == '__main__':
    main()
