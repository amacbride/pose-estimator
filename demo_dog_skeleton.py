#!/usr/bin/env python3
"""
Demo script for the parametric dog skeleton model.

This script demonstrates the dog wireframe model by displaying
various predefined poses and allowing interactive exploration.

Usage:
    python demo_dog_skeleton.py                    # Show all poses
    python demo_dog_skeleton.py --pose sitting     # Show specific pose
    python demo_dog_skeleton.py --interactive      # Interactive pose viewer
    python demo_dog_skeleton.py --breeds           # Compare breed body types
    python demo_dog_skeleton.py --compare play_bow begging  # Compare two poses
"""

import argparse
import sys

# Add parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from animal_models import DogSkeleton, SkeletonVisualizer


def show_single_pose(pose_name: str, breed: str = 'medium', show_labels: bool = False) -> None:
    """Display a single pose."""
    dog = DogSkeleton(scale=1.0, breed=breed)

    if pose_name not in dog.get_available_poses():
        print(f"Unknown pose: {pose_name}")
        print(f"Available poses: {', '.join(dog.get_available_poses())}")
        return

    dog.set_pose(pose_name)

    viz = SkeletonVisualizer(dog)
    viz.draw_skeleton(
        show_labels=show_labels,
        title=f"Dog Skeleton ({breed}) - {pose_name.replace('_', ' ').title()} Pose"
    )
    viz.set_view(elevation=15, azimuth=-60)
    viz.show()


def show_all_poses(breed: str = 'medium') -> None:
    """Display all available poses in a grid."""
    import matplotlib.pyplot as plt

    dog = DogSkeleton(scale=1.0, breed=breed)
    poses = dog.get_available_poses()

    # Calculate grid dimensions
    n_poses = len(poses)
    n_cols = 4
    n_rows = (n_poses + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 4 * n_rows))
    fig.suptitle(f"Dog Skeleton ({breed.title()}) - All Predefined Poses", fontsize=16)

    for idx, pose_name in enumerate(poses):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        dog.set_pose(pose_name)

        # Draw bones
        for start_type, end_type in dog.BONES:
            if start_type in dog.joints and end_type in dog.joints:
                start_pos = dog.joints[start_type].world_position
                end_pos = dog.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#AA6633',
                    linewidth=2
                )

        # Draw joints
        for joint in dog.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#3366AA', s=30)

        # Set equal aspect
        min_corner, max_corner = dog.get_bounding_box()
        range_vals = max_corner - min_corner
        max_range = max(range_vals) * 1.3
        center = (min_corner + max_corner) / 2

        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(pose_name.replace('_', ' ').title(), fontsize=10)
        ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.show()


def show_breed_comparison() -> None:
    """Compare different breed body types."""
    import matplotlib.pyplot as plt

    breeds = DogSkeleton.get_available_breeds()

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Dog Breed Body Type Comparison (Standing Pose)", fontsize=16)

    for idx, breed in enumerate(breeds):
        ax = fig.add_subplot(1, len(breeds), idx + 1, projection='3d')

        dog = DogSkeleton(scale=1.0, breed=breed)
        dog.set_pose('standing')

        # Draw bones
        for start_type, end_type in dog.BONES:
            if start_type in dog.joints and end_type in dog.joints:
                start_pos = dog.joints[start_type].world_position
                end_pos = dog.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#AA6633',
                    linewidth=2.5
                )

        # Draw joints
        for joint in dog.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#3366AA', s=40)

        # Use consistent scale for comparison
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1.8)
        ax.set_zlim(-1, 0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{breed.title()}", fontsize=12)
        ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.show()


def compare_poses(pose_a: str, pose_b: str, breed: str = 'medium') -> None:
    """Compare two poses side by side."""
    dog_a = DogSkeleton(scale=1.0, breed=breed)
    dog_b = DogSkeleton(scale=1.0, breed=breed)

    if pose_a not in dog_a.get_available_poses():
        print(f"Unknown pose: {pose_a}")
        return
    if pose_b not in dog_b.get_available_poses():
        print(f"Unknown pose: {pose_b}")
        return

    dog_a.set_pose(pose_a)
    dog_b.set_pose(pose_b)

    viz = SkeletonVisualizer(dog_a)
    viz.draw_pose_comparison(
        dog_b,
        title_a=pose_a.replace('_', ' ').title(),
        title_b=pose_b.replace('_', ' ').title()
    )
    viz.show()


def interactive_viewer(breed: str = 'medium') -> None:
    """Interactive pose viewer with keyboard controls."""
    import matplotlib.pyplot as plt

    dog = DogSkeleton(scale=1.0, breed=breed)
    poses = dog.get_available_poses()
    current_pose_idx = 0

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    def draw_current_pose():
        ax.clear()

        pose_name = poses[current_pose_idx]
        dog.set_pose(pose_name)

        # Draw bones
        for start_type, end_type in dog.BONES:
            if start_type in dog.joints and end_type in dog.joints:
                start_pos = dog.joints[start_type].world_position
                end_pos = dog.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#AA6633',
                    linewidth=2.5
                )

        # Draw joints
        for joint in dog.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#3366AA', s=50, edgecolors='black')

        # Set equal aspect
        min_corner, max_corner = dog.get_bounding_box()
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
            f"Dog Skeleton ({breed}) - {pose_name.replace('_', ' ').title()}\n"
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


def show_interpolation(pose_a: str, pose_b: str, breed: str = 'medium', steps: int = 6) -> None:
    """Show interpolation between two poses."""
    import matplotlib.pyplot as plt

    dog = DogSkeleton(scale=1.0, breed=breed)

    if pose_a not in dog.get_available_poses() or pose_b not in dog.get_available_poses():
        print(f"Invalid poses. Available: {', '.join(dog.get_available_poses())}")
        return

    fig = plt.figure(figsize=(18, 4))
    fig.suptitle(
        f"Pose Interpolation: {pose_a.replace('_', ' ').title()} → {pose_b.replace('_', ' ').title()}",
        fontsize=14
    )

    for idx, t in enumerate([i / (steps - 1) for i in range(steps)]):
        ax = fig.add_subplot(1, steps, idx + 1, projection='3d')

        dog.interpolate_pose(pose_a, pose_b, t)

        # Draw bones
        for start_type, end_type in dog.BONES:
            if start_type in dog.joints and end_type in dog.joints:
                start_pos = dog.joints[start_type].world_position
                end_pos = dog.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#AA6633',
                    linewidth=2
                )

        # Draw joints
        for joint in dog.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#3366AA', s=25)

        # Set consistent bounds
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.5, 1.5)
        ax.set_zlim(-1.0, 0.5)

        ax.set_title(f"t = {t:.1f}")
        ax.view_init(elev=15, azim=-60)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()


def show_tail_wag_animation() -> None:
    """Show tail wagging animation frames."""
    import matplotlib.pyplot as plt

    dog = DogSkeleton(scale=1.0)
    steps = 8

    fig = plt.figure(figsize=(18, 4))
    fig.suptitle("Tail Wagging Animation Frames", fontsize=14)

    for idx in range(steps):
        ax = fig.add_subplot(1, steps, idx + 1, projection='3d')

        # Alternate between left and right wag
        t = idx / (steps - 1)
        if idx % 2 == 0:
            dog.interpolate_pose('tail_wag_left', 'tail_wag_right', t)
        else:
            dog.interpolate_pose('tail_wag_right', 'tail_wag_left', t)

        # Draw bones
        for start_type, end_type in dog.BONES:
            if start_type in dog.joints and end_type in dog.joints:
                start_pos = dog.joints[start_type].world_position
                end_pos = dog.joints[end_type].world_position

                ax.plot3D(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color='#AA6633',
                    linewidth=2
                )

        # Draw joints
        for joint in dog.joints.values():
            pos = joint.world_position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='#3366AA', s=25)

        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.5, 1.5)
        ax.set_zlim(-0.8, 0.5)

        ax.set_title(f"Frame {idx + 1}")
        ax.view_init(elev=15, azim=-60)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()


def print_skeleton_info(breed: str = 'medium') -> None:
    """Print information about the skeleton structure."""
    dog = DogSkeleton(scale=1.0, breed=breed)

    print("\n" + "=" * 60)
    print(f"DOG SKELETON MODEL ({breed.upper()})")
    print("=" * 60)

    print(f"\nTotal joints: {len(dog.joints)}")
    print(f"Total bones: {len(dog.BONES)}")
    print(f"Available breeds: {', '.join(DogSkeleton.get_available_breeds())}")
    print(f"\nAvailable poses ({len(dog.get_available_poses())}):")

    # Group poses by category
    poses = dog.get_available_poses()
    categories = {
        'Basic': ['standing', 'sitting', 'lying', 'sleeping'],
        'Movement': ['walking', 'running'],
        'Playful': ['play_bow', 'begging', 'roll_over', 'tail_wag_left', 'tail_wag_right'],
        'Expressive': ['alert', 'happy', 'scared', 'shake'],
        'Actions': ['sniffing', 'pointing', 'stretching'],
    }

    for category, category_poses in categories.items():
        matching = [p for p in category_poses if p in poses]
        if matching:
            print(f"  {category}: {', '.join(matching)}")

    # Any uncategorized
    all_categorized = set()
    for cp in categories.values():
        all_categorized.update(cp)
    uncategorized = [p for p in poses if p not in all_categorized]
    if uncategorized:
        print(f"  Other: {', '.join(uncategorized)}")

    print("\n" + "-" * 60)
    print("SHAPE PARAMETERS")
    print("-" * 60)
    shape = dog.shape
    print(f"  Body length: {shape.body_length}")
    print(f"  Body width: {shape.body_width}")
    print(f"  Neck length: {shape.neck_length}")
    print(f"  Head length: {shape.head_length}")
    print(f"  Snout length: {shape.snout_length}")
    print(f"  Tail length: {shape.tail_length}")
    print(f"  Front leg (upper/lower/paw): {shape.front_upper_leg_length}/{shape.front_lower_leg_length}/{shape.front_paw_length}")
    print(f"  Back leg (upper/lower/paw): {shape.back_upper_leg_length}/{shape.back_lower_leg_length}/{shape.back_paw_length}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dog Skeleton Wireframe Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_dog_skeleton.py                          Show all poses in a grid
  python demo_dog_skeleton.py --pose play_bow          Show the play bow pose
  python demo_dog_skeleton.py --pose sitting --breed slim
  python demo_dog_skeleton.py --breeds                 Compare breed body types
  python demo_dog_skeleton.py --compare sitting begging
  python demo_dog_skeleton.py --interpolate standing play_bow
  python demo_dog_skeleton.py --tail-wag               Show tail wagging animation
  python demo_dog_skeleton.py --interactive            Interactive pose viewer
  python demo_dog_skeleton.py --info                   Print skeleton structure info
        """
    )

    parser.add_argument(
        '--pose',
        type=str,
        help='Show a specific pose'
    )

    parser.add_argument(
        '--breed',
        type=str,
        default='medium',
        choices=['medium', 'small', 'large', 'long', 'slim'],
        help='Breed/body type preset (default: medium)'
    )

    parser.add_argument(
        '--labels',
        action='store_true',
        help='Show joint labels'
    )

    parser.add_argument(
        '--breeds',
        action='store_true',
        help='Compare different breed body types'
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
        '--tail-wag',
        action='store_true',
        help='Show tail wagging animation frames'
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
        print_skeleton_info(args.breed)
    elif args.pose:
        show_single_pose(args.pose, breed=args.breed, show_labels=args.labels)
    elif args.breeds:
        show_breed_comparison()
    elif args.compare:
        compare_poses(args.compare[0], args.compare[1], breed=args.breed)
    elif args.interpolate:
        show_interpolation(args.interpolate[0], args.interpolate[1], breed=args.breed)
    elif args.tail_wag:
        show_tail_wag_animation()
    elif args.interactive:
        interactive_viewer(args.breed)
    else:
        # Default: show all poses
        show_all_poses(args.breed)


if __name__ == '__main__':
    main()
