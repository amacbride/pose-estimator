"""
3D visualization for quadruped skeleton models.

This module provides visualization utilities using matplotlib for
rendering skeleton wireframes from multiple viewpoints.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List
from .skeleton import QuadrupedSkeleton, JointType


# Joint colors by category
JOINT_COLORS = {
    # Head - red tones
    JointType.HEAD: '#FF4444',
    JointType.NECK: '#FF6666',
    JointType.NOSE: '#FF8888',
    JointType.LEFT_EAR: '#FF5555',
    JointType.RIGHT_EAR: '#FF5555',
    JointType.LEFT_EYE: '#FF7777',
    JointType.RIGHT_EYE: '#FF7777',

    # Spine - orange/yellow
    JointType.ROOT: '#FFAA00',
    JointType.SPINE_LOWER: '#FFBB33',
    JointType.SPINE_MID: '#FFCC44',
    JointType.SPINE_UPPER: '#FFDD55',

    # Tail - purple
    JointType.TAIL_BASE: '#AA44FF',
    JointType.TAIL_MID: '#BB66FF',
    JointType.TAIL_TIP: '#CC88FF',

    # Front legs - green
    JointType.LEFT_SHOULDER: '#44AA44',
    JointType.LEFT_FRONT_UPPER: '#55BB55',
    JointType.LEFT_FRONT_LOWER: '#66CC66',
    JointType.LEFT_FRONT_PAW: '#77DD77',
    JointType.RIGHT_SHOULDER: '#44AA44',
    JointType.RIGHT_FRONT_UPPER: '#55BB55',
    JointType.RIGHT_FRONT_LOWER: '#66CC66',
    JointType.RIGHT_FRONT_PAW: '#77DD77',

    # Back legs - blue
    JointType.LEFT_HIP: '#4444AA',
    JointType.LEFT_BACK_UPPER: '#5555BB',
    JointType.LEFT_BACK_LOWER: '#6666CC',
    JointType.LEFT_BACK_PAW: '#7777DD',
    JointType.RIGHT_HIP: '#4444AA',
    JointType.RIGHT_BACK_UPPER: '#5555BB',
    JointType.RIGHT_BACK_LOWER: '#6666CC',
    JointType.RIGHT_BACK_PAW: '#7777DD',
}

# Bone colors by type
BONE_COLORS = {
    'spine': '#FF8800',
    'head': '#FF4444',
    'tail': '#AA44FF',
    'front_leg': '#44AA44',
    'back_leg': '#4444AA',
}


def get_bone_color(start_type: JointType, end_type: JointType) -> str:
    """Determine bone color based on connected joints."""
    name = start_type.value + end_type.value

    if 'spine' in name or 'root' in name.lower():
        if 'hip' in name or 'shoulder' in name or 'tail' in name:
            return BONE_COLORS['spine']
        return BONE_COLORS['spine']
    if 'head' in name or 'neck' in name or 'ear' in name or 'eye' in name or 'nose' in name:
        return BONE_COLORS['head']
    if 'tail' in name:
        return BONE_COLORS['tail']
    if 'front' in name or 'shoulder' in name:
        return BONE_COLORS['front_leg']
    if 'back' in name or 'hip' in name:
        return BONE_COLORS['back_leg']

    return '#888888'


class SkeletonVisualizer:
    """
    Visualizer for quadruped skeleton models.

    Provides methods for rendering skeleton wireframes as 3D plots
    with configurable viewpoints and styles.
    """

    def __init__(
        self,
        skeleton: QuadrupedSkeleton,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize the visualizer.

        Args:
            skeleton: The skeleton model to visualize
            figsize: Figure size in inches (width, height)
        """
        self.skeleton = skeleton
        self.figsize = figsize
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None

    def _setup_axes(self, title: str = "Skeleton Wireframe") -> Axes3D:
        """Create and configure 3D axes."""
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set labels
        self.ax.set_xlabel('X (Left/Right)')
        self.ax.set_ylabel('Y (Forward/Back)')
        self.ax.set_zlabel('Z (Up/Down)')
        self.ax.set_title(title)

        return self.ax

    def _set_equal_aspect(self) -> None:
        """Set equal aspect ratio for all axes."""
        if self.ax is None:
            return

        # Get bounding box
        min_corner, max_corner = self.skeleton.get_bounding_box()

        # Add padding
        padding = 0.2
        range_vals = max_corner - min_corner
        max_range = max(range_vals) * (1 + padding)

        center = (min_corner + max_corner) / 2

        self.ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        self.ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        self.ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

    def draw_skeleton(
        self,
        show_joints: bool = True,
        show_bones: bool = True,
        show_labels: bool = False,
        joint_size: int = 50,
        bone_width: float = 2.0,
        title: Optional[str] = None
    ) -> None:
        """
        Draw the skeleton wireframe.

        Args:
            show_joints: Whether to draw joint markers
            show_bones: Whether to draw bone lines
            show_labels: Whether to show joint name labels
            joint_size: Size of joint markers
            bone_width: Width of bone lines
            title: Plot title
        """
        ax = self._setup_axes(title or "Skeleton Wireframe")

        # Draw bones first (so joints appear on top)
        if show_bones:
            for start_type, end_type in self.skeleton.BONES:
                if start_type in self.skeleton.joints and end_type in self.skeleton.joints:
                    start_pos = self.skeleton.joints[start_type].world_position
                    end_pos = self.skeleton.joints[end_type].world_position

                    color = get_bone_color(start_type, end_type)

                    ax.plot3D(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        [start_pos[2], end_pos[2]],
                        color=color,
                        linewidth=bone_width,
                        solid_capstyle='round'
                    )

        # Draw joints
        if show_joints:
            for joint_type, joint in self.skeleton.joints.items():
                pos = joint.world_position
                color = JOINT_COLORS.get(joint_type, '#888888')

                ax.scatter(
                    [pos[0]], [pos[1]], [pos[2]],
                    c=color,
                    s=joint_size,
                    marker='o',
                    edgecolors='black',
                    linewidths=0.5
                )

                if show_labels:
                    ax.text(
                        pos[0], pos[1], pos[2] + 0.05,
                        joint_type.value,
                        fontsize=6,
                        ha='center'
                    )

        self._set_equal_aspect()

        # Draw ground plane reference
        self._draw_ground_plane(ax)

    def _draw_ground_plane(self, ax: Axes3D) -> None:
        """Draw a subtle ground plane grid."""
        min_corner, max_corner = self.skeleton.get_bounding_box()

        # Find ground level (lowest Z)
        ground_z = min_corner[2] - 0.02

        # Create grid
        x_range = np.linspace(min_corner[0] - 0.3, max_corner[0] + 0.3, 5)
        y_range = np.linspace(min_corner[1] - 0.3, max_corner[1] + 0.3, 5)

        for x in x_range:
            ax.plot3D(
                [x, x],
                [y_range[0], y_range[-1]],
                [ground_z, ground_z],
                color='#CCCCCC',
                linewidth=0.5,
                alpha=0.5
            )

        for y in y_range:
            ax.plot3D(
                [x_range[0], x_range[-1]],
                [y, y],
                [ground_z, ground_z],
                color='#CCCCCC',
                linewidth=0.5,
                alpha=0.5
            )

    def set_view(
        self,
        elevation: float = 20,
        azimuth: float = -60
    ) -> None:
        """
        Set the camera viewpoint.

        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
        """
        if self.ax:
            self.ax.view_init(elev=elevation, azim=azimuth)

    def show(self) -> None:
        """Display the plot."""
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str, dpi: int = 150) -> None:
        """Save the plot to a file."""
        if self.fig:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

    def draw_multi_view(
        self,
        title: str = "Skeleton Views",
        show_labels: bool = False
    ) -> None:
        """
        Draw the skeleton from multiple viewpoints in a single figure.

        Shows front, side, top, and 3D perspective views.
        """
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(title, fontsize=14)

        views = [
            ('3D Perspective', 20, -60),
            ('Front View', 0, 0),
            ('Side View', 0, -90),
            ('Top View', 90, -90),
        ]

        for idx, (view_name, elev, azim) in enumerate(views):
            ax = self.fig.add_subplot(2, 2, idx + 1, projection='3d')
            self.ax = ax

            # Draw bones
            for start_type, end_type in self.skeleton.BONES:
                if start_type in self.skeleton.joints and end_type in self.skeleton.joints:
                    start_pos = self.skeleton.joints[start_type].world_position
                    end_pos = self.skeleton.joints[end_type].world_position
                    color = get_bone_color(start_type, end_type)

                    ax.plot3D(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        [start_pos[2], end_pos[2]],
                        color=color,
                        linewidth=2
                    )

            # Draw joints
            for joint_type, joint in self.skeleton.joints.items():
                pos = joint.world_position
                color = JOINT_COLORS.get(joint_type, '#888888')
                ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=30)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(view_name)
            ax.view_init(elev=elev, azim=azim)

            self._set_equal_aspect()

    def draw_pose_comparison(
        self,
        skeleton_b: QuadrupedSkeleton,
        title_a: str = "Pose A",
        title_b: str = "Pose B"
    ) -> None:
        """
        Draw two skeletons side by side for comparison.

        Args:
            skeleton_b: Second skeleton to compare
            title_a: Title for first skeleton
            title_b: Title for second skeleton
        """
        self.fig = plt.figure(figsize=(14, 6))

        for idx, (skel, title) in enumerate([(self.skeleton, title_a), (skeleton_b, title_b)]):
            ax = self.fig.add_subplot(1, 2, idx + 1, projection='3d')

            # Draw bones
            for start_type, end_type in skel.BONES:
                if start_type in skel.joints and end_type in skel.joints:
                    start_pos = skel.joints[start_type].world_position
                    end_pos = skel.joints[end_type].world_position
                    color = get_bone_color(start_type, end_type)

                    ax.plot3D(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        [start_pos[2], end_pos[2]],
                        color=color,
                        linewidth=2
                    )

            # Draw joints
            for joint_type, joint in skel.joints.items():
                pos = joint.world_position
                color = JOINT_COLORS.get(joint_type, '#888888')
                ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=40)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            ax.view_init(elev=20, azim=-60)

            # Set equal aspect
            min_corner, max_corner = skel.get_bounding_box()
            range_vals = max_corner - min_corner
            max_range = max(range_vals) * 1.2
            center = (min_corner + max_corner) / 2
            ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
            ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)


def create_pose_animation_frames(
    skeleton: QuadrupedSkeleton,
    pose_sequence: List[str],
    frames_per_transition: int = 10
) -> List[dict]:
    """
    Generate interpolated frames for a sequence of poses.

    Args:
        skeleton: The skeleton to animate
        pose_sequence: List of pose names to transition through
        frames_per_transition: Number of frames between each pose

    Returns:
        List of joint position dictionaries for each frame
    """
    from .cat import CatSkeleton

    if not isinstance(skeleton, CatSkeleton):
        raise TypeError("Animation requires CatSkeleton with predefined poses")

    frames = []

    for i in range(len(pose_sequence) - 1):
        pose_a = pose_sequence[i]
        pose_b = pose_sequence[i + 1]

        for t in range(frames_per_transition):
            progress = t / frames_per_transition
            skeleton.interpolate_pose(pose_a, pose_b, progress)
            frames.append(skeleton.get_all_joint_positions())

    # Add final pose
    skeleton.set_pose(pose_sequence[-1])
    frames.append(skeleton.get_all_joint_positions())

    return frames
