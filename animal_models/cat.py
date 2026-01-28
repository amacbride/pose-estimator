"""
Cat-specific skeleton model with realistic proportions and predefined poses.

This module extends the base QuadrupedSkeleton to create a parametric
cat model suitable for animation.
"""

import numpy as np
from typing import Dict
from .skeleton import QuadrupedSkeleton, ShapeParameters, JointType


class CatShapeParameters(ShapeParameters):
    """
    Shape parameters tuned for a domestic cat.

    Based on average cat proportions:
    - Body length (shoulder to hip): ~40cm
    - Standing height at shoulder: ~25cm
    - Tail length: ~25-30cm

    All values are normalized with body_length = 1.0 as reference.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()

        # Core body - cats are relatively long and flexible
        self.body_length = 1.0
        self.body_width = 0.25
        self.body_height = 0.18

        # Spine - cats have very flexible spines
        self.spine_segments = 3

        # Head and neck - cats have proportionally small heads
        self.neck_length = 0.22
        self.head_length = 0.18
        self.head_width = 0.14
        self.ear_length = 0.1  # Cats have prominent ears
        self.snout_length = 0.08  # Short snout

        # Tail - cats have long, expressive tails
        self.tail_length = 0.7
        self.tail_segments = 3

        # Front legs - relatively short, digitigrade stance
        self.front_shoulder_width = 0.2
        self.front_upper_leg_length = 0.22
        self.front_lower_leg_length = 0.18
        self.front_paw_length = 0.06

        # Back legs - longer than front, powerful for jumping
        self.back_hip_width = 0.18
        self.back_upper_leg_length = 0.26
        self.back_lower_leg_length = 0.2
        self.back_paw_length = 0.08

        self.scale = scale


class CatSkeleton(QuadrupedSkeleton):
    """
    A parametric cat skeleton model.

    Provides cat-specific proportions and a library of predefined poses
    for common cat positions and behaviors.
    """

    def __init__(self, scale: float = 1.0):
        """
        Create a new cat skeleton.

        Args:
            scale: Overall scale factor (1.0 = normalized size)
        """
        shape = CatShapeParameters(scale=scale)
        super().__init__(shape)

        # Store predefined poses
        self._poses: Dict[str, Dict[JointType, np.ndarray]] = self._define_poses()

    def _define_poses(self) -> Dict[str, Dict[JointType, np.ndarray]]:
        """Define a library of cat poses."""
        poses = {}

        # Standing pose (neutral)
        poses['standing'] = {
            # Slight head tilt up
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([0.15, 0, 0]),
            # Tail slightly raised
            JointType.TAIL_BASE: np.array([0.3, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
            JointType.TAIL_TIP: np.array([-0.1, 0, 0]),
        }

        # Sitting pose - weight on haunches, front legs straight
        poses['sitting'] = {
            # Spine curves down toward rear
            JointType.SPINE_LOWER: np.array([-0.3, 0, 0]),
            JointType.SPINE_MID: np.array([-0.1, 0, 0]),
            # Head up and alert
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.2, 0, 0]),
            # Tail wraps around
            JointType.TAIL_BASE: np.array([-0.2, 0, 0.3]),
            JointType.TAIL_MID: np.array([0, 0, 0.5]),
            JointType.TAIL_TIP: np.array([0, 0, 0.3]),
            # Front legs straight down
            JointType.LEFT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.3, 0, 0]),
            # Back legs folded
            JointType.LEFT_HIP: np.array([-0.8, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([1.5, 0, 0.2]),
            JointType.LEFT_BACK_LOWER: np.array([-1.8, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.8, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([1.5, 0, -0.2]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.8, 0, 0]),
        }

        # Lying down pose (sphinx position)
        poses['lying_sphinx'] = {
            # Spine low and flat
            JointType.SPINE_LOWER: np.array([-0.1, 0, 0]),
            JointType.SPINE_MID: np.array([0.05, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            # Head up
            JointType.NECK: np.array([0.5, 0, 0]),
            JointType.HEAD: np.array([0.3, 0, 0]),
            # Tail relaxed
            JointType.TAIL_BASE: np.array([-0.3, 0, 0.2]),
            JointType.TAIL_MID: np.array([0, 0, 0.1]),
            # Front legs tucked forward
            JointType.LEFT_SHOULDER: np.array([0.5, 0, 0.2]),
            JointType.LEFT_FRONT_UPPER: np.array([0.8, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.9, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.5, 0, -0.2]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.8, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.9, 0, 0]),
            # Back legs folded underneath
            JointType.LEFT_HIP: np.array([-0.6, 0, 0.3]),
            JointType.LEFT_BACK_UPPER: np.array([1.2, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.6, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.6, 0, -0.3]),
            JointType.RIGHT_BACK_UPPER: np.array([1.2, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.6, 0, 0]),
        }

        # Sleeping curled up
        poses['sleeping'] = {
            # Spine curves into ball
            JointType.SPINE_LOWER: np.array([-0.2, 0, 0.4]),
            JointType.SPINE_MID: np.array([0, 0, 0.3]),
            JointType.SPINE_UPPER: np.array([0.2, 0, 0.2]),
            # Head tucked down
            JointType.NECK: np.array([0.3, 0, 0.4]),
            JointType.HEAD: np.array([0.5, 0, 0.3]),
            # Tail wraps around body
            JointType.TAIL_BASE: np.array([-0.4, 0, 0.6]),
            JointType.TAIL_MID: np.array([0, 0, 0.8]),
            JointType.TAIL_TIP: np.array([0.3, 0, 0.6]),
            # Front legs tucked in
            JointType.LEFT_SHOULDER: np.array([0.6, 0, 0.3]),
            JointType.LEFT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            # Back legs tucked
            JointType.LEFT_HIP: np.array([-0.5, 0, 0.4]),
            JointType.LEFT_BACK_UPPER: np.array([1.4, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.8, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.5, 0, -0.4]),
            JointType.RIGHT_BACK_UPPER: np.array([1.4, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.8, 0, 0]),
        }

        # Walking pose - mid-stride
        poses['walking'] = {
            # Slight body motion
            JointType.SPINE_MID: np.array([0.05, 0.02, 0]),
            # Head forward
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail for balance
            JointType.TAIL_BASE: np.array([0.2, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0.1, 0]),
            # Left front forward, right front back
            JointType.LEFT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([-0.15, 0, 0]),
            # Right back forward, left back backward (diagonal gait)
            JointType.LEFT_HIP: np.array([-0.2, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.25, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([-0.15, 0, 0]),
        }

        # Alert/curious pose
        poses['alert'] = {
            # Spine slightly arched up
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            # Head very up and forward
            JointType.NECK: np.array([0.5, 0, 0]),
            JointType.HEAD: np.array([0.3, 0, 0]),
            # Ears up (implied by head position)
            JointType.LEFT_EAR: np.array([0.2, 0, 0.1]),
            JointType.RIGHT_EAR: np.array([0.2, 0, -0.1]),
            # Tail up and slightly curved
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([-0.1, 0, 0]),
        }

        # Stretching pose
        poses['stretching'] = {
            # Front low, back high
            JointType.SPINE_LOWER: np.array([0.3, 0, 0]),
            JointType.SPINE_MID: np.array([0.2, 0, 0]),
            JointType.SPINE_UPPER: np.array([-0.1, 0, 0]),
            # Head low
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.1, 0, 0]),
            # Front legs extended forward
            JointType.LEFT_SHOULDER: np.array([0.6, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.3, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.3, 0, 0]),
            # Back legs straight
            JointType.LEFT_HIP: np.array([0.2, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.2, 0, 0]),
            # Tail up
            JointType.TAIL_BASE: np.array([0.6, 0, 0]),
            JointType.TAIL_MID: np.array([0.3, 0, 0]),
        }

        # Arched back (defensive/scared)
        poses['arched'] = {
            # Spine arches up dramatically
            JointType.SPINE_LOWER: np.array([0.4, 0, 0]),
            JointType.SPINE_MID: np.array([0.5, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.3, 0, 0]),
            # Head down, facing forward
            JointType.NECK: np.array([-0.3, 0, 0]),
            JointType.HEAD: np.array([-0.2, 0, 0]),
            # Legs slightly bent
            JointType.LEFT_FRONT_UPPER: np.array([0.2, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.3, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.2, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.3, 0, 0]),
            # Tail puffed up (implied)
            JointType.TAIL_BASE: np.array([0.8, 0, 0]),
            JointType.TAIL_MID: np.array([0.4, 0, 0]),
            JointType.TAIL_TIP: np.array([0.2, 0, 0]),
        }

        return poses

    def set_pose(self, pose_name: str) -> bool:
        """
        Set the skeleton to a predefined pose.

        Args:
            pose_name: Name of the pose (e.g., 'standing', 'sitting', 'sleeping')

        Returns:
            True if pose was found and applied, False otherwise
        """
        if pose_name not in self._poses:
            return False

        # Reset to neutral first
        self.reset_pose()

        # Apply pose rotations
        pose = self._poses[pose_name]
        for joint_type, rotation in pose.items():
            self.set_joint_rotation(joint_type, rotation)

        # Update world positions
        self.update_world_positions()
        return True

    def get_available_poses(self) -> list:
        """Get list of available predefined pose names."""
        return list(self._poses.keys())

    def interpolate_pose(
        self,
        pose_name_a: str,
        pose_name_b: str,
        t: float
    ) -> bool:
        """
        Interpolate between two poses.

        Args:
            pose_name_a: Starting pose name
            pose_name_b: Ending pose name
            t: Interpolation factor (0.0 = pose_a, 1.0 = pose_b)

        Returns:
            True if successful, False if poses not found
        """
        if pose_name_a not in self._poses or pose_name_b not in self._poses:
            return False

        pose_a = self._poses[pose_name_a]
        pose_b = self._poses[pose_name_b]

        # Reset first
        self.reset_pose()

        # Get all joint types that appear in either pose
        all_joints = set(pose_a.keys()) | set(pose_b.keys())

        for joint_type in all_joints:
            rot_a = pose_a.get(joint_type, np.zeros(3))
            rot_b = pose_b.get(joint_type, np.zeros(3))

            # Linear interpolation (could use SLERP for better results)
            interpolated = rot_a * (1 - t) + rot_b * t
            self.set_joint_rotation(joint_type, interpolated)

        self.update_world_positions()
        return True
