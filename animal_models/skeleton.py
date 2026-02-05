"""
Base skeleton classes for parametric quadruped models.

This module provides the foundational Joint and QuadrupedSkeleton classes
that can be extended for specific animals like cats and dogs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class JointType(Enum):
    """Enumeration of joint types in a quadruped skeleton."""
    # Core body
    ROOT = "root"  # Pelvis/hip center - the origin of the skeleton
    SPINE_LOWER = "spine_lower"
    SPINE_MID = "spine_mid"
    SPINE_UPPER = "spine_upper"

    # Head and neck
    NECK = "neck"
    HEAD = "head"
    NOSE = "nose"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"

    # Tail
    TAIL_BASE = "tail_base"
    TAIL_MID = "tail_mid"
    TAIL_TIP = "tail_tip"

    # Front left leg
    LEFT_SHOULDER = "left_shoulder"
    LEFT_FRONT_UPPER = "left_front_upper"
    LEFT_FRONT_LOWER = "left_front_lower"
    LEFT_FRONT_PAW = "left_front_paw"

    # Front right leg
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_FRONT_UPPER = "right_front_upper"
    RIGHT_FRONT_LOWER = "right_front_lower"
    RIGHT_FRONT_PAW = "right_front_paw"

    # Back left leg
    LEFT_HIP = "left_hip"
    LEFT_BACK_UPPER = "left_back_upper"
    LEFT_BACK_LOWER = "left_back_lower"
    LEFT_BACK_PAW = "left_back_paw"

    # Back right leg
    RIGHT_HIP = "right_hip"
    RIGHT_BACK_UPPER = "right_back_upper"
    RIGHT_BACK_LOWER = "right_back_lower"
    RIGHT_BACK_PAW = "right_back_paw"


@dataclass
class Joint:
    """
    Represents a single joint in the skeleton hierarchy.

    Attributes:
        joint_type: The type/name of this joint
        local_offset: Position offset from parent joint (in parent's local space)
        rotation: Euler angles (x, y, z) in radians for this joint's rotation
        children: List of child joints
        parent: Reference to parent joint (None for root)
    """
    joint_type: JointType
    local_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    children: List['Joint'] = field(default_factory=list)
    parent: Optional['Joint'] = None

    # Computed world position (set by forward kinematics)
    _world_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _world_rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def name(self) -> str:
        return self.joint_type.value

    @property
    def world_position(self) -> np.ndarray:
        return self._world_position.copy()

    def add_child(self, child: 'Joint') -> 'Joint':
        """Add a child joint and set its parent reference."""
        child.parent = self
        self.children.append(child)
        return child


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Create rotation matrix for rotation around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Create rotation matrix for rotation around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix for rotation around Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ order) to rotation matrix.

    Args:
        euler: Array of [x, y, z] rotation angles in radians

    Returns:
        3x3 rotation matrix
    """
    rx = rotation_matrix_x(euler[0])
    ry = rotation_matrix_y(euler[1])
    rz = rotation_matrix_z(euler[2])
    return rz @ ry @ rx


@dataclass
class ShapeParameters:
    """
    Shape parameters that define the proportions of a quadruped.

    All measurements are in arbitrary units (can be scaled).
    Default values represent a generic quadruped.
    """
    # Core body dimensions
    body_length: float = 1.0  # Length from shoulders to hips
    body_width: float = 0.3   # Width at shoulders/hips
    body_height: float = 0.2  # Chest depth

    # Spine
    spine_segments: int = 3

    # Head and neck
    neck_length: float = 0.25
    head_length: float = 0.2
    head_width: float = 0.15
    ear_length: float = 0.08
    snout_length: float = 0.1

    # Tail
    tail_length: float = 0.4
    tail_segments: int = 3

    # Front legs
    front_shoulder_width: float = 0.25
    front_upper_leg_length: float = 0.25
    front_lower_leg_length: float = 0.2
    front_paw_length: float = 0.08

    # Back legs
    back_hip_width: float = 0.22
    back_upper_leg_length: float = 0.28
    back_lower_leg_length: float = 0.22
    back_paw_length: float = 0.1

    # Overall scale
    scale: float = 1.0


class QuadrupedSkeleton:
    """
    Base class for parametric quadruped skeleton models.

    This class defines the joint hierarchy and provides methods for:
    - Building the skeleton from shape parameters
    - Setting joint rotations (pose)
    - Computing world positions via forward kinematics
    - Accessing joint positions for visualization

    Coordinate system:
    - X: Right (positive) / Left (negative)
    - Y: Forward (positive) / Backward (negative)
    - Z: Up (positive) / Down (negative)
    - Origin: At the root (pelvis center)
    """

    # Define the skeleton connections (bones) as pairs of joint types
    BONES: List[Tuple[JointType, JointType]] = [
        # Spine
        (JointType.ROOT, JointType.SPINE_LOWER),
        (JointType.SPINE_LOWER, JointType.SPINE_MID),
        (JointType.SPINE_MID, JointType.SPINE_UPPER),

        # Neck and head
        (JointType.SPINE_UPPER, JointType.NECK),
        (JointType.NECK, JointType.HEAD),
        (JointType.HEAD, JointType.NOSE),
        (JointType.HEAD, JointType.LEFT_EAR),
        (JointType.HEAD, JointType.RIGHT_EAR),
        (JointType.HEAD, JointType.LEFT_EYE),
        (JointType.HEAD, JointType.RIGHT_EYE),

        # Tail
        (JointType.ROOT, JointType.TAIL_BASE),
        (JointType.TAIL_BASE, JointType.TAIL_MID),
        (JointType.TAIL_MID, JointType.TAIL_TIP),

        # Front left leg
        (JointType.SPINE_UPPER, JointType.LEFT_SHOULDER),
        (JointType.LEFT_SHOULDER, JointType.LEFT_FRONT_UPPER),
        (JointType.LEFT_FRONT_UPPER, JointType.LEFT_FRONT_LOWER),
        (JointType.LEFT_FRONT_LOWER, JointType.LEFT_FRONT_PAW),

        # Front right leg
        (JointType.SPINE_UPPER, JointType.RIGHT_SHOULDER),
        (JointType.RIGHT_SHOULDER, JointType.RIGHT_FRONT_UPPER),
        (JointType.RIGHT_FRONT_UPPER, JointType.RIGHT_FRONT_LOWER),
        (JointType.RIGHT_FRONT_LOWER, JointType.RIGHT_FRONT_PAW),

        # Back left leg
        (JointType.ROOT, JointType.LEFT_HIP),
        (JointType.LEFT_HIP, JointType.LEFT_BACK_UPPER),
        (JointType.LEFT_BACK_UPPER, JointType.LEFT_BACK_LOWER),
        (JointType.LEFT_BACK_LOWER, JointType.LEFT_BACK_PAW),

        # Back right leg
        (JointType.ROOT, JointType.RIGHT_HIP),
        (JointType.RIGHT_HIP, JointType.RIGHT_BACK_UPPER),
        (JointType.RIGHT_BACK_UPPER, JointType.RIGHT_BACK_LOWER),
        (JointType.RIGHT_BACK_LOWER, JointType.RIGHT_BACK_PAW),
    ]

    def __init__(self, shape: Optional[ShapeParameters] = None):
        """
        Initialize the skeleton with given shape parameters.

        Args:
            shape: ShapeParameters defining the proportions. Uses defaults if None.
        """
        self.shape = shape or ShapeParameters()
        self.root: Optional[Joint] = None
        self.joints: Dict[JointType, Joint] = {}

        # Global position and orientation of the entire skeleton
        self.global_position = np.array([0.0, 0.0, 0.0])
        self.global_rotation = np.array([0.0, 0.0, 0.0])

        self._build_skeleton()
        self.update_world_positions()

    def _build_skeleton(self) -> None:
        """Build the joint hierarchy based on shape parameters."""
        s = self.shape  # Shorthand
        scale = s.scale

        # Create root joint at origin
        self.root = Joint(JointType.ROOT)
        self.joints[JointType.ROOT] = self.root

        # Spine - goes forward from root
        spine_segment_length = s.body_length / s.spine_segments

        spine_lower = self._add_joint(
            JointType.SPINE_LOWER,
            self.root,
            np.array([0, spine_segment_length * scale, 0])
        )

        spine_mid = self._add_joint(
            JointType.SPINE_MID,
            spine_lower,
            np.array([0, spine_segment_length * scale, 0])
        )

        spine_upper = self._add_joint(
            JointType.SPINE_UPPER,
            spine_mid,
            np.array([0, spine_segment_length * scale, 0])
        )

        # Neck and head
        neck = self._add_joint(
            JointType.NECK,
            spine_upper,
            np.array([0, s.neck_length * 0.5 * scale, s.neck_length * 0.3 * scale])
        )

        head = self._add_joint(
            JointType.HEAD,
            neck,
            np.array([0, s.neck_length * 0.5 * scale, s.neck_length * 0.2 * scale])
        )

        # Nose (snout)
        self._add_joint(
            JointType.NOSE,
            head,
            np.array([0, s.snout_length * scale, -s.head_length * 0.1 * scale])
        )

        # Ears
        self._add_joint(
            JointType.LEFT_EAR,
            head,
            np.array([-s.head_width * 0.4 * scale, 0, s.ear_length * scale])
        )

        self._add_joint(
            JointType.RIGHT_EAR,
            head,
            np.array([s.head_width * 0.4 * scale, 0, s.ear_length * scale])
        )

        # Eyes
        self._add_joint(
            JointType.LEFT_EYE,
            head,
            np.array([-s.head_width * 0.25 * scale, s.head_length * 0.3 * scale, 0])
        )

        self._add_joint(
            JointType.RIGHT_EYE,
            head,
            np.array([s.head_width * 0.25 * scale, s.head_length * 0.3 * scale, 0])
        )

        # Tail - goes backward from root
        tail_segment_length = s.tail_length / s.tail_segments

        tail_base = self._add_joint(
            JointType.TAIL_BASE,
            self.root,
            np.array([0, -tail_segment_length * 0.3 * scale, s.body_height * 0.3 * scale])
        )

        tail_mid = self._add_joint(
            JointType.TAIL_MID,
            tail_base,
            np.array([0, -tail_segment_length * scale, tail_segment_length * 0.2 * scale])
        )

        self._add_joint(
            JointType.TAIL_TIP,
            tail_mid,
            np.array([0, -tail_segment_length * scale, tail_segment_length * 0.1 * scale])
        )

        # Front legs
        # Left front leg
        left_shoulder = self._add_joint(
            JointType.LEFT_SHOULDER,
            spine_upper,
            np.array([-s.front_shoulder_width * scale, 0, -s.body_height * 0.3 * scale])
        )

        left_front_upper = self._add_joint(
            JointType.LEFT_FRONT_UPPER,
            left_shoulder,
            np.array([0, 0, -s.front_upper_leg_length * scale])
        )

        left_front_lower = self._add_joint(
            JointType.LEFT_FRONT_LOWER,
            left_front_upper,
            np.array([0, 0, -s.front_lower_leg_length * scale])
        )

        self._add_joint(
            JointType.LEFT_FRONT_PAW,
            left_front_lower,
            np.array([0, 0, -s.front_paw_length * scale])
        )

        # Right front leg
        right_shoulder = self._add_joint(
            JointType.RIGHT_SHOULDER,
            spine_upper,
            np.array([s.front_shoulder_width * scale, 0, -s.body_height * 0.3 * scale])
        )

        right_front_upper = self._add_joint(
            JointType.RIGHT_FRONT_UPPER,
            right_shoulder,
            np.array([0, 0, -s.front_upper_leg_length * scale])
        )

        right_front_lower = self._add_joint(
            JointType.RIGHT_FRONT_LOWER,
            right_front_upper,
            np.array([0, 0, -s.front_lower_leg_length * scale])
        )

        self._add_joint(
            JointType.RIGHT_FRONT_PAW,
            right_front_lower,
            np.array([0, 0, -s.front_paw_length * scale])
        )

        # Back legs
        # Left back leg
        left_hip = self._add_joint(
            JointType.LEFT_HIP,
            self.root,
            np.array([-s.back_hip_width * scale, 0, -s.body_height * 0.2 * scale])
        )

        left_back_upper = self._add_joint(
            JointType.LEFT_BACK_UPPER,
            left_hip,
            np.array([0, -s.back_upper_leg_length * 0.2 * scale, -s.back_upper_leg_length * 0.8 * scale])
        )

        left_back_lower = self._add_joint(
            JointType.LEFT_BACK_LOWER,
            left_back_upper,
            np.array([0, s.back_lower_leg_length * 0.3 * scale, -s.back_lower_leg_length * 0.8 * scale])
        )

        self._add_joint(
            JointType.LEFT_BACK_PAW,
            left_back_lower,
            np.array([0, 0, -s.back_paw_length * scale])
        )

        # Right back leg
        right_hip = self._add_joint(
            JointType.RIGHT_HIP,
            self.root,
            np.array([s.back_hip_width * scale, 0, -s.body_height * 0.2 * scale])
        )

        right_back_upper = self._add_joint(
            JointType.RIGHT_BACK_UPPER,
            right_hip,
            np.array([0, -s.back_upper_leg_length * 0.2 * scale, -s.back_upper_leg_length * 0.8 * scale])
        )

        right_back_lower = self._add_joint(
            JointType.RIGHT_BACK_LOWER,
            right_back_upper,
            np.array([0, s.back_lower_leg_length * 0.3 * scale, -s.back_lower_leg_length * 0.8 * scale])
        )

        self._add_joint(
            JointType.RIGHT_BACK_PAW,
            right_back_lower,
            np.array([0, 0, -s.back_paw_length * scale])
        )

    def _add_joint(self, joint_type: JointType, parent: Joint, offset: np.ndarray) -> Joint:
        """Helper to create a joint, add it to parent, and register it."""
        joint = Joint(joint_type=joint_type, local_offset=offset)
        parent.add_child(joint)
        self.joints[joint_type] = joint
        return joint

    def update_world_positions(self) -> None:
        """
        Compute world positions for all joints using forward kinematics.

        This traverses the joint hierarchy and computes each joint's
        world position based on parent transforms and local rotations.
        """
        if self.root is None:
            return

        # Start with global transform
        global_rotation_matrix = euler_to_rotation_matrix(self.global_rotation)

        self._update_joint_recursive(
            self.root,
            self.global_position.copy(),
            global_rotation_matrix
        )

    def _update_joint_recursive(
        self,
        joint: Joint,
        parent_world_pos: np.ndarray,
        parent_world_rot: np.ndarray
    ) -> None:
        """Recursively update world positions for a joint and its children."""
        # Compute this joint's world position
        local_rotated_offset = parent_world_rot @ joint.local_offset
        joint._world_position = parent_world_pos + local_rotated_offset

        # Compute this joint's world rotation
        local_rotation_matrix = euler_to_rotation_matrix(joint.rotation)
        joint._world_rotation_matrix = parent_world_rot @ local_rotation_matrix

        # Recurse to children
        for child in joint.children:
            self._update_joint_recursive(
                child,
                joint._world_position,
                joint._world_rotation_matrix
            )

    def set_joint_rotation(self, joint_type: JointType, rotation: np.ndarray) -> None:
        """
        Set the rotation for a specific joint.

        Args:
            joint_type: Which joint to rotate
            rotation: Euler angles [x, y, z] in radians
        """
        if joint_type in self.joints:
            self.joints[joint_type].rotation = np.array(rotation)

    def get_joint_position(self, joint_type: JointType) -> np.ndarray:
        """Get the world position of a specific joint."""
        if joint_type in self.joints:
            return self.joints[joint_type].world_position
        raise KeyError(f"Unknown joint type: {joint_type}")

    def get_all_joint_positions(self) -> Dict[JointType, np.ndarray]:
        """Get world positions of all joints."""
        return {jt: j.world_position for jt, j in self.joints.items()}

    def get_bone_segments(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all bone segments as pairs of world positions.

        Returns:
            List of (start_pos, end_pos) tuples for each bone
        """
        segments = []
        for start_type, end_type in self.BONES:
            if start_type in self.joints and end_type in self.joints:
                start_pos = self.joints[start_type].world_position
                end_pos = self.joints[end_type].world_position
                segments.append((start_pos, end_pos))
        return segments

    def reset_pose(self) -> None:
        """Reset all joint rotations to zero (rest pose)."""
        for joint in self.joints.values():
            joint.rotation = np.zeros(3)
        self.update_world_positions()

    def set_global_transform(
        self,
        position: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None
    ) -> None:
        """
        Set the global position and/or rotation of the skeleton.

        Args:
            position: World position [x, y, z]
            rotation: Euler angles [x, y, z] in radians
        """
        if position is not None:
            self.global_position = np.array(position)
        if rotation is not None:
            self.global_rotation = np.array(rotation)
        self.update_world_positions()

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the axis-aligned bounding box of the skeleton.

        Returns:
            Tuple of (min_corner, max_corner) as 3D points
        """
        positions = np.array([j.world_position for j in self.joints.values()])
        return positions.min(axis=0), positions.max(axis=0)
