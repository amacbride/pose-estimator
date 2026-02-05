"""
Motion sequence system for skeletal animation.

This module provides classes for defining and playing back motion sequences
(animations) on quadruped skeletons. Supports keyframe-based animation with
interpolation, looping, and blending.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum

from .skeleton import QuadrupedSkeleton, JointType


class LoopMode(Enum):
    """How a motion sequence should repeat."""
    ONCE = "once"           # Play once and stop
    LOOP = "loop"           # Loop back to start
    PING_PONG = "ping_pong" # Play forward then backward


class EaseType(Enum):
    """Easing functions for interpolation."""
    LINEAR = "linear"
    EASE_IN = "ease_in"       # Slow start
    EASE_OUT = "ease_out"     # Slow end
    EASE_IN_OUT = "ease_in_out"  # Slow start and end
    BOUNCE = "bounce"         # Bouncy overshoot


def ease_linear(t: float) -> float:
    return t


def ease_in(t: float) -> float:
    return t * t


def ease_out(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


def ease_bounce(t: float) -> float:
    # Slight overshoot and settle
    if t < 0.5:
        return 2 * t * t
    elif t < 0.75:
        return 1 + 0.1 * np.sin((t - 0.5) * np.pi * 4)
    return 1


EASING_FUNCTIONS: Dict[EaseType, Callable[[float], float]] = {
    EaseType.LINEAR: ease_linear,
    EaseType.EASE_IN: ease_in,
    EaseType.EASE_OUT: ease_out,
    EaseType.EASE_IN_OUT: ease_in_out,
    EaseType.BOUNCE: ease_bounce,
}


@dataclass
class Keyframe:
    """
    A keyframe in a motion sequence.

    Represents the state of specific joints at a point in time.
    Only joints that should change need to be specified.
    """
    time: float  # Time in seconds from start of motion
    rotations: Dict[JointType, np.ndarray] = field(default_factory=dict)
    easing: EaseType = EaseType.LINEAR  # Easing to next keyframe

    def add_rotation(self, joint: JointType, rotation: np.ndarray) -> 'Keyframe':
        """Add a joint rotation to this keyframe (fluent interface)."""
        self.rotations[joint] = np.array(rotation)
        return self


@dataclass
class MotionSequence:
    """
    A complete motion sequence (animation clip).

    Contains a series of keyframes and metadata about how to play the motion.
    """
    name: str
    keyframes: List[Keyframe] = field(default_factory=list)
    loop_mode: LoopMode = LoopMode.ONCE
    base_pose: Optional[str] = None  # Starting pose name (optional)

    @property
    def duration(self) -> float:
        """Total duration of the motion in seconds."""
        if not self.keyframes:
            return 0.0
        return max(kf.time for kf in self.keyframes)

    def add_keyframe(self, keyframe: Keyframe) -> 'MotionSequence':
        """Add a keyframe (fluent interface)."""
        self.keyframes.append(keyframe)
        # Keep keyframes sorted by time
        self.keyframes.sort(key=lambda kf: kf.time)
        return self

    def get_rotations_at_time(self, time: float) -> Dict[JointType, np.ndarray]:
        """
        Get interpolated joint rotations at a specific time.

        Args:
            time: Time in seconds

        Returns:
            Dictionary of joint rotations
        """
        if not self.keyframes:
            return {}

        # Handle looping
        duration = self.duration
        if duration <= 0:
            return self.keyframes[0].rotations if self.keyframes else {}

        if self.loop_mode == LoopMode.LOOP:
            time = time % duration
        elif self.loop_mode == LoopMode.PING_PONG:
            cycle = time / duration
            if int(cycle) % 2 == 1:
                time = duration - (time % duration)
            else:
                time = time % duration
        else:  # ONCE
            time = min(time, duration)

        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.time <= time:
                prev_kf = kf
            if kf.time >= time:
                next_kf = kf
                break

        # Interpolate between keyframes
        if prev_kf.time == next_kf.time:
            return prev_kf.rotations.copy()

        # Calculate interpolation factor
        t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)

        # Apply easing
        ease_func = EASING_FUNCTIONS.get(prev_kf.easing, ease_linear)
        t = ease_func(t)

        # Interpolate rotations
        result = {}
        all_joints = set(prev_kf.rotations.keys()) | set(next_kf.rotations.keys())

        for joint in all_joints:
            rot_a = prev_kf.rotations.get(joint, np.zeros(3))
            rot_b = next_kf.rotations.get(joint, np.zeros(3))
            result[joint] = rot_a * (1 - t) + rot_b * t

        return result

    def get_all_affected_joints(self) -> set:
        """Get all joints that are animated in this sequence."""
        joints = set()
        for kf in self.keyframes:
            joints.update(kf.rotations.keys())
        return joints


class MotionPlayer:
    """
    Plays motion sequences on a skeleton.

    Handles timing, blending, and applying motions to skeletons.
    """

    def __init__(self, skeleton: QuadrupedSkeleton):
        """
        Initialize the motion player.

        Args:
            skeleton: The skeleton to animate
        """
        self.skeleton = skeleton
        self.current_motion: Optional[MotionSequence] = None
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.playback_speed: float = 1.0

        # For blending between motions
        self._blend_from: Optional[Dict[JointType, np.ndarray]] = None
        self._blend_duration: float = 0.0
        self._blend_time: float = 0.0

    def play(
        self,
        motion: MotionSequence,
        blend_duration: float = 0.0,
        start_time: float = 0.0
    ) -> None:
        """
        Start playing a motion sequence.

        Args:
            motion: The motion to play
            blend_duration: Time to blend from current pose (0 = instant)
            start_time: Time offset to start from
        """
        # Capture current pose for blending
        if blend_duration > 0 and self.current_motion:
            self._blend_from = {
                jt: j.rotation.copy()
                for jt, j in self.skeleton.joints.items()
            }
            self._blend_duration = blend_duration
            self._blend_time = 0.0
        else:
            self._blend_from = None

        self.current_motion = motion
        self.current_time = start_time
        self.is_playing = True

        # Apply base pose if specified
        if motion.base_pose:
            if hasattr(self.skeleton, 'set_pose'):
                self.skeleton.set_pose(motion.base_pose)

    def stop(self) -> None:
        """Stop the current motion."""
        self.is_playing = False

    def update(self, delta_time: float) -> bool:
        """
        Update the animation by a time delta.

        Args:
            delta_time: Time elapsed in seconds

        Returns:
            True if motion is still playing, False if finished
        """
        if not self.is_playing or not self.current_motion:
            return False

        self.current_time += delta_time * self.playback_speed

        # Update blend
        if self._blend_from and self._blend_time < self._blend_duration:
            self._blend_time += delta_time

        # Check if finished (for non-looping motions)
        if self.current_motion.loop_mode == LoopMode.ONCE:
            if self.current_time >= self.current_motion.duration:
                self.is_playing = False

        # Apply current frame
        self._apply_current_frame()

        return self.is_playing

    def set_time(self, time: float) -> None:
        """Set the current playback time."""
        self.current_time = time
        self._apply_current_frame()

    def _apply_current_frame(self) -> None:
        """Apply the current frame's rotations to the skeleton."""
        if not self.current_motion:
            return

        # Get rotations from motion
        rotations = self.current_motion.get_rotations_at_time(self.current_time)

        # Blend with previous pose if needed
        if self._blend_from and self._blend_time < self._blend_duration:
            blend_t = self._blend_time / self._blend_duration
            blend_t = ease_in_out(blend_t)  # Smooth blend

            for joint_type, target_rot in rotations.items():
                if joint_type in self._blend_from:
                    from_rot = self._blend_from[joint_type]
                    rotations[joint_type] = from_rot * (1 - blend_t) + target_rot * blend_t

        # Apply to skeleton
        for joint_type, rotation in rotations.items():
            self.skeleton.set_joint_rotation(joint_type, rotation)

        self.skeleton.update_world_positions()

    def get_frame_data(self, time: float) -> Dict[JointType, np.ndarray]:
        """Get rotation data for a specific time without applying it."""
        if not self.current_motion:
            return {}
        return self.current_motion.get_rotations_at_time(time)


def generate_motion_frames(
    skeleton: QuadrupedSkeleton,
    motion: MotionSequence,
    fps: float = 30.0,
    include_base_pose: bool = True
) -> List[Dict[JointType, np.ndarray]]:
    """
    Generate all frames of a motion as a list of joint positions.

    Args:
        skeleton: The skeleton to animate
        motion: The motion sequence
        fps: Frames per second
        include_base_pose: Whether to apply base pose first

    Returns:
        List of dictionaries mapping joint types to world positions
    """
    player = MotionPlayer(skeleton)

    # Apply base pose
    if include_base_pose and motion.base_pose and hasattr(skeleton, 'set_pose'):
        skeleton.set_pose(motion.base_pose)

    frames = []
    duration = motion.duration
    frame_count = int(duration * fps) + 1
    delta = 1.0 / fps

    player.play(motion)

    for i in range(frame_count):
        time = i * delta
        player.set_time(time)
        frames.append(skeleton.get_all_joint_positions())

    return frames
