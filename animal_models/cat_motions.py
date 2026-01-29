"""
Motion library for cat skeleton animations.

This module provides predefined motion sequences for common cat behaviors
and actions, designed to work with the CatSkeleton model.
"""

import numpy as np
from typing import Dict, List, Optional

from .skeleton import JointType
from .motion import MotionSequence, Keyframe, LoopMode, EaseType


def create_paw_swipe_right() -> MotionSequence:
    """
    Cat swiping with right front paw.

    A quick, sharp swipe motion - perfect for batting at toys or
    defensive swatting.
    """
    motion = MotionSequence(
        name="paw_swipe_right",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Starting position - weight shifts slightly, paw raises
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0, 0]),
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, -0.1, 0]),  # Looking at target
            # Right front leg ready
            JointType.RIGHT_SHOULDER: np.array([0.1, 0, -0.1]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.05, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Wind up - paw pulls back
    motion.add_keyframe(Keyframe(
        time=0.15,
        rotations={
            JointType.SPINE_MID: np.array([0.08, 0, -0.05]),
            JointType.SPINE_UPPER: np.array([0.05, 0, -0.08]),
            JointType.NECK: np.array([0.2, 0, -0.1]),
            JointType.HEAD: np.array([0.15, -0.15, 0]),
            # Paw lifts and pulls back
            JointType.RIGHT_SHOULDER: np.array([0.4, -0.3, -0.2]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.6, 0, 0.2]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.8, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.3, 0, 0]),
            # Weight shifts left
            JointType.LEFT_SHOULDER: np.array([0.05, 0, 0.1]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Strike! - fast forward swipe
    motion.add_keyframe(Keyframe(
        time=0.25,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0, 0.1]),
            JointType.SPINE_UPPER: np.array([0.02, 0, 0.12]),
            JointType.NECK: np.array([0.12, 0, 0.08]),
            JointType.HEAD: np.array([0.08, 0.1, 0]),  # Following through
            # Paw swipes across
            JointType.RIGHT_SHOULDER: np.array([0.35, 0.4, 0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.3, 0.2, -0.1]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.4, 0.1, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([-0.2, 0.2, 0]),
            JointType.LEFT_SHOULDER: np.array([0.02, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Follow through and recover
    motion.add_keyframe(Keyframe(
        time=0.4,
        rotations={
            JointType.SPINE_MID: np.array([0.03, 0, 0.02]),
            JointType.SPINE_UPPER: np.array([0.02, 0, 0.02]),
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Paw returning
            JointType.RIGHT_SHOULDER: np.array([0.15, 0.1, 0.1]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.15, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Return to neutral
    motion.add_keyframe(Keyframe(
        time=0.55,
        rotations={
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.NECK: np.array([0.12, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_paw_swipe_left() -> MotionSequence:
    """
    Cat swiping with left front paw.

    Mirror of right paw swipe.
    """
    motion = MotionSequence(
        name="paw_swipe_left",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Starting position
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0, 0]),
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0.1, 0]),  # Looking at target
            JointType.LEFT_SHOULDER: np.array([0.1, 0, 0.1]),
            JointType.LEFT_FRONT_UPPER: np.array([0.05, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Wind up
    motion.add_keyframe(Keyframe(
        time=0.15,
        rotations={
            JointType.SPINE_MID: np.array([0.08, 0, 0.05]),
            JointType.SPINE_UPPER: np.array([0.05, 0, 0.08]),
            JointType.NECK: np.array([0.2, 0, 0.1]),
            JointType.HEAD: np.array([0.15, 0.15, 0]),
            JointType.LEFT_SHOULDER: np.array([0.4, 0.3, 0.2]),
            JointType.LEFT_FRONT_UPPER: np.array([0.6, 0, -0.2]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.8, 0, 0]),
            JointType.LEFT_FRONT_PAW: np.array([0.3, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.05, 0, -0.1]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Strike!
    motion.add_keyframe(Keyframe(
        time=0.25,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0, -0.1]),
            JointType.SPINE_UPPER: np.array([0.02, 0, -0.12]),
            JointType.NECK: np.array([0.12, 0, -0.08]),
            JointType.HEAD: np.array([0.08, -0.1, 0]),
            JointType.LEFT_SHOULDER: np.array([0.35, -0.4, -0.3]),
            JointType.LEFT_FRONT_UPPER: np.array([0.3, -0.2, 0.1]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.4, -0.1, 0]),
            JointType.LEFT_FRONT_PAW: np.array([-0.2, -0.2, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.02, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Follow through
    motion.add_keyframe(Keyframe(
        time=0.4,
        rotations={
            JointType.SPINE_MID: np.array([0.03, 0, -0.02]),
            JointType.SPINE_UPPER: np.array([0.02, 0, -0.02]),
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.15, -0.1, -0.1]),
            JointType.LEFT_FRONT_UPPER: np.array([0.1, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.15, 0, 0]),
            JointType.LEFT_FRONT_PAW: np.array([0, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Return to neutral
    motion.add_keyframe(Keyframe(
        time=0.55,
        rotations={
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.NECK: np.array([0.12, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_PAW: np.array([0, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_pounce() -> MotionSequence:
    """
    Cat pouncing motion - the crouch, wiggle, and leap.
    """
    motion = MotionSequence(
        name="pounce",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Alert - spotting prey
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.3, 0, 0]),
            JointType.HEAD: np.array([0.2, 0, 0]),
            JointType.LEFT_EAR: np.array([0.2, 0.1, 0]),
            JointType.RIGHT_EAR: np.array([0.2, -0.1, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Crouch down
    motion.add_keyframe(Keyframe(
        time=0.3,
        rotations={
            JointType.SPINE_LOWER: np.array([-0.2, 0, 0]),
            JointType.SPINE_MID: np.array([-0.1, 0, 0]),
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.25, 0, 0]),
            # Front legs bent
            JointType.LEFT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            # Back legs coiled
            JointType.LEFT_HIP: np.array([-0.4, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.0, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.4, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.0, 0, 0]),
            # Tail low
            JointType.TAIL_BASE: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Butt wiggle 1
    motion.add_keyframe(Keyframe(
        time=0.5,
        rotations={
            JointType.SPINE_LOWER: np.array([-0.2, 0, 0.15]),
            JointType.SPINE_MID: np.array([-0.1, 0, 0.1]),
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.25, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.4, 0, 0.1]),
            JointType.LEFT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.0, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.4, 0, -0.1]),
            JointType.RIGHT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.0, 0, 0]),
            JointType.TAIL_BASE: np.array([-0.1, 0, 0.2]),
            JointType.TAIL_MID: np.array([0, 0, 0.15]),
        },
        easing=EaseType.LINEAR
    ))

    # Butt wiggle 2
    motion.add_keyframe(Keyframe(
        time=0.65,
        rotations={
            JointType.SPINE_LOWER: np.array([-0.2, 0, -0.15]),
            JointType.SPINE_MID: np.array([-0.1, 0, -0.1]),
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.25, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.3, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.4, 0, -0.1]),
            JointType.LEFT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.0, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.4, 0, 0.1]),
            JointType.RIGHT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.0, 0, 0]),
            JointType.TAIL_BASE: np.array([-0.1, 0, -0.2]),
            JointType.TAIL_MID: np.array([0, 0, -0.15]),
        },
        easing=EaseType.EASE_IN
    ))

    # Launch! - explosive extension
    motion.add_keyframe(Keyframe(
        time=0.8,
        rotations={
            JointType.SPINE_LOWER: np.array([0.3, 0, 0]),
            JointType.SPINE_MID: np.array([0.25, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0]),
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([0, 0, 0]),
            # Front legs reaching forward
            JointType.LEFT_SHOULDER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.5, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.4, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            # Back legs pushing off
            JointType.LEFT_HIP: np.array([0.4, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([-0.3, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([0.2, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.4, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([-0.3, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([0.2, 0, 0]),
            # Tail streaming
            JointType.TAIL_BASE: np.array([0.1, 0, 0]),
            JointType.TAIL_MID: np.array([-0.1, 0, 0]),
            JointType.TAIL_TIP: np.array([-0.15, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Landing
    motion.add_keyframe(Keyframe(
        time=1.0,
        rotations={
            JointType.SPINE_LOWER: np.array([0.05, 0, 0]),
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.15, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.2, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.LEFT_HIP: np.array([0, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.1, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-0.15, 0, 0]),
            JointType.RIGHT_HIP: np.array([0, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-0.15, 0, 0]),
            JointType.TAIL_BASE: np.array([0.2, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
        }
    ))

    return motion


def create_tail_swish() -> MotionSequence:
    """
    Cat tail swishing back and forth - can indicate irritation or focus.
    """
    motion = MotionSequence(
        name="tail_swish",
        loop_mode=LoopMode.LOOP,
        base_pose="standing"
    )

    # Center
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.TAIL_BASE: np.array([0.25, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
            JointType.TAIL_TIP: np.array([0, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Swish left
    motion.add_keyframe(Keyframe(
        time=0.25,
        rotations={
            JointType.TAIL_BASE: np.array([0.25, 0, -0.4]),
            JointType.TAIL_MID: np.array([0.1, 0, -0.5]),
            JointType.TAIL_TIP: np.array([0, 0, -0.3]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Center
    motion.add_keyframe(Keyframe(
        time=0.5,
        rotations={
            JointType.TAIL_BASE: np.array([0.25, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
            JointType.TAIL_TIP: np.array([0, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Swish right
    motion.add_keyframe(Keyframe(
        time=0.75,
        rotations={
            JointType.TAIL_BASE: np.array([0.25, 0, 0.4]),
            JointType.TAIL_MID: np.array([0.1, 0, 0.5]),
            JointType.TAIL_TIP: np.array([0, 0, 0.3]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Back to center (for loop)
    motion.add_keyframe(Keyframe(
        time=1.0,
        rotations={
            JointType.TAIL_BASE: np.array([0.25, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
            JointType.TAIL_TIP: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_hiss() -> MotionSequence:
    """
    Defensive hiss - arched back, ears flat, mouth open.
    """
    motion = MotionSequence(
        name="hiss",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Start normal
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Arch up and hiss
    motion.add_keyframe(Keyframe(
        time=0.2,
        rotations={
            # Arch the back
            JointType.SPINE_LOWER: np.array([0.35, 0, 0]),
            JointType.SPINE_MID: np.array([0.45, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.3, 0, 0]),
            # Head forward, mouth open (implied)
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            # Ears flat
            JointType.LEFT_EAR: np.array([-0.4, -0.3, 0.2]),
            JointType.RIGHT_EAR: np.array([-0.4, 0.3, -0.2]),
            # Tail puffed (up and slightly arched)
            JointType.TAIL_BASE: np.array([0.7, 0, 0]),
            JointType.TAIL_MID: np.array([0.4, 0, 0]),
            JointType.TAIL_TIP: np.array([0.2, 0, 0]),
            # Legs slightly bent
            JointType.LEFT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.2, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Hold the hiss
    motion.add_keyframe(Keyframe(
        time=0.8,
        rotations={
            JointType.SPINE_LOWER: np.array([0.35, 0, 0]),
            JointType.SPINE_MID: np.array([0.45, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.3, 0, 0]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            JointType.LEFT_EAR: np.array([-0.4, -0.3, 0.2]),
            JointType.RIGHT_EAR: np.array([-0.4, 0.3, -0.2]),
            JointType.TAIL_BASE: np.array([0.7, 0, 0]),
            JointType.TAIL_MID: np.array([0.4, 0, 0]),
            JointType.TAIL_TIP: np.array([0.2, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.2, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Slowly relax
    motion.add_keyframe(Keyframe(
        time=1.2,
        rotations={
            JointType.SPINE_LOWER: np.array([0.1, 0, 0]),
            JointType.SPINE_MID: np.array([0.15, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_EAR: np.array([0, 0, 0]),
            JointType.RIGHT_EAR: np.array([0, 0, 0]),
            JointType.TAIL_BASE: np.array([0.3, 0, 0]),
            JointType.TAIL_MID: np.array([0.15, 0, 0]),
            JointType.TAIL_TIP: np.array([0.05, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_groom_paw() -> MotionSequence:
    """
    Cat grooming - licking front paw motion.
    """
    motion = MotionSequence(
        name="groom_paw",
        loop_mode=LoopMode.LOOP,
        base_pose="sitting"
    )

    # Sitting, paw raised
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.3, 0, 0.2]),
            JointType.HEAD: np.array([0.4, 0, 0.15]),
            # Right paw raised to face
            JointType.RIGHT_SHOULDER: np.array([0.6, 0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.2, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Lick motion 1 - head moves to paw
    motion.add_keyframe(Keyframe(
        time=0.3,
        rotations={
            JointType.NECK: np.array([0.35, 0, 0.25]),
            JointType.HEAD: np.array([0.5, 0, 0.2]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.2, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Lick motion 2 - head pulls back slightly
    motion.add_keyframe(Keyframe(
        time=0.5,
        rotations={
            JointType.NECK: np.array([0.3, 0, 0.2]),
            JointType.HEAD: np.array([0.4, 0, 0.15]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.2, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Lick motion 3
    motion.add_keyframe(Keyframe(
        time=0.8,
        rotations={
            JointType.NECK: np.array([0.35, 0, 0.25]),
            JointType.HEAD: np.array([0.5, 0, 0.2]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.2, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Return
    motion.add_keyframe(Keyframe(
        time=1.0,
        rotations={
            JointType.NECK: np.array([0.3, 0, 0.2]),
            JointType.HEAD: np.array([0.4, 0, 0.15]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([1.0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_FRONT_PAW: np.array([0.2, 0, 0]),
        }
    ))

    return motion


# Motion library - all available cat motions
CAT_MOTIONS: Dict[str, MotionSequence] = {}


def _initialize_cat_motions():
    """Initialize the cat motion library."""
    global CAT_MOTIONS
    CAT_MOTIONS = {
        'paw_swipe_right': create_paw_swipe_right(),
        'paw_swipe_left': create_paw_swipe_left(),
        'pounce': create_pounce(),
        'tail_swish': create_tail_swish(),
        'hiss': create_hiss(),
        'groom_paw': create_groom_paw(),
    }


def get_cat_motion(name: str) -> Optional[MotionSequence]:
    """Get a cat motion by name."""
    if not CAT_MOTIONS:
        _initialize_cat_motions()
    return CAT_MOTIONS.get(name)


def get_available_cat_motions() -> List[str]:
    """Get list of available cat motion names."""
    if not CAT_MOTIONS:
        _initialize_cat_motions()
    return list(CAT_MOTIONS.keys())


# Initialize on import
_initialize_cat_motions()
