"""
Motion library for dog skeleton animations.

This module provides predefined motion sequences for common dog behaviors
and actions, designed to work with the DogSkeleton model.
"""

import numpy as np
from typing import Dict, List, Optional

from .skeleton import JointType
from .motion import MotionSequence, Keyframe, LoopMode, EaseType


def create_grab_shake() -> MotionSequence:
    """
    Dog grabbing something in mouth and shaking it vigorously.

    Classic dog behavior when playing with toys or "killing" prey.
    The motion involves head/neck whipping side to side while
    the body braces.
    """
    motion = MotionSequence(
        name="grab_shake",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Starting - alert, spotted the target
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([0.05, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Lunge forward to grab
    motion.add_keyframe(Keyframe(
        time=0.15,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0]),
            JointType.NECK: np.array([0.05, 0, 0]),
            JointType.HEAD: np.array([-0.2, 0, 0]),  # Head down to grab
            # Front legs brace
            JointType.LEFT_SHOULDER: np.array([0.15, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.15, 0, 0]),
            # Back legs push
            JointType.LEFT_HIP: np.array([0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.1, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Got it! Head comes up with prize
    motion.add_keyframe(Keyframe(
        time=0.3,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.08, 0, 0]),
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.1, 0, 0.05]),
            JointType.RIGHT_SHOULDER: np.array([0.1, 0, -0.05]),
            # Legs spread for stability
            JointType.LEFT_HIP: np.array([0, 0, 0.1]),
            JointType.RIGHT_HIP: np.array([0, 0, -0.1]),
        },
        easing=EaseType.EASE_IN
    ))

    # Shake 1 - violent whip to the left
    motion.add_keyframe(Keyframe(
        time=0.4,
        rotations={
            JointType.SPINE_MID: np.array([0.05, -0.1, -0.15]),
            JointType.SPINE_UPPER: np.array([0.08, -0.15, -0.2]),
            JointType.NECK: np.array([0.15, -0.3, -0.4]),
            JointType.HEAD: np.array([0.1, -0.2, -0.35]),
            # Ears flop
            JointType.LEFT_EAR: np.array([-0.2, 0.3, 0.4]),
            JointType.RIGHT_EAR: np.array([-0.1, 0.2, 0.3]),
            # Body braces
            JointType.LEFT_SHOULDER: np.array([0.12, 0, 0.1]),
            JointType.RIGHT_SHOULDER: np.array([0.08, 0, -0.05]),
            JointType.LEFT_HIP: np.array([0.02, 0, 0.12]),
            JointType.RIGHT_HIP: np.array([-0.02, 0, -0.08]),
        },
        easing=EaseType.LINEAR
    ))

    # Shake 2 - whip to the right
    motion.add_keyframe(Keyframe(
        time=0.5,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0.1, 0.15]),
            JointType.SPINE_UPPER: np.array([0.08, 0.15, 0.2]),
            JointType.NECK: np.array([0.15, 0.3, 0.4]),
            JointType.HEAD: np.array([0.1, 0.2, 0.35]),
            JointType.LEFT_EAR: np.array([-0.1, -0.2, -0.3]),
            JointType.RIGHT_EAR: np.array([-0.2, -0.3, -0.4]),
            JointType.LEFT_SHOULDER: np.array([0.08, 0, -0.05]),
            JointType.RIGHT_SHOULDER: np.array([0.12, 0, 0.1]),
            JointType.LEFT_HIP: np.array([-0.02, 0, -0.08]),
            JointType.RIGHT_HIP: np.array([0.02, 0, 0.12]),
        },
        easing=EaseType.LINEAR
    ))

    # Shake 3 - back to left (faster)
    motion.add_keyframe(Keyframe(
        time=0.58,
        rotations={
            JointType.SPINE_MID: np.array([0.05, -0.12, -0.18]),
            JointType.SPINE_UPPER: np.array([0.08, -0.18, -0.25]),
            JointType.NECK: np.array([0.15, -0.35, -0.45]),
            JointType.HEAD: np.array([0.1, -0.25, -0.4]),
            JointType.LEFT_EAR: np.array([-0.25, 0.35, 0.45]),
            JointType.RIGHT_EAR: np.array([-0.15, 0.25, 0.35]),
            JointType.LEFT_SHOULDER: np.array([0.14, 0, 0.12]),
            JointType.RIGHT_SHOULDER: np.array([0.06, 0, -0.08]),
            JointType.LEFT_HIP: np.array([0.03, 0, 0.14]),
            JointType.RIGHT_HIP: np.array([-0.03, 0, -0.1]),
        },
        easing=EaseType.LINEAR
    ))

    # Shake 4 - back to right
    motion.add_keyframe(Keyframe(
        time=0.66,
        rotations={
            JointType.SPINE_MID: np.array([0.05, 0.12, 0.18]),
            JointType.SPINE_UPPER: np.array([0.08, 0.18, 0.25]),
            JointType.NECK: np.array([0.15, 0.35, 0.45]),
            JointType.HEAD: np.array([0.1, 0.25, 0.4]),
            JointType.LEFT_EAR: np.array([-0.15, -0.25, -0.35]),
            JointType.RIGHT_EAR: np.array([-0.25, -0.35, -0.45]),
            JointType.LEFT_SHOULDER: np.array([0.06, 0, -0.08]),
            JointType.RIGHT_SHOULDER: np.array([0.14, 0, 0.12]),
            JointType.LEFT_HIP: np.array([-0.03, 0, -0.1]),
            JointType.RIGHT_HIP: np.array([0.03, 0, 0.14]),
        },
        easing=EaseType.LINEAR
    ))

    # Shake 5 - one more to left
    motion.add_keyframe(Keyframe(
        time=0.74,
        rotations={
            JointType.SPINE_MID: np.array([0.05, -0.1, -0.12]),
            JointType.SPINE_UPPER: np.array([0.08, -0.12, -0.18]),
            JointType.NECK: np.array([0.15, -0.25, -0.35]),
            JointType.HEAD: np.array([0.1, -0.18, -0.3]),
            JointType.LEFT_EAR: np.array([-0.18, 0.25, 0.35]),
            JointType.RIGHT_EAR: np.array([-0.1, 0.18, 0.28]),
            JointType.LEFT_SHOULDER: np.array([0.12, 0, 0.08]),
            JointType.RIGHT_SHOULDER: np.array([0.08, 0, -0.04]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Settle down - still holding prize proudly
    motion.add_keyframe(Keyframe(
        time=0.9,
        rotations={
            JointType.SPINE_MID: np.array([0.03, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.05, 0, 0]),
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.15, 0, 0]),
            JointType.LEFT_EAR: np.array([0, 0, 0]),
            JointType.RIGHT_EAR: np.array([0, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.05, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.05, 0, 0]),
            JointType.LEFT_HIP: np.array([0, 0, 0]),
            JointType.RIGHT_HIP: np.array([0, 0, 0]),
            # Tail up - proud!
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.25, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Return to neutral
    motion.add_keyframe(Keyframe(
        time=1.1,
        rotations={
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.02, 0, 0]),
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0, 0, 0]),
            JointType.TAIL_BASE: np.array([0.35, 0, 0]),
            JointType.TAIL_MID: np.array([0.15, 0, 0]),
        }
    ))

    return motion


def create_bark() -> MotionSequence:
    """
    Dog barking motion - the characteristic lunge and vocalization pose.
    """
    motion = MotionSequence(
        name="bark",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Alert stance
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.15, 0, 0]),
            JointType.LEFT_EAR: np.array([0.1, 0.1, 0]),
            JointType.RIGHT_EAR: np.array([0.1, -0.1, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Inhale - slight crouch, head pulls back
    motion.add_keyframe(Keyframe(
        time=0.1,
        rotations={
            JointType.SPINE_MID: np.array([-0.05, 0, 0]),
            JointType.NECK: np.array([0.25, 0, 0]),
            JointType.HEAD: np.array([0.2, 0, 0]),
            JointType.LEFT_EAR: np.array([0.15, 0.1, 0]),
            JointType.RIGHT_EAR: np.array([0.15, -0.1, 0]),
            # Slight leg bend
            JointType.LEFT_FRONT_UPPER: np.array([0.05, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.05, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # BARK! - head thrusts forward, body follows
    motion.add_keyframe(Keyframe(
        time=0.18,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0]),
            JointType.NECK: np.array([0.05, 0, 0]),
            JointType.HEAD: np.array([-0.1, 0, 0]),  # Head forward/down for bark
            JointType.LEFT_EAR: np.array([-0.05, 0.15, 0]),
            JointType.RIGHT_EAR: np.array([-0.05, -0.15, 0]),
            # Front extends slightly
            JointType.LEFT_FRONT_UPPER: np.array([-0.05, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([-0.05, 0, 0]),
            # Tail reacts
            JointType.TAIL_BASE: np.array([0.4, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Recover
    motion.add_keyframe(Keyframe(
        time=0.35,
        rotations={
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.03, 0, 0]),
            JointType.NECK: np.array([0.18, 0, 0]),
            JointType.HEAD: np.array([0.12, 0, 0]),
            JointType.LEFT_EAR: np.array([0.1, 0.1, 0]),
            JointType.RIGHT_EAR: np.array([0.1, -0.1, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.TAIL_BASE: np.array([0.35, 0, 0]),
        }
    ))

    return motion


def create_bark_sequence() -> MotionSequence:
    """
    Multiple barks in sequence - more realistic barking behavior.
    """
    motion = MotionSequence(
        name="bark_sequence",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    bark_times = [0.0, 0.4, 0.75]

    for i, start_time in enumerate(bark_times):
        intensity = 1.0 - (i * 0.1)  # Slightly less intense each bark

        # Alert/inhale
        motion.add_keyframe(Keyframe(
            time=start_time,
            rotations={
                JointType.NECK: np.array([0.25 * intensity, 0, 0]),
                JointType.HEAD: np.array([0.2 * intensity, 0, 0]),
            },
            easing=EaseType.EASE_IN
        ))

        # Bark!
        motion.add_keyframe(Keyframe(
            time=start_time + 0.1,
            rotations={
                JointType.SPINE_MID: np.array([0.1 * intensity, 0, 0]),
                JointType.SPINE_UPPER: np.array([0.15 * intensity, 0, 0]),
                JointType.NECK: np.array([0.05, 0, 0]),
                JointType.HEAD: np.array([-0.1 * intensity, 0, 0]),
                JointType.TAIL_BASE: np.array([0.4 * intensity, 0, 0]),
            },
            easing=EaseType.EASE_OUT
        ))

        # Recover
        motion.add_keyframe(Keyframe(
            time=start_time + 0.25,
            rotations={
                JointType.SPINE_MID: np.array([0.02, 0, 0]),
                JointType.SPINE_UPPER: np.array([0.03, 0, 0]),
                JointType.NECK: np.array([0.18, 0, 0]),
                JointType.HEAD: np.array([0.12, 0, 0]),
                JointType.TAIL_BASE: np.array([0.35, 0, 0]),
            }
        ))

    return motion


def create_dig() -> MotionSequence:
    """
    Dog digging motion - alternating front paws scratching at ground.
    """
    motion = MotionSequence(
        name="dig",
        loop_mode=LoopMode.LOOP,
        base_pose="standing"
    )

    # Starting position - head down, ready to dig
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            # Weight back slightly
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Left paw dig - lift
    motion.add_keyframe(Keyframe(
        time=0.1,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0.05]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0.05]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            # Left paw lifts
            JointType.LEFT_SHOULDER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.6, 0, 0]),
            # Right paw planted
            JointType.RIGHT_SHOULDER: np.array([0.1, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Left paw dig - scratch back
    motion.add_keyframe(Keyframe(
        time=0.2,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, -0.05]),
            JointType.SPINE_UPPER: np.array([0.15, 0, -0.03]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            # Left paw digs back
            JointType.LEFT_SHOULDER: np.array([0.15, -0.2, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.1, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.15, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.1, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Right paw dig - lift
    motion.add_keyframe(Keyframe(
        time=0.35,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, -0.05]),
            JointType.SPINE_UPPER: np.array([0.15, 0, -0.05]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.1, 0, 0]),
            # Right paw lifts
            JointType.RIGHT_SHOULDER: np.array([0.4, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.6, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Right paw dig - scratch back
    motion.add_keyframe(Keyframe(
        time=0.45,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0.05]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0.03]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0.1, 0, 0]),
            # Right paw digs back
            JointType.RIGHT_SHOULDER: np.array([0.15, 0.2, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.15, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN
    ))

    # Return to start position for loop
    motion.add_keyframe(Keyframe(
        time=0.55,
        rotations={
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.15, 0, 0]),
            JointType.NECK: np.array([-0.2, 0, 0]),
            JointType.HEAD: np.array([-0.15, 0, 0]),
            JointType.LEFT_SHOULDER: np.array([0, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([0, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([0, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.1, 0, 0]),
        }
    ))

    return motion


def create_head_tilt() -> MotionSequence:
    """
    Curious head tilt - the adorable questioning gesture.
    """
    motion = MotionSequence(
        name="head_tilt",
        loop_mode=LoopMode.ONCE,
        base_pose="standing"
    )

    # Normal position
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Tilt head to the right
    motion.add_keyframe(Keyframe(
        time=0.3,
        rotations={
            JointType.NECK: np.array([0.2, 0, 0.1]),
            JointType.HEAD: np.array([0.15, 0, 0.35]),
            # Ears perk up
            JointType.LEFT_EAR: np.array([0.15, 0.1, 0.1]),
            JointType.RIGHT_EAR: np.array([0.2, -0.05, -0.1]),
        },
        easing=EaseType.EASE_OUT
    ))

    # Hold the tilt
    motion.add_keyframe(Keyframe(
        time=0.8,
        rotations={
            JointType.NECK: np.array([0.2, 0, 0.1]),
            JointType.HEAD: np.array([0.15, 0, 0.35]),
            JointType.LEFT_EAR: np.array([0.15, 0.1, 0.1]),
            JointType.RIGHT_EAR: np.array([0.2, -0.05, -0.1]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Return to normal
    motion.add_keyframe(Keyframe(
        time=1.1,
        rotations={
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            JointType.LEFT_EAR: np.array([0, 0, 0]),
            JointType.RIGHT_EAR: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_tail_wag() -> MotionSequence:
    """
    Happy tail wagging - fast and enthusiastic.
    """
    motion = MotionSequence(
        name="tail_wag",
        loop_mode=LoopMode.LOOP,
        base_pose="standing"
    )

    # Center (tail up)
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0]),
            # Slight body sway
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Wag left
    motion.add_keyframe(Keyframe(
        time=0.08,
        rotations={
            JointType.TAIL_BASE: np.array([0.5, 0, -0.5]),
            JointType.TAIL_MID: np.array([0.2, 0, -0.6]),
            JointType.TAIL_TIP: np.array([0.1, 0, -0.4]),
            JointType.SPINE_MID: np.array([0.02, 0, -0.03]),
            # Hips sway opposite
            JointType.LEFT_HIP: np.array([0, 0, 0.05]),
            JointType.RIGHT_HIP: np.array([0, 0, 0.05]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Center
    motion.add_keyframe(Keyframe(
        time=0.16,
        rotations={
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0]),
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.LEFT_HIP: np.array([0, 0, 0]),
            JointType.RIGHT_HIP: np.array([0, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Wag right
    motion.add_keyframe(Keyframe(
        time=0.24,
        rotations={
            JointType.TAIL_BASE: np.array([0.5, 0, 0.5]),
            JointType.TAIL_MID: np.array([0.2, 0, 0.6]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0.4]),
            JointType.SPINE_MID: np.array([0.02, 0, 0.03]),
            JointType.LEFT_HIP: np.array([0, 0, -0.05]),
            JointType.RIGHT_HIP: np.array([0, 0, -0.05]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Back to center for loop
    motion.add_keyframe(Keyframe(
        time=0.32,
        rotations={
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0]),
            JointType.SPINE_MID: np.array([0.02, 0, 0]),
            JointType.LEFT_HIP: np.array([0, 0, 0]),
            JointType.RIGHT_HIP: np.array([0, 0, 0]),
        }
    ))

    return motion


def create_sniff_ground() -> MotionSequence:
    """
    Dog sniffing the ground - nose down, investigating.
    """
    motion = MotionSequence(
        name="sniff_ground",
        loop_mode=LoopMode.LOOP,
        base_pose="standing"
    )

    # Head down, sniffing
    motion.add_keyframe(Keyframe(
        time=0.0,
        rotations={
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([-0.35, 0, 0]),
            JointType.HEAD: np.array([-0.25, 0, 0]),
            # Tail relaxed, level
            JointType.TAIL_BASE: np.array([0.1, 0, 0]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    # Sniff - small head movements
    motion.add_keyframe(Keyframe(
        time=0.15,
        rotations={
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([-0.35, 0.05, 0]),
            JointType.HEAD: np.array([-0.28, 0.08, 0]),
            JointType.TAIL_BASE: np.array([0.12, 0, 0.05]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    motion.add_keyframe(Keyframe(
        time=0.3,
        rotations={
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([-0.35, -0.05, 0]),
            JointType.HEAD: np.array([-0.25, -0.08, 0]),
            JointType.TAIL_BASE: np.array([0.1, 0, -0.05]),
        },
        easing=EaseType.EASE_IN_OUT
    ))

    motion.add_keyframe(Keyframe(
        time=0.45,
        rotations={
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            JointType.NECK: np.array([-0.35, 0, 0]),
            JointType.HEAD: np.array([-0.25, 0, 0]),
            JointType.TAIL_BASE: np.array([0.1, 0, 0]),
        }
    ))

    return motion


# Motion library - all available dog motions
DOG_MOTIONS: Dict[str, MotionSequence] = {}


def _initialize_dog_motions():
    """Initialize the dog motion library."""
    global DOG_MOTIONS
    DOG_MOTIONS = {
        'grab_shake': create_grab_shake(),
        'bark': create_bark(),
        'bark_sequence': create_bark_sequence(),
        'dig': create_dig(),
        'head_tilt': create_head_tilt(),
        'tail_wag': create_tail_wag(),
        'sniff_ground': create_sniff_ground(),
    }


def get_dog_motion(name: str) -> Optional[MotionSequence]:
    """Get a dog motion by name."""
    if not DOG_MOTIONS:
        _initialize_dog_motions()
    return DOG_MOTIONS.get(name)


def get_available_dog_motions() -> List[str]:
    """Get list of available dog motion names."""
    if not DOG_MOTIONS:
        _initialize_dog_motions()
    return list(DOG_MOTIONS.keys())


# Initialize on import
_initialize_dog_motions()
