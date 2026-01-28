"""
Dog-specific skeleton model with realistic proportions and predefined poses.

This module extends the base QuadrupedSkeleton to create a parametric
dog model suitable for animation. Default proportions are based on a
medium-sized dog (like a Labrador or German Shepherd).
"""

import numpy as np
from typing import Dict
from .skeleton import QuadrupedSkeleton, ShapeParameters, JointType


class DogShapeParameters(ShapeParameters):
    """
    Shape parameters tuned for a domestic dog (medium breed).

    Based on average medium dog proportions:
    - Body length (shoulder to hip): ~50cm
    - Standing height at shoulder: ~55cm
    - Tail length: ~30cm

    All values are normalized with body_length = 1.0 as reference.
    Can be adjusted for different breeds via the breed presets.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()

        # Core body - dogs are generally stockier than cats
        self.body_length = 1.0
        self.body_width = 0.32  # Wider chest than cats
        self.body_height = 0.25  # Deeper chest

        # Spine - less flexible than cats
        self.spine_segments = 3

        # Head and neck - dogs have longer snouts
        self.neck_length = 0.28
        self.head_length = 0.25
        self.head_width = 0.18
        self.ear_length = 0.12  # Varies greatly by breed
        self.snout_length = 0.18  # Longer than cats

        # Tail - varies by breed, moderate default
        self.tail_length = 0.5
        self.tail_segments = 3

        # Front legs - longer than cats, more upright stance
        self.front_shoulder_width = 0.24
        self.front_upper_leg_length = 0.28
        self.front_lower_leg_length = 0.24
        self.front_paw_length = 0.08

        # Back legs - powerful, longer than front
        self.back_hip_width = 0.22
        self.back_upper_leg_length = 0.30
        self.back_lower_leg_length = 0.26
        self.back_paw_length = 0.10

        self.scale = scale


# Breed presets for different dog types
BREED_PRESETS = {
    'medium': DogShapeParameters,  # Default (Lab/Shepherd size)

    'small': lambda scale=1.0: _create_small_dog_params(scale),
    'large': lambda scale=1.0: _create_large_dog_params(scale),
    'long': lambda scale=1.0: _create_long_dog_params(scale),  # Dachshund-like
    'slim': lambda scale=1.0: _create_slim_dog_params(scale),  # Greyhound-like
}


def _create_small_dog_params(scale: float = 1.0) -> DogShapeParameters:
    """Create parameters for a small dog (Terrier/Beagle size)."""
    params = DogShapeParameters(scale)
    params.body_length = 0.8
    params.body_width = 0.25
    params.body_height = 0.2
    params.neck_length = 0.2
    params.head_length = 0.22  # Proportionally larger head
    params.snout_length = 0.12
    params.tail_length = 0.35
    params.front_upper_leg_length = 0.2
    params.front_lower_leg_length = 0.18
    params.back_upper_leg_length = 0.22
    params.back_lower_leg_length = 0.2
    return params


def _create_large_dog_params(scale: float = 1.0) -> DogShapeParameters:
    """Create parameters for a large dog (Great Dane/Mastiff size)."""
    params = DogShapeParameters(scale)
    params.body_length = 1.2
    params.body_width = 0.4
    params.body_height = 0.35
    params.neck_length = 0.35
    params.head_length = 0.3
    params.head_width = 0.22
    params.snout_length = 0.22
    params.tail_length = 0.6
    params.front_upper_leg_length = 0.38
    params.front_lower_leg_length = 0.32
    params.back_upper_leg_length = 0.4
    params.back_lower_leg_length = 0.35
    return params


def _create_long_dog_params(scale: float = 1.0) -> DogShapeParameters:
    """Create parameters for a long-bodied dog (Dachshund/Corgi)."""
    params = DogShapeParameters(scale)
    params.body_length = 1.3  # Longer body
    params.body_width = 0.28
    params.body_height = 0.22
    params.neck_length = 0.2
    params.head_length = 0.22
    params.snout_length = 0.15
    params.tail_length = 0.4
    # Short legs
    params.front_upper_leg_length = 0.15
    params.front_lower_leg_length = 0.12
    params.back_upper_leg_length = 0.16
    params.back_lower_leg_length = 0.14
    return params


def _create_slim_dog_params(scale: float = 1.0) -> DogShapeParameters:
    """Create parameters for a slim/athletic dog (Greyhound/Whippet)."""
    params = DogShapeParameters(scale)
    params.body_length = 1.1
    params.body_width = 0.22  # Narrow chest
    params.body_height = 0.2  # Deep but narrow
    params.neck_length = 0.35  # Long elegant neck
    params.head_length = 0.28
    params.head_width = 0.12  # Narrow head
    params.snout_length = 0.2  # Long snout
    params.tail_length = 0.55  # Long thin tail
    # Long legs
    params.front_upper_leg_length = 0.35
    params.front_lower_leg_length = 0.3
    params.back_upper_leg_length = 0.38
    params.back_lower_leg_length = 0.32
    return params


class DogSkeleton(QuadrupedSkeleton):
    """
    A parametric dog skeleton model.

    Provides dog-specific proportions and a library of predefined poses
    for common dog positions and behaviors.
    """

    def __init__(self, scale: float = 1.0, breed: str = 'medium'):
        """
        Create a new dog skeleton.

        Args:
            scale: Overall scale factor (1.0 = normalized size)
            breed: Breed preset ('medium', 'small', 'large', 'long', 'slim')
        """
        if breed in BREED_PRESETS:
            if callable(BREED_PRESETS[breed]):
                if breed == 'medium':
                    shape = DogShapeParameters(scale=scale)
                else:
                    shape = BREED_PRESETS[breed](scale=scale)
            else:
                shape = BREED_PRESETS[breed](scale=scale)
        else:
            shape = DogShapeParameters(scale=scale)

        super().__init__(shape)
        self.breed = breed

        # Store predefined poses
        self._poses: Dict[str, Dict[JointType, np.ndarray]] = self._define_poses()

    def _define_poses(self) -> Dict[str, Dict[JointType, np.ndarray]]:
        """Define a library of dog poses."""
        poses = {}

        # Standing pose (neutral, alert)
        poses['standing'] = {
            # Head up, alert
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail up and slightly curved (happy)
            JointType.TAIL_BASE: np.array([0.4, 0, 0]),
            JointType.TAIL_MID: np.array([0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0]),
            # Ears forward (alert)
            JointType.LEFT_EAR: np.array([0.1, 0.1, 0]),
            JointType.RIGHT_EAR: np.array([0.1, -0.1, 0]),
        }

        # Sitting pose - weight on haunches, front legs straight
        poses['sitting'] = {
            # Spine angles down toward rear
            JointType.SPINE_LOWER: np.array([-0.35, 0, 0]),
            JointType.SPINE_MID: np.array([-0.15, 0, 0]),
            # Head up
            JointType.NECK: np.array([0.45, 0, 0]),
            JointType.HEAD: np.array([0.2, 0, 0]),
            # Tail down and relaxed
            JointType.TAIL_BASE: np.array([-0.3, 0, 0]),
            JointType.TAIL_MID: np.array([-0.1, 0, 0.2]),
            JointType.TAIL_TIP: np.array([0, 0, 0.1]),
            # Front legs straight
            JointType.LEFT_SHOULDER: np.array([0.35, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.35, 0, 0]),
            # Back legs folded
            JointType.LEFT_HIP: np.array([-0.9, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([1.6, 0, 0.15]),
            JointType.LEFT_BACK_LOWER: np.array([-1.9, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.9, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([1.6, 0, -0.15]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.9, 0, 0]),
        }

        # Lying down pose (relaxed, on belly)
        poses['lying'] = {
            # Spine low
            JointType.SPINE_LOWER: np.array([-0.15, 0, 0]),
            JointType.SPINE_MID: np.array([0, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.1, 0, 0]),
            # Head up but relaxed
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.25, 0, 0]),
            # Tail relaxed
            JointType.TAIL_BASE: np.array([-0.3, 0, 0.15]),
            JointType.TAIL_MID: np.array([0, 0, 0.1]),
            # Front legs extended forward
            JointType.LEFT_SHOULDER: np.array([0.5, 0, 0.15]),
            JointType.LEFT_FRONT_UPPER: np.array([0.7, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.8, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.5, 0, -0.15]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.7, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.8, 0, 0]),
            # Back legs folded to side
            JointType.LEFT_HIP: np.array([-0.5, 0, 0.4]),
            JointType.LEFT_BACK_UPPER: np.array([1.1, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.5, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.5, 0, -0.4]),
            JointType.RIGHT_BACK_UPPER: np.array([1.1, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.5, 0, 0]),
        }

        # Sleeping pose (curled or stretched)
        poses['sleeping'] = {
            # Spine slightly curved
            JointType.SPINE_LOWER: np.array([-0.1, 0, 0.2]),
            JointType.SPINE_MID: np.array([0, 0, 0.15]),
            # Head down, resting
            JointType.NECK: np.array([0.1, 0, 0.2]),
            JointType.HEAD: np.array([0.3, 0, 0.3]),
            # Tail curled
            JointType.TAIL_BASE: np.array([-0.4, 0, 0.5]),
            JointType.TAIL_MID: np.array([0, 0, 0.6]),
            JointType.TAIL_TIP: np.array([0.2, 0, 0.4]),
            # Legs tucked
            JointType.LEFT_SHOULDER: np.array([0.5, 0, 0.2]),
            JointType.LEFT_FRONT_UPPER: np.array([0.9, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-1.1, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.5, 0, -0.2]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.9, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.1, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.4, 0, 0.35]),
            JointType.LEFT_BACK_UPPER: np.array([1.3, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-1.7, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.4, 0, -0.35]),
            JointType.RIGHT_BACK_UPPER: np.array([1.3, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-1.7, 0, 0]),
        }

        # Play bow - classic invitation to play
        poses['play_bow'] = {
            # Front low, rear high
            JointType.SPINE_LOWER: np.array([0.4, 0, 0]),
            JointType.SPINE_MID: np.array([0.25, 0, 0]),
            JointType.SPINE_UPPER: np.array([-0.2, 0, 0]),
            # Head low and forward, looking up
            JointType.NECK: np.array([-0.3, 0, 0]),
            JointType.HEAD: np.array([0.4, 0, 0]),  # Looking up
            # Tail high and wagging position
            JointType.TAIL_BASE: np.array([0.7, 0, 0]),
            JointType.TAIL_MID: np.array([0.3, 0.2, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0.1, 0]),
            # Front legs extended forward, elbows down
            JointType.LEFT_SHOULDER: np.array([0.6, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.4, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.6, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.4, 0, 0]),
            # Back legs straight, rear up
            JointType.LEFT_HIP: np.array([0.15, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.15, 0, 0]),
        }

        # Begging pose - sitting with front paws raised
        poses['begging'] = {
            # Spine very upright
            JointType.SPINE_LOWER: np.array([-0.5, 0, 0]),
            JointType.SPINE_MID: np.array([-0.3, 0, 0]),
            JointType.SPINE_UPPER: np.array([-0.1, 0, 0]),
            # Head up, looking at person
            JointType.NECK: np.array([0.3, 0, 0]),
            JointType.HEAD: np.array([0.2, 0, 0]),
            # Front legs raised
            JointType.LEFT_SHOULDER: np.array([0.8, 0, 0.2]),
            JointType.LEFT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.8, 0, -0.2]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.5, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-1.2, 0, 0]),
            # Back legs folded under (sitting)
            JointType.LEFT_HIP: np.array([-1.0, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([1.7, 0, 0.15]),
            JointType.LEFT_BACK_LOWER: np.array([-2.0, 0, 0]),
            JointType.RIGHT_HIP: np.array([-1.0, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([1.7, 0, -0.15]),
            JointType.RIGHT_BACK_LOWER: np.array([-2.0, 0, 0]),
            # Tail wagging position
            JointType.TAIL_BASE: np.array([-0.2, 0, 0.3]),
            JointType.TAIL_MID: np.array([0.1, 0.2, 0]),
        }

        # Walking pose - mid-stride
        poses['walking'] = {
            # Slight body motion
            JointType.SPINE_MID: np.array([0.05, 0.03, 0]),
            # Head forward, alert
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail up for balance
            JointType.TAIL_BASE: np.array([0.35, 0, 0]),
            JointType.TAIL_MID: np.array([0.15, 0.1, 0]),
            # Diagonal gait: left front + right back forward
            JointType.LEFT_SHOULDER: np.array([0.25, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([-0.15, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.2, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.25, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([-0.15, 0, 0]),
        }

        # Running/trotting pose
        poses['running'] = {
            # Body stretched forward
            JointType.SPINE_LOWER: np.array([0.1, 0, 0]),
            JointType.SPINE_MID: np.array([0.15, 0, 0]),
            # Head forward and down slightly
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([-0.1, 0, 0]),
            # Tail streaming behind
            JointType.TAIL_BASE: np.array([0.1, 0, 0]),
            JointType.TAIL_MID: np.array([-0.2, 0, 0]),
            JointType.TAIL_TIP: np.array([-0.1, 0, 0]),
            # Extended stride
            JointType.LEFT_SHOULDER: np.array([0.4, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.35, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([-0.35, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([-0.25, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.35, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.2, 0, 0]),
            JointType.RIGHT_HIP: np.array([0.4, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([-0.3, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([0.2, 0, 0]),
        }

        # Alert/watching pose
        poses['alert'] = {
            # Spine straight and tense
            JointType.SPINE_MID: np.array([0.08, 0, 0]),
            # Head very up, ears forward
            JointType.NECK: np.array([0.4, 0, 0]),
            JointType.HEAD: np.array([0.25, 0, 0]),
            JointType.LEFT_EAR: np.array([0.2, 0.15, 0]),
            JointType.RIGHT_EAR: np.array([0.2, -0.15, 0]),
            # Tail up and still
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.25, 0, 0]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0]),
            # Legs slightly tensed
            JointType.LEFT_FRONT_UPPER: np.array([0.05, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.05, 0, 0]),
        }

        # Sniffing pose - head down, investigating
        poses['sniffing'] = {
            # Spine level
            JointType.SPINE_MID: np.array([0.05, 0, 0]),
            # Head down to ground
            JointType.NECK: np.array([-0.4, 0, 0]),
            JointType.HEAD: np.array([-0.3, 0, 0]),
            # Tail level or slightly up
            JointType.TAIL_BASE: np.array([0.2, 0, 0]),
            JointType.TAIL_MID: np.array([0.1, 0, 0]),
            # Front legs slightly bent
            JointType.LEFT_SHOULDER: np.array([0.15, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.1, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.15, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.1, 0, 0]),
        }

        # Pointing pose (hunting breed stance)
        poses['pointing'] = {
            # Body tense and forward-leaning
            JointType.SPINE_MID: np.array([0.1, 0, 0]),
            JointType.SPINE_UPPER: np.array([0.05, 0, 0]),
            # Head extended forward, intense focus
            JointType.NECK: np.array([0.25, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail straight back (classic point)
            JointType.TAIL_BASE: np.array([0.05, 0, 0]),
            JointType.TAIL_MID: np.array([-0.05, 0, 0]),
            JointType.TAIL_TIP: np.array([-0.05, 0, 0]),
            # One front leg raised
            JointType.LEFT_SHOULDER: np.array([0.5, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.6, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.8, 0, 0]),
            # Other legs planted
            JointType.RIGHT_SHOULDER: np.array([0.1, 0, 0]),
        }

        # Tail wagging - happy position (tail high, mid-wag)
        poses['tail_wag_left'] = {
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail high and curved to left
            JointType.TAIL_BASE: np.array([0.5, 0, -0.3]),
            JointType.TAIL_MID: np.array([0.2, 0, -0.5]),
            JointType.TAIL_TIP: np.array([0.1, 0, -0.3]),
        }

        poses['tail_wag_right'] = {
            JointType.NECK: np.array([0.15, 0, 0]),
            JointType.HEAD: np.array([0.1, 0, 0]),
            # Tail high and curved to right
            JointType.TAIL_BASE: np.array([0.5, 0, 0.3]),
            JointType.TAIL_MID: np.array([0.2, 0, 0.5]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0.3]),
        }

        # Shake off pose (body twisted, about to shake)
        poses['shake'] = {
            # Body twisted
            JointType.SPINE_LOWER: np.array([0, 0.15, 0.2]),
            JointType.SPINE_MID: np.array([0, -0.1, -0.15]),
            JointType.SPINE_UPPER: np.array([0, 0.1, 0.1]),
            # Head turned
            JointType.NECK: np.array([0.1, 0.2, 0]),
            JointType.HEAD: np.array([0.15, 0.15, 0]),
            # Ears floppy
            JointType.LEFT_EAR: np.array([-0.2, 0.3, 0.2]),
            JointType.RIGHT_EAR: np.array([-0.2, -0.1, -0.2]),
            # Legs braced
            JointType.LEFT_FRONT_UPPER: np.array([0.1, 0, 0.1]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.1, 0, -0.1]),
        }

        # Rolling over (playful submission)
        poses['roll_over'] = {
            # Body twisted onto back
            JointType.ROOT: np.array([0, 0, 0]),  # Will use global rotation
            JointType.SPINE_LOWER: np.array([0.1, 0, 0.3]),
            JointType.SPINE_MID: np.array([0.05, 0, 0.2]),
            # Head to side
            JointType.NECK: np.array([0.2, 0.3, 0.4]),
            JointType.HEAD: np.array([0.3, 0.2, 0]),
            # Legs in air
            JointType.LEFT_SHOULDER: np.array([0.3, 0.2, 0.5]),
            JointType.LEFT_FRONT_UPPER: np.array([0.8, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.6, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.3, -0.2, -0.3]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.7, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.5, 0, 0]),
            JointType.LEFT_HIP: np.array([-0.2, 0.2, 0.5]),
            JointType.LEFT_BACK_UPPER: np.array([0.9, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-0.7, 0, 0]),
            JointType.RIGHT_HIP: np.array([-0.2, -0.2, -0.4]),
            JointType.RIGHT_BACK_UPPER: np.array([0.8, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-0.6, 0, 0]),
            # Tail relaxed
            JointType.TAIL_BASE: np.array([-0.2, 0.3, 0.4]),
            JointType.TAIL_MID: np.array([0, 0.2, 0.3]),
        }

        # Stretching pose (front stretch)
        poses['stretching'] = {
            # Front low, back normal
            JointType.SPINE_LOWER: np.array([0.25, 0, 0]),
            JointType.SPINE_MID: np.array([0.15, 0, 0]),
            JointType.SPINE_UPPER: np.array([-0.1, 0, 0]),
            # Head low
            JointType.NECK: np.array([-0.25, 0, 0]),
            JointType.HEAD: np.array([-0.1, 0, 0]),
            # Front legs extended
            JointType.LEFT_SHOULDER: np.array([0.55, 0, 0]),
            JointType.LEFT_FRONT_UPPER: np.array([0.45, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.25, 0, 0]),
            JointType.RIGHT_SHOULDER: np.array([0.55, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.45, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.25, 0, 0]),
            # Tail up
            JointType.TAIL_BASE: np.array([0.5, 0, 0]),
            JointType.TAIL_MID: np.array([0.25, 0, 0]),
        }

        # Panting/happy pose
        poses['happy'] = {
            # Relaxed stance
            JointType.SPINE_MID: np.array([0.03, 0, 0]),
            # Head up, mouth open (implied)
            JointType.NECK: np.array([0.2, 0, 0]),
            JointType.HEAD: np.array([0.15, 0, 0]),
            # Ears relaxed
            JointType.LEFT_EAR: np.array([-0.1, 0.1, 0.1]),
            JointType.RIGHT_EAR: np.array([-0.1, -0.1, -0.1]),
            # Tail wagging high
            JointType.TAIL_BASE: np.array([0.55, 0, 0.15]),
            JointType.TAIL_MID: np.array([0.25, 0, 0.2]),
            JointType.TAIL_TIP: np.array([0.1, 0, 0.1]),
        }

        # Scared/submissive pose
        poses['scared'] = {
            # Body low, crouching
            JointType.SPINE_LOWER: np.array([-0.2, 0, 0]),
            JointType.SPINE_MID: np.array([-0.1, 0, 0]),
            # Head low, avoiding eye contact
            JointType.NECK: np.array([0.1, 0, 0]),
            JointType.HEAD: np.array([-0.2, 0, 0]),
            # Ears back
            JointType.LEFT_EAR: np.array([-0.3, -0.2, 0.3]),
            JointType.RIGHT_EAR: np.array([-0.3, 0.2, -0.3]),
            # Tail tucked
            JointType.TAIL_BASE: np.array([-0.6, 0, 0]),
            JointType.TAIL_MID: np.array([-0.4, 0, 0.4]),
            JointType.TAIL_TIP: np.array([-0.2, 0, 0.3]),
            # Legs bent, ready to flee
            JointType.LEFT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.LEFT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.RIGHT_FRONT_UPPER: np.array([0.15, 0, 0]),
            JointType.RIGHT_FRONT_LOWER: np.array([-0.2, 0, 0]),
            JointType.LEFT_BACK_UPPER: np.array([0.2, 0, 0]),
            JointType.LEFT_BACK_LOWER: np.array([-0.25, 0, 0]),
            JointType.RIGHT_BACK_UPPER: np.array([0.2, 0, 0]),
            JointType.RIGHT_BACK_LOWER: np.array([-0.25, 0, 0]),
        }

        return poses

    def set_pose(self, pose_name: str) -> bool:
        """
        Set the skeleton to a predefined pose.

        Args:
            pose_name: Name of the pose

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

            # Linear interpolation
            interpolated = rot_a * (1 - t) + rot_b * t
            self.set_joint_rotation(joint_type, interpolated)

        self.update_world_positions()
        return True

    @staticmethod
    def get_available_breeds() -> list:
        """Get list of available breed presets."""
        return list(BREED_PRESETS.keys())
