"""
Animal Models - Parametric 3D wireframe models for quadruped animals.

This package provides parametric skeleton models for cats, dogs, and other
quadruped animals, designed for animation and pose estimation applications.
"""

from .skeleton import Joint, QuadrupedSkeleton
from .cat import CatSkeleton
from .dog import DogSkeleton
from .visualizer import SkeletonVisualizer
from .motion import MotionSequence, Keyframe, MotionPlayer, LoopMode, EaseType
from .cat_motions import get_cat_motion, get_available_cat_motions, CAT_MOTIONS
from .dog_motions import get_dog_motion, get_available_dog_motions, DOG_MOTIONS

__all__ = [
    'Joint', 'QuadrupedSkeleton', 'CatSkeleton', 'DogSkeleton', 'SkeletonVisualizer',
    'MotionSequence', 'Keyframe', 'MotionPlayer', 'LoopMode', 'EaseType',
    'get_cat_motion', 'get_available_cat_motions', 'CAT_MOTIONS',
    'get_dog_motion', 'get_available_dog_motions', 'DOG_MOTIONS',
]
