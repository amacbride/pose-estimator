"""
Animal Models - Parametric 3D wireframe models for quadruped animals.

This package provides parametric skeleton models for cats, dogs, and other
quadruped animals, designed for animation and pose estimation applications.
"""

from .skeleton import Joint, QuadrupedSkeleton
from .cat import CatSkeleton
from .visualizer import SkeletonVisualizer

__all__ = ['Joint', 'QuadrupedSkeleton', 'CatSkeleton', 'SkeletonVisualizer']
