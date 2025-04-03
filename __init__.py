"""
Screw-Panda: A framework for robotic manipulation planning using dual quaternions
and screw theory for primitive motions.

This framework provides tools for representing object poses, planning
manipulations, and controlling robot manipulator arms efficiently.
"""

# Core submodules
from . import utils
from . import planning
from . import models

# Re-export key classes and functions from submodules
from .utils import (
    SpatialObject,
    ManipulableBox, 
    ManipulableCylinder,
    vec_angle,
    DualQuaternionUtils,
    sclerp,
    generate_trajectory,
    pose_to_screw_parameters,
    dq_to_se3,
    se3_to_dq,
    twist_from_poses
)

from .planning import (
    # Contact planning
    ContactPlanner,
    
    # Motion primitives
    ManipulationPrimitive,
    Grasping,
    Sliding,
    Pivoting,
    StraightRolling,
    CurvedRolling
)

# Version information
__version__ = '0.1.0'

__all__ = [
    # Classes from utils
    'SpatialObject',
    'ManipulableBox', 
    'ManipulableCylinder',
    'DualQuaternionUtils',
    
    # Functions from utils
    'vec_angle',
    'sclerp',
    'generate_trajectory',
    'pose_to_screw_parameters',
    'dq_to_se3',
    'se3_to_dq',
    'twist_from_poses',
    
    # Classes from planning
    'ContactPlanner',
    'ManipulationPrimitive',
    'Grasping',
    'Sliding',
    'Pivoting',
    'StraightRolling',
    'CurvedRolling',
] 