"""
Sliding primitive for maintaining face contact during manipulation.
"""

import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d_trans
from .base import ManipulationPrimitive

class Sliding(ManipulationPrimitive):
    """
    A manipulation primitive for sliding objects while maintaining face contact.
    
    This primitive generates a trajectory that slides an object along a supporting
    surface while keeping a specified face in contact with the surface.
    """
    
    def __init__(self, obj, goal_pose, contact_face_idx=0, 
                 duration=2.0, frequency=100):
        """
        Initialize a sliding primitive.
        
        Args:
            obj: The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            contact_face_idx (int): Index of the face in contact with the surface.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        self.contact_face_idx = contact_face_idx
        
    def _validate_poses(self):
        """
        Validate that the start and goal poses maintain the contact face constraint.
        
        For sliding, the contact face normal should be perpendicular to the supporting
        surface (assumed to be the XY plane with Z pointing up).
        """
        # Placeholder - will be implemented later
        pass
    
    def _maintain_contact_constraint(self, pose):
        """
        Ensure object maintains contact with the supporting surface.
        
        Args:
            pose (SE3): The pose to constrain.
            
        Returns:
            SE3: The constrained pose that maintains surface contact.
        """
        # Placeholder - will be implemented later
        pass
    
    def plan(self):
        """
        Plan a trajectory for sliding the object.
        
        The sliding motion ensures that:
        1. The contact face remains in contact with the supporting surface
        2. The object's orientation relative to the supporting surface is maintained
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Placeholder - will be implemented later
        pass 