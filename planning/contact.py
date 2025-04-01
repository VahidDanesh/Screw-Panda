import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d

from utils.objects import SpatialObject, Box, Cylinder
from .primitives import (
    Grasping, Sliding, Pivoting, StraightRolling, CurvedRolling
)

class ContactPlanner:
    """
    A planner for contact-based manipulation tasks.
    
    This class provides a convenient interface for planning and executing
    various contact-based manipulations with different objects.
    """
    
    def __init__(self, obj=None):
        """
        Initialize a contact planner.
        
        Args:
            obj (SpatialObject, optional): The object to manipulate.
        """
        self.object = obj
        self.planned_trajectory = None
        self.primitives = []
    
    def set_object(self, obj):
        """
        Set the object to manipulate.
        
        Args:
            obj (SpatialObject): The object to manipulate.
        """
        if not isinstance(obj, SpatialObject):
            raise TypeError("Object must be an instance of SpatialObject")
        
        self.object = obj
        return self
    
    def grasp(self, goal_pose, duration=2.0, frequency=100):
        """
        Plan a grasping manipulation.
        
        Args:
            goal_pose (SE3): Goal pose for the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
            
        Returns:
            Grasping: The grasping primitive.
        """
        if self.object is None:
            raise ValueError("No object set for manipulation")
        
        primitive = Grasping(self.object, goal_pose, duration=duration, frequency=frequency)
        self.primitives.append(primitive)
        return primitive
    
    def slide(self, goal_pose, contact_face_idx=0, duration=2.0, frequency=100):
        """
        Plan a sliding manipulation.
        
        Args:
            goal_pose (SE3): Goal pose for the object.
            contact_face_idx (int): Index of the face in contact with the surface.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
            
        Returns:
            Sliding: The sliding primitive.
        """
        if self.object is None:
            raise ValueError("No object set for manipulation")
        
        primitive = Sliding(self.object, goal_pose, contact_face_idx, duration, frequency)
        self.primitives.append(primitive)
        return primitive
    
    def pivot(self, goal_pose, pivot_edge_idx=0, pivot_param=0.5, duration=2.0, frequency=100):
        """
        Plan a pivoting manipulation.
        
        Args:
            goal_pose (SE3): Goal pose for the object.
            pivot_edge_idx (int): Index of the edge to pivot around.
            pivot_param (float): Parameter along the edge for the pivot point.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
            
        Returns:
            Pivoting: The pivoting primitive.
        """
        if self.object is None:
            raise ValueError("No object set for manipulation")
        
        primitive = Pivoting(self.object, goal_pose, pivot_edge_idx, pivot_param, duration, frequency)
        self.primitives.append(primitive)
        return primitive
    
    def roll_straight(self, goal_pose, duration=2.0, frequency=100):
        """
        Plan a straight rolling manipulation.
        
        Args:
            goal_pose (SE3): Goal pose for the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
            
        Returns:
            StraightRolling: The straight rolling primitive.
        """
        if self.object is None:
            raise ValueError("No object set for manipulation")
        
        if not hasattr(self.object, 'radius'):
            raise TypeError("Object must be a cylinder for rolling manipulation")
        
        primitive = StraightRolling(self.object, goal_pose, duration, frequency)
        self.primitives.append(primitive)
        return primitive
    
    def roll_curved(self, goal_pose, path_points=None, duration=2.0, frequency=100):
        """
        Plan a curved rolling manipulation.
        
        Args:
            goal_pose (SE3): Goal pose for the object.
            path_points (list, optional): List of waypoints defining the curved path.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
            
        Returns:
            CurvedRolling: The curved rolling primitive.
        """
        if self.object is None:
            raise ValueError("No object set for manipulation")
        
        if not hasattr(self.object, 'radius'):
            raise TypeError("Object must be a cylinder for rolling manipulation")
        
        primitive = CurvedRolling(self.object, goal_pose, path_points, duration, frequency)
        self.primitives.append(primitive)
        return primitive
    
    def execute_all(self):
        """
        Execute all planned primitives in sequence.
        
        Returns:
            tuple: (object_poses, ee_poses) Combined lists of object and end-effector poses.
        """
        if not self.primitives:
            raise ValueError("No primitives planned")
        
        all_object_poses = []
        all_ee_poses = []
        
        for primitive in self.primitives:
            obj_poses, ee_poses = primitive.execute()
            all_object_poses.extend(obj_poses)
            all_ee_poses.extend(ee_poses)
        
        self.planned_trajectory = (all_object_poses, all_ee_poses)
        return self.planned_trajectory
    
    def get_robot_trajectory(self):
        """
        Get the robot end-effector trajectory for the planned manipulation.
        
        Returns:
            list: List of SE3 poses for the robot end-effector.
        """
        if self.planned_trajectory is None:
            raise ValueError("No trajectory has been planned. Call execute_all() first.")
            
        return self.planned_trajectory[1] 