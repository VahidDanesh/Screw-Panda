import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d
from .base import ManipulationPrimitive

class Grasping(ManipulationPrimitive):
    """
    A manipulation primitive for simple grasping (pick and place) operations.
    
    This primitive generates a trajectory that moves an object from its current
    pose to a goal pose without any specific contact constraints.
    """
    
    def __init__(self, obj, goal_pose, approach_distance=0.1, 
                 duration=2.0, frequency=100):
        """
        Initialize a grasping primitive.
        
        Args:
            obj (SpatialObject): The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            approach_distance (float): Distance for the pre-grasp approach.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        self.approach_distance = approach_distance
    
    def plan(self):
        """
        Plan a trajectory for grasping the object.
        
        The grasping motion consists of:
        1. Pre-grasp approach to the grasp pose
        2. Grasp the object
        3. Move the object to the goal pose
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Get time scaling
        tau, _ = self._set_time_scaling(method="quintic")
        
        # Calculate grasp pose
        grasp_pose = self.object.grasp_pose
        
        # Plan object trajectory directly from start to goal
        self._object_dquats = self._create_dquat_trajectory(
            self.start_pose, self.goal_pose, tau
        )
        
        # Convert dual quaternions to SE3 for object poses
        self._object_poses = [self._dquat_to_se3(dq) for dq in self._object_dquats]
        
        # Calculate the end-effector poses
        # The end-effector follows the object's grasp point
        self._ee_poses = []
        self._ee_dquats = []
        
        for obj_pose in self._object_poses:
            # Calculate the grasp pose at each step
            ee_pose = obj_pose * self.object.grasp_offset
            self._ee_poses.append(ee_pose)
            
            # Convert to dual quaternion for the end-effector
            ee_dquat = pt3d.dual_quaternion_from_transformation_matrix(ee_pose.A)
            self._ee_dquats.append(ee_dquat)
        
        return self._object_poses, self._ee_poses 