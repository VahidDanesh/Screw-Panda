import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d
from .base import ManipulationPrimitive

class Pivoting(ManipulationPrimitive):
    """
    A manipulation primitive for pivoting objects around a fixed point or line.
    
    This primitive generates a trajectory that rotates an object around a
    fixed point or line of contact with a supporting surface.
    """
    
    def __init__(self, obj, goal_pose, pivot_edge_idx=0, pivot_param=0.5,
                 duration=2.0, frequency=100):
        """
        Initialize a pivoting primitive.
        
        Args:
            obj (SpatialObject): The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            pivot_edge_idx (int): Index of the edge to pivot around.
            pivot_param (float): Parameter along the edge for the pivot point.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        self.pivot_edge_idx = pivot_edge_idx
        self.pivot_param = pivot_param
        
        # Calculate the pivot point
        self._calculate_pivot_point()
    
    def _calculate_pivot_point(self):
        """
        Calculate the pivot point in world coordinates.
        
        For a box, the pivot point is a point on one of its edges, determined by
        the edge index and a parameter along that edge.
        """
        # Get the edge in world coordinates
        start_point, end_point = self.object.get_edge_in_world(self.pivot_edge_idx)
        
        # Calculate the pivot point by interpolating along the edge
        self.pivot_point = start_point + self.pivot_param * (end_point - start_point)
        
        # Calculate the pivot direction (assumed to be vertical)
        self.pivot_direction = np.array([0, 0, 1])
    
    def plan(self):
        """
        Plan a trajectory for pivoting the object.
        
        The pivoting motion ensures that:
        1. The pivot point remains fixed in space
        2. The object rotates around the pivot point along a specified axis
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Get time scaling
        tau, _ = self._set_time_scaling(method="quintic")
        
        # Extract the transformation from the pivot point to the object center
        # This will be preserved throughout the pivot
        pivot_to_center = self.start_pose.inv() * SE3(self.pivot_point)
        
        # Calculate the total rotation angle needed
        # Here we're assuming rotation around the vertical (Z) axis
        start_rot = self.start_pose.R
        goal_rot = self.goal_pose.R
        
        # Generate object poses
        self._object_poses = []
        self._object_dquats = []
        
        for t in tau:
            # Interpolate rotation using quaternion slerp
            if np.allclose(start_rot, goal_rot):
                rot = start_rot
            else:
                dq_rot = pt3d.quaternion_slerp(
                    pt3d.quaternion_from_matrix(start_rot),
                    pt3d.quaternion_from_matrix(goal_rot),
                    t
                )
                rot = pt3d.matrix_from_quaternion(dq_rot)
            
            # Calculate the new object pose
            # First create a pose at the pivot point with the interpolated rotation
            pivot_pose = SE3(self.pivot_point) * SE3(R=rot)
            
            # Then apply the preserved pivot-to-center transformation
            obj_pose = pivot_pose * pivot_to_center
            
            self._object_poses.append(obj_pose)
            
            # Convert to dual quaternion
            dq = pt3d.dual_quaternion_from_transformation_matrix(obj_pose.A)
            self._object_dquats.append(dq)
        
        # Calculate the end-effector poses
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