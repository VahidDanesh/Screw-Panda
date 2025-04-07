"""
Base class for manipulation primitives.
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from utils.objects import MBox, MCylinder
from pytransform3d import (
    batch_rotations as pbr3d,
    transformations as pt3d,
    trajectories as ptr3d,
)
import modern_robotics as mr

class ManipulationPrimitive:
    """
    Base class for manipulation primitives that generate motion plans for objects.
    
    A manipulation primitive represents a specific way to manipulate an object
    (e.g., grasping, sliding, pivoting, rolling) and provides methods to
    generate trajectories for both the object and the robot end-effector.
    
    Attributes:
        object: The object being manipulated (MBox or MCylinder)
        object_type (str): Type of object ('box', 'cylinder', or 'unknown')
        start_pose (SE3): Initial pose of the object
        goal_pose (SE3): Target pose for the object
        duration (float): Duration of the motion in seconds
        frequency (int): Sampling frequency for the trajectory in Hz
        steps (int): Number of steps in the trajectory
        tvec (np.ndarray): Time vector for the trajectory
        tau (np.ndarray): Time scaling vector (0 to 1)
        object_poses (list): List of object poses along the trajectory
        ee_poses (list): List of end-effector poses along the trajectory
        object_dqs (list): List of object dual quaternions along the trajectory
        ee_dqs (list): List of end-effector dual quaternions along the trajectory
    """
    
    def __init__(self, obj: MBox | MCylinder, goal_pose: SE3 | np.ndarray, duration: float = 2.0, frequency: int = 1000):
        """
        Initialize a manipulation primitive.
        
        Args:
            obj: The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        self.object = obj
        self.object_type = self._determine_object_type(obj)
        self.start_pose = obj.T  # Current pose of the object
        self.goal_pose = goal_pose if isinstance(goal_pose, SE3) else SE3(goal_pose)
        self.duration = duration
        self.frequency = frequency
        
        # Parameters used for path planning
        self.steps = int(self.duration * self.frequency)
        self.tvec = np.linspace(0, self.duration, self.steps)
        self._set_time_scaling()
        
        # Initialize result containers
        self.object_poses = []
        self.ee_poses = []
        self.object_dqs = []
        self.ee_dqs = []
    
    def _determine_object_type(self, obj: MBox | MCylinder):
        """
        Determine object type from name attribute.
        
        Args:
            obj: The object to check.
            
        Returns:
            str: Object type ('box', 'cylinder', or 'unknown').
        """
        name = obj.name.lower()
        if "box" in name:
            return "box"
        elif "cylinder" in name or "cyl" in name:
            return "cylinder"
        else:
            return "unknown"
    
    def _set_time_scaling(self, method: str = "quintic"):
        """
        Set the time scaling for interpolation.
        
        Args:
            method (str): Interpolation method ('linear', 'cubic', 'quintic').
            
        Returns:
            self.tau: time scaling vector (0 to 1).
        """
        tau = np.linspace(0, 1, self.steps)
        if method == "linear":
            self.tau = tau
        elif method == "cubic":
            # TODO: Implement cubic time scaling
            raise NotImplementedError("Cubic time scaling not implemented")
        elif method == "quintic":
            self.tau = rtb.quintic(q0=0, qf=1, t=tau).q
    
    def _create_trajectory(self, start_dq, end_dq):
        """
        Create a trajectory between two dual quaternions using ScLERP.
        
        Args:
            start_dq (np.ndarray): Starting dual quaternion or SE3 pose.
            end_dq (np.ndarray): Ending dual quaternion or SE3 pose.
            
        Returns:
            np.ndarray: Array of dual quaternions along the trajectory.
        """
        # Convert poses to dual quaternions if needed
        if isinstance(start_dq, SE3) or start_dq.shape == (4, 4):
            start_dq = ptr3d.dual_quaternions_from_transforms(start_dq.A)
            start_dq = pt3d.check_dual_quaternion(start_dq)
        if isinstance(end_dq, SE3) or end_dq.shape == (4, 4):
            end_dq = ptr3d.dual_quaternions_from_transforms(end_dq.A)
            end_dq = pt3d.check_dual_quaternion(end_dq)
        # Create trajectory using ScLERP
        dq_traj = [ptr3d.dual_quaternions_sclerp(start_dq, end_dq, t) for t in self.tau]
            
        return dq_traj
    
    def plan(self):
        """
        Plan a trajectory for the manipulation primitive.
        
        This method should be implemented by each specific primitive subclass.
        It should populate self.object_poses and self.ee_poses with the
        planned trajectory for both the object and the end-effector.
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def execute(self, update_object=True):
        """
        Execute the manipulation primitive by generating a trajectory.
        
        Args:
            update_object (bool): Whether to update the object's pose to the
                                  final pose after planning.
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Plan the trajectory
        self.object_poses, self.ee_poses = self.plan()
        
        # Convert poses to dual quaternions for smooth interpolation
        self.object_dqs = [ptr3d.dual_quaternion_from_transform(pose.A) for pose in self.object_poses]
        self.ee_dqs = [ptr3d.dual_quaternion_from_transform(pose.A) for pose in self.ee_poses]
        
        # Update object pose if requested
        if update_object:
            self.object.T = self.object_poses[-1].A
            
        return self.object_poses, self.ee_poses 