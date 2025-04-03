import numpy as np
from spatialmath import SE3, UnitDualQuaternion
import pytransform3d.transformations as pt3d
from utils.objects import SpatialObject
from utils.screw_utils import sclerp, dq_to_se3, se3_to_dq
from abc import ABC, abstractmethod
from roboticstoolbox import quintic

class ManipulationPrimitive(ABC):
    """
    Base class for manipulation primitives that generate motion plans for objects.
    
    A manipulation primitive represents a specific way to manipulate an object
    (e.g., grasping, sliding, pivoting, rolling) and provides methods to
    generate trajectories for both the object and the robot end-effector.
    
    Attributes:
        object (SpatialObject): The object to manipulate.
        start_pose (SE3): Starting pose of the object.
        goal_pose (SE3): Goal pose of the object.
        duration (float): Duration of the motion in seconds.
        frequency (int): Sampling frequency for the trajectory.
    """
    
    def __init__(self, obj, goal_pose, duration=2.0, frequency=100):
        """
        Initialize a manipulation primitive.
        
        Args:
            obj (SpatialObject): The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        if not isinstance(obj, SpatialObject):
            raise TypeError("Object must be an instance of SpatialObject")
        
        self.object = obj
        self.start_pose = obj.pose  # Current pose of the object
        self.goal_pose = goal_pose if isinstance(goal_pose, SE3) else SE3(goal_pose)
        self.duration = duration
        self.frequency = frequency
        
        # Parameters used for path planning
        self.steps = int(self.duration * self.frequency)
        self.tvec = np.linspace(0, self.duration, self.steps)
        
        # Initialize result containers
        self._object_poses = []
        self._ee_poses = []
        self._object_dquats = []
        self._ee_dquats = []
    
    @property
    def object_poses(self):
        """Get the planned object poses."""
        return self._object_poses
    
    @property
    def ee_poses(self):
        """Get the planned end-effector poses."""
        return self._ee_poses
    
    @property
    def object_dquats(self):
        """Get the planned object poses as dual quaternions."""
        return self._object_dquats
    
    @property
    def ee_dquats(self):
        """Get the planned end-effector poses as dual quaternions."""
        return self._ee_dquats
    
    def _set_time_scaling(self, method="quintic"):
        """
        Set the time scaling for interpolation.
        
        Args:
            method (str): Interpolation method ('linear', 'cubic', 'quintic').
            
        Returns:
            tuple: (tau, taud) time scaling and its derivative.
        """
        if method.lower() == "linear":
            tau = np.linspace(0, 1, self.steps)
            taud = np.ones(self.steps) / self.duration
        elif method.lower() == "cubic":
            # Implementation would use cubic time scaling
            raise NotImplementedError("Cubic time scaling not yet implemented")
        elif method.lower() == "quintic":
            tg = quintic(0, 1, self.tvec)
            tau = tg.q
            taud = tg.qd
        else:
            raise ValueError("Method must be 'linear', 'cubic', or 'quintic'")
        
        return tau, taud
    
    def _create_dquat_trajectory(self, start_pose, end_pose, tau):
        """
        Create a dual quaternion trajectory between two poses using pytransform3d.
        
        Args:
            start_pose (SE3): Starting pose.
            end_pose (SE3): Ending pose.
            tau (np.ndarray): Time scaling vector (0 to 1).
            
        Returns:
            list: List of dual quaternions along the trajectory.
        """
        # Convert SE3 to transformation matrices for pytransform3d
        T1 = start_pose.A
        T2 = end_pose.A
        
        # Generate trajectory using screw interpolation
        dquats = []
        for t in tau:
            # Use pytransform3d's screw interpolation directly
            T_interp = pt3d.transform_sclerp(T1, T2, t)
            dq = pt3d.dual_quaternion_from_transform(T_interp)
            dquats.append(dq)
        
        return dquats
    
    def _dquat_to_se3(self, dquat):
        """
        Convert a dual quaternion to an SE3 pose.
        
        Args:
            dquat: Dual quaternion from pytransform3d.
            
        Returns:
            SE3: The equivalent SE3 pose.
        """
        T = pt3d.transform_from_dual_quaternion(dquat)
        return SE3(T)
    
    @abstractmethod
    def plan(self):
        """
        Plan a trajectory for the manipulation primitive.
        
        This method should be implemented by each specific primitive subclass.
        It should populate self._object_poses and self._ee_poses with the
        planned trajectory for both the object and the end-effector.
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        pass
    
    def execute(self, update_object=True):
        """
        Execute the manipulation primitive by generating a trajectory.
        
        Args:
            update_object (bool): Whether to update the object's pose to the
                                  final pose after planning.
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        object_poses, ee_poses = self.plan()
        
        if update_object and object_poses:
            # Update the object's pose to the final pose
            self.object.update_pose(object_poses[-1])
        
        return object_poses, ee_poses 