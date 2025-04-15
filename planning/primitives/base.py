"""
Base class for manipulation primitives.
"""

import numpy as np
import roboticstoolbox as rtb
from spatialgeometry import Cuboid, Cylinder
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
    
    def __init__(self, 
                 obj: MBox | MCylinder | None = None, 
                 goal_pose: SE3 | np.ndarray | None = None,
                 start_pose: SE3 | np.ndarray | None = None, 
                 duration: float = 2.0, 
                 frequency: int = 1000,
                 **kwargs):
        """
        Initialize a manipulation primitive.
        
        Args:
            obj: The object to manipulate.
            goal_pose (SE3 | np.ndarray | None): Goal pose for the object.
            start_pose (SE3 | np.ndarray | None): Initial pose of the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        self._object = obj
        self._start_pose = start_pose
        self._goal_pose = goal_pose 
        self._duration = duration
        self._frequency = frequency
        self._time_scaling = kwargs.get("time_scaling", "quintic")
        
        # Parameters used for path planning
        self._update_trajectory_parameters()
        self._set_time_scaling(self._time_scaling)
        
        # Initialize result containers
        self.object_poses = []
        self.ee_poses = []
        self.object_dqs = []
        self.ee_dqs = []
        
        
    @property
    def object(self):
        """The object being manipulated."""
        return self._object
    @object.setter
    def object(self, obj):
        """Set a new object and update dependent parameters."""
        if not isinstance(obj, (MBox, MCylinder)):
            if isinstance(obj, Cuboid):
                self._object = MBox(obj)
            elif isinstance(obj, Cylinder):
                self._object = MCylinder(obj)
            else:
                raise ValueError("Object must be of type MBox or MCylinder")
        else:
            self._object = obj
            self._object_type = self._determine_object_type(self._object)
            self._start_pose = obj.T if self._start_pose is None else self._start_pose
            self._update_trajectory_parameters()
    
    @property
    def start_pose(self):
        """Current start pose for the manipulation."""
        return self._start_pose

    @start_pose.setter
    def start_pose(self, pose):
        """Set a new start pose and update dependent parameters."""
        self._start_pose = pose if isinstance(pose, SE3) else SE3(pose)

    @property
    def goal_pose(self):
        """Goal pose for the manipulation."""
        return self._goal_pose

    @goal_pose.setter
    def goal_pose(self, pose):
        """Set a new goal pose and update dependent parameters."""
        self._goal_pose = pose if isinstance(pose, SE3) else SE3(pose)

    @property
    def duration(self):
        """Duration of the motion in seconds."""
        return self._duration

    @duration.setter
    def duration(self, value):
        """Set a new duration and update dependent parameters."""
        if value <= 0:
            raise ValueError("Duration must be positive")
        self._duration = value
        self._update_trajectory_parameters()

    @property
    def frequency(self):
        """Sampling frequency for the trajectory in Hz."""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """Set a new frequency and update dependent parameters."""
        if value <= 0:
            raise ValueError("Frequency must be positive")
        self._frequency = value
        self._update_trajectory_parameters()

    
    @property
    def time_scaling(self):
        """Time scaling method for the trajectory."""
        return self._time_scaling
    
    @time_scaling.setter
    def time_scaling(self, method):
        """Set a new time scaling method and update dependent parameters."""
        if method not in ["linear", "cubic", "quintic"]:
            raise ValueError("Invalid time scaling method")
        self._time_scaling = method
        self._set_time_scaling(method)
        
        
    def _update_trajectory_parameters(self):
        """Update trajectory parameters when duration or frequency changes."""
        self.steps = int(self._duration * self._frequency)
        self.tvec = np.linspace(0, self._duration, self.steps)
        self._set_time_scaling()
        
        
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
            self.tau = 3 * tau**2 - 2 * tau**3
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