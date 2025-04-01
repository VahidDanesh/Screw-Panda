import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d
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
            obj (SpatialObject): The object to manipulate.
            goal_pose (SE3): Goal pose for the object.
            contact_face_idx (int): Index of the face in contact with the surface.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        self.contact_face_idx = contact_face_idx
        
        # Verify that the object's pose and goal pose maintain the contact face constraint
        self._validate_poses()
    
    def _validate_poses(self):
        """
        Validate that the start and goal poses maintain the contact face constraint.
        
        For sliding, the contact face normal should be perpendicular to the supporting
        surface (assumed to be the XY plane with Z pointing up).
        """
        # Get the face normal in world frame for both start and goal poses
        start_normal = self.object.get_face_normal(self.contact_face_idx)
        
        # For the goal pose, we need to temporarily update the object's pose,
        # get the normal, then restore the original pose
        original_pose = self.object.pose
        self.object.update_pose(self.goal_pose)
        goal_normal = self.object.get_face_normal(self.contact_face_idx)
        self.object.update_pose(original_pose)
        
        # Check if the face normal is aligned with the Z axis (up)
        z_axis = np.array([0, 0, 1])
        start_alignment = np.abs(np.dot(start_normal, z_axis))
        goal_alignment = np.abs(np.dot(goal_normal, z_axis))
        
        # Allow a small tolerance for numerical errors
        tolerance = 0.01
        
        if start_alignment < 1 - tolerance or goal_alignment < 1 - tolerance:
            print(f"Warning: Contact face does not maintain alignment with supporting surface.")
            print(f"Start alignment: {start_alignment}, Goal alignment: {goal_alignment}")
    
    def plan(self):
        """
        Plan a trajectory for sliding the object.
        
        The sliding motion ensures that:
        1. The contact face remains in contact with the supporting surface
        2. The object's orientation relative to the supporting surface is maintained
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Get time scaling
        tau, _ = self._set_time_scaling(method="quintic")
        
        # Extract translational and rotational components
        start_pos = self.start_pose.t
        start_rot = self.start_pose.R
        goal_pos = self.goal_pose.t
        goal_rot = self.goal_pose.R
        
        # For sliding, we'll constrain the height (z) to maintain contact
        contact_height = start_pos[2]  # Assume that start position is valid
        
        # Generate object poses
        self._object_poses = []
        self._object_dquats = []
        
        for t in tau:
            # Interpolate position (maintain z height)
            pos = start_pos + t * (goal_pos - start_pos)
            pos[2] = contact_height  # Fix z coordinate to maintain contact
            
            # Interpolate rotation (using slerp via pytransform3d)
            if np.allclose(start_rot, goal_rot):
                rot = start_rot
            else:
                dq_rot = pt3d.quaternion_slerp(
                    pt3d.quaternion_from_matrix(start_rot),
                    pt3d.quaternion_from_matrix(goal_rot),
                    t
                )
                rot = pt3d.matrix_from_quaternion(dq_rot)
            
            # Create SE3 pose
            pose = SE3(pos) * SE3(R=rot)
            self._object_poses.append(pose)
            
            # Convert to dual quaternion
            dq = pt3d.dual_quaternion_from_transformation_matrix(pose.A)
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