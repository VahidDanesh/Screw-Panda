"""
Utility functions for screw theory operations using pytransform3d.

This module provides simple wrappers around pytransform3d functions to
maintain compatibility with the Screw-Panda framework while using the 
efficient implementations from pytransform3d.
"""

import numpy as np
import pytransform3d.transformations as pt3d
from spatialmath import SE3, UnitDualQuaternion

def sclerp(pose1, pose2, s):
    """
    Screw Linear Interpolation (ScLERP) between two SE3 poses.
    
    Args:
        pose1 (SE3): Starting pose
        pose2 (SE3): Ending pose
        s (float or numpy.ndarray): Interpolation parameter(s) in range [0, 1]
        
    Returns:
        SE3 or list: Interpolated pose(s)
    """
    # Convert poses to transformation matrices
    T1 = pose1.A if isinstance(pose1, SE3) else pose1
    T2 = pose2.A if isinstance(pose2, SE3) else pose2
    
    # Handle array input for s
    if isinstance(s, np.ndarray) or isinstance(s, list):
        return [SE3(pt3d.transform_sclerp(T1, T2, si)) for si in s]
    else:
        # Use pytransform3d's screw interpolation
        T_interp = pt3d.transform_sclerp(T1, T2, s)
        return SE3(T_interp)

def generate_trajectory(start_pose, end_pose, num_steps):
    """
    Generate a trajectory between two poses using screw interpolation.
    
    Args:
        start_pose (SE3): Starting pose
        end_pose (SE3): Ending pose
        num_steps (int): Number of steps in the trajectory
        
    Returns:
        tuple: (poses, twists) where:
            - poses is a list of SE3 poses along the trajectory
            - twists is a list of 6D spatial velocity vectors between consecutive poses
    """
    # Get transformation matrices
    T_start = start_pose.A
    T_end = end_pose.A
    
    # Generate interpolation parameters
    s_values = np.linspace(0, 1, num_steps)
    
    # Initialize results
    poses = []
    twists = []
    
    # Previous pose for twist calculation
    prev_pose = T_start
    
    # Generate trajectory using pytransform3d's screw interpolation
    for s in s_values:
        # Interpolate transformation using transform_sclerp
        T_current = pt3d.transform_sclerp(T_start, T_end, s)
        poses.append(SE3(T_current))
        
        # Calculate twist between consecutive poses if not the first pose
        if len(poses) > 1:
            # Time step (assuming uniform time spacing)
            dt = 1.0 / (num_steps - 1)
            
            # Get exponential coordinates between consecutive poses
            exp_coords = pt3d.exponential_coordinates_from_transform(
                pt3d.concat(pt3d.invert_transform(prev_pose), T_current))
            
            # Convert to twist (spatial velocity)
            twist = exp_coords / dt
            twists.append(twist)
        
        # Update previous pose
        prev_pose = T_current
    
    return poses, twists

def pose_to_screw_parameters(T_start, T_end):
    """
    Extract screw parameters between two poses.
    
    Args:
        T_start (SE3): Starting transformation
        T_end (SE3): Ending transformation
        
    Returns:
        dict: Screw parameters
            - theta: rotation angle
            - d: translation distance
            - axis: screw axis
            - point: point on the axis
            - pitch: screw pitch
    """
    # Get relative transform from start to end
    T_rel = T_end * T_start.inv()
    
    # Get screw axis
    exp_coords = pt3d.exponential_coordinates_from_transform(T_rel.A)
    
    # Extract components [v, w]
    v = exp_coords[:3]  # Linear part
    w = exp_coords[3:]  # Angular part
    
    # Calculate theta (rotation angle)
    theta = np.linalg.norm(w)
    
    # Calculate screw parameters
    if theta < 1e-10:
        # Pure translation case
        axis = np.array([0, 0, 1])  # Default direction
        d = np.linalg.norm(v)
        
        # If there's a preferred direction in the translation
        if d > 1e-10:
            axis = v / d
        
        # For pure translation, point and pitch aren't well-defined
        # but we can use these values
        point = np.zeros(3)
        pitch = float('inf')
    else:
        # General motion case with rotation
        axis = w / theta
        d = np.dot(v, axis)  # Translation along screw axis
        
        # Compute a point on the axis
        # p = cross(axis, (v - d*axis)) / theta
        point = np.cross(axis, v - d*axis) / theta
        
        # Calculate pitch
        pitch = d / theta
    
    return {
        'theta': theta,
        'd': d,
        'axis': axis,
        'point': point,
        'pitch': pitch
    }

def dq_to_se3(dq):
    """
    Convert a dual quaternion from pytransform3d to an SE3 pose.
    
    Args:
        dq: Dual quaternion from pytransform3d.
        
    Returns:
        SE3: The equivalent SE3 pose.
    """
    T = pt3d.transform_from_dual_quaternion(dq)
    return SE3(T)

def se3_to_dq(pose):
    """
    Convert SE3 pose to a dual quaternion in pytransform3d format.
    
    Args:
        pose (SE3): The pose to convert
        
    Returns:
        ndarray: Dual quaternion in pytransform3d format
    """
    T = pose.A if isinstance(pose, SE3) else pose
    return pt3d.dual_quaternion_from_transform(T)

def twist_from_poses(pose1, pose2, dt=1.0):
    """
    Calculate twist (spatial velocity) between two poses.
    
    Args:
        pose1 (SE3): Starting pose
        pose2 (SE3): Ending pose
        dt (float): Time difference between poses
        
    Returns:
        ndarray: 6D twist vector [vx, vy, vz, wx, wy, wz]
    """
    # Convert poses to transformation matrices
    T1 = pose1.A if isinstance(pose1, SE3) else pose1
    T2 = pose2.A if isinstance(pose2, SE3) else pose2
    
    # Get exponential coordinates
    exp_coords = pt3d.exponential_coordinates_from_transform(
        pt3d.concat(pt3d.invert_transform(T1), T2))
    
    # Convert to twist by dividing by dt
    return exp_coords / dt 