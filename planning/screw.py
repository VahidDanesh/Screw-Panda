# planning/screw.py

import numpy as np
from spatialmath import DualQuaternion, UnitDualQuaternion, Quaternion

class Screw:
    """
    Represents a screw motion in 3D space.
    
    Attributes:
        theta (float): Rotation angle about the screw axis (in radians)
        d (float): Translation distance along the screw axis
        u (numpy.ndarray): Unit vector along the screw axis (3x1)
        r (numpy.ndarray): Point on the screw axis (3x1)
        m (numpy.ndarray): Moment vector (r × u)
    """
    
    def __init__(self, theta, d, u, r=None, m=None):
        """
        Initialize a screw motion.
        
        Args:
            theta (float): Rotation angle about the screw axis (in radians)
            d (float): Translation distance along the screw axis
            u (numpy.ndarray): Unit vector along the screw axis (3x1)
            r (numpy.ndarray, optional): Point on the screw axis (3x1)
            m (numpy.ndarray, optional): Moment vector. If not provided and r is given,
                                        it will be calculated as r × u
        """
        self.theta = theta
        self.d = d
        self.u = np.array(u, dtype=float)
        
        # Normalize u if it's not already a unit vector
        norm_u = np.linalg.norm(self.u)
        if abs(norm_u - 1.0) > 1e-10:
            self.u = self.u / norm_u
        
        if m is not None:
            self.m = np.array(m, dtype=float)
            self.r = None  # We don't calculate r from m automatically
        elif r is not None:
            self.r = np.array(r, dtype=float)
            self.m = np.cross(self.r, self.u)
        else:
            raise ValueError("Either r or m must be provided")

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        """
        String representation of the screw motion.
        Returns:
            str: A string describing the screw motion
        """
        return "Screw(theta={}, d={}, u={}, m={})".format(
            self.theta, self.d, self.u, self.m
        )


    def dquat(self):
        """
        Convert screw motion parameters to a unit dual quaternion.
        
        Args:
            screw (Screw): Screw motion parameters
            
        Returns:
            UnitDualQuaternion: The equivalent unit dual quaternion
        """
        theta = self.theta
        d = self.d
        u = self.u
        m = self.m
        
        qr = Quaternion(np.r_[np.cos(theta / 2), np.sin(theta / 2) * u])
        
        # Dual part
        qd_scalar = -d/2 * np.sin(theta / 2)
        qd_vector = d/2 * np.cos(theta / 2) * u + np.sin(theta / 2) * m
        qd = Quaternion(np.r_[qd_scalar, qd_vector])
        
        return UnitDualQuaternion(qr, qd)
    
    def sclerp(self, dq2, s):
        """
        Screw Linear Interpolation (ScLERP) between this screw and another unit dual quaternion.
        
        Args:
            dq2 (UnitDualQuaternion): Ending configuration
            s (float or numpy.ndarray): Interpolation parameter(s) in range [0, 1]
            
        Returns:
            UnitDualQuaternion or list: Interpolated configuration(s)
        """
        return sclerp(self.dquat(), dq2, s)
    
def sclerp(dq1, dq2, s):
    """
    Screw Linear Interpolation (ScLERP) between two unit dual quaternions.
    
    The smooth path in SE(3) provided by ScLERP is given by D(s) = D_1 * D_12^s,
    where s ∈ [0, 1] is a scalar path parameter, and D_12 is the transformation
    of configuration 2 relative to configuration 1. 
    
    Args:
        dq1 (UnitDualQuaternion): Starting configuration
        dq2 (UnitDualQuaternion): Ending configuration
        s (float or numpy.ndarray): Interpolation parameter(s) in range [0, 1]
        
    Returns:
        UnitDualQuaternion or list: Interpolated configuration(s)
    """
    # Handle array input for s
    
    if isinstance(s, np.ndarray):
        return [sclerp(dq1, dq2, si) for si in s]
    
    # Calculate the relative transformation from dq1 to dq2
    dq12 = dq1.conj() * dq2
    
    # Extract screw parameters from dq12
    screw12 = dual_quaternion_to_screw(dq12)
    
    # Create a new screw with scaled parameters
    scaled_screw = Screw(
        theta=s * screw12.theta,
        d=s * screw12.d,
        u=screw12.u,
        m=screw12.m
    )
    
    # Convert scaled screw to dual quaternion
    dq12_s = scaled_screw.dquat()
    
    # Apply the scaled transformation to dq1
    return dq1 * dq12_s


def dual_quaternion_to_screw(dq):
    """
    Extract screw parameters from a unit dual quaternion.
    
    Args:
        dq (UnitDualQuaternion): Unit dual quaternion
        
    Returns:
        Screw: The equivalent screw motion parameters
    """
    # Extract the real and dual parts
    qr = dq.real
    qd = dq.dual
    
    # Extract rotation parameters
    qr_v = np.array([qr.x, qr.y, qr.z])
    qr_w = qr.w
    qr_v_norm = np.linalg.norm(qr_v)
    
    # Handle pure translation case
    if qr_v_norm < 1e-10:
        # Pure translation
        theta = 0
        u = np.array([0, 0, 1])  # Default direction
        d = 2 * np.linalg.norm(np.array([qd.x, qd.y, qd.z]))
        m = np.zeros(3)
        
        # If there's a preferred direction in the dual part
        qd_v = np.array([qd.x, qd.y, qd.z])
        if np.linalg.norm(qd_v) > 1e-10:
            u = qd_v / np.linalg.norm(qd_v)
            
        return Screw(theta, d, u, m=m)
    
    # Normal case with rotation
    theta = 2 * np.arctan2(qr_v_norm, qr_w)
    u = qr_v / qr_v_norm
    
    # Extract translation parameters
    qd_v = np.array([qd.x, qd.y, qd.z])
    qd_w = qd.w
    
    # Calculate d (translation along screw axis)
    d = -2 * (qd_w * qr_v_norm / qr_w - np.dot(qd_v, u) * qr_w / qr_v_norm)
    
    # Calculate m (moment vector)
    sin_half_theta = np.sin(theta/2)
    if abs(sin_half_theta) < 1e-10:
        # Handle small angle case
        m = np.zeros(3)
    else:
        m_term1 = qd_v / sin_half_theta
        m_term2 = u * (d/2 * np.cos(theta/2) / sin_half_theta)
        m = m_term1 - m_term2
    
    return Screw(theta, d, u, m=m)
