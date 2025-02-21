import numpy as np
from spatialmath import SE3, DualQuaternion, UnitQuaternion
from spatialgeometry import Cuboid, Cylinder, Shape
from roboticstoolbox import quintic
import matplotlib.pyplot as plt

class SpatialObject:
    """
    A class to represent spatial objects with geometries and generate smooth motion paths using dual quaternions.

    Attributes:
        geometry (Shape): The spatial geometry (e.g., Cuboid, Cylinder) from spatialgeometry.
        pose (SE3): The object's current pose as an SE3 transformation.
        name (str): A name identifier for the object.
    """

    def __init__(self, geometry_type="box", dimensions=(.1, .07, .3), pose=SE3(), name="object"):
        """
        Initialize a SpatialObject with a specified geometry, dimensions, and pose.

        Args:
            geometry_type (str): Type of geometry ('box' or 'cylinder'). Default is 'box'.
            dimensions (tuple): Dimensions of the object (length, width, height) for box,
                                or (radius, height) for cylinder. Default is (.1, .07, 0.3).
            pose (SE3): Initial pose of the object as an SE3 instance. Default is identity.
            name (str): Name of the object. Default is 'object'.

        Raises:
            ValueError: If geometry_type is unsupported or dimensions are invalid.
        """
        self.name = name
        self.pose = pose if isinstance(pose, SE3) else SE3(pose)

        if geometry_type.lower() == "box":
            if len(dimensions) != 3:
                raise ValueError("Box requires 3 dimensions: (length, width, height)")
            self.geometry = Cuboid(scale=dimensions, pose=self.pose)
        elif geometry_type.lower() == "cylinder":
            if len(dimensions) != 2:
                raise ValueError("Cylinder requires 2 dimensions: (radius, height)")
            self.geometry = Cylinder(radius=dimensions[0], height=dimensions[1], pose=self.pose)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
        
        # Update the geometry's pose to match the object's pose
        self.geometry.T = self.pose

    @property
    def position(self):
        """Get the object's position (translation vector) as a numpy array."""
        return self.pose.t

    @property
    def orientation(self):
        """Get the object's orientation as a rotation matrix (3x3 numpy array)."""
        return self.pose.R

    def generate_path(self, target_pose, tilt_time=1.0, method="quintic", frequency=1000):
        """
        Generate a smooth motion path from the current pose to a target pose using dual quaternions.

        Args:
            target_pose (SE3): The target pose to move to.
            tilt_time (float): Duration of the motion in seconds. Default is 1.0.
            method (str): Interpolation method ('linear', 'cubic', 'quintic'). Default is 'quintic'.
            frequency (int): Sampling frequency in Hz. Default is 1000.

        Returns:
            tuple: (poses, velocities)
                - poses (list of SE3): List of poses along the path.
                - velocities (np.ndarray): 6xN array of twist velocities (v, omega).

        Raises:
            ValueError: If method is unsupported.
        """
        if not isinstance(target_pose, SE3):
            target_pose = SE3(target_pose)

        # Convert poses to dual quaternions
        dq_start = self.pose.UnitQuaternion()
        dq_end = target_pose.UnitQuaternion()

        # Time vector and steps
        steps = int(tilt_time * frequency)
        tvec = np.linspace(0, tilt_time, steps)

        # Generate time-scaling profile
        if method.lower() == "linear":
            tau = np.linspace(0, 1, steps)
            taud = np.ones(steps) / tilt_time  # Constant velocity
        elif method.lower() == "cubic":
            # tau, taud, _ = cubic(0, 1, tvec)
            ...
        elif method.lower() == "quintic":
            tg= quintic(0, 1, tvec)
            tau = tg.q
            taud = tg.qd
        else:
            raise ValueError("Method must be 'linear', 'cubic', or 'quintic'")

        # Interpolate using dual quaternions
        poses = []
        velocities = []
        for i in range(steps):
            # Dual quaternion interpolation (slerp-like)
            dq_interp = dq_start * (1 - tau[i]) + dq_end * tau[i]
            dq_interp = UnitQuaternion(dq_interp.s, dq_interp.v, norm=True)  # Normalize
            pose = dq_interp.SE3()  # Convert back to SE3
            poses.append(pose)

            # Compute twist (assume screw motion approximation)
            # For simplicity, use a finite difference approximation of velocity
            if i > 0:
                delta_pose = poses[i].inv() * poses[i - 1]
                twist = delta_pose.log() / (tvec[i] - tvec[i - 1])  # 6x1 twist vector
                velocities.append(twist)
            else:
                velocities.append(np.zeros(6))  # Initial velocity is zero

        # Update the object's final pose
        self.pose = poses[-1]
        self.geometry.T = self.pose

        return poses  # Return velocities as 6xN

# Example Usage
if __name__ == "__main__":
    # Create a box
    box = SpatialObject(
        geometry_type="box",
        dimensions=(0.5, 0.3, 0.2),
        pose=SE3(0, 0, 0),
        name="Box1"
    )

    # Define a target pose (rotate 90 degrees around Z and translate)
    target = SE3(1, 1, 0) * SE3.Rz(np.pi / 2)

    # Generate a path
    poses = box.generate_path(
        target_pose=target,
        tilt_time=2.0,
        method="quintic",
        frequency=100
    )

    # Print some info
    print(f"Initial Pose:\n{poses[0]}")
    print(f"Final Pose:\n{poses[-1]}")
    print(f"Number of steps: {len(poses)}")

    # Visualize the path
    # box.visualize_path(poses)