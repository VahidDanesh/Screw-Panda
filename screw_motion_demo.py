#!/usr/bin/env python3
"""
Demonstration of the Screw-Panda framework for efficient motion planning
using dual quaternions and screw theory.
"""

import numpy as np
from spatialmath import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
from swift import Swift
import pytransform3d.transformations as pt3d

# Import from our framework
from utils.dual_quaternion import DualQuaternionUtils
from utils.screw_utils import (
    sclerp, 
    generate_trajectory, 
    pose_to_screw_parameters, 
    dq_to_se3, 
    se3_to_dq, 
    twist_from_poses
)


def visualize_trajectory(poses, ax=None):
    """Visualize a trajectory of poses."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract position and plot trajectory
    positions = np.array([pose.t for pose in poses])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    
    # Plot coordinate frames at selected points
    stride = max(1, len(poses) // 5)  # Show at most 5 frames
    for i in range(0, len(poses), stride):
        pose = poses[i]
        origin = pose.t
        
        # X, Y, Z axes
        axis_length = 0.05
        colors = ['r', 'g', 'b']
        
        for j, color in enumerate(colors):
            direction = pose.R[:, j]
            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0] * axis_length, direction[1] * axis_length, direction[2] * axis_length,
                color=color, arrow_length_ratio=0.15
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory with Coordinate Frames')
    
    # Make the plot more square/equal
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    return ax


def demonstrate_screw_motion():
    """Demonstrate screw motion for manipulation planning."""
    # Define start and end poses
    start_pose = SE3(0.5, 0.0, 0.1) * SE3.RPY(0, 0, 0)
    end_pose = SE3(0.5, 0.4, 0.3) * SE3.RPY(np.pi/2, 0, np.pi/4)
    
    # Extract screw parameters
    screw_params = pose_to_screw_parameters(start_pose, end_pose)
    
    print(f"Screw parameters:")
    print(f"  Rotation angle: {screw_params['theta']:.4f} rad ({np.degrees(screw_params['theta']):.2f} deg)")
    print(f"  Translation distance: {screw_params['d']:.4f} m")
    print(f"  Screw axis direction: {screw_params['axis']}")
    print(f"  Point on axis: {screw_params['point']}")
    print(f"  Pitch: {screw_params['pitch']:.4f}")
    
    # Generate trajectory using screw motion
    num_steps = 20
    poses, twists = generate_trajectory(start_pose, end_pose, num_steps)
    
    # Visualize the trajectory
    ax = visualize_trajectory(poses)
    plt.show()
    
    # Print the first few twists
    print("\nFirst few twist vectors (spatial velocity):")
    for i, twist in enumerate(twists[:3]):
        print(f"  Twist {i+1}: v=[{twist[0]:.4f}, {twist[1]:.4f}, {twist[2]:.4f}], "
              f"ω=[{twist[3]:.4f}, {twist[4]:.4f}, {twist[5]:.4f}]")
    
    return poses, twists


def demonstrate_dual_quaternion_operations():
    """Demonstrate dual quaternion operations for pose representation."""
    # Create some SE3 poses
    pose1 = SE3(0.3, 0.2, 0.1) * SE3.RPY(0.1, 0.2, 0.3)
    pose2 = SE3(0.5, 0.4, 0.3) * SE3.RPY(0.4, 0.5, 0.6)
    
    # Convert to dual quaternions using pytransform3d
    dq1 = se3_to_dq(pose1)
    dq2 = se3_to_dq(pose2)
    
    print(f"Dual quaternion 1:")
    print(f"  {dq1}")
    
    # Convert back to SE3
    pose1_recovered = dq_to_se3(dq1)
    
    print(f"\nOriginal pose vs recovered pose:")
    print(f"  Original: {pose1}")
    print(f"  Recovered: {pose1_recovered}")
    
    # Perform dual quaternion multiplication
    dq_product = pt3d.concatenate_dual_quaternions(dq1, dq2)
    pose_product = dq_to_se3(dq_product)
    se3_product = pose1 * pose2
    
    print(f"\nSE3 product vs Dual quaternion product:")
    print(f"  SE3 direct: {se3_product}")
    print(f"  Via DQ: {pose_product}")
    
    # Perform sclerp
    interpolated_poses = [sclerp(pose1, pose2, t) for t in np.linspace(0, 1, 5)]
    
    print(f"\nInterpolated poses at t=0, 0.25, 0.5, 0.75, 1.0:")
    for i, pose in enumerate(interpolated_poses):
        t = i / (len(interpolated_poses) - 1)
        print(f"  t={t:.2f}: {pose}")
    
    # Visualize interpolated poses
    ax = visualize_trajectory(interpolated_poses)
    plt.show()
    
    return interpolated_poses


def demonstrate_robot_control():
    """Demonstrate robot control using dual quaternion trajectories."""
    # Create a Swift instance
    env = Swift()
    env.launch(realtime=True)
    
    # Create a Panda robot
    panda = rtb.models.Panda()
    panda.q = panda.qr  # Set to ready position
    
    # Add the robot to the environment
    env.add(panda)
    
    # Define start and end poses
    Tstart = panda.fkine(panda.q)
    Tgoal = Tstart * SE3.Tx(0.2) * SE3.Ty(0.2) * SE3.Tz(0.2) * SE3.RPY(0.3, 0.2, 0.1)
    
    # Generate a trajectory using screw interpolation
    num_steps = 50
    poses, twists = generate_trajectory(Tstart, Tgoal, num_steps)
    
    print("Starting robot trajectory...")
    
    # Follow the trajectory
    for i, (pose, twist) in enumerate(zip(poses[1:], twists)):
        # Display progress
        if i % 10 == 0:
            completion = (i / len(twists)) * 100
            print(f"Trajectory progress: {completion:.1f}%")
        
        # Calculate joint velocities using resolved rate motion control
        J = panda.jacob0(panda.q)
        J_pinv = np.linalg.pinv(J)
        
        # Current state
        Te = panda.fkine(panda.q)
        
        # Use twist from our trajectory
        Vep = twist
        
        # Calculate error twist
        ev, arrived = rtb.p_servo(Te, pose, gain=1.0, threshold=0.001, method="twist")
        
        # Secondary task (joint limit avoidance)
        n = panda.n
        q0 = secondary_task(panda, panda.q)
        
        # Calculate joint velocities
        qd = J_pinv @ (Vep + ev) + (np.eye(n) - J_pinv @ J) @ q0
        
        # Update robot state
        dt = 0.02
        panda.q = panda.q + dt * qd
        
        # Step the simulation
        env.step(dt)
        
        # Check if we've arrived at the target
        if arrived:
            break
    
    print("Trajectory complete!")
    
    # Keep the environment open
    while True:
        env.step(0.05)


def secondary_task(robot, q):
    """
    Calculate the gradient of a cost function for joint limit avoidance.
    
    Parameters:
        robot: The robot model
        q: Current joint configuration
        
    Returns:
        Gradient of the cost function for secondary task
    """
    n = robot.n
    qlim = robot.qlim
    
    q_mid = (qlim[1] + qlim[0]) / 2
    q_range = qlim[1] - qlim[0]
    
    # Vectorized calculation with singularity protection
    with np.errstate(divide='ignore', invalid='ignore'):
        dw_dq = -1/n * (q - q_mid) / (q_range)**2
        
    # Handle unlimited joints (range = 0)
    dw_dq[np.isinf(dw_dq)] = 0  
    dw_dq[np.isnan(dw_dq)] = 0
    
    return dw_dq


if __name__ == "__main__":
    print("=" * 50)
    print("Screw-Panda Framework Demonstration")
    print("=" * 50)
    print("\n1. Demonstrating screw motion\n")
    poses, twists = demonstrate_screw_motion()
    
    print("\n" + "=" * 50)
    print("\n2. Demonstrating dual quaternion operations\n")
    interpolated_poses = demonstrate_dual_quaternion_operations()
    
    print("\n" + "=" * 50)
    print("\n3. Demonstrating robot control with dual quaternions\n")
    demonstrate_robot_control() 