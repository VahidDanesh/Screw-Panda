#!/usr/bin/env python3
"""
Demonstration of rotational slippage using the Screw-Panda framework with Swift simulation.
This example shows how to rotate a box around a screw axis that passes through 
the bottom back edge, and controls a Panda robot to follow the corresponding trajectory.
"""

import numpy as np
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import pytransform3d.transformations as pt3d
import spatialgeometry as sg
from swift import Swift
import time

# Import from Screw-Panda framework
from utils.objects import ManipulableBox
from models.panda_model import create_virtual_panda
# Import planning primitives
from planning.primitives.pivoting import Pivoting

def create_box_and_screw_axis():
    """Create a box and define a screw axis through its bottom back edge."""
    # Create box with dimensions (x, y, z) in meters
    box = ManipulableBox(dimensions=(0.1, 0.07, 0.03), name="target_box")
    
    # Initial pose - slightly above the ground
    initial_pose = SE3(0.5, 0, 0.03) * SE3.Rz(0)
    box.update_pose(initial_pose)
    
    # Get the bottom back edge (edge 1 in the ManipulableBox class)
    edge_start, edge_end = box.get_edge_in_world(1)
    
    # Middle point of the edge as the point on the screw axis
    point_on_axis = (edge_start + edge_end) / 2
    
    # Direction of the edge - this will be our screw axis direction
    axis_direction = edge_end - edge_start
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    
    print(f"Box dimensions: {box.length} x {box.width} x {box.height} meters")
    print(f"Initial pose: {box.pose}")
    print(f"Bottom back edge: from {edge_start} to {edge_end}")
    print(f"Point on screw axis: {point_on_axis}")
    print(f"Screw axis direction: {axis_direction}")
    
    return box, point_on_axis, axis_direction


def create_screw_rotation(box, point_on_axis, axis_direction, angle_deg=45):
    """
    Create a screw motion that rotates the box around the defined axis.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create a rotation matrix around the axis direction
    R = SO3.AngleAxis(angle_rad, axis_direction).R
    
    # Current pose of the box
    start_pose = box.pose
    
    # We need to transform the rotation to be around the point on the axis
    # rather than around the origin
    p = point_on_axis
    
    # Translation part: p - R*p gives the translation needed
    # to keep the point p fixed during rotation
    t = p - R @ p
    
    # Create the final transformation
    end_pose = SE3.Rt(R, t) * start_pose
    
    return end_pose


def init_env():
    """
    Initialize the Swift environment with the Panda robot and visualization elements.
    """
    # Make the environment
    env = Swift()
    env.launch(realtime=True)
    
    # Import robot
    panda_virtual = create_virtual_panda()
    # Set the robot config to ready position
    panda_virtual.q = panda_virtual.qr
    # Open the fingers
    panda_virtual.grippers[0].q = [0.035, 0.035]
    
    # Add robot to environment
    env.add(panda_virtual)
    
    return env, panda_virtual


def simulate_rotational_slippage():
    """
    Main function to simulate rotational slippage with Swift.
    """
    print("=" * 50)
    print("Rotational Slippage Demonstration using Swift Simulation")
    print("=" * 50)
    
    # Initialize the Swift environment
    env, panda_virtual = init_env()
    
    # Create a box and define the screw axis
    box, point_on_axis, axis_direction = create_box_and_screw_axis()
    
    # Create a Swift-compatible visual box
    swift_box = sg.Cuboid(scale=[box.length, box.width, box.height], color='blue')
    swift_box.T = box.pose
    swift_box.set_alpha(0.5)
    env.add(swift_box)
    
    # Create the final pose after rotation
    angle_deg = 70  # Rotation angle in degrees
    final_pose = create_screw_rotation(box, point_on_axis, axis_direction, angle_deg)
    
    # Use the Pivoting primitive to plan the trajectory
    # For the pivoting edge, we'll use edge index 3 (bottom front edge)
    pivoting_primitive = Pivoting(
        obj=box,
        goal_pose=final_pose,
        pivot_edge_idx=3,  # The bottom front edge
        pivot_param=0.5,   # Middle of the edge
        duration=2.0,      # Duration in seconds
        frequency=50       # Same as num_steps in original code
    )
    
    # Plan the trajectory using the primitive
    box_poses, ee_poses = pivoting_primitive.plan()
    
    # Create visual elements for screw axis and visualization
    # Edge of the box (screw axis)
    edge_start, edge_end = box.get_edge_in_world(1)
    # Instead of using sg.Line which doesn't exist, create a thin cylinder
    axis_length = np.linalg.norm(edge_end - edge_start)
    axis_direction_normalized = (edge_end - edge_start) / axis_length
    # Create a thin cylinder along the edge
    screw_axis = sg.Cylinder(
        radius=0.002,  # Very thin cylinder
        length=axis_length,
        pose=SE3(edge_start) * SE3.AngleAxis(np.pi/2, np.cross([0,0,1], axis_direction_normalized))
    )
    screw_axis.color = (1, 1, 0)  # Yellow
    env.add(screw_axis)
    
    # End-effector axes
    ee_axes = sg.Axes(0.1)
    ee_axes.T = panda_virtual.fkine(panda_virtual.q, end='panda_finger_virtual')
    env.add(ee_axes)
    
    # Goal axes
    goal_axes = sg.Axes(0.1)
    
    # Initial grasp pose
    initial_grasp_pose = box.pose * box.grasp_offset
    goal_axes.T = initial_grasp_pose
    env.add(goal_axes)
    
    # Simulation time step
    dt = 0.05
    
    # PHASE 1: Move robot to grasp the box
    print("\nPhase 1: Moving robot to grasp position...")
    
    # Set the initial goal for the robot to approach the box
    arrived_at_grasp = False
    timeout_counter = 0
    max_timeout = 100  # Prevent infinite loops
    
    while not arrived_at_grasp and timeout_counter < max_timeout:
        # Get current robot end-effector pose
        Te = panda_virtual.fkine(panda_virtual.q, end='panda_finger_virtual')
        ee_axes.T = Te
        
        # P servo to grasp pose
        v, arrived_at_grasp = rtb.p_servo(Te, initial_grasp_pose, threshold=0.005)
        
        # Set the Panda's joint velocities using Jacobian
        J = panda_virtual.jacobe(panda_virtual.q, end='panda_finger_virtual')
        panda_virtual.qd = np.linalg.pinv(J) @ v
        
        # Step the simulator
        env.step(dt)
        
        # Add a small delay for visualization
        time.sleep(dt/2)
        timeout_counter += 1
    
    if timeout_counter >= max_timeout:
        print("Warning: Reached timeout while approaching grasp position")
    
    # Close gripper to grasp the box
    print("Closing gripper to grasp the box...")
    for _ in range(20):  # Animate gripper closing
        # Gradually close the fingers
        finger_positions = panda_virtual.grippers[0].q
        # Reduce finger opening by a small amount each step
        new_positions = [max(pos - 0.003, 0.001) for pos in finger_positions]
        panda_virtual.grippers[0].q = new_positions
        
        # Step the simulator
        env.step(dt)
        time.sleep(dt)
    
    print("Box grasped successfully!")
    time.sleep(1)  # Pause to show successful grasp
    
    # PHASE 2: Execute the planned pivoting trajectory
    print("\nPhase 2: Executing pivoting trajectory...")
    
    # Now iterate through each pose in the trajectory
    for i in range(len(box_poses)):
        # Update box position
        current_box_pose = box_poses[i]
        swift_box.T = current_box_pose
        
        # Update goal for the robot
        current_ee_pose = ee_poses[i]
        goal_axes.T = current_ee_pose
        
        # Calculate the required end-effector velocity to track the goal
        Te = panda_virtual.fkine(panda_virtual.q, end='panda_finger_virtual')
        ee_axes.T = Te
        
        # P servo to current goal pose
        v, _ = rtb.p_servo(Te, current_ee_pose, threshold=0.01)
        
        # Set the Panda's joint velocities using Jacobian
        J = panda_virtual.jacobe(panda_virtual.q, end='panda_finger_virtual')
        panda_virtual.qd = np.linalg.pinv(J) @ v
        
        # Step the simulator
        env.step(dt)
        
        # Add a small delay for visualization
        time.sleep(dt/2)
    
    # Let the robot reach the final pose exactly
    print("\nFinalizing position...")
    arrived = False
    timeout_counter = 0
    max_timeout = 50  # Limit maximum iterations
    
    while not arrived and timeout_counter < max_timeout:
        # Work out the required end-effector velocity to go towards the goal
        Te = panda_virtual.fkine(panda_virtual.q, end='panda_finger_virtual')
        ee_axes.T = Te
        v, arrived = rtb.p_servo(Te, ee_poses[-1], threshold=0.01)
        
        # Set the Panda's joint velocities
        panda_virtual.qd = np.linalg.pinv(panda_virtual.jacobe(panda_virtual.q, end='panda_finger_virtual')) @ v
        
        # Step the simulator
        env.step(dt)
        time.sleep(dt/2)
        
        # Increment timeout counter
        timeout_counter += 1
    
    print("Simulation complete!")
    
    # Keep the window open
    while True:
        env.step(dt)
        time.sleep(dt)


if __name__ == "__main__":
    simulate_rotational_slippage() 