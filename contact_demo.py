"""
Demonstration of contact-based manipulation primitives.
This file contains example code for manipulating objects using various
contact-based primitives like sliding, pivoting, and rolling.
"""

import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d
import spatialgeometry as sg
from spatialgeometry import Axes
from swift import Swift

from planning.contact import ContactPlanner
from utils.objects import Box, Cylinder

def init_contact_env():
    """
    Initialize a simulation environment for contact-based manipulation.
    
    Returns:
        Swift: The simulation environment.
    """
    # Make the environment
    env = Swift()
    env.launch(realtime=True, browser="notebook")
    
    # Create a ground plane
    ground = sg.Cuboid(scale=[2.0, 2.0, 0.01], pose=SE3(0, 0, -0.005))
    ground.color = [0.8, 0.8, 0.8, 0.3]  # Light gray, semi-transparent
    
    # Create axes for reference
    world_axes = Axes(0.2)
    
    # Add to environment
    env.add(ground)
    env.add(world_axes)
    
    return env

def demonstrate_box_sliding():
    """
    Demonstrate sliding a box on a surface.
    
    Returns:
        tuple: (Box, list of SE3) The box object and end-effector poses.
    """
    print("Demonstrating box sliding...")
    env = init_contact_env()
    
    # Create a box
    box = Box(
        dimensions=(0.1, 0.07, 0.03),   # Length, width, height
        pose=SE3(0.5, -0.3, 0.015),     # Initial pose
        name="box"
    )
    
    # Set the grasp point to be at the top center
    box.set_grasp_offset(SE3(0, 0, box.height/2))
    
    # Create a planner
    planner = ContactPlanner(box)
    
    # Define a goal pose for sliding
    goal_pose = SE3(0.5, 0.3, 0.015) * SE3.Rz(np.pi/4)
    
    # Plan the sliding motion (face 0 is the bottom face)
    sliding = planner.slide(goal_pose, contact_face_idx=0, duration=3.0, frequency=50)
    
    # Execute the plan
    object_poses, ee_poses = planner.execute_all()
    
    # Add the box to the environment
    env.add(box.geometry)
    
    # Add an axis marker for the end-effector
    ee_axes = Axes(0.1)
    env.add(ee_axes)
    
    # Simulate the motion
    for i, (obj_pose, ee_pose) in enumerate(zip(object_poses, ee_poses)):
        # Update poses
        box.update_pose(obj_pose)
        ee_axes.T = ee_pose
        
        # Step the environment
        env.step(0)
        
        # Slow down visualization
        import time
        time.sleep(0.05)
    
    return box, ee_poses

def demonstrate_cylinder_rolling():
    """
    Demonstrate rolling a cylinder along a curved path.
    
    Returns:
        tuple: (Cylinder, list of SE3) The cylinder object and end-effector poses.
    """
    print("Demonstrating cylinder rolling...")
    env = init_contact_env()
    
    # Create a cylinder
    cylinder = Cylinder(
        radius=0.02,
        height=0.1,
        pose=SE3(-0.3, -0.3, 0.02),  # Initial pose
        name="cylinder"
    )
    
    # Set the grasp point to be at the side, slightly above center
    cylinder.set_grasp_offset(SE3(cylinder.radius, 0, cylinder.height/4))
    
    # Create a planner
    planner = ContactPlanner(cylinder)
    
    # Define a goal pose for rolling
    goal_pose = SE3(0.3, 0.3, 0.02)
    
    # Plan the rolling motion along a curved path
    rolling = planner.roll_curved(goal_pose, duration=5.0, frequency=100)
    
    # Execute the plan
    object_poses, ee_poses = planner.execute_all()
    
    # Add the cylinder to the environment
    env.add(cylinder.geometry)
    
    # Add an axis marker for the end-effector
    ee_axes = Axes(0.1)
    env.add(ee_axes)
    
    # Simulate the motion
    for i, (obj_pose, ee_pose) in enumerate(zip(object_poses, ee_poses)):
        # Update poses
        cylinder.update_pose(obj_pose)
        ee_axes.T = ee_pose
        
        # Step the environment
        env.step(0)
        
        # Slow down visualization
        import time
        time.sleep(0.02)
    
    return cylinder, ee_poses

def demonstrate_box_pivoting():
    """
    Demonstrate pivoting a box around an edge.
    
    Returns:
        tuple: (Box, list of SE3) The box object and end-effector poses.
    """
    print("Demonstrating box pivoting...")
    env = init_contact_env()
    
    # Create a box
    box = Box(
        dimensions=(0.1, 0.07, 0.03),  # Length, width, height
        pose=SE3(0, 0, 0.015),          # Initial pose
        name="box"
    )
    
    # Set the grasp point to be at the top-right corner
    box.set_grasp_offset(SE3(box.length/2, box.width/2, box.height/2))
    
    # Create a planner
    planner = ContactPlanner(box)
    
    # Define a goal pose for pivoting (90 degree rotation around Z)
    goal_pose = SE3(0, 0, 0.015) * SE3.Rz(np.pi/2)
    
    # Plan the pivoting motion (edge 0 is the bottom-back edge)
    pivoting = planner.pivot(goal_pose, pivot_edge_idx=0, pivot_param=0.5, duration=2.0, frequency=50)
    
    # Execute the plan
    object_poses, ee_poses = planner.execute_all()
    
    # Add the box to the environment
    env.add(box.geometry)
    
    # Add an axis marker for the end-effector
    ee_axes = Axes(0.1)
    env.add(ee_axes)
    
    # Simulate the motion
    for i, (obj_pose, ee_pose) in enumerate(zip(object_poses, ee_poses)):
        # Update poses
        box.update_pose(obj_pose)
        ee_axes.T = ee_pose
        
        # Step the environment
        env.step(0)
        
        # Slow down visualization
        import time
        time.sleep(0.05)
    
    return box, ee_poses

def extract_dual_quaternions(ee_poses):
    """
    Extract dual quaternion representations of poses.
    
    Args:
        ee_poses (list): List of SE3 poses.
        
    Returns:
        list: List of dual quaternions.
    """
    dquats = []
    for pose in ee_poses:
        dq = pt3d.dual_quaternion_from_transform(pose.A)
        dquats.append(dq)
    
    return dquats

def control_robot_with_trajectory(ee_poses, init_env_fn, secondary_object_fn, duration=5.0):
    """
    Control the Panda robot to follow the end-effector trajectory.
    
    Args:
        ee_poses (list): List of SE3 poses for the end-effector.
        init_env_fn: Function to initialize the robot environment.
        secondary_object_fn: Function to calculate joint limit avoidance gradient.
        duration (float): Duration of the motion in seconds.
        
    Returns:
        object: The environment.
    """
    import roboticstoolbox as rtb
    
    # Initialize the environment and robot
    env, _, panda_virtual, ee_axes, goal_axes = init_env_fn()
    
    # Calculate time step
    dt = duration / len(ee_poses)
    
    # Control gains
    kt = 1.5
    kr = 1.0
    k = np.array([kt, kt, kt, kr, kr, kr])
    n = panda_virtual.n
    
    # Secondary task gain
    k0 = 1
    
    # Loop through the trajectory
    for i, Tep in enumerate(ee_poses):
        # Update the goal axes
        goal_axes.T = Tep
        
        # Current state
        J = panda_virtual.jacobe(panda_virtual.q, end='panda_finger_virtual')
        J_pinv = np.linalg.pinv(J)
        Te = panda_virtual.fkine(panda_virtual.q, end='panda_finger_virtual').A
        
        # Calculate the required end-effector velocity
        ev, _ = rtb.p_servo(Te, Tep.A, gain=k, threshold=0.001, method="twist")
        
        # Secondary task for joint limit avoidance
        dw_dq = secondary_object_fn(panda_virtual, panda_virtual.q)
        q0 = k0 * dw_dq
        
        # Calculate joint velocities
        qd = J_pinv @ ev + (np.eye(n) - J_pinv @ J) @ q0
        
        # Apply the joint velocities to the Panda
        panda_virtual.qd[:n] = qd[:n]
        
        # Update the ee axes
        ee_axes.T = Te
        
        # Step the simulator
        env.step(dt)
    
    return env 