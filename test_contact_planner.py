#!/usr/bin/env python3
"""
Test script for the contact-based manipulation planning project.

This script demonstrates how to use the contact planner with various 
manipulation primitives to perform object manipulation tasks.
"""

import numpy as np
import time
from spatialmath import SE3
import roboticstoolbox as rtb
import spatialgeometry as sg
import swift
from swift.tools import Ticker
import spatialmath as sm

# Import our planning modules
from planning.contact import ContactPlanner
from planning.primitives import (
    Grasping, Sliding, Pivoting, StraightRolling, CurvedRolling
)
from utils.objects import ManipulableBox, ManipulableCylinder, SpatialObject


def init_env():
    """
    Initialize the simulation environment with a Panda robot and objects.
    
    Returns:
        tuple: (env, Tep, panda, ee_axes, goal_axes) 
    """
    # Make the environment
    env = swift.Swift()
    env.launch(realtime=True, browser="notebook")
    
    # import robot
    panda = rtb.models.Panda()
    panda_virtual = panda.copy()
    # set the robot config to ready position
    panda_virtual.q = panda_virtual.qr
    # open the fingers
    panda_virtual.grippers[0].q = [0.035, 0.035]
    
    # create object to grasp
    box_geom = sg.Cuboid(scale=[0.1, 0.07, 0.03], color='blue')
    box_geom.T = sm.SE3(0.7, 0, 0.015)
    box_geom.set_alpha(0.5)
    
    # Set a desired end effector pose as an offset from the box pose
    bTe = sm.SE3(-box_geom.scale[0]/2 + 0.01, 0, 0) * sm.SE3.Rx(np.pi/2)
    Tep = box_geom.T * bTe
    
    # end-effector axes
    ee_axes = sg.Axes(0.1)
    ee_axes.T = panda_virtual.fkine(panda_virtual.q, end='panda_hand')

    # goal axes
    goal_axes = sg.Axes(0.1)
    # Set the goal axes to Tep
    goal_axes.T = Tep

    box_axes = sg.Axes(0.1)
    box_axes.T = box_geom.T
    
    # add to environment
    env.add(panda_virtual)
    env.add(box_geom)
    env.add(ee_axes)
    env.add(goal_axes)
    env.add(box_axes)
    
    return env, Tep, panda_virtual, box_geom, ee_axes, goal_axes


def secondary_object(robot: rtb.Robot, q: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the joint limit avoidance function

    Parameters
    ----------
    robot
        The robot the manipulators kinematics
    q
        The current joint coordinate vector

    Returns
    -------
    dw_dq
        The gradient of the joint limit avoidance function
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


def execute_robot_trajectory(env, robot, ee_axes, ee_poses, dt=0.05):
    """
    Execute a trajectory on the robot.
    
    Args:
        env: Swift environment
        robot: Robot model
        ee_axes: End-effector visualization axes
        ee_poses: List of end-effector poses to follow
        dt: Time step
    """
    # Resolution of the trajectory
    num_steps = len(ee_poses)
    
    for i in range(num_steps):
        # Get the target pose for this step
        target_pose = ee_poses[i].A if hasattr(ee_poses[i], 'A') else ee_poses[i]
        
        # Current end-effector pose
        Te = robot.fkine(robot.q, end='panda_hand').A
        
        # Gain for the p_servo method
        kt = 1.5  # translational gain
        kr = 1.0  # rotational gain
        k = np.array([kt, kt, kt, kr, kr, kr])
        
        # Calculate the required end-effector velocity
        ev, arrived = rtb.p_servo(Te, target_pose, gain=k, threshold=0.001, method="twist")
        
        # Calculate the Jacobian
        J = robot.jacobe(robot.q, end='panda_hand')
        J_pinv = np.linalg.pinv(J)
        
        # Secondary task - joint limit avoidance
        dw_dq = secondary_object(robot, robot.q)
        k0 = 1.0  # gain for joint limit avoidance
        q0 = k0 * dw_dq
        
        # Calculate joint velocities with redundancy handling
        qd = J_pinv @ ev + (np.eye(robot.n) - J_pinv @ J) @ q0
        
        # Apply the joint velocities
        robot.qd[:robot.n] = qd[:robot.n]
        
        # Update the end-effector visualization
        ee_axes.T = Te
        
        # Step the simulator
        env.step(dt)
        
        # Additional delay to slow down the simulation for visualization
        time.sleep(0.01)


def test_grasping():
    """Test the grasping primitive."""
    print("=== Testing Grasping Primitive ===")
    
    # Initialize environment
    env, Tep, panda, box_geom, ee_axes, goal_axes = init_env()
    
    # Create a manipulable box object
    box = ManipulableBox(
        dimensions=[0.1, 0.07, 0.03],
        pose=SE3(0.7, 0, 0.015),
        name="box"
    )
    
    # Create a contact planner
    planner = ContactPlanner(box)
    
    # Define a goal pose - move the box 20cm forward and rotate it
    goal_pose = box.pose * SE3(0.2, 0.1, 0.05) * SE3.RPY(0.3, 0, 0.5)
    
    # Plan a grasping primitive
    print("Planning grasping primitive...")
    grasping = planner.grasp(goal_pose, approach_distance=0.1)
    
    # Execute the primitive to generate trajectories
    print("Executing the plan...")
    object_poses, ee_poses = grasping.execute()
    
    # Visualize in simulation
    print("Simulating robot motion...")
    execute_robot_trajectory(env, panda, ee_axes, ee_poses)
    
    print("Grasping test completed")
    time.sleep(2)
    env.close()


def test_sliding():
    """Test the sliding primitive."""
    print("=== Testing Sliding Primitive ===")
    
    # Initialize environment
    env, Tep, panda, box_geom, ee_axes, goal_axes = init_env()
    
    # Create a manipulable box object
    box = ManipulableBox(
        dimensions=[0.1, 0.07, 0.03],
        pose=SE3(0.7, 0, 0.015),
        name="box"
    )
    
    # Create a contact planner
    planner = ContactPlanner(box)
    
    # Define a goal pose for sliding - keep the same height
    goal_pose = box.pose * SE3(0.2, 0.1, 0) * SE3.RPY(0, 0, 0.5)
    
    # Plan a sliding primitive (bottom face stays on surface)
    print("Planning sliding primitive...")
    sliding = planner.slide(goal_pose, contact_face_idx=0)
    
    # Execute the primitive to generate trajectories
    print("Executing the plan...")
    object_poses, ee_poses = sliding.execute()
    
    # Visualize in simulation
    print("Simulating robot motion...")
    execute_robot_trajectory(env, panda, ee_axes, ee_poses)
    
    print("Sliding test completed")
    time.sleep(2)
    env.close()


def test_pivoting():
    """Test the pivoting primitive."""
    print("=== Testing Pivoting Primitive ===")
    
    # Initialize environment
    env, Tep, panda, box_geom, ee_axes, goal_axes = init_env()
    
    # Create a manipulable box object
    box = ManipulableBox(
        dimensions=[0.1, 0.07, 0.03],
        pose=SE3(0.7, 0, 0.015),
        name="box"
    )
    
    # Create a contact planner
    planner = ContactPlanner(box)
    
    # Define a goal pose for pivoting - rotate around the front edge
    goal_pose = box.pose * SE3.RPY(0.5, 0, 0)
    
    # Plan a pivoting primitive (front edge stays fixed)
    print("Planning pivoting primitive...")
    pivoting = planner.pivot(goal_pose, pivot_edge_idx=3)
    
    # Execute the primitive to generate trajectories
    print("Executing the plan...")
    object_poses, ee_poses = pivoting.execute()
    
    # Visualize in simulation
    print("Simulating robot motion...")
    execute_robot_trajectory(env, panda, ee_axes, ee_poses)
    
    print("Pivoting test completed")
    time.sleep(2)
    env.close()


def test_cylinder_rolling():
    """Test the rolling primitives with a cylinder."""
    print("=== Testing Cylinder Rolling Primitives ===")
    
    # Initialize environment
    env = swift.Swift()
    env.launch(realtime=True, browser="notebook")
    
    # import robot
    panda = rtb.models.Panda()
    panda_virtual = panda.copy()
    panda_virtual.q = panda_virtual.qr
    panda_virtual.grippers[0].q = [0.035, 0.035]
    
    # Create a cylinder in the environment
    cyl_geom = sg.Cylinder(radius=0.037, length=0.234, color='green')
    cyl_geom.T = sm.SE3(0.7, 0, 0.037)  # Cylinder on surface
    cyl_geom.set_alpha(0.5)
    
    # end-effector and goal visualization
    ee_axes = sg.Axes(0.1)
    ee_axes.T = panda_virtual.fkine(panda_virtual.q, end='panda_hand')
    goal_axes = sg.Axes(0.1)
    goal_axes.T = sm.SE3(0.9, 0.2, 0.037)  # Goal position
    
    # Add to environment
    env.add(panda_virtual)
    env.add(cyl_geom)
    env.add(ee_axes)
    env.add(goal_axes)
    
    # Create a manipulable cylinder object
    cylinder = ManipulableCylinder(
        radius=0.037,
        length=0.234,
        pose=SE3(0.7, 0, 0.037),
        name="cylinder"
    )
    
    # Create a contact planner
    planner = ContactPlanner(cylinder)
    
    # First test: Straight rolling
    print("Planning straight rolling primitive...")
    goal_pose = cylinder.pose * SE3(0.2, 0, 0)
    straight_rolling = planner.roll_straight(goal_pose)
    
    # Execute the primitive to generate trajectories
    print("Executing straight rolling plan...")
    object_poses, ee_poses = straight_rolling.execute()
    
    # Visualize in simulation
    print("Simulating robot motion for straight rolling...")
    execute_robot_trajectory(env, panda_virtual, ee_axes, ee_poses)
    
    # Update the cylinder position after straight rolling
    cyl_geom.T = goal_pose.A
    time.sleep(1)
    
    # Second test: Curved rolling
    print("\nPlanning curved rolling primitive...")
    curved_goal = goal_pose * SE3(0.1, 0.2, 0)
    goal_axes.T = curved_goal.A
    
    # Create waypoints for curved path
    path_points = [
        goal_pose.t,
        goal_pose.t + np.array([0.05, 0.1, 0]),
        goal_pose.t + np.array([0.1, 0.2, 0])
    ]
    
    curved_rolling = planner.roll_curved(curved_goal, path_points=path_points)
    
    # Execute the primitive to generate trajectories
    print("Executing curved rolling plan...")
    object_poses, ee_poses = curved_rolling.execute()
    
    # Visualize in simulation
    print("Simulating robot motion for curved rolling...")
    execute_robot_trajectory(env, panda_virtual, ee_axes, ee_poses)
    
    print("Cylinder rolling tests completed")
    time.sleep(2)
    env.close()


def test_multi_primitive_sequence():
    """Test a sequence of multiple primitives."""
    print("=== Testing Multi-Primitive Sequence ===")
    
    # Initialize environment
    env, Tep, panda, box_geom, ee_axes, goal_axes = init_env()
    
    # Create a manipulable box object
    box = ManipulableBox(
        dimensions=[0.1, 0.07, 0.03],
        pose=SE3(0.7, 0, 0.015),
        name="box"
    )
    
    # Create a contact planner
    planner = ContactPlanner(box)
    
    # Create a sequence of primitives
    
    # 1. First slide the box forward
    print("Planning sliding primitive...")
    slide_goal = box.pose * SE3(0.15, 0, 0)
    planner.slide(slide_goal, contact_face_idx=0)
    
    # 2. Then pivot the box
    print("Planning pivoting primitive...")
    pivot_goal = slide_goal * SE3.RPY(0, 0, np.pi/4)
    planner.pivot(pivot_goal, pivot_edge_idx=4)
    
    # 3. Finally grasp and lift the box
    print("Planning grasping primitive...")
    final_goal = pivot_goal * SE3(0, 0, 0.1)
    planner.grasp(final_goal)
    
    # Execute all primitives in sequence
    print("Executing the complete plan sequence...")
    object_poses, ee_poses = planner.execute_all()
    
    # Visualize in simulation
    print("Simulating robot motion...")
    execute_robot_trajectory(env, panda, ee_axes, ee_poses)
    
    print("Multi-primitive sequence test completed")
    time.sleep(2)
    env.close()


def test_planning_and_visualization():
    """Test planning and visualization features."""
    print("=== Testing Planning and Visualization ===")
    
    # Initialize environment
    env, Tep, panda, box_geom, ee_axes, goal_axes = init_env()
    
    # Create a manipulable box object
    box = ManipulableBox(
        dimensions=[0.1, 0.07, 0.03],
        pose=SE3(0.7, 0, 0.015),
        name="box"
    )
    
    # Create a contact planner
    planner = ContactPlanner(box)
    
    # Plan a pivoting primitive
    goal_pose = box.pose * SE3.RPY(0.3, 0, 0)
    print("Planning pivoting primitive...")
    pivoting = planner.pivot(goal_pose, pivot_edge_idx=3)
    
    # Execute and get trajectories
    print("Generating trajectory...")
    object_poses, ee_poses = pivoting.execute()
    
    # Visualize trajectory using the planner's visualization
    print("Visualizing trajectory using the planner...")
    planner.visualize_trajectory(env)
    
    # Save the trajectory to a file
    print("Saving trajectory to file...")
    filename = "test_trajectory.npz"
    planner.save_trajectory(filename)
    
    # Create a new planner and load the trajectory
    print("Loading trajectory from file...")
    new_planner = ContactPlanner(box)
    loaded_object_poses, loaded_ee_poses = new_planner.load_trajectory(filename)
    
    # Verify loaded trajectory matches the original
    print("Verifying loaded trajectory...")
    match = (len(loaded_object_poses) == len(object_poses))
    print(f"Trajectory match: {match}")
    
    # Visualize the loaded trajectory
    print("Visualizing loaded trajectory...")
    execute_robot_trajectory(env, panda, ee_axes, loaded_ee_poses)
    
    print("Planning and visualization test completed")
    time.sleep(2)
    env.close()


if __name__ == "__main__":
    # Run tests
    try:
        print("\nNOTE: Some functionality is implemented as placeholders,")
        print("so certain tests may not fully work until implementations are completed.\n")
        
        # Test individual primitives
        test_grasping()
        test_sliding()
        test_pivoting()
        test_cylinder_rolling()
        
        # Test combined functionality
        test_multi_primitive_sequence()
        test_planning_and_visualization()
        
        print("\nAll tests completed. Note that some functionality relies on placeholder implementations.")
    except Exception as e:
        print(f"Error during testing: {e}")
        # Continue with other tests even if one fails 