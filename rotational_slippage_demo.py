#!/usr/bin/env python3
"""
Demonstration of rotational slippage using the Screw-Panda framework.
This example shows how to rotate a box around a screw axis that passes
through the bottom back edge, and computes the corresponding end-effector
trajectory for a Panda robot.
"""

import numpy as np
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
import pytransform3d.transformations as pt3d

# Import from Screw-Panda framework
from utils.objects import ManipulableBox, SpatialObject
from utils.screw_utils import generate_trajectory, pose_to_screw_parameters, twist_from_poses


def create_box_and_screw_axis():
    """Create a box and define a screw axis through its bottom back edge."""
    # Create box with dimensions (x, y, z) in meters
    box = ManipulableBox(dimensions=(0.1, 0.07, 0.03), name="target_box")
    
    # Initial pose - slightly above the ground
    initial_pose = SE3(0.5, 0, 0.03) * SE3.Rz(0)
    box.update_pose(initial_pose)
    
    # Get the bottom back edge (edge 1 in the ManipulableBox class)
    # The indices are 1-based in the get_edge_in_world method
    edge_start, edge_end = box.get_edge_in_world(1)
    
    # Middle point of the edge as the point on the screw axis
    point_on_axis = (edge_start + edge_end) / 2
    
    # Direction of the edge - this will be our screw axis direction
    axis_direction = edge_end - edge_start
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    
    print(f"Box dimensions: {box.length} x {box.width} x {box.height} meters")
    print(f"Initial pose: \n{box.pose}")
    print(f"Bottom back edge: from {edge_start} to {edge_end}")
    print(f"Point on screw axis: {point_on_axis}")
    print(f"Screw axis direction: {axis_direction}")
    
    return box, point_on_axis, axis_direction


def create_screw_rotation(box, point_on_axis, axis_direction, angle_deg=45):
    """
    Create a screw motion that rotates the box around the defined axis.
    
    Args:
        box: The ManipulableBox object
        point_on_axis: A point on the screw axis
        axis_direction: Direction vector of the screw axis
        angle_deg: Rotation angle in degrees
        
    Returns:
        Final pose after rotation
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create a rotation matrix around the axis direction
    # Use SO3.AngleAxis to create rotation around arbitrary axis
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
    end_pose = SE3(R, t) * start_pose
    
    return end_pose


def visualize_box_trajectory(box, poses, ax=None):
    """Visualize the box and its trajectory."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract box corners for each pose
    corners_trajectories = []
    
    # The indices are as follows (from ManipulableBox._create_edges):
    # 0: back-left-bottom, 1: back-right-bottom, 2: front-right-bottom, 3: front-left-bottom
    # 4: back-left-top, 5: back-right-top, 6: front-right-top, 7: front-left-top
    
    # Get corner coordinates in local frame
    hl, hw, hh = box.length/2, box.width/2, box.height/2
    local_corners = [
        np.array([-hl, hw, -hh]),     # 0: back-left-bottom
        np.array([-hl, -hw, -hh]),    # 1: back-right-bottom
        np.array([hl, -hw, -hh]),     # 2: front-right-bottom
        np.array([hl, hw, -hh]),     # 3: front-left-bottom
        np.array([-hl, hw, hh]),      # 4: back-left-top
        np.array([-hl, -hw, hh]),     # 5: back-right-top
        np.array([hl, -hw, hh]),      # 6: front-right-top
        np.array([hl, hw, hh])        # 7: front-left-top
    ]
    
    # For each pose, transform all corners
    for pose in poses:
        transformed_corners = [pose * corner for corner in local_corners]
        corners_trajectories.append(transformed_corners)
    
    # Plot the trajectories of the corners
    for i in range(8):
        points = np.array([corners[i] for corners in corners_trajectories])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.3)
    
    # Plot the initial and final box
    for idx, corners in [(0, 'initial'), (-1, 'final')]:
        # Get corners from the corresponding pose
        box_corners = corners_trajectories[idx]
        
        # Define box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        
        # Plot each edge
        for i, j in edges:
            xs = [box_corners[i][0], box_corners[j][0]]
            ys = [box_corners[i][1], box_corners[j][1]]
            zs = [box_corners[i][2], box_corners[j][2]]
            
            if idx == 0:  # Initial box
                ax.plot(xs, ys, zs, 'g-', linewidth=2)
            else:  # Final box
                ax.plot(xs, ys, zs, 'r-', linewidth=2)
    
    # Plot the grasp point trajectory
    grasp_trajectory = [pose * box.grasp_offset for pose in poses]
    grasp_points = np.array([pose.t for pose in grasp_trajectory])
    ax.plot(grasp_points[:, 0], grasp_points[:, 1], grasp_points[:, 2], 'm-', linewidth=3, label='Grasp Point')
    
    # Improve the plot appearance
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Box Rotation around Edge')
    ax.legend()
    
    # Make axis equal
    limits = []
    for dim in [0, 1, 2]:
        min_val = np.min([c[dim] for corners in corners_trajectories for c in corners])
        max_val = np.max([c[dim] for corners in corners_trajectories for c in corners])
        limits.append((min_val, max_val))
    
    # Find the range
    ranges = [max_val - min_val for min_val, max_val in limits]
    max_range = max(ranges)
    
    # Set equal aspect ratio
    for i, (min_val, max_val) in enumerate(limits):
        mid = (min_val + max_val) / 2
        ax.set_xlim3d(mid - max_range/2, mid + max_range/2) if i==0 else None
        ax.set_ylim3d(mid - max_range/2, mid + max_range/2) if i==1 else None
        ax.set_zlim3d(min_val, max_val) if i==2 else None
    
    return ax


def calculate_robot_trajectory(box, box_poses):
    """
    Calculate the robot end-effector trajectory based on box poses.
    
    Returns:
        tuple: (ee_poses, ee_twists) End-effector poses and twists
    """
    # Initialize lists to store end-effector poses and twists
    ee_poses = []
    ee_twists = []
    
    # For each box pose, calculate the corresponding end-effector pose
    for i, box_pose in enumerate(box_poses):
        # Get the grasp pose
        grasp_offset = box.grasp_offset
        ee_pose = box_pose * grasp_offset
        ee_poses.append(ee_pose)
        
        # Calculate twist if not the first pose
        if i > 0:
            prev_ee_pose = ee_poses[i-1]
            # Assuming uniform time steps (dt=1)
            twist = twist_from_poses(prev_ee_pose, ee_pose)
            ee_twists.append(twist)
    
    return ee_poses, ee_twists


def demonstrate_robot_motion(ee_poses, ee_twists):
    """
    Demonstrate the robot motion using the calculated end-effector trajectory.
    Since Swift browser doesn't work on WSL, we'll use matplotlib for visualization.
    """
    # Create a Panda robot
    panda = rtb.models.Panda()
    
    # Calculate IK for each pose in the trajectory
    q_trajectory = []
    q_current = panda.qr  # Start from ready pose
    
    for ee_pose in ee_poses:
        # Use inverse kinematics to find joint angles
        sol = panda.ik_lm(ee_pose, q0=q_current)
        
        if sol[0]:  # If IK found a solution
            q_current = sol[1]
            q_trajectory.append(q_current)
        else:
            print(f"Warning: IK failed for pose {ee_pose}")
            # Use previous joint angles if IK fails
            q_trajectory.append(q_current)
    
    # Visualize the robot at selected points in the trajectory
    fig = plt.figure(figsize=(15, 10))
    
    # Show 5 key poses
    num_poses = len(q_trajectory)
    for i, idx in enumerate([0, num_poses//4, num_poses//2, 3*num_poses//4, -1]):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        panda.plot(q_trajectory[idx], ax=ax, jointaxes=False, eeframe=True)
        ax.set_title(f"Step {idx}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate rotational slippage."""
    print("=" * 50)
    print("Rotational Slippage Demonstration using Screw Theory")
    print("=" * 50)
    
    # Create a box and define the screw axis
    box, point_on_axis, axis_direction = create_box_and_screw_axis()
    
    # Create the final pose after rotation
    angle_deg = 70  # Rotation angle in degrees
    final_pose = create_screw_rotation(box, point_on_axis, axis_direction, angle_deg)
    
    # Generate a trajectory between the initial and final poses
    num_steps = 30
    box_poses, box_twists = generate_trajectory(box.pose, final_pose, num_steps)
    
    # Visualize the box trajectory
    plt.figure(figsize=(10, 8))
    ax = visualize_box_trajectory(box, box_poses)
    plt.show()
    
    # Calculate the robot end-effector trajectory
    ee_poses, ee_twists = calculate_robot_trajectory(box, box_poses)
    
    print("\nTrajectory information:")
    print(f"  Generated {len(box_poses)} poses for the box")
    print(f"  Starting box pose: {box.pose}")
    print(f"  Final box pose: {final_pose}")
    print(f"  Starting grasp pose: {box.pose * box.grasp_offset}")
    print(f"  Final grasp pose: {final_pose * box.grasp_offset}")
    
    # Extract screw parameters for the overall motion
    screw_params = pose_to_screw_parameters(box.pose, final_pose)
    print("\nScrew parameters for the overall motion:")
    print(f"  Rotation angle: {screw_params['theta']:.4f} rad ({np.degrees(screw_params['theta']):.2f} deg)")
    print(f"  Translation distance: {screw_params['d']:.4f} m")
    print(f"  Screw axis direction: {screw_params['axis']}")
    print(f"  Point on axis: {screw_params['point']}")
    print(f"  Pitch: {screw_params['pitch']:.4f}")
    
    # Demonstrate robot motion
    demonstrate_robot_motion(ee_poses, ee_twists)


if __name__ == "__main__":
    # Import SO3 here to avoid circular imports
    from spatialmath import SO3
    main() 