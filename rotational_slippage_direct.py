#!/usr/bin/env python3
"""
Demonstration of rotational slippage using direct spatialmath and pytransform3d functions.
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

# Import only the box object from the Screw-Panda framework
from utils.objects import ManipulableBox


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
    R = SO3.AngleAxis(angle_rad, axis_direction).R
    
    # Current pose of the box
    start_pose = box.pose
    
    # We need to transform the rotation to be around the point on the axis
    # rather than around the origin
    p = point_on_axis
    
    # Translation part: p - R*p gives the translation needed
    # to keep the point p fixed during rotation
    t = p - R @ p
    
    # Create the final transformation - using correct SE3 constructor format
    end_pose = SE3.Rt(R, t) * start_pose
    
    return end_pose


def visualize_box_trajectory(box, poses, ax=None):
    """Visualize the box and its trajectory."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract box corners for each pose
    corners_trajectories = []
    
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
    
    # Plot the screw axis
    edge_start, edge_end = box.get_edge_in_world(1)
    ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], [edge_start[2], edge_end[2]], 
            'y-', linewidth=3, label='Screw Axis')
    
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


def calculate_screw_parameters(start_pose, end_pose):
    """
    Extract screw parameters between two poses.
    
    Args:
        start_pose (SE3): Starting transformation
        end_pose (SE3): Ending transformation
        
    Returns:
        dict: Screw parameters
            - theta: rotation angle
            - d: translation distance
            - axis: screw axis
            - point: point on the axis
            - pitch: screw pitch
    """
    # Get relative transform from start to end
    T_rel = end_pose * start_pose.inv()
    
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


def calculate_twist_from_poses(pose1, pose2, dt=1.0):
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
            twist = calculate_twist_from_poses(prev_ee_pose, ee_pose)
            ee_twists.append(twist)
    
    return ee_poses, ee_twists


def main():
    """Main function to demonstrate rotational slippage."""
    print("=" * 50)
    print("Rotational Slippage Demonstration using Direct Library Functions")
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
    screw_params = calculate_screw_parameters(box.pose, final_pose)
    print("\nScrew parameters for the overall motion:")
    print(f"  Rotation angle: {screw_params['theta']:.4f} rad ({np.degrees(screw_params['theta']):.2f} deg)")
    print(f"  Translation distance: {screw_params['d']:.4f} m")
    print(f"  Screw axis direction: {screw_params['axis']}")
    print(f"  Point on axis: {screw_params['point']}")
    print(f"  Pitch: {screw_params['pitch']:.4f}")
    
    # Skip robot visualization due to potential issues with roboticstoolbox
    print("\nEnd-effector trajectory information:")
    print(f"  Number of poses: {len(ee_poses)}")
    print(f"  Starting ee pose: {ee_poses[0]}")
    print(f"  Final ee pose: {ee_poses[-1]}")
    
    # Calculate the first twist
    if len(ee_twists) > 0:
        print(f"\nFirst twist vector (spatial velocity):")
        print(f"  Linear: {ee_twists[0][:3]} m/s")
        print(f"  Angular: {ee_twists[0][3:]} rad/s")


if __name__ == "__main__":
    main() 