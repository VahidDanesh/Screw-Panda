import numpy as np
from spatialmath import SE3
import pytransform3d.transformations as pt3d
from .base import ManipulationPrimitive

class StraightRolling(ManipulationPrimitive):
    """
    A manipulation primitive for rolling a cylindrical object along a straight line.
    
    This primitive generates a trajectory that rolls a cylinder along a straight
    line on a supporting surface.
    """
    
    def __init__(self, obj, goal_pose, duration=2.0, frequency=100):
        """
        Initialize a straight rolling primitive.
        
        Args:
            obj (SpatialObject): The cylindrical object to roll.
            goal_pose (SE3): Goal pose for the object.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        
        # Verify that the object is a cylinder
        if not hasattr(obj, 'radius'):
            raise TypeError("Object must be a cylinder for rolling primitive")
        
        # Calculate the rolling path
        self._calculate_rolling_path()
    
    def _calculate_rolling_path(self):
        """
        Calculate the straight rolling path.
        
        For rolling, we need to ensure that:
        1. The cylinder maintains contact with the supporting surface
        2. The rotation is proportional to the distance traveled
        """
        # Extract start and goal positions
        start_pos = self.start_pose.t
        goal_pos = self.goal_pose.t
        
        # Calculate the rolling direction and distance
        direction_xy = goal_pos[:2] - start_pos[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy < 1e-6:
            # No significant horizontal movement, no rolling needed
            self.rolling_direction = np.array([1, 0])  # Default direction
            self.rolling_distance = 0
            self.rolling_angle = 0
        else:
            # Normalize the rolling direction
            self.rolling_direction = direction_xy / distance_xy
            self.rolling_distance = distance_xy
            
            # Calculate the rolling angle (radians)
            # For a cylinder, rolling angle = distance / radius
            self.rolling_angle = self.rolling_distance / self.object.radius
    
    def plan(self):
        """
        Plan a trajectory for rolling the cylindrical object.
        
        The rolling motion ensures that:
        1. The cylinder maintains contact with the supporting surface
        2. The cylinder rotates proportionally to the distance traveled
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Get time scaling
        tau, _ = self._set_time_scaling(method="quintic")
        
        # Extract start components
        start_pos = self.start_pose.t
        start_rot = self.start_pose.R
        
        # Generate object poses
        self._object_poses = []
        self._object_dquats = []
        
        # Create rotation axis (perpendicular to rolling direction)
        rotation_axis = np.array([-self.rolling_direction[1], self.rolling_direction[0], 0])
        
        for t in tau:
            # Calculate the distance traveled at this step
            distance = t * self.rolling_distance
            
            # Calculate the rolling angle at this step
            angle = t * self.rolling_angle
            
            # Calculate the new position
            new_pos = start_pos.copy()
            new_pos[0] += distance * self.rolling_direction[0]
            new_pos[1] += distance * self.rolling_direction[1]
            
            # Create the rotation for rolling
            roll_matrix = pt3d.matrix_from_axis_angle(
                np.hstack([rotation_axis, angle])
            )
            
            # Combine with the starting rotation
            new_rot = roll_matrix @ start_rot
            
            # Create the pose
            pose = SE3(new_pos) * SE3(R=new_rot)
            self._object_poses.append(pose)
            
            # Convert to dual quaternion
            dq = pt3d.dual_quaternion_from_transform(pose.A)
            self._object_dquats.append(dq)
        
        # Calculate the end-effector poses
        self._ee_poses = []
        self._ee_dquats = []
        
        for obj_pose in self._object_poses:
            # Calculate the grasp pose at each step
            ee_pose = obj_pose * self.object.grasp_offset
            self._ee_poses.append(ee_pose)
            
            # Convert to dual quaternion for the end-effector
            ee_dquat = pt3d.dual_quaternion_from_transform(ee_pose.A)
            self._ee_dquats.append(ee_dquat)
        
        return self._object_poses, self._ee_poses


class CurvedRolling(ManipulationPrimitive):
    """
    A manipulation primitive for rolling a cylindrical object along a curved path.
    
    This primitive generates a trajectory that rolls a cylinder along a curved
    path on a supporting surface, with pivoting to adjust orientation.
    """
    
    def __init__(self, obj, goal_pose, path_points=None, 
                 duration=2.0, frequency=100):
        """
        Initialize a curved rolling primitive.
        
        Args:
            obj (SpatialObject): The cylindrical object to roll.
            goal_pose (SE3): Goal pose for the object.
            path_points (list, optional): List of waypoints defining the curved path.
                If None, a path will be generated from start to goal.
            duration (float): Duration of the motion in seconds.
            frequency (int): Sampling frequency for the trajectory in Hz.
        """
        super().__init__(obj, goal_pose, duration, frequency)
        
        # Verify that the object is a cylinder
        if not hasattr(obj, 'radius'):
            raise TypeError("Object must be a cylinder for rolling primitive")
        
        # Store path points or generate them
        self.path_points = path_points
        if path_points is None:
            self._generate_path_points()
        
        # Calculate the rolling segments
        self._calculate_rolling_segments()
    
    def _generate_path_points(self):
        """
        Generate waypoints for a curved path from start to goal.
        
        This method creates a simple curved path by adding an intermediate point.
        For more complex paths, the user should provide custom waypoints.
        """
        start_pos = self.start_pose.t
        goal_pos = self.goal_pose.t
        
        # Calculate midpoint with an offset for curvature
        direction = goal_pos - start_pos
        perpendicular = np.array([-direction[1], direction[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        midpoint = (start_pos + goal_pos) / 2 + perpendicular * np.linalg.norm(direction) * 0.25
        midpoint[2] = start_pos[2]  # Maintain same height
        
        # Create waypoints
        self.path_points = [start_pos, midpoint, goal_pos]
    
    def _calculate_rolling_segments(self):
        """
        Calculate the rolling segments between waypoints.
        
        For each segment, calculate:
        1. The rolling direction
        2. The rolling distance
        3. The required pivot at the waypoint
        """
        self.segments = []
        
        for i in range(len(self.path_points) - 1):
            # Extract segment start and end
            start = self.path_points[i]
            end = self.path_points[i + 1]
            
            # Calculate direction and distance
            direction_xy = end[:2] - start[:2]
            distance_xy = np.linalg.norm(direction_xy)
            
            if distance_xy < 1e-6:
                # Skip negligible segments
                continue
            
            # Normalize direction
            direction = direction_xy / distance_xy
            
            # Calculate rolling angle
            angle = distance_xy / self.object.radius
            
            # Store segment information
            self.segments.append({
                'start': start,
                'end': end,
                'direction': np.array([direction[0], direction[1], 0]),
                'distance': distance_xy,
                'angle': angle
            })
            
        # Calculate required pivots at waypoints
        for i in range(len(self.segments) - 1):
            # Extract segments
            curr_segment = self.segments[i]
            next_segment = self.segments[i + 1]
            
            # Calculate the change in direction
            curr_dir = curr_segment['direction']
            next_dir = next_segment['direction']
            
            # Calculate the pivot angle needed (rotate from curr_dir to next_dir)
            pivot_angle = np.arctan2(
                curr_dir[0] * next_dir[1] - curr_dir[1] * next_dir[0],
                curr_dir[0] * next_dir[0] + curr_dir[1] * next_dir[1]
            )
            
            # Store the pivot information in the current segment
            curr_segment['pivot_angle'] = pivot_angle
    
    def plan(self):
        """
        Plan a trajectory for rolling the cylindrical object along a curved path.
        
        The curved rolling motion consists of:
        1. Rolling along straight segments
        2. Pivoting at waypoints to adjust orientation
        
        Returns:
            tuple: (object_poses, ee_poses) Lists of object and end-effector poses.
        """
        # Get time scaling
        tau, _ = self._set_time_scaling(method="quintic")
        
        # Distribute time steps among segments proportionally to segment length
        total_distance = sum(segment['distance'] for segment in self.segments)
        segment_times = []
        
        for segment in self.segments:
            # Allocate time steps proportionally to distance
            segment_time = segment['distance'] / total_distance
            segment_times.append(segment_time)
        
        # Normalize segment times
        segment_times = np.array(segment_times)
        segment_times = segment_times / np.sum(segment_times)
        
        # Cumulative segment times
        cum_segment_times = np.cumsum(segment_times)
        
        # Extract start components
        start_pose = self.start_pose
        
        # Generate object poses
        self._object_poses = []
        self._object_dquats = []
        
        # Initialize current pose
        current_pose = start_pose
        
        for t in tau:
            # Determine which segment this time step belongs to
            segment_idx = np.searchsorted(cum_segment_times, t)
            
            if segment_idx >= len(self.segments):
                # Use the last segment if t is exactly 1.0
                segment_idx = len(self.segments) - 1
            
            # Get the current segment
            segment = self.segments[segment_idx]
            
            # Calculate the local time within this segment
            if segment_idx == 0:
                local_t = t / cum_segment_times[0]
            else:
                local_t = (t - cum_segment_times[segment_idx - 1]) / (cum_segment_times[segment_idx] - cum_segment_times[segment_idx - 1])
            
            # Constrain local_t to [0, 1]
            local_t = max(0, min(1, local_t))
            
            # Calculate new position along the segment
            direction = segment['direction']
            distance = local_t * segment['distance']
            
            new_pos = segment['start'] + distance * direction
            
            # Create rotation axis (perpendicular to rolling direction)
            rotation_axis = np.array([-direction[1], direction[0], 0])
            
            # Calculate the rolling angle
            angle = local_t * segment['angle']
            
            # Create the rotation for rolling
            roll_matrix = pt3d.matrix_from_axis_angle(
                np.hstack([rotation_axis, angle])
            )
            
            # For the first segment, combine with the starting rotation
            if segment_idx == 0:
                base_rot = start_pose.R
            else:
                # For subsequent segments, account for pivoting
                # This is a simplified approach - a more physically accurate
                # implementation would need to account for the exact pivot mechanics
                base_rot = self._object_poses[-1].R
            
            # Create the pose
            new_rot = roll_matrix @ base_rot
            pose = SE3(new_pos) * SE3(R=new_rot)
            
            self._object_poses.append(pose)
            
            # Convert to dual quaternion
            dq = pt3d.dual_quaternion_from_transform(pose.A)
            self._object_dquats.append(dq)
        
        # Calculate the end-effector poses
        self._ee_poses = []
        self._ee_dquats = []
        
        for obj_pose in self._object_poses:
            # Calculate the grasp pose at each step
            ee_pose = obj_pose * self.object.grasp_offset
            self._ee_poses.append(ee_pose)
            
            # Convert to dual quaternion for the end-effector
            ee_dquat = pt3d.dual_quaternion_from_transform(ee_pose.A)
            self._ee_dquats.append(ee_dquat)
        
        return self._object_poses, self._ee_poses 