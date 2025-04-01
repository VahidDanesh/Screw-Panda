import numpy as np
from spatialmath import SE3, SO3, UnitQuaternion, DualQuaternion
from spatialgeometry import Cuboid, Cylinder, Shape
import pytransform3d.rotations as pr3d
import pytransform3d.transformations as pt3d
import pytransform3d.coordinates as pc3d

class SpatialObject:
    """
    Base class for spatial objects with geometries that can be manipulated with environment contact.
    
    Attributes:
        geometry (Shape): The spatial geometry (e.g., Cuboid, Cylinder) from spatialgeometry.
        pose (SE3): The object's current pose as an SE3 transformation.
        name (str): A name identifier for the object.
        grasp_offset (SE3): Rigid transformation from object center to grasp point.
    """
    
    def __init__(self, pose=SE3(), name="object"):
        """Base initialization for spatial objects"""
        self.name = name
        self.pose = pose if isinstance(pose, SE3) else SE3(pose)
        # Default grasp offset is identity (grasp at center)
        self.grasp_offset = SE3()
        self.geometry = None
    
    @property
    def position(self):
        """Get the object's position (translation vector) as a numpy array."""
        return self.pose.t
    
    @property
    def orientation(self):
        """Get the object's orientation as a rotation matrix (3x3 numpy array)."""
        return self.pose.R
    
    @property
    def grasp_pose(self):
        """Get the pose at the grasp point."""
        return self.pose * self.grasp_offset
    
    def set_grasp_offset(self, offset):
        """
        Set the grasp offset from the center of the object.
        
        Args:
            offset (SE3): Rigid transformation from object center to grasp point.
        """
        self.grasp_offset = offset if isinstance(offset, SE3) else SE3(offset)
        return self
    
    def update_pose(self, pose):
        """
        Update the object's pose.
        
        Args:
            pose (SE3): New pose for the object.
        """
        self.pose = pose if isinstance(pose, SE3) else SE3(pose)
        if self.geometry:
            self.geometry.T = self.pose
        return self
    
    def to_pt3d_transform(self):
        """Convert the object's pose to pytransform3d transform."""
        T = np.eye(4)
        T[:3, :3] = self.orientation
        T[:3, 3] = self.position
        return T
    
    def from_pt3d_transform(self, T):
        """
        Update the object's pose from a pytransform3d transform.
        
        Args:
            T (numpy.ndarray): 4x4 homogeneous transformation matrix.
        """
        self.pose = SE3(T)
        if self.geometry:
            self.geometry.T = self.pose
        return self


class Box(SpatialObject):
    """
    A box object that can be manipulated with edge contact.
    
    Attributes:
        length (float): Length of the box (x dimension).
        width (float): Width of the box (y dimension).
        height (float): Height of the box (z dimension).
        edges (list): List of edge vectors in object frame.
    """
    
    def __init__(self, dimensions=(0.1, 0.07, 0.03), pose=SE3(), name="box"):
        """
        Initialize a Box object.
        
        Args:
            dimensions (tuple): (length, width, height) dimensions of the box.
            pose (SE3): Initial pose of the box.
            name (str): Name identifier for the box.
        """
        super().__init__(pose, name)
        
        if len(dimensions) != 3:
            raise ValueError("Box requires 3 dimensions: (length, width, height)")
        
        self.length, self.width, self.height = dimensions
        self.geometry = Cuboid(scale=dimensions, pose=self.pose)
        
        # Create a list of the 12 edges of the box in object frame
        # Each edge is a tuple of (start_point, end_point)
        self._create_edges()
        
        # Default grasp offset is at the middle of the back edge (negative x direction)
        bTe = SE3(-self.length/2 + 0.01, 0, 0) * SE3.Rx(np.pi/2)
        self.set_grasp_offset(bTe)
    
    def _create_edges(self):
        """Create the edges of the box in object frame."""
        # Half-dimensions for easier calculations
        hl, hw, hh = self.length/2, self.width/2, self.height/2
        
        # 8 corners of the box, x direction if pointing forward, y direction if pointing left, z direction if pointing up
        corners = [
            (-hl, hw, -hh),     # 0: back-left-bottom
            (-hl, -hw, -hh),    # 1: back-right-bottom
            (hl, -hw, -hh),     # 2: front-right-bottom
            (hl, hw, -hh),     # 3: front-left-bottom
            (-hl, hw, hh),      # 4: back-left-top
            (-hl, -hw, hh),     # 5: back-right-top
            (hl, -hw, hh),      # 6: front-right-top
            (hl, hw, hh)        # 7: front-left-top
        ]
        
        # 12 edges defined by pairs of corner indices
        edge_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        
        self.edges = [(np.array(corners[i]), np.array(corners[j])) for i, j in edge_indices]
    
    def get_edge_in_world(self, edge_idx):
        """
        Get the edge in world coordinates.
        
        Args:
            edge_idx (int): Index of the edge to get (1-12).
            edge_idx = {
            1: bottom back, 2: bottom right, 3: bottom front, 4: bottom left, 
            5: top back, 6: top right, 7: top front, 8: top left, 
            9: back left, 10: back right, 11: front right, 12: front left
            }
            
        Returns:
            tuple: (start_point, end_point) in world coordinates.
        """
        if edge_idx < 0 or edge_idx >= len(self.edges):
            raise ValueError(f"Edge index must be between 1 and {len(self.edges)}")
            
        start_local, end_local = self.edges[edge_idx - 1]
        
        # Transform to world frame
        start_world = self.pose * start_local
        end_world = self.pose * end_local
        
        return start_world, end_world
    
    def get_face_normal(self, face_idx):
        """
        Get the normal vector of a face in world coordinates.
        
        Args:
            face_idx (int): Index of the face (0=bottom, 1=top, 2=front, 3=back, 4=left, 5=right)
            
        Returns:
            numpy.ndarray: Normal vector in world coordinates.
        """
        # Face normals in object frame
        normals = [
            np.array([0, 0, -1]),   # bottom (-z)
            np.array([0, 0, 1]),    # top (+z)
            np.array([1, 0, 0]),    # front (+x)
            np.array([-1, 0, 0]),   # back (-x)
            np.array([0, 1, 0]),    # left (+y)
            np.array([0, -1, 0])    # right (-y)
        ]
        
        if face_idx < 0 or face_idx >= len(normals):
            raise ValueError(f"Face index must be between 0 and {len(normals)-1}")
        
        # Transform normal to world frame (only rotation)
        normal_local = normals[face_idx]
        normal_world = self.orientation @ normal_local
        
        return normal_world
    
    def get_edge_contact_pose(self, edge_idx, contact_param=0.5):
        """
        Get the pose at a contact point along an edge.
        
        Args:
            edge_idx (int): Index of the edge to contact (1-12).
            contact_param (float): Parameter along the edge (0 = start, 1 = end, 0.5 = middle).
            
        Returns:
            SE3: Pose at the contact point.
        """
        start_world, end_world = self.get_edge_in_world(edge_idx)
        
        # Interpolate along the edge
        contact_point = start_world + contact_param * (end_world - start_world)
        
        # TODO: Calculate orientation based on edge direction
        
        # For now, just use object orientation
        return SE3(contact_point) * SE3(SO3(self.orientation))


class Cylinder(SpatialObject):
    """
    A cylinder object that can be manipulated with point or line contact.
    
    Attributes:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
    """
    
    def __init__(self, radius=0.02, height=0.1, pose=SE3(), name="cylinder"):
        """
        Initialize a Cylinder object.
        
        Args:
            radius (float): Radius of the cylinder.
            height (float): Height of the cylinder.
            pose (SE3): Initial pose of the cylinder.
            name (str): Name identifier for the cylinder.
        """
        super().__init__(pose, name)
        
        self.radius = radius
        self.height = height
        self.geometry = Cylinder(radius=radius, height=height, pose=self.pose)
        
        # Default grasp offset is at the top face center
        self.set_grasp_offset(SE3(0, 0, self.height/2))
    
    def get_rim_point(self, angle, on_top=True):
        """
        Get a point on the rim of the cylinder.
        
        Args:
            angle (float): Angle in radians around the cylinder axis.
            on_top (bool): Whether to get a point on the top rim (True) or bottom rim (False).
            
        Returns:
            numpy.ndarray: Point coordinates in world frame.
        """
        # Point on rim in object frame
        z = self.height/2 if on_top else -self.height/2
        local_point = np.array([
            self.radius * np.cos(angle),
            self.radius * np.sin(angle),
            z
        ])
        
        # Transform to world frame
        world_point = self.pose * local_point
        
        return world_point
    
    def get_side_point(self, angle, height_param=0.5):
        """
        Get a point on the side of the cylinder.
        
        Args:
            angle (float): Angle in radians around the cylinder axis.
            height_param (float): Parameter along the height (0 = bottom, 1 = top, 0.5 = middle).
            
        Returns:
            numpy.ndarray: Point coordinates in world frame.
        """
        # Point on side in object frame
        z = -self.height/2 + height_param * self.height
        local_point = np.array([
            self.radius * np.cos(angle),
            self.radius * np.sin(angle),
            z
        ])
        
        # Transform to world frame
        world_point = self.pose * local_point
        
        return world_point
    
    def get_axis_vector(self):
        """
        Get the axis vector of the cylinder in world coordinates.
        
        Returns:
            numpy.ndarray: Unit vector along the cylinder axis in world frame.
        """
        # Cylinder axis is along z in object frame
        local_axis = np.array([0, 0, 1])
        
        # Transform to world frame (only rotation)
        world_axis = self.orientation @ local_axis
        
        return world_axis
    
    def get_rolling_contact_pose(self, angle, contact_height=0):
        """
        Get the pose at a rolling contact point.
        
        Args:
            angle (float): Angle in radians around the cylinder axis.
            contact_height (float): Height parameter for the contact point.
            
        Returns:
            SE3: Pose at the rolling contact point.
        """
        # Get contact point
        height_param = 0.5 + contact_height/self.height
        contact_point = self.get_side_point(angle, height_param)
        
        # Calculate orientation: 
        # - Z axis is along the cylinder axis
        # - X axis points outward from the cylinder center at the given angle
        # - Y axis completes the right-handed frame
        
        z_axis = self.get_axis_vector()
        local_x = np.array([np.cos(angle), np.sin(angle), 0])
        world_x = self.orientation @ local_x
        y_axis = np.cross(z_axis, world_x)
        x_axis = np.cross(y_axis, z_axis)
        
        # Normalize axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Create rotation matrix
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        return SE3(contact_point) * SE3(R=R) 