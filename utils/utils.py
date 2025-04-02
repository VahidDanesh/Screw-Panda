import numpy as np

def vec_angle(v1, v2, n=np.array([0, 0, 1])):
    """
    Calculate the angle between two vectors.
    """
    x = np.cross(v1, v2)
    c = np.sign(np.dot(x, n)) * np.linalg.norm(x)
    a = np.arctan2(c, np.dot(v1, v2))
    return a
