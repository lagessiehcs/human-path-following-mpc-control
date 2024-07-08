import numpy as np

def Rz(rad: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for a rotation around the z-axis by `rad` radians.

    Parameters:
    rad : float
        Rotation angle in radians.

    Returns:
    R : np.ndarray
        3x3 rotation matrix.
    """
    tol = np.finfo(float).eps
    
    ct = np.cos(rad)
    st = np.sin(rad)
    
    # Make almost zero elements exactly zero
    if abs(st) < tol:
        st = 0
    if abs(ct) < tol:
        ct = 0

    # Create the rotation matrix
    R = np.array([
        [ct, -st,  0],
        [st,  ct,  0],
        [0,    0,  1]
    ])
    return R

def Tz(rad: float, t: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix for a rotation around the z-axis
    by `rad` radians and a translation by `t`.

    Parameters:
    rad : float
        Rotation angle in radians.
    t : np.ndarray
        Translation vector [tx, ty].

    Returns:
    T : np.ndarray
        4x4 homogeneous transformation matrix.
    """
    tol = np.finfo(float).eps
    
    ct = np.cos(rad)
    st = np.sin(rad)
    
    # Make almost zero elements exactly zero
    if abs(st) < tol:
        st = 0
    if abs(ct) < tol:
        ct = 0

    # Create the homogeneous transformation matrix
    T = np.array([
        [ct, -st,  0, t[0]],
        [st,  ct,  0, t[1]],
        [0,    0,  1,    0],
        [0,    0,  0,    1]
    ])
    
    return T

def body_to_world(velB: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Convert a 2D velocity input velB = [vx; vy; w] from body (vehicle) 
    coordinates to world (global) coordinates.

    Parameters:
    velB : np.ndarray
        2D velocity input in body coordinates [vx, vy, w].
    pose : np.ndarray
        Pose of the vehicle [x, y, theta] where theta is the orientation.

    Returns:
    velW : np.ndarray
        Velocity in world coordinates [vx_w, vy_w, w_w].
    """
    theta = pose[2]
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    velW = np.dot(rotation_matrix, velB)
    
    return velW
