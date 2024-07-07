import numpy as np

def Rz(rad):
    tol = np.finfo(float).eps
    
    ct = np.cos(rad)
    st = np.sin(rad)
    
    # Make almost zero elements exactly zero
    if abs(st) < tol:
        st = 0
    if abs(ct) < tol:
        ct = 0

    # Create the rotational matrix
    R = np.array([
        [ct, -st,  0],
        [st,  ct,  0],
        [0,    0,  1]
    ])
    return R

def Tz(rad, t):
    tol = np.finfo(float).eps
    
    ct = np.cos(rad)
    st = np.sin(rad)
    
    # Make almost zero elements exactly zero
    if abs(st) < tol:
        st = 0
    if abs(ct) < tol:
        ct = 0

    # Create the homogenous transformation matrix
    T = np.array([
        [ct, -st,  0, t[0]],
        [st,  ct,  0, t[1]],
        [0,    0,  1,    0],
        [0,    0,  0,    1]
    ])
    
    return T

import numpy as np

def body_to_world(velB, pose):
    """
    Converts a 2D velocity input velB = [vx;vy;w] from body (vehicle) 
    coordinates to world (global) coordinates.

    Parameters:
    velB : numpy array
        2D velocity input in body coordinates [vx, vy, w].
    pose : numpy array
        Pose of the vehicle [x, y, theta] where theta is the orientation.

    Returns:
    velW : numpy array
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


