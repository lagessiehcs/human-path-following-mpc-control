import numpy as np

class DifferentialDrive:
    # Class variables for the distance between wheels and radius of wheels
    wheel_base = 0.5  # distance between wheels (m)
    wheel_radius = 0.1  # radius of wheels (m)
    
    def __init__(self, wheel_base=None, wheel_radius=None):
        if wheel_base is not None:
            DifferentialDrive.wheel_base = wheel_base
        if wheel_radius is not None:
            DifferentialDrive.wheel_radius = wheel_radius

    @staticmethod
    def inverse_kinematics(v, omega):
        b = DifferentialDrive.wheel_base
        r = DifferentialDrive.wheel_radius
        
        omega_L = (v - omega * b * 0.5) / r
        omega_R = (v + omega * b * 0.5) / r
        
        return omega_L, omega_R
    
    @staticmethod
    def forward_kinematics(v_L, v_R):
        b = DifferentialDrive.wheel_base
        r = DifferentialDrive.wheel_radius
        
        v = r * (v_L + v_R) * 0.5
        omega = r * (v_R - v_L) / b
        
        return v, omega

