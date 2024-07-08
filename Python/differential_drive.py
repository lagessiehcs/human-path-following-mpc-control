class DifferentialDrive:
    def __init__(self, wheel_base=0.5, wheel_radius=0.1):
        """
        Initialize the DifferentialDrive model.

        Parameters:
        - wheel_base (float): Distance between the wheels (default is 0.5 m).
        - wheel_radius (float): Radius of the wheels (default is 0.1 m).
        """
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius

    def inverse_kinematics(self, v, omega):
        """
        Calculate the wheel velocities given the linear and angular velocities.

        Parameters:
        - v (float): Linear velocity of the robot (m/s).
        - omega (float): Angular velocity of the robot (rad/s).

        Returns:
        - omega_L (float): Angular velocity of the left wheel (rad/s).
        - omega_R (float): Angular velocity of the right wheel (rad/s).
        """
        b = self.wheel_base
        r = self.wheel_radius
        
        omega_L = (v - omega * b * 0.5) / r
        omega_R = (v + omega * b * 0.5) / r
        
        return omega_L, omega_R
    
    def forward_kinematics(self, omega_L, omega_R):
        """
        Calculate the linear and angular velocities given the wheel velocities.

        Parameters:
        - omega_L (float): Angular velocity of the left wheel (rad/s).
        - omega_R (float): Angular velocity of the right wheel (rad/s).

        Returns:
        - v (float): Linear velocity of the robot (m/s).
        - omega (float): Angular velocity of the robot (rad/s).
        """
        b = self.wheel_base
        r = self.wheel_radius
        
        v = r * (omega_L + omega_R) * 0.5
        omega = r * (omega_R - omega_L) / b
        
        return v, omega
