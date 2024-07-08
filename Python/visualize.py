import matplotlib.pyplot as plt
from numpy import sin, cos
import numpy as np

class Visualize:
    def __init__(self):
        """
        Initialize the visualization environment.
        """
        self.fig, self.ax = plt.subplots()
        self.xMin = 0
        self.xMax = 0
        self.yMin = 0
        self.yMax = 0

    def setup_plot(self, waypoints: np.ndarray, shape: str):
        """
        Setup the plot dimensions and scale based on waypoints and the shape of the path.

        Parameters:
        waypoints : np.ndarray
            Array of waypoints (x, y) for the path.
        shape : str
            The shape of the path (e.g., 'Straight', 'Sample').
        """
        self.xMax = 1.2 * np.max(waypoints[:, 0])
        self.xMin = np.min(waypoints[:, 0]) - 0.2 * np.abs(np.max(waypoints[:, 0]))
        offset = 0
        if shape == "Straight":
            offset = 5
        self.yMax = 1.3 * np.max(waypoints[:, 1]) + offset
        self.yMin = np.min(waypoints[:, 1]) - 0.3 * np.abs(np.max(waypoints[:, 1])) - offset
        self.ax.set_aspect('equal')
        plt.xlim(self.xMin, self.xMax)
        plt.ylim(self.yMin, self.yMax)

    def update_plot(self, waypoints: np.ndarray, positions: np.ndarray, current_orientation: float, sampleTime: float):
        """
        Update the plot with the current position and orientation of the robot.

        Parameters:
        waypoints : np.ndarray
            Array of waypoints (x, y) for the path.
        positions : np.ndarray
            Array of past positions (x, y) of the robot.
        current_orientation : float
            Current orientation (theta) of the robot in radians.
        sampleTime : float
            Time interval between updates for animation effect.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        theta = current_orientation

        # Clear previous patches and lines
        self.ax.clear()

        # Set plot limits and aspect ratio again
        self.ax.set_aspect('equal')
        plt.xlim(self.xMin, self.xMax)
        plt.ylim(self.yMin, self.yMax)

        # Plot the path of the robot
        self.ax.plot(x, y, 'b.-', linewidth=1, markersize=2)
        self.ax.plot(waypoints[:, 0], waypoints[:, 1], 'r.', markersize=2)
        self.ax.plot(x[-1], y[-1], 'b.', markersize=14)  # Highlight the current position

        # Calculate and draw the new line indicating orientation
        line_length = 1
        end_x = x[-1] + line_length * cos(theta)
        end_y = y[-1] + line_length * sin(theta)
        self.ax.plot([x[-1], end_x], [y[-1], end_y], 'b-', linewidth=2)

        # Redraw the plot
        plt.draw()
        plt.pause(sampleTime)  # Pause to create an animation effect
