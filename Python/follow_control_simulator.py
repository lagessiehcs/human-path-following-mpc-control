import numpy as np
import matplotlib.pyplot as plt
from Paths.path import gen_path, arclength
from mpc_optimizer import MpcOptimizer
from transform import body_to_world, Tz, Rz
from differential_drive import DifferentialDrive
from visualize import Visualize

# Load route
shape = "Sample"  # Options: Sinus, Sample, Straight, 8
route = gen_path(shape)

# Choose control method
control = "MPC"  # Options: MPC, PID

# Define waypoints and scale the path
L = arclength(route[:, 0], route[:, 1], 'spline')[0]  # Calculate length of the path
scale = 0.05 / (L / route.shape[0])  # Scale the path for a walking speed of 1 m/s
waypoints = route * scale

# Define differential drive kinematics parameters
R = 0.1  # Wheel radius [m]
L = 0.35  # Wheelbase [m]
dd = DifferentialDrive(R, L)

# Simulation parameters
sampleTime = 0.05  # Sample time [s], equals 20 Hz
initPose = np.array([0, 0, 0])  # Initial pose (x, y, theta) of the robot
currentPose = initPose
lastPose = initPose

path_storage = np.zeros((2, 1))

# Setup visualization
positions = [initPose[:2]]
vis = Visualize()
vis.setup_plot(waypoints, shape)

# Define error accumulation variable
err = 0

# Define control and velocity limits
numPos = 20  # Number of stored positions corresponding to a 1 m distance to human
v_max = 1.5
omega_max = np.pi
v_min = -v_max
omega_min = -omega_max

# Setup MPC if selected
if control == "MPC":
    N = 5
    u = np.zeros((N, 2))

    optimizeProblem = MpcOptimizer(N, sampleTime, u,
                                   WeightX=100, WeightY=100, WeightTheta=1,
                                   WeightV=0.1, WeightOmega=0.01,
                                   VelLim=[v_min, v_max], AngVelLim=[omega_min, omega_max])
    optimizeProblem.setup_problem()

# Simulation loop
step = 1
velB = np.array([0, 0, 0])

while step <= waypoints.shape[0]:
    # Calculate relative distance between current pose and operator
    dx = waypoints[step - 1, 0] - currentPose[0]
    dy = waypoints[step - 1, 1] - currentPose[1]
    d_rel = np.linalg.inv(Rz(currentPose[2])) @ np.array([dx, dy, 1])

    # Store path history
    current_pose_last = -velB * sampleTime
    current_T_last = Tz(current_pose_last[2], current_pose_last)
    path_storage = current_T_last @ np.vstack([path_storage, np.zeros((1, path_storage.shape[1])), np.ones((1, path_storage.shape[1]))])
    path_storage = path_storage[:2, :]

    if np.any(np.abs(path_storage[:, -1] - d_rel[:2]) > 1e-3):
        path_storage = np.hstack([path_storage, d_rel[:2].reshape(-1, 1)])

    if path_storage.shape[1] > numPos:
        if control == "MPC":
            # Perform MPC optimization
            optimizeProblem.solve_problem(path_storage)
            vRef = optimizeProblem.Controls[0, 0]  # Linear velocity [m/s]
            wRef = optimizeProblem.Controls[0, 1]  # Angular velocity [rad/s]
        elif control == "PID":
            # Perform PID control
            vRef = 3 * np.linalg.norm(path_storage[:, 0])  # Linear velocity [m/s]
            phi = np.arctan2(path_storage[1, 0], path_storage[0, 0])
            wRef = 5 * phi  # Angular velocity [rad/s]

            vRef = np.clip(vRef, v_min, v_max)
            wRef = np.clip(wRef, omega_min, omega_max)

        path_storage = path_storage[:, 1:]
    else:
        vRef = 0
        wRef = 0

    # Calculate motor velocities and perform forward kinematics
    wL, wR = dd.inverse_kinematics(vRef, wRef)
    v, w = dd.forward_kinematics(wL, wR)

    # Perform odometry
    velB = np.array([v, 0, w])
    vel = body_to_world(velB, lastPose)
    currentPose = lastPose + vel * sampleTime

    if np.linalg.norm(vel) > 1e-3 and step > numPos:
        err += np.linalg.norm(currentPose[:2] - waypoints[step - numPos - 1, :])

    # Update positions for visualization
    positions = np.vstack((positions, currentPose[:2]))
    vis.update_plot(waypoints, positions, currentPose[2], sampleTime)

    lastPose = currentPose
    step += 1

print('Total time:', step * sampleTime)
print('Error sum:', err)
plt.show()
