import numpy as np
from scipy.spatial.transform import Rotation as R
import casadi as ca
import matplotlib.pyplot as plt

from path import gen_path, arclength
from mpc_optimizer import MpcOptimizer
from transform import body_to_world, Tz, Rz
from differential_drive import DifferentialDrive

# Load route
shape = "Sinus"  # Option: Sinus, Sample and Straight, 8
route = gen_path(shape)

L = arclength(route[:, 0], route[:, 1],'spline')[0]  # calculate length of the path
scale = 0.05 / (L / route.shape[0])  # scale the path so as the human walks at 1 m/s
# Define waypoints
waypoints = route * scale  # Exp. format of the waypoints


# Choose Control
control = "MPC"  # Option: MPC, PID

# Define Vehicle Kinematics
R = 0.1;                # Wheel radius [m]
L = 0.35;               # Wheelbase [m]
dd = DifferentialDrive(R,L)


# Simulation parameters
sampleTime = 0.05  # Sample time [s], equals 20 Hz
totalTime = 0  # Total time in [s]

initPose = np.array([0, 0, 0])  # Initial pose (x, y, theta) of the robot
currentpose = initPose
lastpose = initPose



path_storage = np.zeros((2, 1))

# Visualization preparation
traj_x = []
traj_y = []
# x and y limits for visualization
xMax = 1.2 * np.max(waypoints[:, 0])
xMin = np.min(waypoints[:, 0]) - 0.2 * np.abs(np.max(waypoints[:, 0]))
offset = 0
if shape == "Straight":
    offset = 5
yMax = 1.3 * np.max(waypoints[:, 1]) + offset
yMin = np.min(waypoints[:, 1]) - 0.3 * np.abs(np.max(waypoints[:, 1])) - offset
traj_handle, = plt.plot(0, 0, 'b.-', linewidth=1,markersize=4)
plt.ion()


err = 0

numPos = 20  # Number of stored position (Corresponding to a 1 m distance to human)
v_max = 1.5
omega_max = np.pi / 1.2
v_min = -v_max
omega_min = -omega_max

# Setup Problem for MPC
if control == "MPC":
    N = 5
    u = np.zeros((N, 2))

    optimizeProblem = MpcOptimizer(N, sampleTime, u,
                                   WeightX=100, WeightY=100, WeightTheta=1,
                                   WeightV=0.1, WeightOmega=0.01,
                                   VelLim=[v_min, v_max], AngVelLim=[omega_min, omega_max])

    optimizeProblem = optimizeProblem.setup_problem()

# Simulation loop
step = 1

# Initialization
velB = np.array([0, 0, 0])

while step <= waypoints.shape[0]:
    # Sensor Output relative distance between current pose and operator
    dx = waypoints[step - 1, 0] - currentpose[0]  # global coordinate
    dy = waypoints[step - 1, 1] - currentpose[1]

    # Derive relative distances from human to robot relative to the robot's Frame
    d_rel = np.linalg.inv(Rz(currentpose[2])) @ np.array([dx, dy, 1])

    # Design controller based on sensor outputs
    current_pose_last = -velB * sampleTime
    current_T_last = Tz(current_pose_last[2],current_pose_last)
    path_storage = current_T_last @ np.vstack([path_storage, np.zeros((1, path_storage.shape[1])), np.ones((1, path_storage.shape[1]))])
    path_storage = path_storage[:2, :]

    if np.any(np.abs(path_storage[:, -1] - d_rel[:2]) > 1e-3):
        path_storage = np.hstack([path_storage, d_rel[:2].reshape(-1, 1)])

    if path_storage.shape[1] > numPos:
        if control == "MPC":
            # ------------------MPC Optimization-------------------------
            optimizeProblem = optimizeProblem.solve_problem(path_storage)
            vRef = optimizeProblem.Controls[0, 0]  # v in [m/s]
            wRef = optimizeProblem.Controls[0, 1]  # w in [rad/s]
       

            path_storage = path_storage[:, 1:]

        elif control == "PID":
            vRef = 3 * np.linalg.norm(path_storage[:, 0])  # v in [m/s]
            phi = np.arctan2(path_storage[1, 0], path_storage[0, 0])
            wRef = 5 * phi  # w in [rad/s]

            if np.abs(vRef) > v_max:
                vRef = np.sign(vRef) * 1.6
            if np.abs(wRef) > omega_max:
                wRef = np.sign(wRef) * 3.4

            path_storage = path_storage[:, 1:]

    else:
        vRef = 0  # v in [m/s]
        wRef = 0  # w in [rad/s]

    wL, wR = dd.inverse_kinematics(vRef, wRef)  # calculate motor rpms

    # Compute the velocities for simulation
    v, w = dd.forward_kinematics(wL, wR)

    # perform Odometry
    velB = np.array([v, 0, w])  # Body velocities [vx, vy, w]
    vel = body_to_world(velB,lastpose)

    # Perform forward discrete integration step
    currentpose = lastpose + vel * sampleTime

    if np.linalg.norm(vel) > 1e-3 and step > numPos:
        err = err + np.linalg.norm(currentpose[:2] - waypoints[step - numPos, :])
        print(err)

    # T = Tz(lastpose[2],lastpose[:2])
    # s = T @ np.vstack([path_storage, np.zeros((1, path_storage.shape[1])), np.ones((1, path_storage.shape[1]))])
    # s = s[:2, :]

    # Calculate Distance
    distance = np.linalg.norm(currentpose[:2] - waypoints[-1, :])
    lastpose = currentpose
    totalTime = totalTime + sampleTime

   
    # plt.clf()  # Clear the current figure
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'r.', markersize=2)
    # plt.plot(s[0], s[1], 'b')
    plt.gca().set_aspect('equal')
    traj_x.append(currentpose[0])
    traj_y.append(currentpose[1])
    traj_handle.set_data(traj_x, traj_y)

    

    step += 1
    plt.pause(sampleTime)
plt.show(block=True)
