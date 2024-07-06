%% Differential drive vehicle following waypoints using the 
% simulator for testing follow controller
% path of human (contains only position (x,y), no rotation (phi))
%
% Copyright 2018-2019 The MathWorks, Inc.

clear
clc

addpath("arclength\")
addpath("Paths\")
addpath("casadi-3.6.5-windows64-matlab2018b\")

import casadi.*

%% Load route
shape = "Sample"; % Option: Sinus, Sample and Straight, 8
route = gen_path(shape); 

L = arclength(route(:,1),route(:,2),'spline'); % calculate length of the path
scale = 0.05/(L/size(route,1)); % scale the path so as the human walks at 1 m/s
% Define waypoints
waypoints = route*scale; % Exp. format of the waypoints:[1,1; 10,1; 20,1; 30,1; 30,15;20,15;10,15;1,15];

%% Choose Control
control = "MPC"; % Option: MPC, PID

%% Define Vehicle Kinematics
R = 0.1;                % Wheel radius [m]
L = 0.35;                % Wheelbase [m]
dd = DifferentialDrive(R,L);

%% Simulation parameters
sampleTime = 0.05;              % Sample time [s], euqlas 20 Hz
totalTime = 0;                 % Total time in [s]

initPose = [0;0;0];             % Initial pose (x y theta) of the robot
currentpose = initPose;
lastpose = initPose;

path_storage = zeros(2,1);

% Create visualizer
viz = Visualizer2D;
viz.hasWaypoints = true;
% x and y limits for visualization
xMax = 1.2 * max(waypoints(:,1));
xMin = min(waypoints(:,1)) - 0.2 * abs(max(waypoints(:,1)));
offset = 0;
if shape  == "Straight"
    offset = 5;
end
yMax = 1.3 * max(waypoints(:,2)) + offset;
yMin = min(waypoints(:,2)) - 0.3 * abs(max(waypoints(:,2))) - offset;

err = 0;

numPos = 20; % Number of stored position (Coresponding to a 1 m distance to human)
v_max = 1.5; 
omega_max = pi/1.2;
v_min = -v_max;
omega_min = -omega_max;

%% Setup Problem for MPC
if control == "MPC"    
    N = 5;    
    u = zeros(N,2);
    
    optimizeProblem = mpcOptimizer(N, sampleTime, u, ...
               WeightX=100, WeightY=100, WeightTheta=1, ...
               WeightV= 0.1, WeightOmega=0.01, ...
               VelLim=[v_min v_max], AngVelLim=[omega_min omega_max]);
    
    optimizeProblem = optimizeProblem.setupProblem;
end

%% Simulation loop

close all
r = rateControl(1/sampleTime); % fixed sample time of 0.05 s-> 20 Hz
step = 1;

% Initialization
velB = [0; 0; 0];

while step <= size(waypoints,1)
    
    % Sensor Output relative distance between current pose and operator
    dx = waypoints(step,1) - currentpose(1); % global coordinate
    dy = waypoints(step,2) - currentpose(2);
   
    %% Derive relative distances from human to robot relative to the robot's Frame    
    d_rel = rotz(double(currentpose(3)))^-1*[dx dy 1]';

    %% Design controller based on sensor outputs    
    current_pose_last = -velB * sampleTime;
    current_T_last = Tz(current_pose_last(3), current_pose_last(1:2));
    path_storage = current_T_last * [path_storage; zeros(1,size(path_storage,2)); ones(1,size(path_storage,2))];
    path_storage = path_storage(1:2,:);  
  
    if any(abs(path_storage(:,end) - d_rel(1:2)) > 1e-3)
        path_storage = [path_storage d_rel(1:2)];
    end
     
    if size(path_storage,2) > numPos
        if control == "MPC"       
            % ------------------MPC Optimization-------------------------            
            optimizeProblem = optimizeProblem.solveProblem(path_storage);
            vRef = optimizeProblem.Controls(1,1); % v in [m/s]
            wRef = optimizeProblem.Controls(1,2); % w in [rad/s]
            
            path_storage(:,1) = [];

        elseif control == "PID"
            vRef = 3*(norm(path_storage(:,1))); % v in [m/s]
            phi = atan2(path_storage(2,1), path_storage(1,1));
            wRef = 5*phi; % w in [rad/s]
            if abs(vRef) > v_max
                vRef = sign(vRef)*1.6;
            end
            if abs(wRef) > omega_max
                wRef = sign(wRef)*3.4;
            end

            path_storage(:,1) = [];                
            
        end
    else
        vRef = 0; % v in [m/s]
        wRef = 0; % w in [rad/s]
    end        

    [wL,wR] = inverseKinematics(dd,vRef,wRef);  % calculate motor rpms
   
    % % Add noise
    % if any(abs(path_storage(:,end-1) - d_rel(1:2)) > 1e-2)
    %     wL = wL + double((-1)^int8(rand))*rand*5;
    %     wR = wR + double((-1)^int8(rand))*rand*5;
    % end
    %% Compute the velocities for simulation
    [v,w] = forwardKinematics(dd,wL,wR);

    %% perform Odometry
    velB = [v;0;w]; % Body velocities [vx;vy;w]
    vel = bodyToWorld(velB,lastpose);  % Convert from body to world

    % Perform forward discrete integration step
    currentpose = lastpose + vel*sampleTime; 
    
    % T = Tz(lastpose(3),lastpose(1:2));
    % s = T * [path_storage; zeros(1,size(path_storage,2)); ones(1,size(path_storage,2))];
    % s = s(1:2,:);

    if norm(vel) > 1e-3 && step > numPos
        err = err + norm(currentpose(1:2)-waypoints(step-numPos,:)')
    end
    
    % Calculate Distance
    distance = norm(currentpose(1:2)-waypoints(end,:)');
    lastpose = currentpose;
    totalTime = totalTime + sampleTime;

    % Update visualization
    viz(currentpose,[waypoints(:,1) waypoints(:,2)])

    % if step > 20
    % viz([s(:,1); 0],[waypoints(:,1) waypoints(:,2)])
    % end

    xlim([xMin xMax])
    ylim([yMin yMax])

    step = step+1; 
   
    waitfor(r);
end

