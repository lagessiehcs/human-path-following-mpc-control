%% Differential drive vehicle following waypoints using the 
% simulator for testing follow controller path of human (contains only position (x,y), no rotation (phi))
%
% Copyright 2018-2019 The MathWorks, Inc.

clear;
clc;

% Add necessary paths
addpath("arclength\");
addpath("Paths\");
addpath("casadi-3.6.5-windows64-matlab2018b\");

import casadi.*;

%% Load route and control options
shape = "Sinus"; % Options: Sinus|Sample|Straight|8
control = "MPC"; % Options: MPC|PID

route = gen_path(shape);

%% Generate GIF
gen_gif = false; % Options: true|false

%% Define waypoints
L = arclength(route(:,1), route(:,2), 'spline'); % Calculate length of the path
scale = 0.05 / (L / size(route, 1)); % Scale the path so the human walks at 1 m/s

waypoints = route * scale; % Waypoints in the format: [1,1; 10,1; ...]

%% Define Vehicle Kinematics
R = 0.1; % Wheel radius [m]
wheelbase = 0.35; % Wheelbase [m]
dd = DifferentialDrive(R, wheelbase);

%% Simulation parameters
sampleTime = 0.05; % Sample time [s], equals 20 Hz
totalTime = 0; % Total simulation time [s]

initPose = [0; 0; 0]; % Initial pose (x, y, theta) of the robot
currentPose = initPose;
lastPose = initPose;

pathStorage = zeros(2, 1);

% Create visualizer
viz = Visualizer2D;
viz.hasWaypoints = true;

% Set visualization limits
xMax = 1.2 * max(waypoints(:,1));
xMin = min(waypoints(:,1)) - 0.2 * abs(max(waypoints(:,1)));

% Offset adjustment based on the shape
if shape == "Straight"
    offset = 5;
else
    offset = 0;
end

yMax = 1.3 * max(waypoints(:,2)) + offset;
yMin = min(waypoints(:,2)) - 0.3 * abs(max(waypoints(:,2))) - offset;

%% Control limits and storage parameters
numPos = 20; % Number of stored positions (corresponding to a 1 m distance to human)
vMax = 1.5; 
omegaMax = pi;
vMin = -vMax;
omegaMin = -omegaMax;

%% Setup MPC problem
if control == "MPC"    
    N = 5; % Prediction horizon
    u = zeros(N, 2); % Initial control inputs

    optimizeProblem = mpcOptimizer(N, sampleTime, u, ...
        WeightX=100, WeightY=100, WeightTheta=1, ...
        WeightV=0.1, WeightOmega=0.01, ...
        VelLim=[vMin vMax], AngVelLim=[omegaMin omegaMax]);

    optimizeProblem = optimizeProblem.setupProblem;
end

%% Simulation loop
close all;
r = rateControl(1 / sampleTime); % Fixed sample time of 0.05 s -> 20 Hz
step = 1;

% Initialization
velB = [0; 0; 0];
err = 0;

% Prepare for GIF creation
if gen_gif
    gifFilename = 'Animation/animated_plot.gif';
end

while step <= size(waypoints, 1)
    % Sensor output: relative distance between current pose and operator
    dx = waypoints(step, 1) - currentPose(1); % Global coordinate
    dy = waypoints(step, 2) - currentPose(2);
   
    % Derive relative distances from human to robot relative to the robot's frame    
    dRel = rotz(double(currentPose(3)))^-1 * [dx dy 1]';

    % Design controller based on sensor outputs    
    currentPoseLast = -velB * sampleTime;
    currentTLast = Tz(currentPoseLast(3), currentPoseLast(1:2));
    pathStorage = currentTLast * [pathStorage; zeros(1, size(pathStorage, 2)); ones(1, size(pathStorage, 2))];
    pathStorage = pathStorage(1:2,:);  
  
    if any(abs(pathStorage(:, end) - dRel(1:2)) > 1e-3)
        pathStorage = [pathStorage dRel(1:2)];
    end
     
    if size(pathStorage, 2) > numPos
        if control == "MPC"       
            % MPC Optimization            
            optimizeProblem = optimizeProblem.solveProblem(pathStorage);
            vRef = optimizeProblem.Controls(1, 1); % v in [m/s]
            wRef = optimizeProblem.Controls(1, 2); % w in [rad/s]
            
            pathStorage(:, 1) = [];

        elseif control == "PID"
            vRef = 3 * norm(pathStorage(:, 1)); % v in [m/s]
            phi = atan2(pathStorage(2, 1), pathStorage(1, 1));
            wRef = 5 * phi; % w in [rad/s]
            vRef = min(max(vRef, vMin), vMax);
            wRef = min(max(wRef, omegaMin), omegaMax);

            pathStorage(:, 1) = [];                
        end
    else
        vRef = 0; % v in [m/s]
        wRef = 0; % w in [rad/s]
    end        

    [wL, wR] = inverseKinematics(dd, vRef, wRef);  % Calculate motor RPMs
   
    % Compute the velocities for simulation
    [v, w] = forwardKinematics(dd, wL, wR);

    % Perform Odometry
    velB = [v; 0; w]; % Body velocities [vx; vy; w]
    vel = bodyToWorld(velB, lastPose); % Convert from body to world

    % Perform forward discrete integration step
    currentPose = lastPose + vel * sampleTime; 
    
    if norm(vel) > 1e-3 && step > numPos
        err = err + norm(currentPose(1:2) - waypoints(step - numPos, :)');
    end
    
    % Calculate distance to the final waypoint
    distance = norm(currentPose(1:2) - waypoints(end, :)');
    lastPose = currentPose;
    totalTime = totalTime + sampleTime;

    % Update visualization
    viz(currentPose, [waypoints(:, 1) waypoints(:, 2)]);
    xlim([xMin xMax]);
    ylim([yMin yMax]);
    
    if gen_gif
        % Capture frame and write to GIF
        set(gcf, 'Position', [100, 100, 500*1.1*(xMax-xMin)/(yMax-yMin), 500]);
        frame = getframe(gcf);
        img = frame2im(frame);
        [imgInd, cmap] = rgb2ind(img, 256);
        
        if step == 1
            imwrite(imgInd, cmap, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', sampleTime);
        else
            imwrite(imgInd, cmap, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', sampleTime);
        end
    end
    step = step + 1; 
    waitfor(r);
end
fprintf("Error sum: %.2f\n", err)
