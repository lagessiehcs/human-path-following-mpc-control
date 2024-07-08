classdef mpcOptimizer
    % MPCOPTIMIZER - Control a differential drive mobile robot to follow a predefined path using MPC
    %
    % This class uses Model Predictive Control (MPC) to control a
    % differential drive mobile robot to follow a predefined path.
    % 
    % Required Properties:
    %   N          - Prediction horizon, scalar [-]
    %   sampleTime - Sample time, scalar [s]
    %   Controls   - Control inputs for N timesteps in the future, Nx2 matrix [m/s rad/s]
    %
    % Optional Properties:
    %   VelLim    - Maximum and minimum linear velocity [v_min v_max], 1x2 vector [m/s]
    %   AngVelLim - Maximum and minimum angular velocity [omega_min omega_max], 1x2 vector [rad/s]
    %   PosXLim   - Maximum and minimum x value [x_min x_max], 1x2 vector [m]
    %   PosYLim   - Maximum and minimum y value [y_min y_max], 1x2 vector [m]
    %   OrientLim - Maximum and minimum theta value [theta_min theta_max], 1x2 vector [rad]
    %
    % Internal Properties:
    %   Q           - States weighting matrix
    %   R           - Controls weighting matrix
    %   NumStates   - Number of states [x, y, theta], scalar (default = 3)
    %   NumControls - Number of controls [v, omega], scalar (default = 2)
    %   Args        - Arguments for solver function
    %   Solver      - Solver function
    %
    % Methods:
    %   setupProblem - Setup the optimization problem
    %   solveProblem - Solve the optimization problem
    
    properties
        % Required Properties
        N        
        sampleTime
        Controls
        
        % Optional Properties
        VRef
        OmegaRef
        PosXLim
        PosYLim
        OrientLim
        VelLim
        AngVelLim

        % Weighting Matrices
        Q
        R

        % Internal Properties
        NumStates = 3 % [x, y, theta]
        NumControls = 2 % [v, omega]

        Args
        Solver
    end
    
    methods
        %% Constructor
        function obj = mpcOptimizer(n, t, u, options)        
            arguments
                n (1, 1) {mustBeInteger}
                t (1, 1) {mustBePositive}
                u (:, 2) {mustBeNumeric}
                options.VRef        (1, 1) {mustBeNumeric} = 1
                options.OmegaRef    (1, 1) {mustBeNumeric} = 0
                options.PosXLim     (1, 2) {mustBeNumeric} = [-inf inf]
                options.PosYLim     (1, 2) {mustBeNumeric} = [-inf inf]
                options.OrientLim   (1, 2) {mustBeNumeric} = [-inf inf]
                options.VelLim      (1, 2) {mustBeNumeric} = [-inf inf]
                options.AngVelLim   (1, 2) {mustBeNumeric} = [-inf inf]
                options.WeightX     (1, 1) {mustBeNumeric} = 1
                options.WeightY     (1, 1) {mustBeNumeric} = 1
                options.WeightTheta (1, 1) {mustBeNumeric} = 1
                options.WeightV     (1, 1) {mustBeNumeric} = 1
                options.WeightOmega (1, 1) {mustBeNumeric} = 1
            end

            obj.N          = n;
            obj.sampleTime = t;
            obj.Controls   = u;
    
            obj.VRef      = options.VRef;
            obj.OmegaRef  = options.OmegaRef;
            obj.PosXLim   = options.PosXLim;
            obj.PosYLim   = options.PosYLim; 
            obj.OrientLim = options.OrientLim;
            obj.VelLim    = options.VelLim;
            obj.AngVelLim = options.AngVelLim;

            obj.Q     = diag([options.WeightX options.WeightY options.WeightTheta]);
            obj.R     = diag([options.WeightV options.WeightOmega]);
        end

        %% Setup Problem
        function obj = setupProblem(obj)
            %SETUPPROBLEM - Setup the optimization problem using Casadi
            
            import casadi.*
            
            % Define symbolic variables
            x = SX.sym('x'); 
            y = SX.sym('y'); 
            theta = SX.sym('theta');
            states = [x; y; theta];
            
            v = SX.sym('v'); 
            omega = SX.sym('omega');
            controls = [v; omega]; 
            rhs = [v * cos(theta); v * sin(theta); omega];

            % Nonlinear mapping function f(x,u)
            f = Function('f', {states, controls}, {rhs}); 

            % Decision variables (controls)
            U = SX.sym('U', obj.NumControls, obj.N); 

            % Parameters (initial state + reference along predicted trajectory)
            P = SX.sym('P', obj.NumStates + obj.N * (obj.NumStates + obj.NumControls)); 
                 
            % States over the optimization problem
            X = SX.sym('X', obj.NumStates, (obj.N + 1)); 

            % Make the decision variable one column vector
            OPT_variables = [reshape(X, obj.NumStates * (obj.N + 1), 1); reshape(U, obj.NumControls * obj.N, 1)];                 
   
            % Calculate cost function
            cost = CostFun(obj, X, U, P);

            % Calculate nonlinear constraints
            g = nonlinearConstraints(obj, f, X, U, P);
            
            obj = setSolver(obj, cost, g, P, OPT_variables);

            obj = setArgs(obj);            
        end
        
        %% Solve Problem
        function obj = solveProblem(obj, path)
            %SOLVEPROBLEM - Solve the optimization problem using Casadi
            
            import casadi.*
            
            x0 = [0; 0; 0]; % Initial state
            X0 = repmat(x0, 1, obj.N + 1)'; % Initialization of state decision variables
            u0 = obj.Controls;        
            xs = path; % Reference path
            obj.Args.p(1:obj.NumStates) = x0; % Initial condition of the robot posture
            
            % Set reference path and controls
            for k = 1:obj.N 
                x_next = xs(1, 5); 
                y_next = xs(2, 5);
                x_ref = xs(1, 1); 
                y_ref = xs(2, 1); 
                theta_ref = atan2(y_next - y_ref, x_next - x_ref);
                v_ref = obj.VRef; 
                omega_ref = obj.OmegaRef;
                
                obj.Args.p((obj.NumStates + obj.NumControls) * k - 1 : (obj.NumStates + obj.NumControls) * k - 1 + (obj.NumStates - 1)) = [x_ref, y_ref, theta_ref];
                obj.Args.p((obj.NumStates + obj.NumControls) * k + 2 : (obj.NumStates + obj.NumControls) * k + 2 + (obj.NumControls - 1)) = [v_ref, omega_ref];
            end
            
            % Initial value of the optimization variables
            obj.Args.x0  = [reshape(X0', obj.NumStates * (obj.N + 1), 1); reshape(u0', obj.NumControls * obj.N, 1)];
            sol = obj.Solver('x0', obj.Args.x0, 'lbx', obj.Args.lbx, 'ubx', obj.Args.ubx,...
                'lbg', obj.Args.lbg, 'ubg', obj.Args.ubg, 'p', obj.Args.p);
            obj.Controls = reshape(full(sol.x(obj.NumStates * (obj.N + 1) + 1:end))', 2, obj.N)'; % Get controls only from the solution
        end

        %% Set Solver
        function obj = setSolver(obj, cost, g, P, OPT_variables)
            %SETSOLVER - Setup the solver for the optimization problem
            
            import casadi.*
            
            nlp_prob = struct('f', cost, 'x', OPT_variables, 'g', g, 'p', P);
            opts = struct;
            opts.ipopt.max_iter = 2000;
            opts.ipopt.print_level = 0; %0,3
            opts.print_time = 0;
            opts.ipopt.acceptable_tol = 1e-8;
            opts.ipopt.acceptable_obj_change_tol = 1e-6;
            
            obj.Solver = nlpsol('solver', 'ipopt', nlp_prob, opts);
        end

        %% Set Arguments
        function obj = setArgs(obj)
            %SETARGS - Setup the arguments for the solver
            
            obj.Args = struct;
            
            obj.Args.lbg(1:obj.NumStates * (obj.N + 1)) = 0;  % Equality constraints
            obj.Args.ubg(1:obj.NumStates * (obj.N + 1)) = 0;  % Equality constraints
            
            obj.Args.lbx(1:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.PosXLim(1); % state x lower bound
            obj.Args.ubx(1:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.PosXLim(2); % state x upper bound
            obj.Args.lbx(2:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.PosYLim(1); % state y lower bound
            obj.Args.ubx(2:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.PosYLim(2); % state y upper bound
            obj.Args.lbx(3:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.OrientLim(1); % state theta lower bound
            obj.Args.ubx(3:obj.NumStates:obj.NumStates * (obj.N + 1), 1) = obj.OrientLim(2); % state theta upper bound
            
            obj.Args.lbx(obj.NumStates * (obj.N + 1) + 1:obj.NumControls:obj.NumStates * (obj.N + 1) + obj.NumControls * obj.N, 1) = obj.VelLim(1); % v lower bound
            obj.Args.ubx(obj.NumStates * (obj.N + 1) + 1:obj.NumControls:obj.NumStates * (obj.N + 1) + obj.NumControls * obj.N, 1) = obj.VelLim(2); % v upper bound
            obj.Args.lbx(obj.NumStates * (obj.N + 1) + 2:obj.NumControls:obj.NumStates * (obj.N + 1) + obj.NumControls * obj.N, 1) = obj.AngVelLim(1); % omega lower bound
            obj.Args.ubx(obj.NumStates * (obj.N + 1) + 2:obj.NumControls:obj.NumStates * (obj.N + 1) + obj.NumControls * obj.N, 1) = obj.AngVelLim(2); % omega upper bound
        end
    end
end

%% Cost Function
function cost = CostFun(obj, X, U, P)
    %COSTFUN - Calculate the cost function for the optimization problem
    
    cost = 0; 

    for k = 1:obj.N
        state = X(:, k);
        control = U(:, k);

        % Sum of obj.NumStates + obj.NumControls
        n_var = obj.NumStates + obj.NumControls;

        % Cost function to minimize
        cost = cost + (state - P(n_var * k - 1:n_var * k - 1 + (obj.NumStates - 1)))' * obj.Q * (state - P(n_var * k - 1:n_var * k - 1 + (obj.NumStates - 1))) + ...
                     (control - P(n_var * k + 2:n_var * k + 2 + (obj.NumControls - 1)))' * obj.R * (control - P(n_var * k + 2:n_var * k + 2 + (obj.NumControls - 1)));            
    end
end

%% Nonlinear Constraints
function constraints = nonlinearConstraints(obj, f, X, U, P)
    %NONLINEARCONSTRAINTS - Calculate the nonlinear constraints for the optimization problem
    
    % Initial condition constraints
    constraints = X(:, 1) - P(1:obj.NumStates);

    for k = 1:obj.N
        state = X(:, k);
        control = U(:, k);

        st_next = X(:, k + 1);
        f_value = f(state, control);
        
        st_next_euler = state + (obj.sampleTime * f_value);
        % Compute constraints
        constraints = [constraints; st_next - st_next_euler]; 
    end
end
