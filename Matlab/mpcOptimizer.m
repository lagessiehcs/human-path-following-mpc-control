classdef mpcOptimizer
    %MPCPATHFOLLOWER - Control a differential drive mobile robot to follow a
    %predefined path
    %   This class use Model Predictive Control (MPC) to control a
    %   differential drive mobile robot to follow a predefined path.
    % 
    %   Required Properties
    %     N          - Prediction horizon, scalar [-]
    %     sampleTime - Sample time, scalar [s]    
    %     Controls   - Control inputs for N timestep in the future, Nx2
    %                matrix [m/s rad/s]
    %     
    %   Optional Properties
    %     VelLim    - Maximum and minimum linear velocity [v_min v_max],
    %               1x2 vector [m/s]
    %     AngVelLim - Maximum and minimum angular velocity 
    %               [omega_min omega_max], 1x2 vector [m/s] 
    %     PosXLim   - Maximum and minimum x value [x_min x_max], 
    %               1x2 vector [m]
    %     PosYLim   - Maximum and minimum y value [y_min y_max], 
    %               1x2 vector [m]
    %     OrientLim - Maximum and minimum theta value [theta_min theta_max], 
    %               1x2 vector [m/s]
    % 
    %   Other Properties  
    %     The remaining properties are assigned in the class methods
    %     
    %     Assigned in Contructor
    %     Q - States weighting matrix 
    %     R - Controls weighting matrix 
    %     
    %     Assigned in setupProblem
    %     Args   - Arguments for solver function
    %     Solver - Solver function
    % 
    %   
    %   Methods
    %     setupProblem - Setup the optimization problem
    %     solveProblem - Solve the optimization problem
    
    properties
        N
        sampleTime
        Controls
        VRef
        OmegaRef
        PosXLim
        PosYLim
        OrientLim
        VelLim
        AngVelLim
        Q
        R

        Args
        Solver
    end
    
    methods
        %% ----------- Constructor ---------------------------------------
        function fl = mpcOptimizer(n, t, u, options)        
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
                options.WeightY   	(1, 1) {mustBeNumeric} = 1
                options.WeightTheta (1, 1) {mustBeNumeric} = 1
                options.WeightV     (1, 1) {mustBeNumeric} = 1
                options.WeightOmega (1, 1) {mustBeNumeric} = 1
            end

            fl.N          = n;
            fl.sampleTime = t;
            fl.Controls   = u;
    
            fl.VRef      = options.VRef;
            fl.OmegaRef  = options.OmegaRef;
            fl.PosXLim   = options.PosXLim;
            fl.PosYLim   = options.PosYLim; 
            fl.OrientLim = options.OrientLim;
            fl.VelLim    = options.VelLim;
            fl.AngVelLim = options.AngVelLim;

            fl.Q     = diag([options.WeightX options.WeightY options.WeightTheta]);
            fl.R     = diag([options.WeightV options.WeightOmega]);
        end

        %% ----------- Setup Problem --------------------------------------
        function obj = setupProblem(obj)
            %SETUPPROBLEM - Setup the optimization problem
            %   This function setups the optimization problem using Casadi  
            
            import casadi.*
            
            % Define symbolic variables
            x = SX.sym('x'); 
            y = SX.sym('y'); 
            theta = SX.sym('theta');

            states = [x;y;theta]; 
            numStates = length(states);
            
            v = SX.sym('v'); 
            omega = SX.sym('omega');
            controls = [v;omega]; 
            numControls = length(controls);
            rhs = [v*cos(theta);v*sin(theta);omega];

            % Nonlinear mapping function f(x,u)
            f = Function('f',{states,controls},{rhs}); 

            % Decision variables (controls)
            U = SX.sym('U',numControls,obj.N); 

            % Parameters (which include the initial state and the reference 
            % along the predicted trajectory (reference states and
            % reference controls))
            P = SX.sym('P',numStates + obj.N*(numStates+numControls)); 
                 
            % States over the optimization problem.
            X = SX.sym('X',numStates,(obj.N+1)); 

            % Make the decision variable one column  vector
            OPT_variables = [reshape(X,3*(obj.N+1),1);reshape(U,2*obj.N,1)];                 
   
            % Calculate cost function
            cost = CostFun(obj, X, U, P, numStates, numControls);

            % Calculate nonlinear constraints
            g = nonlinearConstraints(obj, f, X, U, P);
            
            obj = setSolver(obj, cost, g, P, OPT_variables);

            obj = setArgs(obj);            
            
        end
        
        %% ----------- Solve Problem --------------------------------------
        function obj = solveProblem(obj, path)
            %SOLVEPROBLEM - Solve the optimization problem
            %   This function solves the optimization problem using Casadi
            
            import casadi.*
            
            x0 = [0;0;0];
            X0 = repmat(x0,1,obj.N+1)'; % initialization of the states decision variables
            u0 = obj.Controls;        
            xs = path;
            obj.Args.p(1:3) = x0; % initial condition of the robot posture
            for k = 1:obj.N 
                x_next = xs(1,5); y_next = xs(2,5);
                x_ref = xs(1,1); y_ref = xs(2,1); theta_ref = atan2(y_next-y_ref,x_next-x_ref);
            
            
                v_ref = obj.VRef; omega_ref = obj.OmegaRef;
                obj.Args.p(5*k-1:5*k+1) = [x_ref, y_ref, theta_ref];
                obj.Args.p(5*k+2:5*k+3) = [v_ref, omega_ref];
            end
       
            % initial value of the optimization variables
            obj.Args.x0  = [reshape(X0',3*(obj.N+1),1);reshape(u0',2*obj.N,1)];
            sol = obj.Solver('x0', obj.Args.x0, 'lbx', obj.Args.lbx, 'ubx', obj.Args.ubx,...
                'lbg', obj.Args.lbg, 'ubg', obj.Args.ubg,'p',obj.Args.p);
            obj.Controls = reshape(full(sol.x(3*(obj.N+1)+1:end))',2,obj.N)'; % get controls only from the solution
        end


         %% ----------- Set Solver --------------------------------------
        function obj = setSolver(obj, cost, g, P, OPT_variables)

            import casadi.*
            
            nlp_prob = struct('f', cost, 'x', OPT_variables, 'g', g, 'p', P);
            
            opts = struct;
            opts.ipopt.max_iter = 2000;
            opts.ipopt.print_level = 0; %0,3
            opts.print_time = 0;
            opts.ipopt.acceptable_tol = 1e-8;
            opts.ipopt.acceptable_obj_change_tol = 1e-6;
            
            obj.Solver = nlpsol('solver', 'ipopt', nlp_prob,opts);
        end


        %% ----------- Set Arguments --------------------------------------
        function obj = setArgs(obj)
            obj.Args = struct;
            
            obj.Args.lbg(1:3*(obj.N+1)) = 0;  % -1e-20  % Equality constraints
            obj.Args.ubg(1:3*(obj.N+1)) = 0;  % 1e-20   % Equality constraints
            
            obj.Args.lbx(1:3:3*(obj.N+1),1) = obj.PosXLim(1); %state x lower bound % new - adapt the bound
            obj.Args.ubx(1:3:3*(obj.N+1),1) = obj.PosXLim(2); %state x upper bound  % new - adapt the bound
            obj.Args.lbx(2:3:3*(obj.N+1),1) = obj.PosYLim(1); %state y lower bound
            obj.Args.ubx(2:3:3*(obj.N+1),1) = obj.PosYLim(2); %state y upper bound
            obj.Args.lbx(3:3:3*(obj.N+1),1) = obj.OrientLim(1); %state theta lower bound
            obj.Args.ubx(3:3:3*(obj.N+1),1) = obj.OrientLim(2); %state theta upper bound
            
            obj.Args.lbx(3*(obj.N+1)+1:2:3*(obj.N+1)+2*obj.N,1) = obj.VelLim(1); % v lower bound
            obj.Args.ubx(3*(obj.N+1)+1:2:3*(obj.N+1)+2*obj.N,1) = obj.VelLim(2); % v upper bound
            obj.Args.lbx(3*(obj.N+1)+2:2:3*(obj.N+1)+2*obj.N,1) = obj.AngVelLim(1); % omega lower bound
            obj.Args.ubx(3*(obj.N+1)+2:2:3*(obj.N+1)+2*obj.N,1) = obj.AngVelLim(2); % omega upper bound
        end
       

    end
end


%% ----------- Cost Function --------------------------------------
function cost = CostFun(obj, X, U, P, n_states, n_controls)

cost = 0; 

for k = 1:obj.N
    state = X(:,k);
    control = U(:,k);

    % Sum of n_states+n_controls
    n_var = n_states + n_controls;

    % Cost function to minimize
    cost = cost+(state-P(n_var*k-1:n_var*k+1))'*obj.Q*(state-P(n_var*k-1:n_var*k+1)) + ...
         (control-P(n_var*k+2:n_var*k+3))'*obj.R*(control-P(n_var*k+2:n_var*k+3)) ;            
end

end

%% ----------- Nonlinear Constraints --------------------------------------
function constraints = nonlinearConstraints(obj, f, X, U, P)

% Initial condition constraints
constraints = X(:,1)-P(1:3);

for k = 1:obj.N
    state = X(:,k);
    control = U(:,k);

    st_next = X(:,k+1);
    f_value = f(state,control);
    
    st_next_euler = state + (obj.sampleTime*f_value);
    % Compute constraints
    constraints = [constraints;st_next-st_next_euler]; 
end
end


