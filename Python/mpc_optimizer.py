import casadi as ca
import numpy as np

class MpcOptimizer:
    def __init__(self, n, t, u, **options):
        # Required properties
        self.N = n
        self.sampleTime = t
        self.Controls = np.array(u)

        # Optional properties with defaults
        self.VRef = options.get('VRef', 1)
        self.OmegaRef = options.get('OmegaRef', 0)
        self.PosXLim = options.get('PosXLim', [-np.inf, np.inf])
        self.PosYLim = options.get('PosYLim', [-np.inf, np.inf])
        self.OrientLim = options.get('OrientLim', [-np.inf, np.inf])
        self.VelLim = options.get('VelLim', [-np.inf, np.inf])
        self.AngVelLim = options.get('AngVelLim', [-np.inf, np.inf])
        
        weight_x = options.get('WeightX', 1)
        weight_y = options.get('WeightY', 1)
        weight_theta = options.get('WeightTheta', 1)
        weight_v = options.get('WeightV', 1)
        weight_omega = options.get('WeightOmega', 1)

        self.Q = np.diag([weight_x, weight_y, weight_theta])
        self.R = np.diag([weight_v, weight_omega])

        self.NumStates = 3
        self.NumControls = 2
        
        self.Args = {}
        self.Solver = None

    def setup_problem(self):
        # Define symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables (controls)
        U = ca.SX.sym('U', self.NumControls, self.N)

        # Parameters (initial state and reference trajectory)
        P = ca.SX.sym('P', self.NumStates + self.N * (self.NumStates + self.NumControls))

        # States over the optimization problem
        X = ca.SX.sym('X', self.NumStates, (self.N + 1))

        # Make the decision variable one column vector
        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )

        # Calculate cost function
        cost = self._cost_function(X, U, P)

        # Calculate nonlinear constraints
        g = self._nonlinear_constraints(f, X, U, P)

        self._set_solver(cost, g, P, OPT_variables)
        self._set_args()

    def solve_problem(self, path):
        x0 = np.array([0, 0, 0])
        X0 = np.tile(x0, (self.N + 1, 1)).T
        u0 = self.Controls
        xs = path

        self.Args['p'] = np.zeros((self.NumStates + (self.NumStates+self.NumControls) * self.N,))
        self.Args['p'][0:self.NumStates] = x0

        for k in range(self.N):
            x_next, y_next = xs[:, 4]
            x_ref, y_ref = xs[:, 0]
            theta_ref = np.arctan2(y_next - y_ref, x_next - x_ref)

            v_ref = self.VRef
            omega_ref = self.OmegaRef
            self.Args['p'][(self.NumStates+self.NumControls)*(k+1)-2 : (self.NumStates+self.NumControls)*(k+1)-2+self.NumStates] = [x_ref, y_ref, theta_ref]
            self.Args['p'][(self.NumStates+self.NumControls)*(k+1)+1 : (self.NumStates+self.NumControls)*(k+1)+1+self.NumControls] = [v_ref, omega_ref]

        # Initial value of the optimization variables
        self.Args['x0'] = ca.vertcat(
            ca.reshape(X0, -1, 1),
            ca.reshape(u0, -1, 1)
        )
        sol = self.Solver(x0=self.Args['x0'], lbx=self.Args['lbx'], ubx=self.Args['ubx'], 
                          lbg=self.Args['lbg'], ubg=self.Args['ubg'], p=self.Args['p'])
        self.Controls = np.reshape(sol['x'][self.NumStates * (self.N + 1):].full().T, (self.N, 2)) 

    def _cost_function(self, X, U, P):
        cost = 0
        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]

            # Cost function to minimize
            cost += ca.mtimes([(state - P[(self.NumStates+self.NumControls)*(k+1)-2 : (self.NumStates+self.NumControls)*(k+1)-2+self.NumStates]).T, self.Q, (state - P[(self.NumStates+self.NumControls)*(k+1)-2 : (self.NumStates+self.NumControls)*(k+1)-2+self.NumStates])]) + \
                    ca.mtimes([(control - P[(self.NumStates+self.NumControls)*(k+1)+1 : (self.NumStates+self.NumControls)*(k+1)+1+self.NumControls]).T, self.R, (control - P[(self.NumStates+self.NumControls)*(k+1)+1 : (self.NumStates+self.NumControls)*(k+1)+1+self.NumControls])])
        return cost

    def _nonlinear_constraints(self, f, X, U, P):
        # Initial condition constraints
        constraints = X[:, 0] - P[:self.NumStates]
        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]

            st_next = X[:, k+1]
            f_value = f(state, control)

            st_next_euler = state + self.sampleTime * f_value
            # Compute constraints
            constraints = ca.vertcat(constraints, st_next - st_next_euler)
        return constraints

    def _set_solver(self, cost, g, P, OPT_variables):
        nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g, 'p': P}

        opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,  # 0,3
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.Solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def _set_args(self):
        self.Args['lbg'] = ca.DM.zeros(self.NumStates * (self.N + 1), 1)  # Equality constraints
        self.Args['ubg'] = ca.DM.zeros(self.NumStates * (self.N + 1), 1)  # Equality constraints

        self.Args['lbx'] = ca.DM.zeros(self.NumStates * (self.N + 1) + self.NumControls * self.N, 1)
        self.Args['ubx'] = ca.DM.zeros(self.NumStates * (self.N + 1) + self.NumControls * self.N, 1)

        self.Args['lbx'][0:self.NumStates * (self.N + 1):self.NumStates] = self.PosXLim[0]  # state x lower bound
        self.Args['ubx'][0:self.NumStates * (self.N + 1):self.NumStates] = self.PosXLim[1]  # state x upper bound
        self.Args['lbx'][1:self.NumStates * (self.N + 1):self.NumStates] = self.PosYLim[0]  # state y lower bound
        self.Args['ubx'][1:self.NumStates * (self.N + 1):self.NumStates] = self.PosYLim[1]  # state y upper bound
        self.Args['lbx'][2:self.NumStates * (self.N + 1):self.NumStates] = self.OrientLim[0]  # state theta lower bound
        self.Args['ubx'][2:self.NumStates * (self.N + 1):self.NumStates] = self.OrientLim[1]  # state theta upper bound

        self.Args['lbx'][self.NumStates * (self.N + 1):self.NumStates * (self.N + 1) + 2 * self.N:2] = self.VelLim[0]  # v lower bound
        self.Args['ubx'][self.NumStates * (self.N + 1):self.NumStates * (self.N + 1) + 2 * self.N:2] = self.VelLim[1]  # v upper bound
        self.Args['lbx'][self.NumStates * (self.N + 1) + 1:self.NumStates * (self.N + 1) + 2 * self.N:2] = self.AngVelLim[0]  # omega lower bound
        self.Args['ubx'][self.NumStates * (self.N + 1) + 1:self.NumStates * (self.N + 1) + 2 * self.N:2] = self.AngVelLim[1]  # omega upper bound
