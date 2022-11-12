import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

domain = np.array([[0, 5]])
v_min = 1.2


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # Kernel f
        self.var_f = 0.5
        self.len_scale_f = 0.5
        self.nu_f = 2.5
        # Kernel v
        self.mean_v = 1.5
        self.var_v = math.sqrt(2)
        self.len_scale_v = 0.5
        self.nu_v = 2.5

        self.alpha = 1e-6
        self.restart = 30
        # GP f
        self.gp_f = GaussianProcessRegressor(
            kernel=Matern(length_scale=self.len_scale_f, nu=self.nu_f), 
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.restart,
            #random_state=self._random_state,
        )
        # GP v
        self.gp_v = GaussianProcessRegressor(
            kernel=ConstantKernel(constant_value=self.mean_v)
            + Matern(length_scale=self.len_scale_v, nu=self.nu_v), 
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.restart,
            #random_state=self._random_state,
        )
        
        
        self.beta = 2
        self.X = np.empty((0,1))
        self.Y = np.empty((0,1))
        self.V = np.empty((0,1))

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
        if self.X.size == 0:
            # if no point has been sampled yet, we can't optimize the acquisition function yet
            # we instead sample a random starting point in the domain
            #next_x = np.array([x0]).reshape(-1, domain.shape[0])
            next_x[np.newaxis, :]
        else:
            if len(self.Y) == 12 and np.all(self.Y < 0.4):
                # if after 10 iterations we have not found a point with a accuracy larger than 0.3
                # we take a random point in the domain as next point
                x0 = (self.X[0] + (domain[:, 1] - domain[:, 0])/2) % domain[:, 1]
                next_x = np.array([x0]).reshape(-1, domain.shape[0])
            else:
                next_x = self.optimize_acquisition_function()

        assert next_x.shape == (1, domain.shape[0])
        return next_x


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])


    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        
        # def update_acquisition_function(self):        
        pred_mean_f, pred_stddev_f = self.gp_f.predict(x[np.newaxis,:], return_std=True)
        ucb_f = pred_mean_f + self.beta * pred_stddev_f  # Calculate UCB.

        return ucb_f.squeeze()
 

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, f))
        self.V = np.vstack((self.V, v))

        self.gp_f.fit(self.X, self.Y)
        self.gp_v.fit(self.X, self.V)



    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        #plt.plot(self.V)

        self.Y[self.V < v_min] = -1e-10 #shorthand for -np.inf
        highest_y = np.argmax(self.Y)

        return self.X[highest_y]
        



""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()
    n_dim = domain.shape[0]
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
