import numpy as np
import os
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

EXTENDED_EVALUATION = False
domain = np.array([[0, 5]])
np.random.seed(0)

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # plottings variables
        self.selected_x = []
        self.index = 0
        
        # Kernel f
        self.var_f = 0.5
        self.len_scale_f = 0.5
        self.nu_f = 2.5

        # Kernel v
        self.mean_v = 1.5
        self.var_v = 2**0.5
        self.len_scale_v = 0.5
        self.nu_v = 2.5

        # GP parameters
        self.gen_rand = False
        self.sigma_f = 0.15
        self.sigma_v = 0.0001
        self.randomstate = 0 if not self.gen_rand else np.random.randint(100)
        self.restart = 0
        length_scale_bounds=(1e-10, 100000.0)

        # General initializations
        self.v_min = 1.2
        self.beta = 2
        self.X = np.empty((0,1))
        self.output_accuracy = np.empty((0,1))
        self.V = np.empty((0,1))

        # GP f
        self.gp_f = GaussianProcessRegressor(
            kernel=
                Matern(length_scale=self.len_scale_f, nu=self.nu_f, length_scale_bounds=length_scale_bounds)+
                    WhiteKernel(noise_level=self.var_f), 
            alpha=self.sigma_f**2,
            normalize_y=True,
            n_restarts_optimizer=self.restart,
        )
        # GP v
        self.gp_v = GaussianProcessRegressor(
            kernel=
                Matern(length_scale=self.len_scale_v, nu=self.nu_v,length_scale_bounds=length_scale_bounds)+
                    ConstantKernel(constant_value=self.mean_v)+
                        WhiteKernel(noise_level=self.var_v), 
            alpha=self.sigma_v**2,
            normalize_y=True,
            n_restarts_optimizer=self.restart,
        )
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if self.X.size == 0:
            # if no point has been sampled yet, we can't optimize the acquisition function yet
            # we instead sample a random starting point in the domain
            x0 = np.random.random()*domain.squeeze()[-1]
            next_x = x0[np.newaxis, :]
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


    def acquisition_function(self, x, method="UCB"):
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
        if method == "UCB":
            pred_mean_f, pred_stddev_f = self.gp_f.predict(x[np.newaxis,:], return_std=True)
            ucb_f = pred_mean_f + self.beta * pred_stddev_f  # Calculate UCB.
        elif method == "EI":
            # de-mean data?
            
            #two strategies for picking t possible
            # a) t is minimum over previous observations
            # b) t is minimum of expected value of the objective
            if self.t_choice == 'observations':
                t = np.array(self.previous_points)[:,2].min()
            elif self.t_choice == 'expectation':
                t = self.objective_model.predict(self.theta_sample).min()
            
            # constraint
            c_mean, c_std = self.constraint_model.predict(np.atleast_2d(x), return_std=True)
            prob_constraint = norm.cdf(0, loc=c_mean, scale=c_std)
            
            # objective
            y_mean, y_std = self.objective_model.predict(np.atleast_2d(x), return_std=True)
            z_x = (t - y_mean - self.xi)/y_std

            ei_x =  y_std *(z_x * norm.cdf(z_x) + norm.pdf(z_x))
            
            return prob_constraint * ei_x

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
        self.output_accuracy = np.vstack((self.output_accuracy, f))
        self.V = np.vstack((self.V, v))

        self.gp_f.fit(self.X, self.output_accuracy)
        self.gp_v.fit(self.X, self.V)



    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        valid_f = self.output_accuracy
        valid_f[self.V < self.v_min] = -1e6 #shorthand for -np.inf
        optmal_x = self.X[np.argmax(valid_f)]
        print(f"valid_f: {valid_f[self.V > self.v_min]} \n \
            with the highest accuracy: {np.max(valid_f)} \n \
                and corresponding x: {optmal_x} \n \
                    and velocity {self.V[self.V > self.v_min]}")
        
        plt.plot(self.X, color='black')
        plt.plot(self.output_accuracy, color='red')
        plt.plot(self.V, color='blue')
        plt.show()
        #plt.savefig(f'{os.getcwd()}/images/x_points_{self.index}.png')
        #plt.savefig(f'x_points_{self.index}.png')


        self.index += 1

        return optmal_x
        

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
