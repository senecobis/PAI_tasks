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

        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn

        # plottings variables
        self.selected_x = []
        self.selected_y = []
        self.selected_v = []
        self.perc_of_unsafe_eval = 0
        
        # Kernel f
        self.var_f = 0.5
        self.len_scale_f = 0.5
        self.nu_f = 2.5
        self.ker_f = self.var_f*Matern(length_scale=self.len_scale_f, nu=self.nu_f)

        # Kernel v
        self.mean_v = 1.5
        self.var_v = np.sqrt(2)
        self.len_scale_v = 0.5
        self.nu_v = 2.5
        self.ker_v = self.var_v*Matern(length_scale=self.len_scale_v, nu=self.nu_v)+ConstantKernel(constant_value=self.mean_v)

        # General initializations
        self.v_min = 1.2
        self.beta = 5
        self.X = np.empty((0,1))
        self.f = np.empty((0,1))
        self.V = np.empty((0,1))

        self.gp_f = GaussianProcessRegressor(kernel=self.ker_f,alpha=0.001)
        self.gp_v = GaussianProcessRegressor(kernel=self.ker_v,alpha=0.001)

    def next_safe_random_point(self):

        # TODO: add the possibility to do unsafe eval but less then 5 percent
        bounds = np.squeeze(domain)
        ind=0
        # take a point at random that is greater than the one evaluated
        for _ in range(1000):
            next_x = np.array([[np.random.uniform(bounds[0], bounds[-1])]])
            if self.gp_v.predict(np.vstack(self.X, next_x)) < self.v_min:
                print(f"random round, predicted x {next_x} \
                    and vel {self.gp_v.predict(next_x)}")
                ind+=1
            else:
                print(f"\n --- save x {next_x}\
                        found with vel {self.gp_v.predict(next_x)}")

        print(f"\n --- reinitialized {ind}\
                times with a final vel of {self.gp_v.predict(next_x)}")

        return next_x

        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # - In implementing this function, you may use optimize_acquisition_function() defined below.
        # - Predict next point optimazing our a(x) function
        # if the evaluation is unsafe then take a point at random

        next_x = self.optimize_acquisition_function()
        corresponding_v = self.gp_v.predict(next_x)
        
        if corresponding_v < self.v_min:
            print(f"\n unsafe next point {next_x} with velocity {corresponding_v}")
            #next_x = self.next_safe_random_point()
        return next_x


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        x_values = np.linspace(domain[:, 0], domain[:, 1], num=1000)
        f_values = np.zeros(x_values.shape)
        for i, x in enumerate(x_values):
            f_values[i] = self.acquisition_function(x)
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
            self.gp_f.fit(self.X, self.f)
            self.gp_v.fit(self.X, self.V)
            pred_mean_f, pred_stddev_f = self.gp_f.predict(x[np.newaxis,:], return_std=True)
            return (pred_mean_f + self.beta * pred_stddev_f).squeeze() 

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

        #return ucb_f.squeeze()
 

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
        self.selected_x.append(int(x.squeeze()))
        self.selected_y.append(int(f.squeeze()))
       #self.selected_v.append(int(v.squeeze()))

        self.X = np.vstack((self.X, x))
        self.f = np.vstack((self.f, f))
        self.V = np.vstack((self.V, v))


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        valid_f = self.f
        #valid_f[self.V < self.v_min] = -1e6 #shorthand for -np.inf
        optmal_x = self.X[np.argmax(valid_f)]

        print(f"Selected {len(self.selected_x)} points: {self.selected_x} \n \
                and corresponding accuracies: {self.selected_y} \n \
                and velocity {self.V}")
        
        #plt.plot(self.X, color='black')
        #plt.plot(self.f, color='red')
        #plt.plot(self.V, color='blue')
        #plt.show()
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
    data = []
    label = []
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
        data.append(x.squeeze())
        label.append(obj_val)

    plt.plot(data, label, 'o', color='black')
    plt.show()

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
