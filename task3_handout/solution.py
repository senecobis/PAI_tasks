import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

EXTENDED_EVALUATION = False
domain = np.array([[0, 5]])
np.random.seed(1)

""" Solution """


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        def warn(*args, **kwargs):
            pass

        import warnings

        warnings.warn = warn

        # Kernel f
        self.var_f = 0.5
        self.len_scale_f = 0.5
        self.nu_f = 2.5
        self.noise_f = 0.15
        self.ker_f = Matern(length_scale=self.len_scale_f, nu=self.nu_f)

        # Kernel v
        self.mean_v = 1.5
        self.var_v = np.sqrt(2)
        self.len_scale_v = 0.5
        self.nu_v = 2.5
        self.noise_v = 1e-4
        self.ker_v = Matern(length_scale=self.len_scale_v, nu=self.nu_v) + self.mean_v

        # GP parameters
        # self.gen_rand = False
        # self.sigma_f = 0.15
        # self.sigma_v = 0.0001
        # self.restart = 0
        # length_scale_bounds=(1e-10, 100000.0)

        # General initializations
        self.v_min = 1.2
        self.beta = 2
        self.X = np.empty((0, 1))
        self.F = np.empty((0, 1))
        self.V = np.empty((0, 1))

        self.gp_f = GaussianProcessRegressor(kernel=self.ker_f, alpha=self.noise_f**2)
        self.gp_v = GaussianProcessRegressor(kernel=self.ker_v, alpha=self.noise_v**2)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # self.plot_stuff()

        self.gp_f.fit(X=self.X.reshape(-1, 1), y=self.F)
        self.gp_v.fit(X=self.X.reshape(-1, 1), y=self.V)

        next_x = self.optimize_acquisition_function()

        return np.array(next_x).reshape(1, 1)

    def optimize_acquisition_function(self, num_points=200):

        test_x = np.linspace(start=0, stop=5, num=num_points)
        test_x = np.array([y for y in test_x if y not in self.X])
        val_x = self.acquisition_function(test_x)
        best_x_idx = np.argmax(val_x)

        return test_x[best_x_idx]

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

        def objective(f_mean, f_std, v_mean, v_std, v_penalty=True):
            ucb_f = f_mean + self.beta * f_std
            v_penalty = (
                2 * ucb_f if (v_mean - v_std < self.v_min and v_penalty) else 0.0
            )
            return ucb_f - v_penalty

        pred_mean_v, pred_stddev_v = self.gp_v.predict(
            x.reshape(-1, 1), return_std=True
        )
        pred_mean_f, pred_stddev_f = self.gp_f.predict(
            x.reshape(-1, 1), return_std=True
        )

        acquisition_val = [
            objective(
                pred_mean_f[i], pred_stddev_f[i], pred_mean_v[i], pred_stddev_v[i]
            )
            for i in range(len(pred_mean_f))
        ]

        return acquisition_val

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
        # print(np.around(self.X.reshape((-1)), decimals=2))
        self.X = np.vstack((self.X, x))
        self.F = np.vstack((self.F, f))
        self.V = np.vstack((self.V, v))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        valid_f = self.F
        valid_f[self.V < self.v_min] = -np.inf
        optimal_x = self.X[np.argmax(valid_f)]

        return optimal_x

    def plot_stuff(self):
        test_x = np.linspace(start=0, stop=5, num=100)

        pred_mean_f, pred_stddev_f = self.gp_f.predict(
            test_x.reshape(-1, 1), return_std=True
        )
        pred_mean_v, pred_stddev_v = self.gp_v.predict(
            test_x.reshape(-1, 1), return_std=True
        )

        # plot f_hat_mean(blue) ucb (red) ucb_modified (green)
        plt.figure()
        id = np.random.randint(0, 100)
        print(f"plotting {id}")
        ucb_normal = [pred_mean_f[i] + self.beta * pred_stddev_f[i] for i in range(100)]
        ucb_mod = [
            ucb_normal[i] if pred_mean_v[i] - pred_stddev_v[i] > self.v_min else 0
            for i in range(100)
        ]
        plt.plot(test_x, ucb_normal, "r")
        plt.plot(test_x, ucb_mod, "g")
        plt.plot(test_x, pred_mean_f, "b")
        plt.plot(self.X, self.F, "k+")
        plt.savefig(f"abc/plot_f{id}.jpg")

        # plot velocity + uncertainty
        plt.figure()
        vel_up = [pred_mean_v[i] + pred_stddev_v[i] for i in range(100)]
        vel_down = [pred_mean_v[i] - pred_stddev_v[i] for i in range(100)]
        plt.plot(test_x, pred_mean_v, "b")
        plt.plot(test_x, vel_up, "r")
        plt.plot(test_x, vel_down, "r")
        plt.plot(self.X, self.V, "k+")
        plt.savefig(f"abc/plot_v{id}.jpg")

        time.sleep(3)


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return -np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


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
        assert x.shape == (1, domain.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {domain.shape[0]})"
        )

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)
        data.append(x.squeeze())
        label.append(obj_val)

    plt.plot(data, label, "o", color="black")
    plt.show()

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), (
        f"The function get solution must return a numpy array of shape ("
        f"1, {domain.shape[0]})"
    )
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the "
        f"domain, {solution} returned instead"
    )

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = 0 - f(solution)

    print(
        f"Optimal value: 0\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret{regret}"
    )


if __name__ == "__main__":
    main()
