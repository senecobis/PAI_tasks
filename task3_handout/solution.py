import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

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

        # General initializations
        self.beta = 2
        self.v_min = 1.2
        self.counter = 0
        self.unsafe_counter = 0
        self.X = np.empty((0, 1))
        self.F = np.empty((0, 1))
        self.V = np.empty((0, 1))

        # Kernel f
        noise_f = 0.15
        self.ker_f = 0.5 * Matern(length_scale=0.5, nu=2.5)

        # Kernel v
        noise_v = 1e-4
        self.ker_v = np.sqrt(2) * Matern(length_scale=0.5, nu=2.5) + ConstantKernel(1.5)

        self.gp_f = GaussianProcessRegressor(kernel=self.ker_f, alpha=noise_f**2)
        self.gp_v = GaussianProcessRegressor(kernel=self.ker_v, alpha=noise_v**2)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        """if self.unsafe_counter > 1:
            return 0"""
        if self.X.size == 0:
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
                domain.shape[0]
            )
            next_x = np.array([x0]).reshape(-1, domain.shape[0])
        else:
            if (
                len(self.F) == 12
                and np.all(self.V < 1.2 + 0.2)
                and np.all(self.F < 0.5)
            ):
                x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
                    domain.shape[0]
                )
                next_x = np.array([x0]).reshape(-1, domain.shape[0])
            else:
                next_x = self.optimize_acquisition_function()


        return np.atleast_2d(next_x)

    def optimize_acquisition_function(self, num_points=300):

        test_x = np.random.uniform(low=0, high=5, size=num_points)
        val_x = [
            self.acquisition_function(x, v_penalty=True)
            for x in test_x
            if x not in self.X
        ]
        best_x_idx = np.argmax(val_x)

        return test_x[best_x_idx]

    def acquisition_function(self, x, UCB=False, v_penalty=False):
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
        if UCB is True:
            mean_f, stddev_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
            mean_v, stddev_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)
            ucb_f = mean_f + self.beta * stddev_f
            min_possible_v = mean_v - (2 + self.unsafe_counter) * stddev_v
            ucb_mod = (0.9 * ucb_f) if (min_possible_v > self.v_min) else 0.1 * ucb_f
            af_value = ucb_mod if (v_penalty) else ucb_f
        else:
            ei = self.expected_improvement(x, eps=0.015)
            constraint_weight = self.constraint_function(x)
            af_value = float(ei * constraint_weight)

        return af_value

    def expected_improvement(self, x, eps=0.01):
        """
        Compute expected improvement at points x based on samples x_samples
        and y_samples using Gaussian process surrogate
        Args:
            x: Points at which EI should be computed
            eps: Exploitation-exploration trade-off parameter
        """

        mean_f, stddev_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
        mean_F = self.gp_f.predict(self.X)

        stddev_f = stddev_f.reshape(-1, 1)
        mean_sample_opt = np.max(mean_F)
        with np.errstate(divide="warn"):
            imp = mean_f - mean_sample_opt - eps
            Z = imp / stddev_f
            ei = imp * norm.cdf(Z) + stddev_f * norm.pdf(Z)
            ei[stddev_f == 0.0] = 0.0

        return ei

    def constraint_function(self, x):
        """
        https://arxiv.org/abs/1403.5607
        """

        # predict distribution of speed v
        mean_v, stddev_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)

        # Gaussian CDF with params from GPR prediction
        if stddev_v != 0:
            pr = 1 - norm.cdf(self.v_min, loc=mean_v, scale=stddev_v)
        else:
            pr = (
                0.98 * (mean_v - self.v_min)
                if mean_v >= self.v_min
                else 0.02 * (self.v_min - mean_v)
            )

        return pr

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
        self.counter += 1
        if v < self.v_min:
            self.unsafe_counter += 1
        self.X = np.vstack((self.X, x))
        self.F = np.vstack((self.F, f))
        self.V = np.vstack((self.V, v))

        self.gp_f.fit(X=self.X.reshape(-1, 1), y=self.F)
        self.gp_v.fit(X=self.X.reshape(-1, 1), y=self.V)

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

        # print("Done!")
        # time.sleep(5)

        return optimal_x

    def plot_stuff(self):
        print(f"plotting {self.counter:03}")
        test_x = np.linspace(start=0, stop=5, num=100)

        pred_mean_f, pred_stddev_f = self.gp_f.predict(
            test_x.reshape(-1, 1), return_std=True
        )
        pred_mean_v, pred_stddev_v = self.gp_v.predict(
            test_x.reshape(-1, 1), return_std=True
        )

        # plot f_hat_mean(blue) ucb (red) ucb_modified (green)
        plt.figure()
        ucb = self.acquisition_function(test_x)
        plt.plot(test_x, ucb, "r")
        plt.plot(test_x, pred_mean_f, "b")
        plt.plot(self.X, self.F, "k+")
        plt.savefig(f"abc/plot_{self.counter:03}f.jpg")

        # plot velocity + uncertainty
        plt.figure()
        vel_up = [pred_mean_v[i] + pred_stddev_v[i] for i in range(100)]
        vel_down = [pred_mean_v[i] - pred_stddev_v[i] for i in range(100)]
        plt.plot(test_x, pred_mean_v, "b")
        plt.plot(test_x, vel_up, "r")
        plt.plot(test_x, vel_down, "r")
        plt.plot(self.X, self.V, "k+")
        plt.savefig(f"abc/plot_{self.counter:03}v.jpg")


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
