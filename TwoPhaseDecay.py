import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")



def exp_decay(x, a, b, c):
    """
    Normal exponential decay
    """
    return c + a * np.exp(-b * x)


def bi_exp_decay(x, Y0, Plateau, PercentFast, b1, b2):
    """
    bi-exponential decay / twophase decay
    """
    return (
        Plateau
        + ((Y0 - Plateau) * PercentFast / 100) * np.exp(-b1 * x)
        + ((Y0 - Plateau) - (((Y0 - Plateau) * PercentFast / 100))) * np.exp(-b2 * x)
    )


class bi_exp_regression:
    """
    Perform bi-exponential fit of input x,y
    Multiple fits are made and the model with lowest sum of squares is chosen
    as the best model

    Parameters:
    -----------
    x : array-like
        x-values
    y : array-like
        y-values
    iterations : int
        number of times to fit data
    p0 : list
        list of initial paramters
    verbose : bool
        whether or not to verbose

    Attributes:
    -----------
    best_model : list
        list of the best parameters to use
    plot_best_model : matplot.lib figure
        plots the data and the best model
    """

    def __init__(self, x, y, iterations, p0=None, verbose=False, max_n_fits=5):
        self.x = x
        self.y = y
        self.iterations = iterations
        self.verbose = verbose

        self.initial_parameters = p0
        self.found_parameters = {}
        self.ssq = {}
        self.best_params = []

        self.fig = 0

        self.n_fits = 0
        self.max_n_fits = max_n_fits

        self.multiparams = []

    def validated(self):
        """
        Check if x,y has same dimensions
        """
        if not self.x.shape == self.y.shape:
            print("Dimension Missmatch")
            return False

    def exp_fit(self):
        """
        Perform normal exponential decay to get inital parameters
        """
        popt, pcov = op.curve_fit(f=exp_decay, xdata=self.x, ydata=self.y)
        a, b, c = popt
        self.initial_parameters = [np.max(self.y), c, a, b, b]

    def bi_exp_fit(self, i):
        """
        Perform bi-exponential decay fit of data
        """
        if self.initial_parameters is None:
            print("Initial parameters not set")
            print("Using parameters from single exp decay")
            self.exp_fit()
        p0 = self.initial_parameters + np.random.normal(
            loc=0, scale=20, size=len(self.initial_parameters)
        )
        popt, pcov = op.curve_fit(f=bi_exp_decay, xdata=self.x, ydata=self.y, p0=p0)
        Y0, Plateau, PercentFast, b1, b2 = popt
        conditions = (Plateau > 0) & (100 > PercentFast > 0) & (b1 > b2 > 0)
        if conditions:
            self.found_parameters[i] = popt
            self.multiparams.append(popt)
            # print()
            # print("------------------------------------------------------------------------------------")
            # print(popt)
            # print()
            # print(self.found_parameters)
            # print(self.multiparams)
            # print("------------------------------------------------------------------------------------")
            # print()
            return (popt, i)

    def sum_squares_model(self, param):
        """
        Check the sum of squares for found parameters
        """
        # Unpack parameters
        Y0, Plateau, PercentFast, KFast, KSlow = param
        # calculate y values from function
        yfunc = list(
            map(
                lambda x: bi_exp_decay(x, Y0, Plateau, PercentFast, KFast, KSlow),
                self.x,
            )
        )
        # Convert to numpy array
        yfunc = np.array(yfunc)

        # Difference
        diff = self.y - yfunc
        # Squared Difference
        sqdiff = diff * diff
        # Sum of squares
        ssq = np.sum(sqdiff)
        return ssq

    def update_ssq(self):
        """
        For each model parameter found calculate the sum of suqares
        """
        for modelID, model_param in self.found_parameters.items():
            # Calculate ssq
            ssq = self.sum_squares_model(model_param)
            # Add to dictionary
            self.ssq[modelID] = ssq

    def best_model(self):
        """
        Get the parameters of the best model
        """
        self.update_ssq()
        try:
            best_model_id = min(self.ssq, key=self.ssq.get)
            best_params = self.found_parameters[best_model_id]
            self.best_params = best_params
            Y0, Plateau, PercentFast, KFast, KSlow = best_params
            if self.verbose:
                print()
                print("Model ID", best_model_id)
                print("Y0: ", Y0)
                print("Plateau: ", Plateau)
                print("PercentFast: ", PercentFast)
                print("KFast", KFast)
                print("KSlow", KSlow)
                print()
                print("Faction Fast: ", ((Y0 - Plateau) * PercentFast / 100))
                print(
                    "Faction Slow: ",
                    ((Y0 - Plateau) - (((Y0 - Plateau) * PercentFast / 100))),
                )
                print("HalfLife Fast: ", np.log(2) / KFast)
                print("HalfLife Slow: ", np.log(2) / KSlow)
                print()
        except:
            print("No model found run again")
            if self.n_fits < self.max_n_fits:
                self.n_fits = self.n_fits + 1
                self.fit()
                # self.best_model()
        return self.best_params

    def multifit(self, i):
        try:
            res = self.bi_exp_fit(i=i)
            return res
        except:
            if self.verbose:
                print("Could not fit model: ", i)

    def update_params(self, lst):
        """
        Update the found params dictionary
        """
        for result in lst:
            popt, idx = result
            self.found_parameters[idx] = popt

    def fit(self):
        """
        Run <self.iterations> iterations of biexpfit in paralell
        """
        self.exp_fit()
        with mp.Pool() as pool:
            results = pool.map(self.multifit, [i for i in range(self.iterations)])
        results = np.array(results)
        results = [value for value in results if value is not None]
        self.update_params(results)
        self.update_ssq()
        self.best_model()

    def kinetics(self):
        """
        Return the special model kinetics
        """
        params = self.best_params
        # Unpack parameters
        Y0, Plateau, PercentFast, KFast, KSlow = params
        fraction_fast = (Y0 - Plateau) * PercentFast / 100
        fraction_slow = (Y0 - Plateau) - (((Y0 - Plateau) * PercentFast / 100))
        half_life_fast = np.log(2) / KFast
        half_life_slow = np.log(2) / KSlow
        if self.verbose:
            print()
            print("Y0: ", Y0)
            print("Plateau: ", Plateau)
            print("PercentFast: ", PercentFast)
            print("KFast", KFast)
            print("KSlow", KSlow)
            print()
            print("Faction Fast: ", fraction_fast)
            print("Faction Slow: ", fraction_slow)
            print("HalfLife Fast: ", half_life_fast)
            print("HalfLife Slow: ", half_life_slow)
            print()
        return fraction_fast, fraction_slow, half_life_fast, half_life_slow

    def plot_best_model(self):
        """
        plot the best model
        """
        param = self.best_params
        # Unpack parameters
        Y0, Plateau, PercentFast, KFast, KSlow = param
        # x,y values to plot
        xs = np.linspace(np.min(self.x), np.max(self.x), 200)
        # calculate y values from function
        ys = list(
            map(lambda x: bi_exp_decay(x, Y0, Plateau, PercentFast, KFast, KSlow), xs)
        )

        self.fig = plt.figure()
        plt.scatter(self.x, self.y, c="gray", alpha=0.5)
        plt.plot(xs, ys)
        plt.title("Best Model")
        plt.show()

    def save_plot(self, filename):
        """
        Save the figure
        """
        if self.fig == 0:
            self.plot_best_model()
        self.fig.savefig(filename + ".png")


if __name__ == "__main__":
    # Scipy example
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    rng = np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise

    bi = bi_exp_regression(x=xdata, y=ydata, iterations=100, verbose=True)
    bi.fit()
    bi.plot_best_model()
    print(bi.best_params)
    print()
    print()