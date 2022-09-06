import datetime as dt
import random
import warnings


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


warnings.filterwarnings("ignore")


class ExponentialRegression:
    """
    Ordinary least squares bi-exponential regression

    ExponentialRegression fits a bi exponential model with coefficients w = #TODO
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the exponential approximation.

    Parameters:
    -----------
    p0 : list or Nonetype, default = None
        intial parameters

    iterations : int
        number of fits to perform before selecting best model

    Attributes:
    -----------
    coef_ : array of shape ()#TODO num of parameters
        Estimated coefficients for the linear regression problem
    """

    def __init__(self, p0=None, iterations=100):
        self.p0 = p0
        self.iterations = iterations

        self.coef_ = np.array([])
        self.ssq_opt = 99999999
        self.model_number_opt = 0

        self.kinetics = np.array([])

        self.refits = 0

    def fit(self, x, y, early_stop=True):
        """
        Fit Bi-exponential model

        Parameters:
        -----------
        x : array of shape (n_samples, 1)
            Training data

        y : array of shape (n_samples, 1)
            Target Values

        early_stop : Bool, default = True
            stop fitting and return model if a better model has not been found in 10 iterations
        Returns:
        --------
        self : object
            Fitted Estimator.
        """
        self.x = x.astype(float)
        self.y = y.astype(float)
        # if self.valid_input():
        if self.p0 is None:
            self.initial_fit()
            self.initial_ssq()
        for model_number in range(self.iterations):
            try:
                random.seed(dt.datetime.now())
                p0 = self.p0 + np.random.normal(loc=0, scale=15, size=len(self.p0))
                p_opt, p_cov, infodict, mesg, ier = op.curve_fit(
                    f=self.bi_exp_decay,
                    xdata=self.x,
                    ydata=self.y,
                    p0=p0,
                    full_output=True,
                    bounds=(0, [100, 100, 100, 100, 100]),
                )
            except:
                continue
            if self.valid_params(p_opt):
                ssq = (infodict["fvec"] ** 2).mean()
                if ssq < self.ssq_opt:
                    self.ssq_opt = ssq
                    self.coef_ = p_opt
                    self.model_number_opt = model_number
                elif (early_stop) and (
                    np.abs(model_number - self.model_number_opt) > 25
                ):
                    print("Stopped early", model_number)
                    break
                else:
                    # print("Could not fit model:", model_number)
                    None
        if len(self.coef_) > 0:
            Y0, Plateau, PercentFast, KFast, KSlow = self.coef_
            fraction_fast = (Y0 - Plateau) * PercentFast / 100
            fraction_slow = (Y0 - Plateau) - (((Y0 - Plateau) * PercentFast / 100))
            half_life_fast = np.log(2) / KFast
            half_life_slow = np.log(2) / KSlow

            self.kinetics = [
                fraction_fast / 100,
                fraction_slow / 100,
                half_life_fast,
                half_life_slow,
            ]

            self.model = lambda x: self.bi_exp_decay(
                x, Y0, Plateau, PercentFast, KFast, KSlow
            )

        elif self.refits<3:
            print("+------------------------------+")
            print("| No model found fitting again |")
            print("+------------------------------+")
            self.fit(self.x, self.y)
            self.refits = self.refits + 1

    def valid_input(self):
        """
        Check if the input is valid
        """
        if len(self.x) != len(self.y):
            print("Dimension missmatch")
            print("Dimension of:")
            print(" - x values: ", np.shape(self.x))
            print(" - y values: ", np.shape(self.y))
            return False

    def exp_decay(self, x, a, b, c):
        """
        Normal exponential decay
        """
        return c + a * np.exp(-b * x)

    def bi_exp_decay(self, x, Y0, Plateau, PercentFast, KFast, KSlow):
        """
        bi-exponential decay / twophase decay
        """
        return (
            Plateau
            + ((Y0 - Plateau) * PercentFast / 100) * np.exp(-KFast * x)
            + ((Y0 - Plateau) - (((Y0 - Plateau) * PercentFast / 100)))
            * np.exp(-KSlow * x)
        )

    def initial_fit(self):
        """
        Fit a standard exponential model and save the optimal parameters
        """
        p_opt, p_cov = op.curve_fit(f=self.exp_decay, xdata=self.x, ydata=self.y)
        a, b, c = p_opt
        max_y = np.max(self.y)
        self.p0 = [max_y, c, a, b * 3, b]

    def initial_ssq(self):
        """
        Initial SSq value from the initial fit p0
        """
        Y0, plateau, pfast, kfast, kslow = self.p0
        ys = self.bi_exp_decay(self.x, Y0, plateau, pfast, kfast, kslow)

        self.ssq_opt = np.mean((self.y - ys) ** 2)

    def valid_params(self, params):
        """
        Validate if a list of parameters are valid for a bi exponential model
        """
        Y0, Plateau, PercentFast, KFast, KSlow = params
        isValid = (Plateau > 0) & (100 > PercentFast > 0) & (KFast > KSlow > 0)
        return isValid

    def model(self):
        """
        Return the model as a lambda expression
        """
        params = self.coef_
        Y0, Plateau, PercentFast, KFast, KSlow = params
        return lambda x: self.bi_exp_decay(
            x=x,
            Y0=Y0,
            Plateau=Plateau,
            PercentFast=PercentFast,
            KFast=KFast,
            KSlow=KSlow,
        )


if __name__ == "__main__":
    # Scipy example
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    rng = np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise

    fig = plt.figure()
    start = dt.datetime.now()

    reg = ExponentialRegression(iterations=200)
    reg.fit(xdata, ydata)
    model = reg.model

    print("Coeficients: ", reg.coef_)
    print("Kinetics: ", reg.kinetics)
    print("model evaluated at 3.1415: ", model(3.1415))

    end = (dt.datetime.now() - start).total_seconds()
    print("Runtime: ", end)

    plt.scatter(xdata, ydata, c="gray")
    plt.plot(xdata, reg.model(xdata))
    plt.show()
