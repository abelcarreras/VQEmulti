from scipy.optimize import OptimizeResult
from vqemulti.preferences import Configuration


class OptimizerParams:
    def __init__(self,
                 method='BFGS',
                 options=None):

        self._method = method
        self._options = options

        if self._options is None:
            self._options = {}

        # set display connected to verbose
        self._options['disp'] = Configuration().verbose

    def __str__(self):
        return '{} / {}'.format(self._method, str(self._options))

    @property
    def method(self):
        return self._method

    @property
    def options(self):
        return self._options


def spsa_minimizer(fun, x0, args=None, **options):
    """
    wrapper of sps implementation module to works as a custom method for scipy minimizer
    :param fun: optimization function
    :param x0: initial guess
    :param args: optimization function additional arguments
    :param options: additional options (not used for now)
    :return:
    """
    import spsa
    global nfev
    nfev = 0
    def full_function(x0):
        global nfev
        nfev += 1
        return fun(x0, *args)

    data = spsa.minimize(full_function, x0)

    return OptimizeResult(fun=full_function(data), x=data, nit=0, nfev=nfev, success=True)
