from scipy.optimize import OptimizeResult
from vqemulti.preferences import Configuration
import numpy as np


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


def disp_message_print(f_val, success, iterations):
    if success:
        print('Optimization terminated successfully.')
    else:
        print('Optimization terminated unsuccessfully.')

    print('     Current function value:', f_val)
    print('     Iterations:', iterations)
    print('     Gradient evaluations:', iterations)


def sgd(fun, x0, jac, args=(), **options):
    """
    ``scipy.optimize.minimize`` compatible implementation of stochastic gradient descent with momentum.
     Adapted from https://gist.github.com/jcmgray/

    :param fun: optimization function
    :param x0: initial guess
    :param args: optimization function additional arguments
    :param options: additional options (not used for now)
    :return: OptimizeResult
    """
    # default parameters
    params = {'learning_rate': 0.01,
              'mass': 0.9,
              'maxiter': 1000,
              'gtol': 1-5}

    params.update(options)

    learning_rate = params['learning_rate']
    mass = params['mass']

    x = x0
    velocity = np.zeros_like(x)
    i = 0
    success = False

    for i in range(params['maxiter']):
        g = np.array(jac(x, *args))

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

        if abs(np.linalg.norm(g)) < params['gtol']:
            success = True
            break

    i += 1

    if options['disp']:
        disp_message_print(fun(x, *args), success, i)

    return OptimizeResult(x=x, fun=fun(x, *args), jac=g, nit=i, nfev=i, success=True)


def rmsprop(fun, x0, jac, args=(), **options):
    """
    ``scipy.optimize.minimize`` compatible implementation of root mean squared prop: See Adagrad paper for details.
     Adapted from https://gist.github.com/jcmgray/

    :param fun: optimization function
    :param x0: initial guess
    :param args: optimization function additional arguments
    :param options: additional options (not used for now)
    :return: OptimizeResult
    """

    # default parameters
    params = {'learning_rate': 0.01,
              'gamma': 0.9,
              'eps': 1e-6,
              'maxiter': 1000,
              'gtol': 1-5}

    params.update(options)

    learning_rate = params['learning_rate']
    gamma = params['gamma']
    eps = params['eps']

    x = x0
    avg_sq_grad = np.ones_like(x)
    i = 0
    success = False

    for i in range(params['maxiter']):
        g = np.array(jac(x, *args))

        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

        if abs(np.linalg.norm(g)) < params['gtol']:
            success = True
            break

    i += 1

    if options['disp']:
        disp_message_print(fun(x, *args), success, i)

    return OptimizeResult(x=x, fun=fun(x, *args), jac=g, nit=i, nfev=i, success=True)


def adam(fun, x0, jac, args=(), **options):
    """
    ``scipy.optimize.minimize`` compatible implementation of ADAM - [http://arxiv.org/pdf/1412.6980.pdf].
     Adapted from https://gist.github.com/jcmgray/

    :param fun: optimization function
    :param x0: initial guess
    :param args: optimization function additional arguments
    :param options: additional options (not used for now)
    :return: OptimizeResult
    """
    # default parameters
    params = {'learning_rate': 0.01,
              'beta1': 0.9,
              'beta2': 0.999,
              'eps': 1e-6,
              'maxiter': 1000,
              'gtol': 1-5}

    params.update(options)

    learning_rate = params['learning_rate']
    beta1 = params['beta1']
    beta2 = params['beta2']
    eps = params['eps']

    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    i = 0
    success = False

    for i in range(params['maxiter']):
        g = np.array(jac(x, *args))

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        if abs(np.linalg.norm(g)) < params['gtol']:
            success = True
            break

    i += 1

    if options['disp']:
        disp_message_print(fun(x, *args), success, i)

    return OptimizeResult(x=x, fun=fun(x, *args), jac=g, nit=i, nfev=i, success=success)


def spsa_minimizer(fun, x0, args=(), **options):
    """
    wrapper of sps implementation module to works as a custom method for scipy minimizer

    :param fun: optimization function
    :param x0: initial guess
    :param args: optimization function additional arguments
    :param options: additional options (not used for now)
    :return: OptimizeResult
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
