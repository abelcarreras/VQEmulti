from scipy.optimize import OptimizeResult


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

    def full_function(x0):
        return fun(x0, *args)

    data = spsa.minimize(full_function, x0)
    print('data:', data)

    return OptimizeResult(fun=full_function(data), x=data, nit=0, nfev=0, success=True)
