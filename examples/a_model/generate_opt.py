import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import scipy.special as sp


def skew_normal(x, p, s, a):
    x = (x-p)/s
    y = 2*np.exp(-x**2/2)/(s*np.sqrt(2*np.pi))
    y *= sp.ndtr(a*x)
    return y

def skew_normal_max(p, s, a):

    fun = lambda x: -skew_normal(x, p, s, a)

    res = minimize(fun, (0), method='COBYLA')

    return res.x[0]


def compute_simulation(sigma=1e-1, precision=1e-1, max_lim=2.0, n_dim=3, n_samples=10000, plot_potential=False):

    # precision = sigma*np.sqrt(n_dim)

    precision = sigma
    sigma = sigma / n_dim**0.8

    def function(x_data):

        c = 1.0
        f = 0.0
        for x in x_data:
            f += c * x**2 + np.random.normal(0, sigma)
        return f


    if plot_potential:
        plt.plot(np.linspace(-max_lim, max_lim, 100), [function([x]) for x in np.linspace(-max_lim, max_lim, 100)])
        plt.show()

    energy_list = []
    guess_list = []
    coeff_list = []
    evaluations_list = []
    for i in range(n_samples):
        # print('i sample', i)
        guess = (np.random.rand(n_dim) - 0.5) * 2 * max_lim
        res = minimize(function, guess, method='COBYLA', tol=precision)
        evaluations_list.append(res.nfev)
        energy_list.append(res.fun)
        for g in guess:
            guess_list.append(g)
        for c in res.x:
            coeff_list.append(c)

    return energy_list, coeff_list, guess_list, evaluations_list


energy_list, coeff_list, guess_list, evaluations_list = compute_simulation(n_samples=10000,
                                                                           sigma=1e-2,
                                                                           #precision=1e-2,
                                                                           max_lim=3.5,
                                                                           n_dim=2,                                                                           plot_potential=False)

print('\nEnergy stats')
print('std: ', np.std(energy_list))
print('average: ', np.average(energy_list))

print('\nCoefficients stats')
print('std: ', np.std(coeff_list))
print('average: ', np.average(coeff_list))


plt.hist(guess_list, bins=20, density=True)
plt.title('Initial guess')
plt.figure()

plt.hist(coeff_list, bins=20, density=True)
plt.title('optimized coeff')
plt.figure()

plt.hist(energy_list, bins=20, density=True, rwidth=0.9)
plt.title('optimized energy')

#fit
bin, freq = np.histogram(energy_list, bins=20, density=True)
h = (freq[1] - freq[0])/2
freq += h


popt, pcov = curve_fit(skew_normal, freq[:-1], bin)
print('\nFitting skew distribution')
print('center: ', popt[0])
print('sigma: ', popt[1])
print('alpha: ', popt[2])
max = skew_normal_max(*popt)

plt.axvline(x=max, ymin=0, ymax=1, color='red', linewidth=2)
# plot nice
#plt.ylim(0, 30)


# plot fit
x = np.linspace(freq[0], freq[-1], 100)
plt.plot(x, skew_normal(x, popt[0], popt[1], popt[2]), linewidth=2)

#plt.savefig('test.pdf')
plt.show()
