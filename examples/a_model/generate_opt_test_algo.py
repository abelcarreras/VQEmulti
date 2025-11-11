import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
#from noisyopt import minimizeCompass, minimizeSPSA
import scipy.special as sp
from simulated_function import get_energy_sampled


def skew_normal(x, p, s, a):
    x = (x-p)/s
    y = 2*np.exp(-x**2/2)/(s*np.sqrt(2*np.pi))
    y *= sp.ndtr(a*x)
    return y


def skew_normal_max(p, s, a):

    fun = lambda x: -skew_normal(x, p, s, a)

    res = minimize(fun, p, method='COBYLA')

    return res.x[0]


def get_delta(alpha):
    return alpha/np.sqrt(1+alpha**2)

def compute_simulation(sigma=1e-1, precision=1e-1, max_lim=2.0, n_dim=3, n_samples=10000, plot_potential=False, k=0.1):

    # precision = sigma*np.sqrt(n_dim)

    precision = sigma
    #sigma = sigma / n_dim**(0.825) # sum( + error) # not used!

    #sigma = sigma / n_dim**(0.33) # sum() + error
    # sigma = sigma / (0.42 * n_dim) # sum() + error
    # sigma = sigma / (0.42 * n_dim + n_dim**(0.33333)) # sum() + error

    def function(x_data):

        c = 5.0
        f = 0.0
        for x in x_data:
            f += c * x**2
        return f + np.random.normal(0, sigma)

    def function_(x_data):
        return get_energy_sampled(x_data, sigma) + 0.5

    if plot_potential:
        n_points = 20
        plt.plot(np.linspace(-max_lim, max_lim, n_points), [function([x]) for x in np.linspace(-max_lim, max_lim, n_points)])
        plt.show()

    energy_list = []
    guess_list = []
    coeff_list = []
    evaluations_list = []
    for i in range(n_samples):
        #print('i sample', i)
        guess = (np.random.rand(n_dim) - 0.5) * 2 * max_lim
        res = minimize(function, guess, method='COBYLA', tol=precision*k)
        # res = minimizeCompass(function, x0=guess, deltatol=precision*0.1, paired=False, feps=precision*0.1, errorcontrol=False)
        # res = minimizeSPSA(function, x0=guess, paired=False, niter=50)

        evaluations_list.append(res.nfev)
        energy_list.append(res.fun)
        for g in guess:
            guess_list.append(g)
        for c in res.x:
            coeff_list.append(c)

    return energy_list, coeff_list, guess_list, evaluations_list


if False:
    std_list = []
    ave_list = []
    sum_list = []
    eval_list = []
    std_list_skew = []
    ave_list_skew = []
    alpha_list_skew = []

    k_range = [10.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.0001]
    for k in k_range:
        print('k: ', k)
        energy_list, coeff_list, guess_list, evaluations_list = compute_simulation(n_samples=100000,
                                                                                   sigma=5e-3,
                                                                                   # precision=1e-2,
                                                                                   max_lim=np.pi,
                                                                                   n_dim=1,
                                                                                   k=k,
                                                                                   plot_potential=False)

        print('\nEnergy stats')
        print('std: ', np.std(energy_list))
        print('average: ', np.average(energy_list))
        print('sum: ', np.std(energy_list) + np.average(energy_list))

        print('\nCoefficients stats')
        print('std: ', np.std(coeff_list))
        print('average: ', np.average(coeff_list))
        print('Evaluations: ', np.average(evaluations_list))

        std_list.append(np.std(energy_list))
        ave_list.append(np.average(energy_list))
        sum_list.append(np.std(energy_list) + np.average(energy_list))
        eval_list.append(np.average(evaluations_list))

        # skew
        n_bins = 20

        bin, freq = np.histogram(energy_list, bins=n_bins, density=True)
        h = (freq[1] - freq[0]) / 2
        freq += h

        popt, pcov = curve_fit(skew_normal, freq[:-1], bin)
        data = [skew_normal(x, *popt) for x in freq]

        if False:
            plt.hist(energy_list, bins=n_bins, density=True, rwidth=0.9)
            plt.title('optimized energy')

            plt.plot(freq, data)
            plt.show()

        print('\nFitting skew distribution')
        print('location: ', popt[0])
        print('scale (w): ', popt[1])
        print('shape (alpha): ', popt[2])

        std_list_skew.append(popt[1])
        ave_list_skew.append( popt[0])
        alpha_list_skew.append(popt[2])

    plt.plot(k_range, std_list, label='std')
    plt.plot(k_range, ave_list, label='ave')
    plt.plot(k_range, sum_list, label='sum')
    plt.xscale('log')
    plt.legend()

    plt.figure()
    plt.plot(k_range, eval_list, label='eval')
    plt.xscale('log')
    plt.legend()

    plt.figure()
    plt.plot(k_range, std_list_skew, label='loc')
    plt.plot(k_range, ave_list_skew, label='scale')
    plt.plot(k_range, alpha_list_skew, label='alpha')
    plt.xscale('log')
    plt.legend()

    plt.show()
    exit()


if True:
    std_list_skew = []
    ave_list_skew = []
    alpha_list_skew = []

    dim_range = range(0, 15)
    for n_dim in dim_range:
        print('n_dim: ', n_dim)
        sigma_value = 0.1e-1
        energy_list, coeff_list, guess_list, evaluations_list = compute_simulation(n_samples=500000,
                                                                                   sigma=sigma_value,
                                                                                   # precision=1e-2,
                                                                                   max_lim=np.pi,
                                                                                   n_dim=n_dim,
                                                                                   k=0.1,
                                                                                   plot_potential=False)

        print('\nEnergy stats')
        print('std: ', np.std(energy_list))
        print('average: ', np.average(energy_list))
        print('sum: ', np.std(energy_list) + np.average(energy_list))

        print('\nCoefficients stats')
        print('std: ', np.std(coeff_list))
        print('average: ', np.average(coeff_list))
        print('Evaluations: ', np.average(evaluations_list))

        # skew
        n_bins = 20

        bin, freq = np.histogram(energy_list, bins=n_bins, density=True)
        h = (freq[1] - freq[0]) / 2
        freq += h
        print('h: ', h)

        popt, pcov = curve_fit(skew_normal, freq[:-1], bin)
        data = [skew_normal(x, *popt) for x in freq]

        if False:
            plt.hist(energy_list, bins=n_bins, density=True, rwidth=0.9)
            plt.title('optimized energy')

            plt.plot(freq, data)
            plt.show()

        print('\nFitting skew distribution')
        print('location: ', popt[0])
        print('scale (w): ', popt[1])
        print('shape (alpha): ', popt[2])

        ave_list_skew.append(popt[0])
        std_list_skew.append(popt[1])
        alpha_list_skew.append(popt[2])

    test_list = [0.25*sigma_value*x + sigma_value for x in dim_range]
    plt.plot(dim_range, test_list, label='test_std')
    test_list = [0.25*sigma_value *x + -sigma_value for x in dim_range]
    plt.plot(dim_range, test_list, label='test_ave')
    test_list = [0.075 *x + 1.75 for x in dim_range]
    plt.plot(dim_range, test_list, label='test_alpha')

    plt.plot(dim_range, ave_list_skew, label='loc')
    plt.plot(dim_range, std_list_skew, label='scale')
    plt.plot(dim_range, alpha_list_skew, label='alpha')
    plt.xlabel('N coefficients')
    # plt.xscale('log')
    plt.legend()

    plt.show()
    exit()




energy_list, coeff_list, guess_list, evaluations_list = compute_simulation(n_samples=1000,
                                                                           sigma=1e-2,
                                                                           # precision=1e-2,
                                                                           max_lim=1.6,
                                                                           # max_lim=3.5,
                                                                           n_dim=1,
                                                                           k=1, # 0.01,
                                                                           plot_potential=True)


print(energy_list)
print('\nEnergy stats')
print('std: ', np.std(energy_list))
print('average: ', np.average(energy_list))
print('sum: ', np.std(energy_list) + np.average(energy_list))

print('\nCoefficients stats')
print('std: ', np.std(coeff_list))
print('average: ', np.average(coeff_list))

n_bins = 20
plt.hist(guess_list, bins=n_bins, density=True)
plt.title('Initial guess')
plt.figure()

plt.hist(coeff_list, bins=n_bins, density=True)
plt.title('optimized coeff')
plt.figure()

plt.hist(energy_list, bins=n_bins, density=True, rwidth=0.9)
plt.title('optimized energy')

#fit
bin, freq = np.histogram(energy_list, bins=n_bins, density=True)
h = (freq[1] - freq[0])/2
freq += h


try:
    popt, pcov = curve_fit(skew_normal, freq[:-1], bin)
    print('\nFitting skew distribution')
    print('location: ', popt[0])
    print('scale (w): ', popt[1])
    print('shape (alpha): ', popt[2])

    # extra data
    max = skew_normal_max(*popt)
    delta = get_delta(popt[2])
    sum_approach = popt[1] * (delta * np.sqrt(2/np.pi) + np.sqrt(1-(2*delta**2/np.pi))) + popt[0]
    print('SUM_approach: ', sum_approach)
    print('coeff at max energy: ', max)

    print('\nevaluations (average):', np.average(evaluations_list))
    #plt.axvline(x=max, ymin=0, ymax=1, color='red', linewidth=2)
    # plot nice
    #plt.ylim(0, 30)


    # plot fit
    x = np.linspace(freq[0], freq[-1], 100)
    plt.plot(x, skew_normal(x, popt[0], popt[1], popt[2]), linewidth=2)

except:
    pass

#plt.savefig('test.pdf')
plt.show()
