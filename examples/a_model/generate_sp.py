import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import scipy.special as sp
from simulated_function import get_energy_sampled


def compute_simulation(sigma=1e-1, max_lim=2.0, n_dim=3, n_samples=10000, plot_potential=False):

    def function(x_data):

        c = 1
        f = 0.0
        for x in x_data:
            f += c * x**2
        return f + np.random.normal(0, sigma)

    def function_(x_data):
        return get_energy_sampled(x_data, sigma)


    if plot_potential:
        plt.plot(np.linspace(-max_lim, max_lim, 100), [function([x]) for x in np.linspace(-max_lim, max_lim, 100)])
        plt.show()

    energy_list = []
    for i in range(n_samples):
        print('i:', i)
        energy_list.append(function([0]*n_dim))

    return energy_list

if False:
    energy_list = compute_simulation(n_samples=100,
                                     sigma=1e-3,
                                     n_dim=2,
                                     plot_potential=False)

    print('\nEnergy stats')
    print('std: ', np.std(energy_list))
    print('average: ', np.average(energy_list))


    plt.hist(energy_list, bins=20, density=True, rwidth=0.9)
    plt.title('optimized energy')

    #fit
    bin, freq = np.histogram(energy_list, bins=20, density=True)
    h = (freq[1] - freq[0])/2
    freq += h

    #plt.axvline(x=max, ymin=0, ymax=1, color='red', linewidth=2)
    # plot nice
    #plt.ylim(0, 30)


    # plot fit
    x = np.linspace(freq[0], freq[-1], 100)
    #plt.plot(x, skew_normal(x, popt[0], popt[1], popt[2]), linewidth=2)

    #plt.savefig('test.pdf')
    plt.show()


# check square law N
s_list = []
x_range = range(5)
for i in x_range:
    print('round: ', i)
    energy_list = compute_simulation(n_samples=100,
                                     sigma=1e-3,
                                     n_dim=i,
                                     plot_potential=False)

    s_list.append(np.std(energy_list))

plt.plot(x_range, s_list, label='cal')
plt.plot(x_range, [1e-3 * np.sqrt(n) for n in x_range], 'o', label='theor')
plt.legend()
plt.show()
