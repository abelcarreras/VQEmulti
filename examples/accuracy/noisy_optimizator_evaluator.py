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

#x = np.linspace(-3, 3, 100)
#plt.plot(x, [skew_normal(xx, 0, 1, 4) for xx in x])
#plt.show()
#exit()

from H4_test_function import get_hamiltonian
from vqemulti.energy import simulate_adapt_vqe_energy
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

pool, hf_reference_fock, hamiltonian = get_hamiltonian()
simulator = Simulator(trotter=True, test_only=False, shots=16)


def compute_simulation(sigma=1e-1, precision=1e-1, max_lim=2.0, n_dim=3, n_samples=10000, plot_potential=False):

    def function(x_data):

        c = 1.0
        f = 0.0
        for x in x_data:
            f += c * x**2

        s = np.random.normal(f, sigma)
        return s

    def function_(x_data):
        """
        function that implements operators

        :param x_data:
        :return:
        """

        coefficients = x_data
        ansatz = pool[9:9+len(x_data)]

        energy = simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
        return energy


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
                                                                           precision=1e-2,
                                                                           max_lim=3.5,
                                                                           n_dim=5,
                                                                           plot_potential=True)

print('\nEnergy')
print('std: ', np.std(energy_list))
print('average: ', np.average(energy_list))

print('\noptimized coeff')
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
print('fit_skew')
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

exit()

if False:
    # full log range
    prec_list = [1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    # 5e-1
    prec_list = [0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58,
                 0.56, 0.54, 0.52, 0.5, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38,
                 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22]

    # 1e-1
    prec_list_ = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23,
                 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13,
                 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]

    # 1e-2
    prec_list_ = [0.1, 0.08, 0.06, 0.04, 0.03, 0.025, 0.024, 0.023,
                 0.022, 0.021, 0.020, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013,
                 0.012, 0.011, 0.010, 0.009, 0.008, 0.007, 0.006, 0.005]

    #prec_list = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    list_std = []
    list_ave = []
    list_eval_ave = []
    list_eval_std = []

    for p in prec_list:
        print('precision points: ', p)
        energy_list, coeff_list, guess_list, eval_list = compute_simulation(sigma=5e-1,
                                                                            precision=p,
                                                                            max_lim=50,
                                                                            n_samples=10000,
                                                                            n_dim=5)

        list_std.append(np.std(energy_list))
        list_ave.append(np.average(energy_list))
        list_eval_std.append(np.std(eval_list))
        list_eval_ave.append(np.average(eval_list))


    # plot data
    plt.title('Energy')
    plt.plot(prec_list, list_ave, label='average')
    plt.plot(prec_list, list_std, label='std')
    plt.plot(prec_list, prec_list, '--')

    # plt.xscale('log')
    plt.legend()

    plt.figure()
    plt.title('Evaluations')
    plt.plot(prec_list, list_eval_ave, label='average')
    plt.plot(prec_list, list_eval_std, label='std')
    #plt.plot(prec_list, prec_list, '--')

    # plt.xscale('log')
    plt.legend()
    plt.show()


if False:
    plt.title('NE = -a(NDim)*X + b(NDim)')
    plt.plot([0] + [1, 2, 3, 4, 5], [0] + [4, 6.5, 7.8, 10.8, 13.4], 'o-', label='a')
    #plt.plot([0] + [1, 2, 3, 4, 5], [0] + [27.5, 42.5, 58, 69, 81], 'o-', label='b')
    plt.plot(np.linspace(0, 5, 100), [2.8*x for x in np.linspace(0, 5, 100)], '--', label='fit')

    A = np.vstack([[1, 2, 3, 4, 5], np.ones(5)]).T
    m, c = np.linalg.lstsq(A, [27.5, 42.5, 58, 69, 81], rcond=None)[0]
    print('m: ', m, 'c', c)

    popt, pcov = curve_fit(lambda x, c, d: c*x**(d), [0, 1, 2, 3, 4, 5], [0, 27.5, 42.5, 58, 69, 81])
    print(popt)
    #plt.plot(np.linspace(0, 5, 100), [popt[0]*x**(popt[1]) for x in np.linspace(0, 5, 100)], '--', label='fit')

    plt.legend()

    plt.show()

if True:
    # full log range
    dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    list_std = []
    list_ave = []
    list_eval_ave = []
    list_eval_std = []
    list_skew_center = []
    list_skew_std = []
    list_skew_alpha = []
    list_skew_max = []

    for n_dim in dim_list:
        p = 1e-2
        # p = p * 0.5
        ss = 1
        a1 = p / (n_dim)**(1/3)
        a2 = p / (0.1 * np.pi * n_dim)
        a_tot = np.min([a2, a2])
        # a_tot = np.abs([a1 - a2])

        print('shots: ', int(1/a_tot)**2)
        print('dim: ', n_dim)
        energy_list, coeff_list, guess_list, eval_list = compute_simulation(sigma=p,
                                                                            precision=p,
                                                                            max_lim=1.5,
                                                                            n_samples=10000,
                                                                            n_dim=n_dim)

        list_std.append(np.std(energy_list))
        list_ave.append(np.average(energy_list))
        list_eval_std.append(np.std(eval_list))
        list_eval_ave.append(np.average(eval_list))

        # print(list_std)
        # print(list_ave)
        # print(list_eval_std)
        # print(list_eval_ave)

        #plt.figure()

        bin, freq = np.histogram(energy_list, bins=20, density=True)
        h = (freq[1] - freq[0]) / 2
        freq += h
        popt, pcov = curve_fit(skew_normal, freq[:-1], bin)
        x = np.linspace(freq[0], freq[-1], 100)
        # plt.plot(x, skew_normal(x, popt[0], popt[1], popt[2]))

        max_sknormal = skew_normal_max(*popt)

        list_skew_max.append(max_sknormal)
        list_skew_center.append(popt[0])
        list_skew_std.append(popt[1])
        list_skew_alpha.append(popt[2])

        #plt.hist(energy_list, bins=20, density=True)
        #plt.title('optimized energy (dim = {})'.format(n_dim))
        #plt.axvline(x=max_sknormal, ymin=0, ymax=1, color='red')

    # plot data
    plt.figure()
    plt.title('Energy (skew)')
    # plt.plot(dim_list, list_skew_center, label='center')
    plt.plot(dim_list, list_skew_std, label='sigma')
    # plt.plot(dim_list, list_skew_alpha, label='alpha')
    plt.plot(dim_list, list_skew_max, label='max')
    plt.plot(dim_list, np.array(list_skew_max) + ss * np.array(list_std), label='sum')

    x_range = np.linspace(0, 10, 100)
    popt, pcov = curve_fit(lambda x, c, d: c*x**(d), dim_list, list_skew_std)
    print('skew_sigma: ', popt)
    plt.plot(x_range, [popt[0]*x**(popt[1]) for x in x_range], '--', label='fit_sig')

    #print(list_skew_max)
    p = 0
    popt, pcov = curve_fit(lambda x, c: c*x + p, dim_list, list_skew_max)
    print('skew_max: ', popt)
    plt.plot(x_range, [popt[0]*x + p for x in x_range], '--', label='fit_max')

    plt.legend()
    plt.figure()
    plt.title('Energy (normal)')
    plt.plot(dim_list, list_ave, label='average')
    plt.plot(dim_list, list_std, label='std')
    plt.legend()

    x_range = np.linspace(0, 10, 100)
    popt, pcov = curve_fit(lambda x, c, d: c*x**(d), dim_list, list_std)
    print('std: ', popt)
    plt.plot(x_range, [popt[0]*x**(popt[1]) for x in x_range], '--', label='fit')

    popt, pcov = curve_fit(lambda x, c: c*x, dim_list, list_ave)
    print('average: ', popt)
    plt.plot(x_range, [popt[0]*x for x in x_range], '--', label='fit')

    plt.figure()
    plt.title('Evaluations')
    plt.plot(dim_list, list_eval_ave, label='average')
    plt.plot(dim_list, list_eval_std, label='std')
    # plt.plot(prec_list, prec_list, '--')

    plt.legend()

    plt.figure()
    # combined
    plt.title('combined')
    p = 0
    plt.plot(dim_list, list_skew_max, 'o', label='max', color='#d62728')
    popt, pcov = curve_fit(lambda x, c: c*x + p, dim_list, list_skew_max)
    print('skew_max: ', popt)
    plt.plot(x_range, [popt[0]*x + p for x in x_range], '--', label='position', color='#ff7f0e')

    plt.plot(dim_list, list_std, 'o', label='std', color='#1f77b4')
    popt, pcov = curve_fit(lambda x, c, d: c*x**(d), dim_list, list_std)
    print('std: ', popt)
    plt.plot(x_range, [popt[0]*x**(popt[1]) for x in x_range], '--', label='deviation', color='#2ca02c')

    plt.ylim(0, 0.035)
    plt.savefig('test_2.pdf')

    plt.show()
