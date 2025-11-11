import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm


def get_delta(alpha):
    return alpha/np.sqrt(1+alpha**2)

def get_m(a0, n_dim):
    return np.sqrt(2/np.pi)*get_delta(a0) + (n_dim - 1) * 0.12

def get_s0(s, a0, n_dim):
    #s = s*n_dim**0.85

    variance = s**2 /(1 - 2*get_delta(a0)**2/np.pi)
    return np.sqrt(variance)


def get_mean_x(s, m, c, n_dim):
    return s * m + c


c0 = -0.0035756# -0.024887 # -0.00015797 #-0.00369646 #-0.0015613 #-0.0035
s = 1e-2
a0 = 1.9179035 # 1.739513576 # 1.71983459093 # 1.8285568 # 1.712
n_dim = 1

m = get_m(a0, n_dim)
s0 = get_s0(s, a0, n_dim)

print('deviation (s0)', s0)
print('deviation (s)', s)

mean_x = get_mean_x(s0, m, c0, n_dim)

print('mean_x: ', mean_x)

x_list = []
s_list = []
s0_list = []

for n_dim in range(1, 10):
    m = get_m(a0, n_dim)

    s0 = get_s0(s, a0, n_dim)
    #print('s', s)
    x_list.append(get_mean_x(s0, m, c0, n_dim))
    s0_list.append(s0)
    s_list.append(s)

print(x_list)

plt.plot(range(1, 10), x_list, label='average')
#plt.plot(range(1, 7), [0.000578876, 0.001217495, 0.002304058, 0.003388202, 0.004780345606, 0.006196382], label='ref_ave')
plt.plot(range(1, 5), [0.00629718, 0.007739, 0.0098258, 0.011291724], label='ref_ave')

plt.plot(range(1, 10), s_list, label='std')
#plt.plot(range(1, 7), [0.00101262219588, 0.00167706435258, 0.002432108688162, 0.0030888738, 0.00385242353057, 0.00450019780], label='ref_std')
#plt.plot(range(1, 7), [0.001014757773652, 0.0016897977138, 0.002429439952, 0.003067897588, 0.00385242353057, 0.00450019780], label='ref_std')

#plt.plot(range(1, 7), [0.07896625959, 0.08847367733, 0.1087064810, 0.12563681, 0.141448168, 0.16115602], label='ref_std')

plt.plot(range(1, 10), s0_list, label='sigma')
#plt.plot(range(1, 7), [0.0014036114, 0.0022939333, 0.003365241, 0.00426317289, 0.00531338982, 0.006224986], label='ref_sigma')

plt.xlim(0, None)
plt.ylim(0, None)

plt.legend()
plt.show()
