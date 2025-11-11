import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm


def normal(x, p=1, s=1):
    rv = norm(loc=p, scale=s)
    return rv.pdf(x)

def skew_normal(x, p=0, s=1, a=1):
    rv = skewnorm(a, loc=p, scale=s)
    return rv.pdf(x)

def skew_normal_sample(n_points, p=0.0, s=1.0, a=1.0):
    return skewnorm.rvs(a, loc=p, scale=s, size=n_points)

def skew_normal_2(x, p=0, s=1, a=1):
    x = (x-p)/s
    y = 2*np.exp(-x**2/2)/(s*np.sqrt(2*np.pi))
    y *= sp.ndtr(a*x)
    return y

def skew_normal_3(x, p=0, s=1, a=1):
    x = (x-p)/s
    y = 2*np.exp(-x**2/2)/(s*np.sqrt(2*np.pi))
    y *= sp.ndtr(a*x)
    return y

import matplotlib.pyplot as plt

#plt.plot([skew_normal(x, a=3) for x in np.linspace(-5, 5, 200)])
#plt.plot([skew_normal_2(x, a=3) for x in np.linspace(-5, 5, 200)])
plt.plot(np.linspace(-5, 5, 200), [sp.ndtr(2.14*x) for x in np.linspace(-5, 5, 200)])

plt.show()

exit()

def skew_normal_max(p, s, a):

    fun = lambda x: -skew_normal(x, p, s, a)

    res = minimize(fun, (0), method='COBYLA')

    return res.x[0]


x_range = np.linspace(-4, 4, 100)

plt.plot(x_range, [normal(x, s=1) for x in x_range])
plt.plot(x_range, [skew_normal(x, s=1, a=4) for x in x_range])

data = skew_normal_sample(2000, s=1, a=4)
plt.hist(data, density=True, bins=20)
plt.show()


a_ranges = np.linspace(0.0, 5, 20)
slopes_x = []
slopes_s = []

pos0 = 0.0
for a in a_ranges:

    averages = []
    deviations = []
    x_range = np.linspace(0.1, 3, 100)
    for s in x_range:
        samples = skew_normal_sample(20000, p=pos0, s=s, a=a)
        averages.append(np.mean(samples))
        deviations.append(np.std(samples))

    m, b = np.polyfit(x_range, averages, 1)
    slopes_x.append(m)
    # print(m, b)

    # sqrt(2/pi)

    plt.plot(x_range, averages)
    plt.plot(x_range, [x*m+pos0 for x in x_range])
    plt.xlabel('deviation skew (s)')
    plt.ylabel('mean x')

    m, b = np.polyfit(x_range, deviations, 1)
    slopes_s.append(m)


plt.figure()

def fit_function(x, c):
    #return x**(-d)*c + e
    #return c * np.log(x*d) + e
    #return c * 1/(1+np.exp(-x*d)) + e
    return c * x/np.sqrt(1+x**2)

popt, pcov = curve_fit(fit_function, a_ranges, slopes_x)

print('fit_x_skew')
print('params: ', popt)

plt.plot(a_ranges, slopes_x)
plt.plot(a_ranges, fit_function(a_ranges, *popt), label='fit')
plt.xlabel('alpha skew')
plt.ylabel('m_x')

plt.legend()

plt.show()

exit()

def fit_function(x, c, d, e, f):
    return c * np.exp(-d*x**2) + f
    #return c * np.cos(x*d) * np.exp(-x*e)# + d

popt, pcov = curve_fit(fit_function, a_ranges, slopes_s)

print('fit_s_skew')
print('params: ', popt)

plt.plot(a_ranges, slopes_s)
plt.plot(a_ranges, fit_function(a_ranges, *popt), label='fit')
plt.xlabel('alpha skew')
plt.ylabel('m_d')

plt.legend()
plt.show()
