import numpy as np


mu, sigma = 0, 0.16 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000000)

e2 = np.average(np.square(s))
e = np.average(s)

variance = e2 - e**2
print('variance', variance, np.var(s))
print('deviation', np.sqrt(variance), np.std(s))

print('check', abs(np.sqrt(variance) - sigma))