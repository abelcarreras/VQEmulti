import numpy as np
from scipy.special import erf

sigma_original = 1
def func(sigma_original, sigma_1):
    return erf(sigma_1/(sigma_original * np.sqrt(2)))


print(func(sigma_original=1, sigma_1=1))



print(func(sigma_original=0.5, sigma_1=1))


