from scipy.linalg import expm
import numpy as np

import matplotlib.pyplot as plt


def generate_unitary(n, params):
    print('param', params)

    anti_symmetric = np.zeros((n, n))
    k = 0
    for i in range(0, n):
        for j in range(i+1, n):
            anti_symmetric[i, j] = params[k]
            anti_symmetric[j, i] = -params[k]
            k += 1


    print('*****')
    print(anti_symmetric)
    print('*****')
    #exit()

    return expm(anti_symmetric)


values = []
for a in np.linspace(3, 10, 100):
    antisymetric = np.array([[ 0, a],
                             [-a, 0]])

    print('--')
    #u_matrix = expm(antisymetric)

    u_matrix = generate_unitary(2, [a])
    print(u_matrix)
    print(u_matrix.T @ u_matrix)
    print(values.append(u_matrix))

values = np.array(values)
plt.plot(values.T[0, 0])
plt.plot(values.T[0, 1])
plt.plot(values.T[1, 0])
plt.plot(-values.T[1, 1])

plt.show()
exit()

a = 0.2 # np.pi
u_matrix_ = np.array([[np.cos(a), np.sin(a)],
                     [-np.sin(a), np.cos(a)]])


print(u_matrix)
print()
print(u_matrix.T @ u_matrix)
exit()
a = 0.1
restrict = np.array([[np.cos(a), np.sin(a)],
                         [-np.sin(a), np.cos(a)]])

def minimize_(a, restrict):
    u_matrix = np.array([[np.cos(a), np.sin(a)],
                         [-np.sin(a), np.cos(a)]])

    restrict = restrict @ u_matrix

    return 0


from scipy.optimize import minimize

res = minimize(minimize_, [0], args=(restrict))
print(res)
