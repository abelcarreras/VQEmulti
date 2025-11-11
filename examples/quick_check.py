import numpy as np
import scipy as sp
#state = [1, 0]

state = [3, 1]
state = np.array(state)/np.linalg.norm(state)

H = np.array([[0, 0],
              [0, 2]])

hermitian = np.array([[4,   1j],
                      [-1j, 1]])

print(hermitian - hermitian.T.conjugate())

anti_hermitian = 1j * hermitian
print('antihermitian: \n', anti_hermitian)
#exit()

print(anti_hermitian + anti_hermitian.T.conjugate())

print('Unitary')
U = sp.linalg.expm(anti_hermitian)
#U = np.identity(2)

print(U.T.conjugate() @ U)


rho_0 = 1.0 * np.outer(state, state)

print(rho_0)
#print('U\n', U)

# H = U @ H @ U.T.conjugate()

energy = np.trace(H @ U @ rho_0 @ U.T.conjugate())

print('energy: ', np.round(energy, decimals=5))
