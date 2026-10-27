import numpy as np
sparsity = 0.000001
n = 10
A = np.zeros((n,n))
for i in range(0,n):
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(n,n)
A = (A.T + A)/2

print("Sparse Hamiltonian:")
print(A)

k = 8 # number of initial guess vectors
eig = 4 # number of eignvalues to solve
t = np.eye(n,k) # set of k unit vectors as guess
V = np.zeros((n,n)) # array of zeros to hold guess vec
I = np.eye(n) # identity matrix same dimen as A