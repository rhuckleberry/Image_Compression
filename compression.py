import numpy as np
import math
import eigenvector as evect

# A = evect.make_symmetric_matrix(3, 1, 10)
# print(A)

A = np.matrix([[2,3, 4], [5,6,7], [8,9,10]])
u, s, vh = np.linalg.svd(A)
#s is a list of eigenvalues
#u is a list of eigenvectors of [A][A^t]
#vh is a list of eigenvectors of [A^t][A]

print("u: ", u, "\n")
print("s: ", s, "\n")
print("vh: ", vh, "\n")

n_dim = s.size #matrix dimension size
m, n = A.shape
A_approx = np.zeros((m, n))
for i in range(n_dim):
    s_i = s[i]
    u_i = u[:,i]
    vh_i = vh[:,i]
    print("s_i: ", s_i, "\n")
    print("u_i: ", u_i, "\n")
    print("vh_i: ", vh_i, "\n")

    print("approx ", A_approx, "\n")
    A_approx += math.sqrt(s_i)*u_i*vh_i.T


print("aprox_final ", A_approx)
