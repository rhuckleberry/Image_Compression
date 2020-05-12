import numpy as np
import math
import eigenvector as evect

def svd_approx(A, e):
    """
    Returns an approximation, A_approx, of A based on the singular value
    decomposition of A and e of A's eigenvalues

    Input:
    A - an input mxn matrix
    e - an integer between (0 <= e <= #eigenvalues of A) that will be the number
        of eigenvalues that A_approx will use to approximate A

    Ouput:
    a_approx - a singular value decomposition approximation of A using e
                eigenvalues

                ~If m is not in bounds returns empty matrix with dimensions of A
    """
    u, s, vh = np.linalg.svd(A)
    #s is a list of squarerooted eigenvalues
    #u is a matrix of eigenvectors of [A][A^t]
    #vh is a matrix of transposed eigenvectors of [A^t][A]

    print("u: ", u, "\n")
    print("s: ", s, "\n")
    print("vh: ", vh, "\n")

    m, n = A.shape #gets dimension sizes of A
    A_approx = np.zeros((m, n)) #creates zero matrix in dimension size of A

    if e > s.size:
        print("*Error: Trying to approximate with more eigenvalues than A has!*")
        return A_approx

    # singular value approximation of A by calculating the summation formula:
    # A = Σ i=1 to n (sqrt(lamdai)* ui * (vi)^T)
    # s_i = sqrt(lamdai)
    # u_i = ui
    # v_i = (vi)^T
    for i in range(e): #summation formula added to initial zero matrix
        s_i = s[i]
        u_i = u[:,i]
        vh_i = vh[i, :]

        A_approx += s_i*u_i*vh_i

    return A_approx

if __name__ == "__main__":
    A = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
    e = 1
    A_approx = svd_approx(A, e)
    print(A_approx)
