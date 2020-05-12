import numpy as np
import image_conversion as ic
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

        print("s_i: ", s_i, "\n")
        print("u_i: ", u_i.shape, "\n")
        print("vh_i: ", vh_i.shape, "\n")

        A_approx += s_i*u_i*vh_i

    return A_approx

# def np_mx_convert(A):
#     """
#     prints matrix A in numpy form
#
#     Input:
#     A - an mxn matrix in OpenCV form
#
#     Output: None (printed matrix)
#     """
#     dim = A.shape
#     m = dim[0]
#     n = dim[1]
#
#     str_val = "["
#     for i in range(m):
#         str_val += "["
#         for j in range(n):
#             a = int(A[i,j])
#             str_val += str(a)
#             if j != n-1:
#                 str_val += ","
#
#         str_val += "]"
#         if i != m-1:
#             str_val += ","
#
#     str_val += "]"
#     print(str_val)

if __name__ == "__main__":
    # A = np.matrix([[1,2], [4,5], [8,9]])
    A = ic.import_img('test1.png', True)
    print(A)
    e = 2
    A_approx = svd_approx(A, e)
    print(A_approx)
