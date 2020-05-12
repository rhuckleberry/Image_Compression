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
    A_approx - a singular value decomposition approximation of A using e
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
    print(s.shape)

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

def np_mx_convert(A):
    """
    prints matrix A in numpy form

    Input:
    A - an mxn matrix in OpenCV form

    Output:
    B - matrix A in numpy form
    """
    m, n = A.shape #matrix dimensions

    convert_A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            convert_A[i,j] = A[i,j]

    B = np.matrix(A)
    return B

def control_flow(image, e):
    """
    Takes a matrix in CV2 form and approximates it with singular value decomposition
    ~Runs a grayscale convertion on the given image!!

    Input:
    image - filename of an png image ex: 'test2.png'
    e - an integer between (0 <= e <= #eigenvalues of A) that will be the number
        of eigenvalues that A_approx will use to approximate A

    Output:
    ~prints the approximated image
    A_approx - a singular value decomposition approximation of A using e eigenvalues
    """
    A = ic.import_img(image, True)
    B = np_mx_convert(A)
    A_approx = svd_approx(B, e)
    A_approx = A_approx / 256
    ic.display_img(A_approx, False)
    return A_approx

if __name__ == "__main__":
    A_approx = control_flow('test1.png', 10)
    print(A_approx)
