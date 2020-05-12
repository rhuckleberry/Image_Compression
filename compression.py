import numpy as np
import image_conversion as ic
import math
import eigenvector as evect
import cv2

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

def control_flow_color(image, e):
    """
    Takes a matrix in CV2 form and approximates it with singular value decomposition
    ~Does with a color image

    Input:
    image - filename of an png image ex: 'test2.png'
    e - an integer between (0 <= e <= #eigenvalues of A) that will be the number
        of eigenvalues that A_approx will use to approximate A

    Output:
    ~prints the approximated image
    A_approx - a singular value decomposition approximation of A using e eigenvalues
    """
    A = ic.import_img(image, False)
    A_red = ic.select_color_channel(A, 2, True)
    A_green = ic.select_color_channel(A, 1, True)
    A_blue = ic.select_color_channel(A, 0, True)
    B_red = np_mx_convert(A_red)
    A_red_approx = svd_approx(B_red, e)
    B_green = np_mx_convert(A_green)
    A_green_approx = svd_approx(B_green, e)
    B_blue = np_mx_convert(A_blue)
    A_blue_approx = svd_approx(B_blue, e)
    A_approx = ic.combine_channels(A_red_approx, A_green_approx, A_blue_approx)
    A_approx = A_approx / 256
    cv2.imshow("red", A_red_approx/256)
    cv2.imshow("green", A_green_approx/256)
    cv2.imshow("blue", A_blue_approx/256)
    cv2.imshow("initial image", A)
    ic.display_img(A_approx, False)
    return A_approx



if __name__ == "__main__":
    A_approx = control_flow_color('test1.png', 20)
    print(A_approx)
