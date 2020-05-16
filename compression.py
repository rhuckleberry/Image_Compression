import numpy as np
import image_conversion as ic
import math
import eigenvector as evect
import cv2
import Image_comparison as comparison

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

def compress_data(A, e, isAll = False):
    """
    Returns compressed data of A in form of eigenvals, u_vects and v_vects

    Input:
    A - an input mxn matrix
    e - an integer between (0 <= e <= #eigenvalues of A) that will be the number
        of eigenvalues that A_approx will use to approximate A
    isAll - boolean: if True, return all eigenvals, u_vect, v_vect
                     if False, return e of the eigenvals, u_vect, v_vect

    Ouput: tuple of the following
    eigenvals - list of eigenvalues of svd matrix A
    u_vects -  list subset of u vectors where u = [A][A^t]
    v_vects - list subset of v vectors where v = [A^t][A]
    """
    u, s, vh = np.linalg.svd(A)
    #s is a list of squarerooted eigenvalues
    #u is a matrix of eigenvectors of [A][A^t]
    #vh is a matrix of transposed eigenvectors of [A^t][A]

    if e > s.size:
        print("*Error: Trying to approximate with more eigenvalues than A has!*")
        return A_approx

    eigenvals =[]
    u_vects =[]
    v_vects = []
    max_val = 0

    if isAll:
        max_val = len(s) #want all e-vals & e-vects in list form
    else:
        max_val = e

    for i in range(max_val):
        s_i = s[i]
        u_i = u[:,i]
        vh_i = vh[i, :]

        eigenvals.append(s_i)
        u_vects.append(u_i)
        v_vects.append(vh_i)

    return (eigenvals, u_vects, v_vects)

def form_approximation(A, eigenvals, u_vects, v_vects):
    """
    Returns an approximation, A_approx, of A based on the singular value
    decomposition of A and e of A's eigenvalues

    Input:
    A - an input mxn matrix
    eigenvals - eigenvalues of svd matrix A
    u_vects -  subset of u vectors where u = [A][A^t]
    v_vects - subset of v vectors where v = [A^t][A]

    Output:
    A_approx - approximate matrix of A based on compressed data
    """

    m, n = A.shape #gets dimension sizes of A
    A_approx = np.zeros((m, n)) #creates zero matrix in dimension size of A

    data_length = len(eigenvals)

    if data_length == len(u_vects) and data_length == len(v_vects):

        # singular value approximation of A by calculating the summation formula:
        # A = Σ i=1 to n (sqrt(lamdai)* ui * (vi)^T)
        # s_i = sqrt(lamdai)
        # u_i = ui
        # v_i = (vi)^T
        for i in range(data_length): #summation formula added to initial zero matrix
            s_i = eigenvals[i]
            u_i = u_vects[i]
            vh_i = v_vects[i]

            A_approx += s_i*u_i*vh_i

        return A_approx

    else:
        print("Error: compression data not of the same length")
        return

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
    (eigenvals, u_vects, v_vects) = compress_data(A, e)
    A_approx = form_approximation(A, eigenvals, u_vects, v_vects)

    return A_approx

def best_approximation(image, error_bound):
    """
    Gives data compression values for A_approx such that A_approx is lower
    bounded in data compression accuracy by error_bound

    Input:
    image - filename of an png image ex: 'test2.png'
        ~A is the corresponding matrix of image
    error_bound - a positive MSE value between A and A_approx that A_approx
                must satisfy

    Ouput: compressed_data -- tuple of the following
    eigenvals - eigenvalues of svd matrix A
    u_vects -  subset of u vectors where u = [A][A^t]
    v_vects - subset of v vectors where v = [A^t][A]
    """

    #matrix initialization
    A = ic.import_img(image, True)
    B = np_mx_convert(A)
    current_MSE = float("inf")

    #save all eigenvals and take subsets
    (all_eigenvals, all_u_vects, all_v_vects) = compress_data(B, 0, True)
    (eigenvals, u_vects, v_vects) = ([], [], [])

    #e values and bounds
    min_e = 0
    max_e = len(all_eigenvals) #max possible value for e
    last_e = -1 #cant be same as any other e value
    current_e = math.ceil((min_e + max_e) / 2)

    #binary search style
    while last_e != current_e:
        print("current_e: ", current_e, "last_e: ", last_e)
        print("min_e: ", min_e, "max_e: ", max_e)
        #find MSE of approximation with e
        (eigenvals, u_vects, v_vects) = (all_eigenvals[:current_e],\
                                            all_u_vects[:current_e], \
                                            all_v_vects[:current_e])
        A_approx = form_approximation(B, eigenvals, u_vects, v_vects)
        current_MSE = comparison.mse_comparison(B, A_approx)

        #update min_e, max_e bounds
        if current_MSE == error_bound:
            return (eigenvals, u_vects, v_vects)
        elif current_MSE > error_bound:
            print("bigger MSE")
            min_e = current_e
        else: #current_MSE < error_bound
            print("smaller MSE")
            max_e = current_e

        last_e = current_e
        current_e = math.ceil((min_e + max_e) / 2) #ceiled average of max and min
        print(current_MSE)

    print(current_e)
    return (eigenvals, u_vects, v_vects)

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
    # #greyscale image
    # A = ic.import_img('test1.png', True)
    # B = np_mx_convert(A)
    # (eigenvals, u_vects, v_vects) = compress_data(B, 100)
    # #compressed data = (eigenvals, u_vects, v_vects)
    # A_approx = form_approximation(A, eigenvals, u_vects, v_vects)
    # #Matrix Approximation = A_approx
    # print(A_approx)
    # ic.display_img(A_approx, True)

    #best approximation
    error_bound = 2
    (eigenvals, u_vects, v_vects) = best_approximation('test1.png', error_bound)

    # #color image
    # A_approx = control_flow_color('test1.png', 50)
    # print(A_approx)
