import numpy as np
import image_conversion as ic
import math
import eigenvector as evect
import cv2

def mse_comparison(A, A_approx):
    """
    Gives the MSE between two matrices

    Input:
    A - an mxn matrix
    A_approx - an mxn matrix

    Output:
    MSE - mean squre error comparison of the two matrices

    ~MSE: sum of square of matrix index value differences
    """

    m, n = A.shape #matrix dimensions
    total_indices = m*n
    MSE = 0

    for i in range(m):
        for j in range(n):
            MSE += (A[i,j] - A_approx[i,j]) ** 2

    MSE /= total_indices

    print("MSE: ", MSE)
    return MSE


if __name__ == "__main__":
    A = np.matrix([[1,0],[1,0]])
    A_approx = np.matrix([[3,0],[0,2]])
    MSE = mse_comparison(A, A_approx)
    print(MSE)
