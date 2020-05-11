import numpy as np
import random

def find_largest_evect(A):
    """
    Given a symmetric matrix A, find the largest eigenvector for the matrix

    Input:
    A - symmetric matrix (means it has orthonormal eigenbasis)

    Output: Largest eigenvector for matrix A
    """
    #Part 1: assume eigenvalues all distinct for A

    #a) Choose random vector x
    x = np.matrix([[1], [2]])
    print("x: ", x)
    #b) Calculate A^k
    k = 20
    large_A = multiply_matrix(A, k)
    print("large_A: ", large_A)
    #c) Calculate v1 = (A^k)v
    return large_A * x
    #d) check if v1 is an eigenvector

def eigenvector_check(v, lambda_v):
    """
    Returns True if v seems to be an eigenvector and False otherwise
    ~Not exact so need to determine what is "close enough"

    Inputs:
    v - a nx1 possible eigenvector
    lambda_v - the nx1 output of Av, if v is an eigenvector it should equal lambda*v

    Ouput: True if eigenvector, False otherwise
    """
    dim_v = v.shape
    dim_lamda_v = lambda_v.shape
    if (dim_v[1] != 1) or (dim_lamda_v[1] != 1) or (dim_v != dim_lamda_v):
        print("not a vector, or vectors not same dim")
        return

    n = dim_v[0]

    #- transform inputs by multiplying the whole matrix by the first index a1
    #- first index will be 1 and the lambdas in lambda_v will be divided out
    #- all indices will be ai/a1, so if v is eigenvector the two vectors should
    #be approximate multiples of each other
    v_a1 = v[0,0]
    print("v_a1 :", v_a1)
    v_prime = v * 1/v_a1
    print("v_prime: \n", v_prime)

    lambda_v_a1 = lambda_v[0,0]
    print("lambda_v_a1 :", v_a1)
    lambda_v_prime = lambda_v * 1/lambda_v_a1
    print("lambda_v_prime: \n", lambda_v_prime)

    #mean squared error test: 1/n Σ(yi - y_bari)^2 --Σ from i=1 to n(all indices)
    TOLERANCE_VAL = n

    sigma = 0
    for i in range(n):
        vi = v_prime[i,0]
        lambda_vi = lambda_v_prime[i,0]
        square_difference = (vi - lambda_vi)**2
        sigma += square_difference
    sigma /= n
    print(sigma)

    if sigma <= TOLERANCE_VAL:
        return True
    return False

def compute_largest_eigenvector(A, m, v):
    """
    Finds a possible largest eigenvector of A without overflowing in
    the UInt64 datatype (64 bits -- holds 2^64-1 values)

    Input:
    A - a symmetric nxn matrix
    m - a positive integer number of times to multiply v by A
    v - an nx1 vector (randomly chosen)

    Output: lambda_v - a possible largest eigenvector of A
    """
    dim = A.shape
    if dim[0] != dim[1]:
        print("not square matrix")
        return

    n = dim[0] #row/col dimension
    L = max_index_val(A, n, n)[0] #maximum value in matrix
    k = (2**64) - 1 #Max value of uint64
    z = 255 #Max value an index of v will have
    #print(log(float(n*L)))

    r = int(np.floor(np.log(k/z) / np.log(float(n*L))))
    #times we can multiply A by itself to not overflow UInt64

    p = m // r #floor division
    q = m % r #remainder of m by r
    print("q: ", q)
    print("r: ", r)
    #**Note: A^m = [AA...A][A^r A^r ... A^r] with q A's in first parenthesis
    #        and p A^r's in the second parenthesis

    #Further, A^m = [A^q][A^r A^r ... A^r], where q < r by modulus properties
    A_q = multiply_matrix(A, q)
    A_r = multiply_matrix(A, r)

    #Multiply vector v by A^m matric decomposition:
    product_z = 1
    lambda_v = v
    for i in range(p):
        v_prime = A_r * lambda_v #A^r v
        z_prime = max_index_val(v_prime, n, 1)[0] #max value of lambda
        z_i = 255/z_prime #make max value of lambda_v 255
        v_prime *= z_i #scale lambda_v
        # if v_prime.all() == lambda_v.all(): #O(n)
        #     return v_prime
        lambda_v = v_prime
        print(lambda_v)

    lambda_v = A_q * lambda_v #no more multiplication, so no need to scale lambda_v
    a = 1/lambda_v[n-1,0]
    print(a)
    return lambda_v * a

def max_index_val(A, m = None, n = None):
    """
    Finds the maximum index value of matrix A

    ~Run-time: O(n^2)
    -Matrix Indexed Like:
    [[A_11, A_12, A_13, ..., A_1n]
     [A_21, A_22, A_23, ..., A_2n]
                ...
     [A_n1, A_n2, A_n3, ..., A_nn]]

    Input:
    A - an real mxn matrix -- *no complex, infinite, etc. index values
    m - #rows of A
    n - #cols of A

    Output: a tuple of the value and firsth index of the maximum value element
            in input matrix A     (max_value, (i_index, j_index))
    """
    if n == None or m == None:
        dim = A.shape
        m = dim[0]
        n = dim[1]

    max_val = (float("-inf"), (-1, -1))
    for i in range(m):
        for j in range(n):
            if A[i, j] > max_val[0]:
                max_val = (A[i,j], (i+1, j+1))

    return max_val


def multiply_matrix(A, m):
    """
    Returns A^m from an input matrix A and positive integer m

    *If running this for same matrix, might want to just save the final result
    ~Runtime: O(log(m)n^3)???

    Inputs:
    A - a nxn matrix
    m - a positive integer (greater than or equal to 1)

    Output: A matrix corresponding to the multiplcation of A^m
    """
    print(m)
    #Base Case:
    if m == 0:
        return np.identity(A.shape[0])
    if m == 1:
        return A

    #Recursive Case:
    B = A*A

    if m % 2 == 0: #even case
        return multiply_matrix(B, m/2)
    else: #odd case: m % 2== 1
        return A * multiply_matrix(B , (m-1)/2)

def make_symmetric_matrix(n, a, b):
    """
    Makes an nxn random symmetric integer matrix

    Input:
    n - positive integer to give the dimensions for the nxn symmetric matrix
    a - lower bound on random integer values
    b - upper bound on random integer values

    Output: A random nxn symmetric integer matrix
    """

    B = np.zeros(shape=(n,n))

    for i in range(n):
        for j in range(i, n):
            r = random.randint(a, b)
            B[i, j] = r
            B[j, i] = r

    return B.astype(int)

def print_matrix(A):
    """
    prints matrix A in wolfram alpha form

    Input:
    A - an mxn matrix

    Output: None (printed matrix)
    """
    dim = A.shape
    m = dim[0]
    n = dim[1]

    str_val = "{"
    for i in range(m):
        str_val += "{"
        for j in range(n):
            a = int(A[i,j])
            str_val += str(a)
            if j != n-1:
                str_val += ","

        str_val += "}"
        if i != m-1:
            str_val += ","

    str_val += "}"
    print(str_val)

# A = np.matrix([[1,0,1, 0],[0,0,0,1], [1,0,1,1],[0,1,1,1]])
# print("A: ", A)
#print(multiply_matrix(A, 1))
# print(find_largest_evect(A))
# v = np.matrix([[1], [2], [3]])
# print("v: \n", v)
# lambda_v = np.matrix([[3], [6], [9]])
# print("lambda_v: \n", lambda_v)
# print(eigenvector_check(v, lambda_v))

A = make_symmetric_matrix(2, 1, 10)
m = 50
v = np.matrix([[3],[7]])
print("e-vector: \n", compute_largest_eigenvector(A, m, v))
print_matrix(A)

# z = float("-inf")
# A = np.matrix([[0],[1]])
# print("A: \n", A)
# print("max_index_val: ", max_index_val(A))
