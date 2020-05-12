import cv2
import numpy as np


def import_img(path, is_grayscale):
    """
    Takes in Image, returns numpy matrix

    INPUT:
    path - string, gives path to Image
    is_grayscale - bool, true corresponds to grayscale Output

    OUTPUT: image matrix (cv2.Mat type)
    """
    if is_grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)


def display_img(img_matrix):
    """
    Displays an image
    """
    cv2.imshow('Image', img_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_color_channel(input_matrix, channel_num, is_grayscale):
    """
    INPUT:
    input_matrix - Mat type corresponding to an image with three
        BGR color channels
    channel_num - number of channel to be selected (0, 1, or 2)

    OUTPUT: matrix with one channel in grayscale
    """
    mat = input_matrix[:,:,channel_num]
    if is_grayscale:
        return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    else:
        img = np.zeros(input_matrix.shape)
        img[:,:,channel_num] = mat


if __name__ == "__main__":
    # Get matrix
    matrix = import_img('test1.png', False)

    # Testing conversion to other colorspaces
    matrix_convert = cv2.cvtColor(matrix, cv2.COLOR_BGR2HSV)
    # print(matrix)
    # print(matrix_convert)

    print(matrix)
    print(matrix[2,:,:])
    #Show Image

    display_img(select_color_channel(matrix,2,False))



    cv2.waitKey(0)
    cv2.destroyAllWindows()
