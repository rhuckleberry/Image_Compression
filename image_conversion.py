import cv2
import numpy as np


def import_img(path, is_grayscale):
### Takes in Image, returns numpy matrix
### path: string, gives path to Image
### is_grayscale: bool, true corresponds to grayscale Output
### RETURNS: image matrix (cv2.Mat type)
    if is_grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)

def display_img(img_matrix):
### Displays the Image
    cv2.imshow('Image', img_matrix)


if __name__ == "__main__":
    # Get matrix
    matrix = import_img('test1.png', False)

    # Testing conversion to other colorspaces
    matrix_convert = cv2.cvtColor(mat, cv2.COLOR_RGB2HSV)
    print(matrix)
    print(matrix_convert)

    #Show Image
    display_img(matrix)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
