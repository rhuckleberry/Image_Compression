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


def display_img(img_matrix, is_color):
    """
    Displays an image
    """
    if is_color:
        cv2.imshow("Image", img_matrix/255)
    else:
        cv2.imshow('Image', img_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_color_channel(input_matrix, channel_num, is_grayscale):
    """
    INPUT:
    input_matrix - Mat type corresponding to an image with three
        BGR color channels
    channel_num - number of channel to be selected (0, 1, or 2)
    is_grayscale - selects if the output should be grayscale or color

    OUTPUT: matrix with one channel in grayscale or reduced to 1 nonzero channel
                in color
    """
    mat = input_matrix[:,:,channel_num]
    if is_grayscale:
        return mat
    else:
        img = np.zeros(input_matrix.shape)
        img[:,:,channel_num] = mat
        return img


def combine_channels(red_channel, green_channel, blue_channel):
    """
    Takes in three single channel images and combines into a color image

    INPUT:
    red_channel - single channel corresponding to the red output channel
    blue_channel - single channel corresponding to the blue output channel
    green_channel - single channel corresponding to the green output channel

    OUTPUT - CV matrix corresponding to a color image
    """
    if red_channel.shape != green_channel.shape or red_channel.shape != blue_channel.shape:
        print("Channel combination error: channel dimensions do not match")
        return
    output = np.zeros((red_channel.shape[0], red_channel.shape[1], 3))
    output[:,:,0] = blue_channel
    # output[:,:,1] = green_channel
    output[:,:,2] = red_channel
    return output


if __name__ == "__main__":
    # Get matrix
    matrix = import_img('test1.png', False)

    # Testing conversion to other colorspaces
    # matrix_convert = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
    # print(matrix)
    # print(matrix_convert)

    #Show Image

    # img = select_color_channel(matrix,1,False)
    # print(img)
    # display_img(img, True)
#    cv2.imshow("image1", select_color_channel(matrix, 0, False)/256)
#    cv2.imshow("image2", select_color_channel(matrix, 1, False)/256)
#    cv2.imshow("image3", select_color_channel(matrix, 2, False)/256)

    red = select_color_channel(matrix, 2, True)
    green = select_color_channel(matrix, 1, True)
    blue = select_color_channel(matrix, 0, True)

    reconstructed = combine_channels(red/256, green/256, blue/256)

    cv2.imshow("image", reconstructed)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
