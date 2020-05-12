import cv2
import numpy as np


def import_img(path, grayscale):
### Takes in Image, returns numpy matrix
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)

# Get matrix
mat = import_img('test1.png', False)

mat_convert = cv2.cvtColor(mat, cv2.COLOR_RGB2HSV)
print(mat)
print(mat_convert)



#Show Image
cv2.imshow('image', mat)

cv2.waitKey(0)
cv2.destroyAllWindows()
