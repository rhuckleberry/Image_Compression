import cv2
import numpy as np

## Takes in Image, returns numpy matrix
def import_img(path, grayscale):
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)

mat = import_img('test1.png', True)
print(mat)
cv2.imshow('image', mat)

cv2.waitKey(0)
cv2.destroyAllWindows()
