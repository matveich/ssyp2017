from scipy import misc
import numpy as np


def save_debug_img(arr, filename):
    arr = np.asarray(arr)
    arr = misc.imrotate(arr, 90)
    arr = np.flipud(arr)
    misc.imsave(filename, arr)
