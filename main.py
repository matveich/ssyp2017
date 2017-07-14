from PIL import Image
import processing
import utils
import sys
import numpy as np

sys.setrecursionlimit(10000)

filename = '54.png'
img = Image.open(filename).convert('L')
w, h = img.size
pixels = img.load()
pixels = np.reshape([pixels[i, j] for j in range(h) for i in range(w)], (w, h))
process = processing.Process(w, h, pixels)
img = process.make()
