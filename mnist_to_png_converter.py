import gzip
import os
from PIL import Image, ImageOps


def read_mnist(path, mode='images'):
    raw_data = gzip.open(path, 'rb')
    int.from_bytes(raw_data.read(4), byteorder='big')
    items = int.from_bytes(raw_data.read(4), byteorder='big')
    if mode == 'images':
        rows = int.from_bytes(raw_data.read(4), byteorder='big')
        cols = int.from_bytes(raw_data.read(4), byteorder='big')
        data = raw_data.read()
        return rows, cols, data
    if mode == 'labels':
        data = raw_data.read()
        return items, data
    raise ValueError("Inappropriate mode: %s. Available mode values: 'images', 'labels'" % str(mode))


def write_png(images_data, labels_data, items, rows, cols, mode="training"):
    path = './mnist/png/' + mode
    id = 1
    images_data = [images_data[i * rows * cols: (i + 1) * rows * cols] for i in range(items)]
    for X, y in zip(images_data, labels_data):
        filedir = path + '/' + str(y) + '/'
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        img = Image.frombytes('L', (rows, cols), X)
        img = ImageOps.invert(img)
        img.save(filedir + str(id) + '.png')
        id += 1


def convert():
    train_rows, train_cols, train_x = read_mnist('./mnist/train-images-idx3-ubyte.gz')
    train_items, train_y = read_mnist('./mnist/train-labels-idx1-ubyte.gz', mode='labels')
    test_rows, test_cols, test_x = read_mnist('./mnist/t10k-images-idx3-ubyte.gz')
    test_items, test_y = read_mnist('./mnist/t10k-labels-idx1-ubyte.gz', mode='labels')
    write_png(train_x, train_y, train_items, train_rows, train_cols)
    write_png(test_x, test_y, test_items, test_rows, test_cols, mode="testing")
