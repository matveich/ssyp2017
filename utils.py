from matplotlib import image


def save_debug_img(arr, filename):
    image.imsave(filename, arr, vmin=0, vmax=255, cmap='gray', origin='upper')


def print_img(arr):
    for x in arr:
        for p in x:
            print(p, end='')
        print()
    print()


def bin_px(px):
    return 0 if px == 255 else 1
