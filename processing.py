import utils
import numpy as np


class Process:

    def __init__(self, width, height, pixels):
        self.w = width
        self.h = height
        self.img = pixels
        self.t = 0  # threshold

    def otsu_threshold(self):
        min_px = np.amin(self.img)
        max_px = np.amax(self.img)
        delta_px = max_px - min_px
        hist = [self.img[self.img == min_px + i].size for i in range(delta_px + 1)]
        m = sum((map(lambda x, y: x * y, range(delta_px), hist)))
        n = sum(hist)
        max_sigma = -1
        alpha = 0
        beta = 0

        for t in range(delta_px):
            alpha += t * hist[t]
            beta += hist[t]
            w = beta / n
            a = alpha / beta - (m - alpha) / (n - beta)

            sigma = w * (1 - w) * a * a
            if sigma > max_sigma:
                max_sigma = sigma
                self.t = t
        self.t += min_px

    def binarize(self):
        self.img[self.img > self.t] = 255
        self.img[self.img <= self.t] = 0

    def preprocess(self):  # реализовываем их предобработку
        # массив с компонентами связности
        cc_map = self.find_cc()
        pixels_for_change = []
        for i in range(1, self.h - 1):
            for j in range(1, self.w - 1):
                if self.img[i][j] == 0:
                    continue
                # если соседние пиксели не подходят ни под одну из предложенных картинок на рис. 3 -
                # - меняем цвет на чёрный
                if self.img[i - 1][j] != self.img[i + 1][j] or self.img[i][j - 1] != self.img[i][j + 1]:
                    self.img[i][j] = 0
                    cc_map_new = self.find_cc()
                    self.img[i][j] = 255
                    if max(map(max, cc_map)) <= max(map(max, cc_map_new)):  # не уменьшилось число компонент связности
                        pixels_for_change.append((i, j))
        for i, j in pixels_for_change:
            self.img[i][j] = 0

    def zhang_suen(self):
        def iteration(step=1):
            pixels_to_change = []
            for i in range(1, self.w - 1):
                for j in range(1, self.h - 1):
                    if self.img[i][j] == 255:
                        continue
                    seq = str(utils.bin_px(self.img[i][j - 1])) + \
                        str(utils.bin_px(self.img[i + 1][j - 1])) + \
                        str(utils.bin_px(self.img[i + 1][j])) + \
                        str(utils.bin_px(self.img[i + 1][j + 1])) + \
                        str(utils.bin_px(self.img[i][j + 1])) + \
                        str(utils.bin_px(self.img[i - 1][j + 1])) + \
                        str(utils.bin_px(self.img[i - 1][j])) + \
                        str(utils.bin_px(self.img[i - 1][j - 1])) + \
                        str(utils.bin_px(self.img[i][j - 1]))
                    cond = [False for _ in range(4)]
                    cond[0] = (2 <= sum(map(int, seq[:-1])) <= 6)
                    cond[1] = (seq.count('01') == 1)
                    if step == 1:
                        cond[2] = any(map(lambda x: x == '0', seq[0] + seq[2] + seq[4]))
                        cond[3] = any(map(lambda x: x == '0', seq[2] + seq[4] + seq[6]))
                    else:
                        cond[2] = any(map(lambda x: x == '0', seq[0] + seq[2] + seq[6]))
                        cond[3] = any(map(lambda x: x == '0', seq[0] + seq[4] + seq[6]))
                    if all(cond):
                        pixels_to_change.append((i, j))
            for (_i, _j) in pixels_to_change:
                self.img[_i][_j] = 255
            return len(pixels_to_change)

        while True:
            pixels_count = iteration()
            pixels_count += iteration(step=2)
            if pixels_count == 0:
                break

    def wu_tsai(self):
        pass

    def thinning(self):
        utils.save_debug_img(self.img, 'step1.png')
        self.preprocess()
        utils.save_debug_img(self.img, 'step2.png')
        self.zhang_suen()
        utils.save_debug_img(self.img, 'step3.png')

    # cc - connectivity components

    def find_cc(self):
        cc = [[0 for _ in range(self.w)] for _ in range(self.h)]
        tmp_img = np.copy(self.img)
        cc_count = 0

        def f(_i, _j):
            if _i < 0 or _i >= self.h or _j < 0 or _j >= self.w or tmp_img[_i][_j] != 0:
                return
            cc[_i][_j] = cc_count
            tmp_img[_i][_j] = 255
            f(_i - 1, _j)
            f(_i, _j - 1)
            f(_i + 1, _j)
            f(_i, _j + 1)

        for i in range(self.h):
            for j in range(self.w):
                if tmp_img[i][j] == 0:
                    cc_count += 1
                    f(i, j)
        return cc

    def make(self):
        self.otsu_threshold()
        self.binarize()
        self.thinning()
