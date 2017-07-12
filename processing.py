import utils


class Process:

    def __init__(self, width, height, pixels):
        self.w  = width
        self.h = height
        self.img = pixels
        self.t = 0  # threshold

    def get_img_as_array(self):
        return [[self.img[i, j] for j in range(self.h)] for i in range(self.w)]

    def otsu_threshold(self):
        img_data = [self.img[i, j] for j in range(self.h) for i in range(self.w)]
        min_px = min(img_data)
        max_px = max(img_data)
        delta_px = max_px - min_px
        hist = [len(list(filter(lambda x: x == min_px + i, img_data))) for i in range(delta_px + 1)]
        m = sum((map(lambda x, y: x * y, range(delta_px), hist)))
        n = sum(hist)

        max_sigma = -1
        self.t = min_px
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

    def binarize(self):
        for i in range(self.w):
            for j in range(self.h):
                if self.img[i, j] >= self.t:
                    self.img[i, j] = 255
                else:
                    self.img[i, j] = 0

    def preprocess(self):  # реализовываем их предобработку
        # массив с компонентами связности
        cc_map = self.find_cc()
        for i in range(1, self.w - 1):
            for j in range(1, self.h - 1):
                if self.img[i, j] == 0:
                    continue
                # если соседние пиксели не подходят ни под одну из предложенных картинок на рис. 3 -
                # - меняем цвет на чёрный
                if self.img[i - 1, j] != self.img[i + 1, j] or self.img[i, j - 1] != self.img[i, j + 1]:
                    self.img[i, j] = 0
                cc_map_new = self.find_cc()
                if max(map(max, cc_map)) > max(map(max, cc_map_new)):  # уменьшилось число компонент связности
                    self.img[i, j] = 255

    def skeletonize(self):
        utils.save_debug_img(self.get_img_as_array(), 'before.png')
        self.preprocess()
        utils.save_debug_img(self.get_img_as_array(), 'after.png')

    # cc - connectivity components

    def find_cc(self):
        img_tmp = self.get_img_as_array()
        cc = [[0 for _ in range(self.h)] for _ in range(self.w)]
        cc_count = 0

        def f(_i, _j):
            if _i < 0 or _i >= self.w or _j < 0 or _j >= self.h or img_tmp[_i][_j] != 0:
                return
            cc[_i][_j] = cc_count
            img_tmp[_i][_j] = 255
            f(_i - 1, _j)
            f(_i, _j - 1)
            f(_i + 1, _j)
            f(_i, _j + 1)

        for i in range(self.w):
            for j in range(self.h):
                if img_tmp[i][j] == 0:
                    cc_count += 1
                    f(i, j)
        return cc

    def make(self):
        self.otsu_threshold()
        self.binarize()
        self.skeletonize()
