import os

import numpy as np
from PIL import Image


def load_image(path):
    # uint8
    return np.array(Image.open(path).convert("RGB"))

def save_image(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)

def to_gray(img):
    # Y = 0.299R + 0.587G + 0.114B
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.uint8(gray)


# методи
def hist_equalize(gray):
    hist = np.zeros(256, dtype=int)
    for pixel in gray.flat:
        hist[pixel] += 1

    sum_hist = []
    total = 0
    for count in hist:
        total += count
        sum_hist.append(total)

    min_sum_hist = sum_hist[gray.min()]

    if min_sum_hist == gray.size:
        return gray.copy()

    table = []
    for value in sum_hist:
        table.append((value - min_sum_hist) * 255 / (gray.size - min_sum_hist))

    return np.uint8(table)[gray]

def sobel(gray, threshold):
    image = np.float64(gray)

    top_left = image[:-2, :-2]
    top = image[:-2, 1:-1]
    top_right = image[:-2, 2:]
    left = image[1:-1, :-2]
    right = image[1:-1, 2:]
    bottom_left = image[2:, :-2]
    bottom = image[2:, 1:-1]
    bottom_right = image[2:, 2:]

    horizontal = (top_right + 2 * right + bottom_right) - (top_left + 2 * left + bottom_left)
    vertical = (bottom_left + 2 * bottom + bottom_right) - (top_left + 2 * top + top_right)
    strength = np.sqrt(horizontal ** 2 + vertical ** 2)

    result = np.zeros(gray.shape, dtype=np.uint8)
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            if strength[y - 1, x - 1] > threshold:
                result[y, x] = 255
    return result

def otsu(gray, cfg):
    return gray.copy()


def morph(gray, cfg):
    return gray.copy()


def lbp(gray, cfg):
    return gray.copy()


# шум, метрики
def add_noise(gray, kind, seed):
    return gray.copy()


def mse(image1, image2):
    # середній квадрат різниці
    diff = np.float64(image1) - np.float64(image2)
    return np.sum(diff ** 2) / diff.size


def psnr(image1, image2):
    return 0


def ssim(a, b):
   return 0


def main():
    sort_files = sorted(os.listdir("images"))
    for filename in sort_files:
        try:
            gray = to_gray(load_image("images/" + filename))
        except Exception:
            continue

        result = hist_equalize(gray)
        name = os.path.splitext(filename)[0]
        save_image(result, "results/hist/" + name + ".png")

        result = sobel(gray, 100)
        save_image(result, "results/sobel/" + name + ".png")


if __name__ == "__main__":
    main()