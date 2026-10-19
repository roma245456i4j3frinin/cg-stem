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

def otsu(gray):
    hist = np.zeros(256, dtype=int)
    for pixel in gray.flat:
        hist[pixel] += 1

    total = gray.size
    sum_all = 0
    for i in range(256):
        sum_all += i * hist[i]

    best_threshold = 0
    best_score = 0
    dark_count = 0
    dark_sum = 0

    for threshold in range(256):
        dark_count += hist[threshold]
        dark_sum += threshold * hist[threshold]
        light_count = total - dark_count
        if dark_count == 0 or light_count == 0:
            continue
        dark_mean = dark_sum / dark_count
        light_mean = (sum_all - dark_sum) / light_count
        score = dark_count * light_count * (dark_mean - light_mean) ** 2
        if score > best_score:
            best_score = score
            best_threshold = threshold

    result = np.zeros(gray.shape, dtype=np.uint8)
    result[gray > best_threshold] = 255

    return result


def morph(gray):
    size = 3
    radius = size // 2
    height, width = gray.shape
    result = np.zeros(gray.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            max_value = 0

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = y + dy
                    nx = x + dx

                    if 0 <= ny < height and 0 <= nx < width:
                        if gray[ny, nx] > max_value:
                            max_value = gray[ny, nx]

            result[y, x] = max_value

    return result

def lbp(gray):
    image = np.pad(gray, 1, mode="edge")
    center = image[1:-1, 1:-1]
    result = np.zeros(gray.shape, dtype=np.uint8)

    neighbors = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2]
    ]

    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        if i == 0:
            result[neighbor >= center] += 1
        else:
            result[neighbor >= center] += 2 ** i

    return result


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

        result = otsu(gray)
        save_image(result, "results/otsu/" + name + ".png")

        result = morph(gray)
        save_image(result, "results/morph/" + name + ".png")

        result = lbp(gray)
        save_image(result, "results/lbp/" + name + ".png")


if __name__ == "__main__":
    main()