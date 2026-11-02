"""Обробка зображень класичними методами: hist, sobel, otsu, morph, lbp."""
import csv
import json
import os

import numpy as np
from PIL import Image


def load_image(path):
    """Завантажує зображення як RGB масив uint8."""
    # uint8
    return np.array(Image.open(path).convert("RGB"))

def save_image(arr, path):
    """Зберігає масив як зображення, створюючи папку."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)

def to_gray(img):
    """Перетворює RGB зображення в градації сірого."""
    # Y = 0.299R + 0.587G + 0.114B
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.uint8(gray)


# методи
def hist_equalize(gray):
    """Вирівнювання гістограми."""
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
    """Виділення країв оператором Собеля з порогом."""
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
    result[1:-1, 1:-1][strength > threshold] = 255
    return result

def otsu(gray):
    """Бінаризація методом Оцу."""
    hist = np.bincount(gray.ravel(), minlength=256)

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
    """Морфологічна дилатація вікном 3x3."""
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

                    if 0 <= ny < height and 0 <= nx < width and gray[ny, nx] > max_value:
                        max_value = gray[ny, nx]

            result[y, x] = max_value

    return result

def lbp(gray):
    """Локальні бінарні шаблони (LBP)."""
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

    for i, neighbor in enumerate(neighbors):
        if i == 0:
            result[neighbor >= center] += 1
        else:
            result[neighbor >= center] += 2 ** i

    return result


# шум, метрики
def add_noise(gray, kind, seed, sigma, amount):
    """Додає gaussian або salt_pepper шум."""
    rng = np.random.default_rng(seed)
    result = gray.copy()

    if kind == "gaussian":
        noisy = gray + rng.normal(0, sigma, gray.shape)
        result = np.uint8(np.clip(noisy, 0, 255))
    elif kind == "salt_pepper":
        chance = rng.random(gray.shape)
        result[chance < amount / 2] = 0
        result[chance > 1 - amount / 2] = 255

    return result


def mse(image1, image2):
    """Середньоквадратична помилка."""
    # середній квадрат різниці
    diff = np.float64(image1) - np.float64(image2)
    return np.sum(diff ** 2) / diff.size


def psnr(image1, image2):
    """Пікове відношення сигнал/шум у dB."""
    # 10 * log10(255^2 / MSE)
    error = mse(image1, image2)
    if error == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / error)


def ssim(image1, image2):
    """Індекс структурної подібності для всього зображення."""
    # SSIM(x, y) = ((2 * μx * μy + C1) * (2 * σxy + C2)) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))
    image1 = np.float64(image1)
    image2 = np.float64(image2)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    cov = np.mean((image1 - image1.mean()) * (image2 - image2.mean()))
    top = (2 * image1.mean() * image2.mean() + c1) * (2 * cov + c2)
    bottom = (image1.mean() ** 2 + image2.mean() ** 2 + c1) * (image1.var() + image2.var() + c2)
    return top / bottom


def run_method(method_name, gray, config):
    """Запускає метод за назвою."""
    if method_name == "hist":
        return hist_equalize(gray)
    if method_name == "sobel":
        return sobel(gray, config["sobel_threshold"])
    if method_name == "otsu":
        return otsu(gray)
    if method_name == "morph":
        return morph(gray)
    return lbp(gray)


def choose_methods():
    """Меню методів: None - вихід, порожній список - невірний номер."""
    print("Вибір методу:")
    print("1 - hist")
    print("2 - sobel")
    print("3 - otsu")
    print("4 - morph")
    print("5 - lbp")
    print("6 - Усі методи")
    print("0 - Вихід")

    choice = input("Введіть номер: ")
    all_methods = {"1": ["hist"], "2": ["sobel"], "3": ["otsu"], "4": ["morph"], "5": ["lbp"],
                   "6": ["hist", "sobel", "otsu", "morph", "lbp"]}

    if choice == "0":
        return None
    if choice in all_methods:
        return all_methods[choice]
    print("Невірний номер")
    return []


def choose_noise():
    """Меню шуму: повертає (шум, суфікс папки) або None при невірному номері."""
    print("Вибір шуму:")
    print("1 - Без шуму")
    print("2 - Gaussian")
    print("3 - Salt and pepper")

    match input("Введіть номер: "):
        case "1":
            return None, ""
        case "2":
            return "gaussian", "_gaussian"
        case "3":
            return "salt_pepper", "_salt_pepper"
        case _:
            print("Невірний номер")
            return False


def process_images(methods, noise, noise_name, config):
    """Обробляє всі зображення, зберігає результати і metrics.csv."""
    metrics = {}
    for method_name in methods:
        metrics[method_name] = []

    for filename in sorted(os.listdir("images")):
        try:
            original = to_gray(load_image("images/" + filename))
            gray = add_noise(original, noise, config["seed"],
                             config["gaussian_sigma"], config["salt_pepper"])
        except (OSError, ValueError):
            print("Пропущено:", filename)
            continue

        if original.shape[0] < 3 or original.shape[1] < 3:
            print("Пропущено (менше 3x3):", filename)
            continue

        name = os.path.splitext(filename)[0]
        for method_name in methods:
            result = run_method(method_name, gray, config)

            folder = "results/" + method_name + noise_name
            path = folder + "/" + name + ".png"

            save_image(result, path)
            metrics[method_name].append([filename, mse(original, result),
                                        psnr(original, result), ssim(original, result)])
            print("Оброблено:", filename, "-", method_name)

    # метрики csv
    for method_name in methods:
        path = "results/" + method_name + noise_name + "/metrics.csv"
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "mse", "psnr", "ssim"])
            writer.writerows(metrics[method_name])


def main():
    """Меню запуску методів на зображеннях з папки images."""
    with open("config.json", encoding="utf-8") as file:
        config = json.load(file)

    if not os.path.isdir("images"):
        print("Папку images не знайдено")
        return

    while True:
        methods = choose_methods()
        if methods is None:
            break
        if len(methods) == 0:
            continue

        noise_choice = choose_noise()
        if noise_choice is False:
            continue

        process_images(methods, noise_choice[0], noise_choice[1], config)
        print("Обробку завершено")


if __name__ == "__main__":
    main()
