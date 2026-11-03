"""Автотести методів на малих зображеннях"""
import numpy as np

import stem


def test_hist_equalize():
    """Значення 0 (6 пікселів) і 100 (3 пікселі): 0 -> 0, 100 -> 255"""
    gray = np.uint8([[0, 0, 0], [0, 0, 0], [100, 100, 100]])
    expected = np.uint8([[0, 0, 0], [0, 0, 0], [255, 255, 255]])
    assert (stem.hist_equalize(gray) == expected).all()


def test_sobel():
    """Вертикальна межа 0|255: сила в центрі 1020 > 100, краї рамки 0"""
    gray = np.uint8([[0, 0, 255]] * 3)
    expected = np.uint8([[0, 0, 0], [0, 255, 0], [0, 0, 0]])
    assert (stem.sobel(gray, 100) == expected).all()


def test_otsu():
    """Темні 10 -> 0, світлі 200 -> 255"""
    gray = np.uint8([[10, 10, 200], [10, 200, 200], [10, 10, 200]])
    expected = np.uint8([[0, 0, 255], [0, 255, 255], [0, 0, 255]])
    assert (stem.otsu(gray) == expected).all()


def test_morph():
    """Максимум у вікні 3x3: одне значення 9 розтікається на всі пікселі"""
    gray = np.uint8([[0, 0, 0], [0, 9, 0], [0, 0, 0]])
    expected = np.uint8([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
    assert (stem.morph(gray) == expected).all()


def test_lbp():
    """Центр 5, сусіди зверху, праворуч, знизу, ліворуч більші: 2 + 8 + 32 + 128 = 170"""
    gray = np.uint8([[1, 9, 1], [9, 5, 9], [1, 9, 1]])
    expected = 170
    assert stem.lbp(gray)[1, 1] == expected


if __name__ == "__main__":
    test_hist_equalize()
    test_sobel()
    test_otsu()
    test_morph()
    test_lbp()
    print("Усі тести пройдено")
