#為正常影像加入雜訊

import cv2 as cv
import numpy as np
import os


def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = h*w
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    for i in range(nums):
        if i % 2 == 1:
            image[rows[i], cols[i]] = (255, 255, 255)
        else:
            image[rows[i], cols[i]] = (0, 0, 0)
    return image


def add_gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    mean = (15, 15, 15)
    std = (30, 30, 30)
    cv.randn(noise, mean, std)
    dst = cv.add(image, noise)
    return dst

dir_path = "dataset"
for f in os.listdir(dir_path):
    filename = f.split('.')[0]
    src = cv.imread(os.path.join(dir_path, f))
    h, w = src.shape[:2]
    copy = np.copy(src)
    pepper = add_salt_pepper_noise(copy)
    guass = add_gaussian_noise(copy)
    cv.imwrite(os.path.join(dir_path, filename+'_guass.bmp'), guass)
    cv.imwrite(os.path.join(dir_path, filename+'_pepper.bmp'), pepper)
