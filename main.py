import numpy as np
import cv2
import matplotlib.pyplot as plt

def midian_blur(img, kernel_size):
	dst = img.copy()
	d = int(kernel_size-1/2)
	img = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_REPLICATE)
	h, w = img.shape[0], img.shape[1]
	for i in  range(d, h-d):
		for j in range(d, w-d):
			dst[i-d][j-d] = np.median(img[i-d:i+d+1, j-d:j+d+1])
	return dst

def max_blur(img, kernel_size):
	dst = img.copy()
	d = int(kernel_size-1/2)
	img = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_REPLICATE)
	h, w = img.shape[0], img.shape[1]
	for i in  range(d, h-d):
		for j in range(d, w-d):
			dst[i-d][j-d] = np.max(img[i-d:i+d+1, j-d:j+d+1])
	return dst

def min_blur(img, kernel_size):
	dst = img.copy()
	d = int(kernel_size-1/2)
	img = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_REPLICATE)
	h, w = img.shape[0], img.shape[1]
	for i in  range(d, h-d):
		for j in range(d, w-d):
			dst[i-d][j-d] = np.min(img[i-d:i+d+1, j-d:j+d+1])
	return dst

img_path = "dataset/Pepper_pepper.bmp"
img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#模糊化(去除雜訊)
#blur_img = midian_blur(gray_img, 3)
blur_img = cv2.ercode(gray_img, kernel_size)



# image segementation
# OTSU
ret, th1 = cv2.threshold(blur_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 自適應二值化
th2 = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)

images = [gray_img, blur_img, th1, th2]
plt.figure()
for i in range(len(images)):
	plt.subplot(1,len(images),i+1)
	#x,y軸各放大一倍
	img = cv2.resize(images[i], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
	plt.imshow(img,'gray')
plt.show()