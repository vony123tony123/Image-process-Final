#找出高斯平滑化最佳參數值

import numpy as np
import cv2
import matplotlib.pyplot as plt

normal_img_path = "dataset/Balloon.bmp"
noise_img_path = "dataset/Balloon_guass.bmp"

noise_img = cv2.imread(noise_img_path)
gray_noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)

normal_img = cv2.imread(normal_img_path)
gray_normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)

#模糊化(去除雜訊)
#高斯模糊
blur_noise_img_3 = cv2.GaussianBlur(gray_noise_img,(3,3),0)
blur_noise_img_5 = cv2.GaussianBlur(gray_noise_img,(5,5),0)
blur_noise_img_7 = cv2.GaussianBlur(gray_noise_img,(7,7),0)
blur_noise_img_9 = cv2.GaussianBlur(gray_noise_img,(9,9),0)
blur_noise_img_11 = cv2.GaussianBlur(gray_noise_img,(11,11),0)

#邊偵測
canny_normal_img = cv2.Canny(gray_normal_img, 30, 200)
canny_noise_img_3 = cv2.Canny(blur_noise_img_3, 30, 200)
canny_noise_img_5 = cv2.Canny(blur_noise_img_5, 30, 200)
canny_noise_img_7 = cv2.Canny(blur_noise_img_7, 30, 200)
canny_noise_img_9 = cv2.Canny(blur_noise_img_9, 30, 200)
canny_noise_img_11 = cv2.Canny(blur_noise_img_11, 30, 200)



images = [[gray_normal_img, blur_noise_img_3, blur_noise_img_5, blur_noise_img_7, blur_noise_img_9, blur_noise_img_11],
		  [canny_normal_img, canny_noise_img_3, canny_noise_img_5, canny_noise_img_7, canny_noise_img_9, canny_noise_img_11]]
labels = {0:"normal",1: "(3,3)", 2:"(5,5)", 3: "(7,7)", 4: "(9,9)", 5: "(11,11)"}

plt.figure()
for i in range(len(images)):
	for j in range(len(images[i])):
		plt.subplot(len(images),len(images[0]),i*len(images[0])+j+1)
		result = 0
		if i == 1:
			plt.xlabel(labels[j], rotation=0)
		ax = plt.gca()
		ax.axes.xaxis.set_ticks([])
		ax.axes.yaxis.set_ticks([])
		#x,y軸各放大一倍
		img = cv2.resize(images[i][j], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
		plt.imshow(img,'gray')
plt.show()