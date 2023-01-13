#找出中值平滑化最佳參數值

import numpy as np
import cv2
import matplotlib.pyplot as plt

#中值濾波法實作
def midian_blur(pepper_img, kernel_size):
	dst = pepper_img.copy()
	d = int(kernel_size-1/2)
	pepper_img = cv2.copyMakeBorder(pepper_img, d, d, d, d, cv2.BORDER_REPLICATE)
	h, w = pepper_img.shape[0], pepper_img.shape[1]
	for i in  range(d, h-d):
		for j in range(d, w-d):
			dst[i-d][j-d] = np.median(pepper_img[i-d:i+d+1, j-d:j+d+1])
	return dst

normal_img_path = "dataset/Balloon.bmp"
noise_img_path = "dataset/Balloon_pepper.bmp"

noise_img = cv2.imread(noise_img_path)
gray_noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)

normal_img = cv2.imread(normal_img_path)
gray_normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)

#模糊化(去除雜訊)
#中值模糊
blur_noise_img_2 = midian_blur(gray_noise_img, 2)
blur_noise_img_3 = midian_blur(gray_noise_img, 3)
blur_noise_img_4 = midian_blur(gray_noise_img, 4)
blur_noise_img_5 = midian_blur(gray_noise_img, 5)
blur_noise_img_6 = midian_blur(gray_noise_img, 6)


#邊偵測
canny_normal_img = cv2.Canny(gray_normal_img, 30, 200)
canny_noise_img_2 = cv2.Canny(blur_noise_img_2, 30, 200)
canny_noise_img_3 = cv2.Canny(blur_noise_img_3, 30, 200)
canny_noise_img_4 = cv2.Canny(blur_noise_img_4, 30, 200)
canny_noise_img_5 = cv2.Canny(blur_noise_img_5, 30, 200)
canny_noise_img_6 = cv2.Canny(blur_noise_img_6, 30, 200)



images = [[gray_normal_img, blur_noise_img_2, blur_noise_img_3, blur_noise_img_4, blur_noise_img_5, blur_noise_img_6],
		  [canny_normal_img, canny_noise_img_2, canny_noise_img_3, canny_noise_img_4, canny_noise_img_5, canny_noise_img_6]]
labels = {0:"normal",1: "(2,2)", 2:"(3,3)", 3: "(4,4)", 4: "(5,5)", 5: "(6,6)"}

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