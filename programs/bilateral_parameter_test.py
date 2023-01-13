# 找出雙側濾波最佳參數組合

import numpy as np
import cv2
import matplotlib.pyplot as plt

#相似度計算
def similiar(img1, img2):
	img1 = np.array(img1)
	img2 = np.array(img2)
	result = (np.count_nonzero(img1-img2)*(-1.0) + (img1.size - np.count_nonzero(img1-img2)))/img1.size
	return result

normal_img_path = "dataset/Balloon.bmp"
noise_img_path = "dataset/Balloon_guass.bmp"

noise_img = cv2.imread(noise_img_path)
gray_noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)

normal_img = cv2.imread(normal_img_path)
gray_normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)

#模糊化(去除雜訊)
#雙側濾波
blur_noise_img_2 = cv2.bilateralFilter(gray_noise_img, d=9, sigmaColor=100, sigmaSpace=100)  
blur_noise_img_3 = cv2.bilateralFilter(gray_noise_img, d=9, sigmaColor=130, sigmaSpace=130)
blur_noise_img_4 = cv2.bilateralFilter(gray_noise_img, d=9, sigmaColor=150, sigmaSpace=150)
blur_noise_img_5 = cv2.bilateralFilter(gray_noise_img, d=9, sigmaColor=160, sigmaSpace=160)
blur_noise_img_6 = cv2.bilateralFilter(gray_noise_img, d=9, sigmaColor=180, sigmaSpace=180)


#邊偵測
canny_normal_img = cv2.Canny(gray_normal_img, 30, 200)
canny_noise_img_2 = cv2.Canny(blur_noise_img_2, 30, 200)
canny_noise_img_3 = cv2.Canny(blur_noise_img_3, 30, 200)
canny_noise_img_4 = cv2.Canny(blur_noise_img_4, 30, 200)
canny_noise_img_5 = cv2.Canny(blur_noise_img_5, 30, 200)
canny_noise_img_6 = cv2.Canny(blur_noise_img_6, 30, 200)



images = [[gray_normal_img, blur_noise_img_2, blur_noise_img_3, blur_noise_img_4, blur_noise_img_5, blur_noise_img_6],
		  [canny_normal_img, canny_noise_img_2, canny_noise_img_3, canny_noise_img_4, canny_noise_img_5, canny_noise_img_6]]
labels = {0:"normal", 1: "d=9\nsigmaColor=100\nsigmaSpace=100", 2:"d=9\nsigmaColor=130\nsigmaSpace=130", 
		3: "d=9\nsigmaColor=150\nsigmaSpace=150", 4: "d=9\nsigmaColor=160\nsigmaSpace=160", 5: "d=9\nsigmaColor=180\nsigmaSpace=180"}

print(similiar(canny_noise_img_2, canny_normal_img))
print(similiar(canny_noise_img_3, canny_normal_img))
print(similiar(canny_noise_img_4, canny_normal_img))
print(similiar(canny_noise_img_5, canny_normal_img))
print(similiar(canny_noise_img_6, canny_normal_img))

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