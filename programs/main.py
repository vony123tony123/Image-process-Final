# 比較不同平滑化方法的邊偵測

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


#相似度計算
def similiar(img1, img2):
	img1 = np.array(img1)
	img2 = np.array(img2)
	result = (np.count_nonzero(img1-img2)*(-1.0) + (img1.size - np.count_nonzero(img1-img2)))/img1.size
	return result



normal_img_path = "dataset/Balloon.bmp"
pepper_img_path = "dataset/Balloon_guass.bmp"

normal_img = cv2.imread(normal_img_path)
gray_normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)

pepper_img = cv2.imread(pepper_img_path)
gray_pepper_img = cv2.cvtColor(pepper_img, cv2.COLOR_BGR2GRAY)

#模糊化(去除雜訊)
#高斯模糊
blur_guass_pepper_img = cv2.GaussianBlur(gray_pepper_img,(7,7),0)

#中值模糊化
blur_median_pepper_img = midian_blur(gray_pepper_img, 3)

#雙側濾波器
blur_bilateral_pepper_img = cv2.bilateralFilter(gray_pepper_img, d=9, sigmaColor=130, sigmaSpace=130)




# image segementation
# OTSU
#ret, th1 = cv2.threshold(blur_pepper_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 自適應二值化
#th2 = cv2.adaptiveThreshold(blur_pepper_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)


#邊偵測
canny_normal_img = cv2.Canny(gray_normal_img, 30, 200)
canny_guass_pepper_img = cv2.Canny(blur_guass_pepper_img, 30, 200)
canny_median_pepper_img = cv2.Canny(blur_median_pepper_img, 30, 200)
canny_bilateral_pepper_img = cv2.Canny(blur_bilateral_pepper_img, 30, 200)
canny__pepper_img = cv2.Canny(gray_pepper_img, 30, 200)


#印出相似度
print(similiar(canny_normal_img, canny_normal_img))
print(similiar(canny_normal_img, canny__pepper_img))
print(similiar(canny_normal_img, canny_guass_pepper_img))
print(similiar(canny_normal_img, canny_median_pepper_img))
print(similiar(canny_normal_img, canny_bilateral_pepper_img))

# img_hash = cv2.img_hash.PHash_create()
# print(img_hash.compare(img_hash.compute(canny_normal_img), img_hash.compute(canny_normal_img)))
# print(img_hash.compare(img_hash.compute(canny_normal_img), img_hash.compute(canny__pepper_img)))
# print(img_hash.compare(img_hash.compute(canny_normal_img), img_hash.compute(canny_guass_pepper_img)))
# print(img_hash.compare(img_hash.compute(canny_normal_img), img_hash.compute(canny_median_pepper_img)))
# print(img_hash.compare(img_hash.compute(canny_normal_img), img_hash.compute(canny_bilateral_pepper_img)))

images = [[gray_normal_img, None, canny_normal_img],
		  [gray_pepper_img, None, canny__pepper_img],
		  [gray_pepper_img, blur_guass_pepper_img, canny_guass_pepper_img],
		  [gray_pepper_img, blur_median_pepper_img, canny_median_pepper_img],
		  [gray_pepper_img, blur_bilateral_pepper_img, canny_bilateral_pepper_img]]

labels = {0: "normal", 1:"noise", 2:"GaussianBlur", 3: "MedianBlur", 4: "bilateralBlur"}

plt.figure()
for i in range(len(images)):
	for j in range(len(images[i])):
		if (i==0 or i==1) and j==1:
			continue
		plt.subplot(len(images),len(images[2]),i*len(images[2])+j+1)
		result = 0
		if j == 0:
			plt.ylabel(labels[i]+"                         ", rotation=0)
		ax = plt.gca()
		ax.axes.xaxis.set_ticks([])
		ax.axes.yaxis.set_ticks([])
		#x,y軸各放大一倍
		img = cv2.resize(images[i][j], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
		plt.imshow(img,'gray')
plt.show()

