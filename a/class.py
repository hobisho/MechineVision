import cv2
import numpy as np

img = cv2.imread('photo.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, output1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# kernel = np.ones((3, 3), np.uint8)
# output1 = cv2.dilate(output1, kernel, iterations=3)  # 膨胀操作
# output1 = cv2.erode(output1, kernel, iterations=3)  # 腐蚀操作

a= np.zeros(img.shape, dtype=np.uint8)  # 创建一个与img同样大小的黑色图像

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if output1[i, j] == 255:
            a[i, j]  = img[i, j]
            


cv2.imshow('a', a)
cv2.waitKey(0)                                    # 按下任意鍵停止
cv2.destroyAllWindows()