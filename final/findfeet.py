import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取城市航拍圖像
image = cv2.imread('a.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 模糊 + 邊緣偵測
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
edges = cv2.Canny(blurred, 100, 200)
edges = cv2.dilate(edges, None, iterations=15)
edges = cv2.erode(edges, None, iterations=15)

# 僅在邊緣像素上進行 Harris 角點檢測
gray_float = np.float32(gray)  # 轉為 float32 格式
harris_corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

# 僅保留邊緣上的角點
harris_corners = harris_corners * (edges == 255)  # 用邊緣遮罩過濾角點

# 增強角點並標記到原圖
harris_corners = cv2.dilate(harris_corners, None)
image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]  # 角點標記為紅色

# 顯示結果
plt.figure(figsize=(10, 5))

# 顯示邊緣
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

# 顯示帶角點的圖像
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners on Edges')
plt.axis('off')

plt.show()