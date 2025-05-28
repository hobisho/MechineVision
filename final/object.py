import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
img = cv2.imread('final/image/bbox_left_left.jpg')  # 改成你的圖片路徑
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉成RGB方便matplotlib顯示

# 灰階轉換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 邊緣檢測 (Canny)
edges = cv2.Canny(blur, 50, 150)

# 基於顏色範圍找物件 (假設偵測藍色)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 找輪廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原圖畫出輪廓
img_contours = img.copy()
for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        cv2.drawContours(img_contours, [cnt], -1, (0,255,0), 2)

# 顯示結果
plt.figure(figsize=(12,8))
plt.subplot(231), plt.imshow(img_rgb), plt.title('原始圖'), plt.axis('off')
plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('灰階'), plt.axis('off')
plt.subplot(233), plt.imshow(blur, cmap='gray'), plt.title('模糊'), plt.axis('off')
plt.subplot(234), plt.imshow(edges, cmap='gray'), plt.title('邊緣檢測'), plt.axis('off')
plt.subplot(235), plt.imshow(mask, cmap='gray'), plt.title('顏色遮罩'), plt.axis('off')
plt.subplot(236), plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)), plt.title('輪廓標註'), plt.axis('off')
plt.tight_layout()
plt.show()
