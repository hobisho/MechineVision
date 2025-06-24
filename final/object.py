import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image = cv2.imread('a.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 模糊 + 邊緣偵測
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred, 50, 200)
edges = cv2.dilate(edges, None, iterations=15)
edges = cv2.erode(edges, None, iterations=15)

# 創建一個與 gray 相同大小的遮罩，初始全為 0
mask = np.zeros_like(gray)

# 將 edges 中值為 255 的地方設為 255
mask[edges == 255] = 255

# 對 gray 進行中值濾波
median_filtered = cv2.medianBlur(gray, 5)  # 使用 5x5 核心，可以根據需要調整大小
# median_filtered = cv2.medianBlur(median_filtered, 5)
# median_filtered = cv2.medianBlur(median_filtered, 5)

# 僅在 edges=255 的地方使用中值濾波結果
gray_filtered = gray.copy()
gray_filtered[mask == 255] = median_filtered[mask == 255]

edgesa = cv2.Canny(gray_filtered, 50, 200)
edgesa = cv2.dilate(edgesa, None, iterations=3)

gray_filtereda = gray.copy()
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if edgesa[i, j] == 255:
            gray_filtereda[i, j] = 0
            


# 找出輪廓（只找外層）
# contours, _ = cv2.findContours(gray_filtereda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 複製一張原圖用來描邊
# contour_img = image.copy()

# 畫出輪廓（真實形狀）
# cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # 綠色線條

_, binary = cv2.threshold(edgesa, 128, 255, cv2.THRESH_BINARY_INV)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

mask = np.zeros(binary.shape, dtype=np.uint8)
for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 5000:
        mask[labels == i] = 255
        
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_mask = np.zeros_like(mask)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

# 顯示過濾後的灰度圖（可選）
plt.imshow(cv2.cvtColor(filled_mask, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Median Filtered Gray (at edges)')
plt.axis('off')
plt.show()