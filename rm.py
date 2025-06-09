import cv2
import numpy as np

# 讀取圖片
img = cv2.imread('photo.jpg')

# --- Step 1: 轉換成 HSV 找咖啡色 ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 咖啡色範圍（可依圖微調）
lower_brown = np.array([10, 50, 20])
upper_brown = np.array([25, 255, 200])
brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

# --- Step 2: 將圖片轉灰階並抓白色區域 ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# --- Step 3: 移除被咖啡色包圍的白區 ---
# 找白色區域輪廓
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 建立最終遮罩
final_mask = np.zeros_like(white_mask)

for cnt in contours:
    # 建立臨時遮罩
    temp_mask = np.zeros_like(white_mask)
    cv2.drawContours(temp_mask, [cnt], -1, 255, -1)  # 填滿白區輪廓

    # 判斷該區域是否與咖啡色有重疊
    intersection = cv2.bitwise_and(temp_mask, brown_mask)
    if cv2.countNonZero(intersection) == 0:  # 沒有被咖啡色包圍才保留
        final_mask = cv2.bitwise_or(final_mask, temp_mask)

# --- Step 4: 把白色區域從原圖取出 ---
result = np.zeros_like(img)
result[final_mask == 255] = img[final_mask == 255]

# 顯示結果
cv2.imshow('Brown Mask', brown_mask)
cv2.imshow('Filtered White Area', final_mask)
cv2.imshow('Final Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
