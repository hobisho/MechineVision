import cv2
import numpy as np

# 讀取圖片
image = cv2.imread('a.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 模糊 + 邊緣偵測
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# 找出輪廓（只找外層）
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 複製一張原圖用來描邊
contour_img = image.copy()

# 畫出輪廓（真實形狀）
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # 綠色線條

# 顯示結果
cv2.imshow('Contour Outline', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
