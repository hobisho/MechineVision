import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 讀入圖像
img = cv2.imread('./HW3/b.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 將長寬調整為4的倍數
h, w = img.shape
h_div4 = h - h % 4
w_div4= w - w % 4
img = img[:h_div4, :w_div4]

img_4 = np.zeros((4*h_div4, 4*w_div4), dtype=np.uint8)
for i in range(4*h_div4):
    for j in range(4*w_div4):
        img_4[i, j] = img[i//4, j//4]

# 3. 建立 Bayer 抖動矩陣
Dithering_array = np.array([[0, 128, 32, 160],
                            [192, 64, 224, 96], 
                            [48, 176, 16, 144],
                            [240, 112, 208, 80]], dtype=np.uint8)

img_dithering = np.zeros((4*h_div4, 4*w_div4), dtype=np.uint8)
for i in range(4*h_div4):
    for j in range(4*w_div4):
        img_dithering[i, j] = Dithering_array[i % 4, j % 4]
        
print(img_dithering)

# 4. 比較影像與抖動陣列進行二值化
dithered = (img_4 > img_dithering).astype(np.uint8) * 255

# 5. 顯示與儲存結果
cv2.imwrite('dithered_city.png', dithered)
plt.imshow(dithered, cmap='gray')
plt.title("Dithering City View")
plt.axis('off')
plt.show()
