import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 讀入圖像
img = cv2.imread('./HW3/b.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 將長寬調整為4的倍數
h, w = img.shape
h_new = h - h % 4
w_new = w - w % 4
img = img[:h_new, :w_new]

# 3. 建立 Bayer 抖動矩陣
Dithering_array = np.array([[0, 128, 32, 160],
                            [192, 64, 224, 96], 
                            [48, 176, 16, 144],
                            [240, 112, 208, 80]], dtype=np.uint8)

img_dithering = np.zeros((h_new, w_new), dtype=np.uint8)
for i in range(h_new):
    for j in range(w_new):
        img_dithering[i, j] = Dithering_array[i % 4, j % 4]

# 4. 比較影像與抖動陣列進行二值化
dithered = (img > img_dithering).astype(np.uint8) * 255
homochrome = (img > 128).astype(np.uint8) * 255

# 5. 建立圖像列表（字典）
image_list = {
    'Original': img,
    'Homochrome Binarization': homochrome,
    'Dithering Result': dithered
}

# 6. 顯示與儲存
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 列 3 張圖
for ax, (title, image) in zip(axs.ravel(), image_list.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig("./HW3/output/dithering_results.png")  # 儲存輸出結果
plt.show()
