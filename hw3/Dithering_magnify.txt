import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 讀入圖像
img = cv2.imread('./HW3/b.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 將長寬調整為4的倍數
size = 8
h, w = img.shape
h_new = h - h % size
w_new = w - w % size
img = img[:h_new, :w_new]

img_magnify = cv2.resize(img, (img.shape[1]*size, img.shape[0]*size), interpolation=cv2.INTER_NEAREST)  # 最近鄰放大

# 3. 建立 Bayer 抖動矩陣
def MakeDitheringarray(magnify):
    Dithering_order = np.array([[0, 2],
                            [3, 1]], dtype=np.uint8)
    array = 64 * Dithering_order
    for k in range ((magnify-1)):
        size = 2*array.shape[0]
        print("size:", size)    
        Dithering_22 = array
        unique_vals = np.unique(Dithering_22)[1]
        Dithering_22 = np.tile(Dithering_22, (2, 2))
        Dithering_array = np.zeros((size, size), dtype=np.uint8)
        wieght = unique_vals / 4
        print("wieght:", unique_vals)
        for i in range (size):
            for j in range(size):
                Dithering_array[i, j] = Dithering_22[i, j] + wieght * Dithering_order[2*i//size][2*j//size]
        array = Dithering_array
    
    return Dithering_array


Dithering_array = MakeDitheringarray(np.log2(size).astype(int))
print("Dithering Array:\n", Dithering_array)

img_dithering = np.tile(Dithering_array, (h_new//size, w_new//size))

img_dithering_magnify = np.tile(img_dithering, (size, size))


# 4. 比較影像與抖動陣列進行二值化
dithered = (img > img_dithering).astype(np.uint8) * 255
dithered_magnify = (img_magnify > img_dithering_magnify).astype(np.uint8) * 255
monochrome = (img > 128).astype(np.uint8) * 255

# 5. 建立圖像列表（字典）
image_list = {
    'Original': img,
    'Monochrome Binarization': monochrome,
    'Dithering Result': dithered,
    'Magnify_Dithering Result' : dithered_magnify,
}

# 6. 顯示與儲存
fig, axs = plt.subplots(2, 3, figsize=(12, 7))  # 1 列 3 張圖
for ax, (title, image) in zip(axs.ravel(), image_list.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for i in range(len(image_list), 6):
    axs.ravel()[i].axis('off')
    
plt.tight_layout()
plt.savefig("./HW3/output/dithering_results.png")  # 儲存輸出結果
plt.show()

cv2.imwrite('./HW3/output/img.png', img)
cv2.imwrite('./HW3/output/monochrome_result.png', monochrome)
cv2.imwrite('./HW3/output/dithered_result.png', dithered)
cv2.imwrite('./HW3/output/dithered_magnify_result.png', dithered_magnify)