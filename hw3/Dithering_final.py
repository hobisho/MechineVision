import cv2
import numpy as np
from matplotlib import pyplot as plt

# MSE PSNR計算函式
def calculate_mse_psnr(processed):
    assert img_resized.shape == processed.shape, "Images must have the same dimensions"

    mse = np.mean((img_resized.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return mse, psnr

# 1. 讀入圖像
img = cv2.imread('m.png', cv2.IMREAD_GRAYSCALE)

size_list = [2, 4, 8, 16]  # 不同的抖動矩陣大小
h, w = img.shape
h_new = h - h % size_list[3]
w_new = w - w % size_list[3]
img_resized = img[:h_new, :w_new]

img_magnify = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST)  # 最近鄰放大
monochrome = (img_resized > 128).astype(np.uint8) * 255
image_list = {
    'Original': img_resized,
    'Monochrome Binarization': monochrome,
}
mse, psnr = calculate_mse_psnr(monochrome)
print(f"homochrome, MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

# 2. 建立 Bayer 抖動矩陣的函式
def MakeDitheringarray(magnify):
    Dithering_order = np.array([[0, 2],
                                [3, 1]], dtype=np.uint8)
    array = 64 * Dithering_order

    if magnify <= 1:
        return array
    
    for k in range((magnify - 1)):
        size = 2 * array.shape[0]
        Dithering_22 = array
        unique_vals = np.unique(Dithering_22)[1]
        Dithering_22 = np.tile(Dithering_22, (2, 2))
        Dithering_array = np.zeros((size, size), dtype=np.uint8)
        weight = unique_vals / 4
        for i in range(size):
            for j in range(size):
                Dithering_array[i, j] = Dithering_22[i, j] + weight * Dithering_order[2 * i // size][2 * j // size]
        array = Dithering_array
    
    return Dithering_array

# 3. 對不同 size 處理
for size in size_list:
    magnify = int(np.log2(size))
    Dithering_array = MakeDitheringarray(magnify)
    
    # 建立與原圖同尺寸的 Bayer Pattern
    tiled_dither = np.tile(Dithering_array, (h_new // size, w_new // size))
    
    # 抖動處理
    dithered = (img_resized > tiled_dither).astype(np.uint8) * 255
    image_list[f'Dithering size={size}'] = dithered

    mse, psnr = calculate_mse_psnr(dithered)
    print(f"Size: {size}, MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")
    cv2.imwrite(f'./HW3/output/dithered_size_{size}.png', dithered)

# 4. 顯示結果
fig, axs = plt.subplots(2, 3, figsize=(12, 7))

for ax, (title, image) in zip(axs.ravel(), image_list.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for i in range(len(image_list), 6):
    axs.ravel()[i].axis('off')

plt.tight_layout()
plt.savefig("./HW3/output/dithering_results.png")  # 儲存輸出結果
plt.show()