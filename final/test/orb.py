import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

@njit
def compute_disparity_map(imgL, imgR, max_disparity=64, block_size=5):
    (h, w,_) = imgL.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)
    half_block = block_size // 2

    for y in range(half_block, h - half_block):
        for x in range(half_block + max_disparity, w - half_block):
            best_offset = 0
            min_sad = float('inf')
            block_left = imgL[y - half_block:y + half_block + 1, x - half_block:x + half_block + 1]

            for d in range(max_disparity):
                x_right = x - d
                if x_right - half_block < 0:
                    continue

                block_right = imgR[y - half_block:y + half_block + 1, x_right - half_block:x_right + half_block + 1]
                sad = np.sum(np.abs(block_left - block_right))

                if sad < min_sad:
                    min_sad = sad
                    best_offset = d

            disparity_map[y, x] = best_offset

    return disparity_map

if __name__ == "__main__":
    # 載入灰階圖像（PNG 或 JPG）
    path = "./final/image/tv"
    imgL = cv2.imread(f"{path}_left.jpg")
    imgR = cv2.imread(f"{path}_right.jpg")

    # 確保圖像為 0-255
    imgL = (imgL * 255).astype(np.uint8) if imgL.max() <= 1.0 else imgL.astype(np.uint8)
    imgR = (imgR * 255).astype(np.uint8) if imgR.max() <= 1.0 else imgR.astype(np.uint8)

    # 計算視差圖
    print(imgL.shape, imgR.shape)
    disparity = compute_disparity_map(imgL, imgR, max_disparity=64, block_size=5)

    # 視覺化結果
    plt.imshow(disparity, cmap='plasma')
    plt.colorbar()
    plt.title("Handmade Disparity Map")
    plt.show()
