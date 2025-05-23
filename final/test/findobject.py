from skimage import io, segmentation, color
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

# 讀取圖片
image = io.imread('left.jpg')

# 使用 Felzenszwalb 分割
segments = segmentation.felzenszwalb(image, scale=400, sigma=5, min_size=3000)

# 隨機上色（固定種子方便重現）
def random_label_colors(num_labels):
    np.random.seed(4)
    colors = np.random.randint(0, 255, size=(num_labels, 3))
    return colors

num_segments = segments.max() + 1
colors = random_label_colors(num_segments)

# 把區塊 label 轉成彩色圖像
segmented_img = np.zeros_like(image)
for label in range(num_segments):
    segmented_img[segments == label] = colors[label]

# 顯示結果
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(segmented_img)
axes[1].set_title("Felzenszwalb Segmentation")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 儲存 segmentation 彩色結果
io.imsave("./final/output/segmented_output.png", segmented_img)
