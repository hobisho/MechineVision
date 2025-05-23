import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from imageio import imread
import os

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
SHIFT_LEFT = +1          # 左圖往中間視角
SHIFT_RIGHT = -1         # 右圖往中間視角
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- Forward Warp ----------
def forward_warp_by_depth(depth_map, image, baseline, focal_length, shift):
    h, w = depth_map.shape
    warped_image = np.zeros_like(image)
    valid_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 1e-3:
                disparity = (baseline * focal_length) / d
                x_virtual = int(round(x + shift * disparity))
                if 0 <= x_virtual < w:
                    warped_image[y, x_virtual] = image[y, x]
                    valid_mask[y, x_virtual] = 1

    # 補洞
    if image.ndim == 3:
        for c in range(3):
            filled = median_filter(warped_image[..., c], size=3)
            mask = (valid_mask == 0)
            warped_image[..., c][mask] = filled[mask]
    else:
        filled = median_filter(warped_image, size=3)
        mask = (valid_mask == 0)
        warped_image[mask] = filled[mask]

    return warped_image, valid_mask

# ---------- 合併左右視角 ----------
def merge_warps(img_l, mask_l, img_r, mask_r):
    h, w = mask_l.shape
    merged = np.zeros_like(img_l)

    both = (mask_l == 1) & (mask_r == 1)
    only_l = (mask_l == 1) & (mask_r == 0)
    only_r = (mask_l == 0) & (mask_r == 1)

    merged[both] = 0.5 * img_l[both] + 0.5 * img_r[both]
    merged[only_l] = img_l[only_l]
    merged[only_r] = img_r[only_r]

    return merged.astype(np.uint8)

# ---------- 顯示 ----------
def visualize_results(img_lw, img_rw, img_merge):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Left Warped")
    plt.imshow(img_lw)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Right Warped")
    plt.imshow(img_rw)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Merged Virtual View")
    plt.imshow(img_merge)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------- 主程式 ----------
if __name__ == "__main__":
    # 改成你的檔案路徑
    depth_l_path = "disparity_left.png"
    depth_r_path = "disparity_right.png"
    img_l_path = "left.jpg"
    img_r_path = "right.jpg"

    if not (os.path.exists(depth_l_path) and os.path.exists(depth_r_path)):
        raise FileNotFoundError("深度圖不存在")

    depth_l_raw = imread(depth_l_path).astype(np.float32)
    depth_r_raw = imread(depth_r_path).astype(np.float32)
    depth_l = (255.0 - depth_l_raw) / 255.0
    depth_r = (255.0 - depth_r_raw) / 255.0

    img_l = imread(img_l_path)
    img_r = imread(img_r_path)
    if img_l.shape[2] == 4: img_l = img_l[..., :3]
    if img_r.shape[2] == 4: img_r = img_r[..., :3]

    warp_l, mask_l = forward_warp_by_depth(depth_l, img_l, BASELINE, FOCAL_LENGTH, SHIFT_LEFT)
    warp_r, mask_r = forward_warp_by_depth(depth_r, img_r, BASELINE, FOCAL_LENGTH, SHIFT_RIGHT)

    merged = merge_warps(warp_l, mask_l, warp_r, mask_r)
    visualize_results(warp_l, warp_r, merged)
