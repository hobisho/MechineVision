import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import os

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 將 0~255 映射到幾公尺範圍（例如 1 公尺）
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 前向投影 (depth + color) ----------
def forward_warp_with_color(depth_map, color_img, baseline, focal_length, shift):
    h, w = depth_map.shape
    warped_depth = np.zeros((h, w), dtype=np.float32)
    warped_color = np.zeros((h, w, 3), dtype=np.uint8)
    valid_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 1e-3:
                disparity = (baseline * focal_length) / d
                x_virtual = int(round(x + shift * disparity))
                if 0 <= x_virtual < w:
                    if warped_depth[y, x_virtual] == 0 or warped_depth[y, x_virtual] > d:
                        warped_depth[y, x_virtual] = d
                        warped_color[y, x_virtual] = color_img[y, x]
                        valid_map[y, x_virtual] = 1


    # 中值濾波補空洞（僅深度）
    filled = median_filter(warped_depth, size=3)
    warped_depth[warped_depth == 0] = filled[warped_depth == 0]

    return warped_depth, warped_color, valid_map

# ---------- 點雲繪製 ----------
def show_point_cloud(depth_map, color_image=None, downsample=4):
    h, w = depth_map.shape
    yy, xx = np.meshgrid(np.arange(0, h, downsample), np.arange(0, w, downsample), indexing='ij')
    z = depth_map[yy, xx]
    x = (xx - w / 2) * z / FOCAL_LENGTH
    y = (yy - h / 2) * z / FOCAL_LENGTH

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    valid = z > 1e-3
    if color_image is not None:
        rgb = color_image[yy, xx][valid] / 255.0
        ax.scatter(x[valid], -y[valid], z[valid], c=rgb, s=0.5)
    else:
        ax.scatter(x[valid], -y[valid], z[valid], c=z[valid], cmap='plasma', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title("3D Point Cloud")
    plt.tight_layout()
    plt.show()

# ---------- 載入圖像 ----------
def load_depth_and_color(depth_path, color_path):
    depth_raw = imread(depth_path).astype(np.float32)
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE

    color_image = imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image, depth_raw

# ---------- 合併兩視角 ----------
def combine_warped(left_depth, left_color, left_valid, right_depth, right_color, right_valid):
    h, w = left_depth.shape
    merged_depth = np.zeros((h, w), dtype=np.float32)
    merged_color = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            lv = left_valid[y, x]
            rv = right_valid[y, x]
            if lv and rv:
                merged_depth[y, x] = (left_depth[y, x] + right_depth[y, x]) / 2.0
                merged_color[y, x] = ((left_color[y, x].astype(np.int32) + right_color[y, x].astype(np.int32)) // 2).astype(np.uint8)
            elif lv:
                merged_depth[y, x] = left_depth[y, x]
                merged_color[y, x] = left_color[y, x]
            elif rv:
                merged_depth[y, x] = right_depth[y, x]
                merged_color[y, x] = right_color[y, x]
            # else: leave as 0

    return merged_depth, merged_color

# ---------- 主程式執行 ----------
left_depth, left_color, left_raw = load_depth_and_color("disparity_left.png", "left.jpg")
right_depth, right_color, right_raw = load_depth_and_color("disparity_right.png", "right.jpg")

# Forward warp to virtual middle view
left_warped_depth, left_warped_color, left_valid = forward_warp_with_color(left_depth, left_color, BASELINE, FOCAL_LENGTH, shift=0)
right_warped_depth, right_warped_color, right_valid = forward_warp_with_color(right_depth, right_color, BASELINE, FOCAL_LENGTH, shift=0)

# Combine
merged_depth, merged_color = combine_warped(left_warped_depth, left_warped_color, left_valid,
                                            right_warped_depth, right_warped_color, right_valid)

# 顯示結果圖
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Warped Left View")
plt.imshow(left_warped_color)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title("Warped Right View")
plt.imshow(right_warped_color)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title("Merged Virtual View")
plt.imshow(merged_color)
plt.axis('off')
plt.tight_layout()
plt.show()

# 顯示點雲
show_point_cloud(merged_depth, merged_color, downsample=DOWNSAMPLE)
