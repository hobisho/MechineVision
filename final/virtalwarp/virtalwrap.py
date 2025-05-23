import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import os

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
SHIFT = 1                # 從左眼往右虛擬視點的 shift (+1 表示從左圖生成虛擬視角)
DEPTH_SCALE = 1.0        # 將 0~255 映射到幾公尺範圍（例如 1 公尺）
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 前向投影 ----------
def forward_warp(depth_map, baseline, focal_length, shift):
    h, w = depth_map.shape
    warped_depth = np.zeros((h, w), dtype=np.float32)
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
                        valid_map[y, x_virtual] = 1

    # 只補空洞，不改動原本有效值
    filled = median_filter(warped_depth, size=3)
    warped_depth[warped_depth == 0] = filled[warped_depth == 0]

    return warped_depth, valid_map

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
    
    if color_image is not None:
        rgb = color_image[yy, xx] / 255.0
        ax.scatter(x, -y, z, c=rgb.reshape(-1, 3), s=0.5)
    else:
        ax.scatter(x, -y, z, c=z.reshape(-1), cmap='plasma', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title("3D Point Cloud")
    plt.tight_layout()
    plt.show()

# ---------- 可視化 ----------
def visualize_all(depth_map, warped_depth, valid_map, color_image=None):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original Depth")
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar()
    plt.axis('off')

    if color_image is not None:
        plt.subplot(1, 4, 2)
        plt.title("Color Image")
        plt.imshow(color_image)
        plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Warped Depth")
    plt.imshow(warped_depth, cmap='plasma')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Valid Map")
    plt.imshow(valid_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------- 主程式 ----------
if __name__ == "__main__":
    # 修改這兩個路徑為你自己的圖
    depth_path = "disparity_left.png"      # 0~255 深度圖 (灰階)
    color_path = "left.jpg"      # 彩色對照圖 (RGB)

    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"找不到深度圖：{depth_path}")
    
    depth_raw = imread(depth_path).astype(np.float32)
    depth_map = (255.0 - depth_raw) / 255.0 

    try:
        color_image = imread(color_path)
        if color_image.ndim == 2 or color_image.shape[2] == 1:
            color_image = np.stack([color_image]*3, axis=-1)
        elif color_image.shape[2] == 4:
            color_image = color_image[:, :, :3]
    except:
        color_image = None

    # Forward Warp
    warped_depth, valid_map = forward_warp(depth_map, BASELINE, FOCAL_LENGTH, SHIFT)

    # 顯示圖像 + 深度圖 + Warp 後結果
    visualize_all(depth_raw, warped_depth, valid_map, color_image)

    # 顯示立體圖（點雲）
    show_point_cloud(depth_map, color_image, downsample=DOWNSAMPLE)
