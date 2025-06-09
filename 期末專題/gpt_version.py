import time
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from def_orb import analyze_displacement

# ---------- 參數設定 ----------
BASELINE = 0.1
FOCAL_LENGTH = 525.0
DEPTH_SCALE = 1.0
DOWNSAMPLE = 4

# ---------- 載入深度和顏色 ----------
def load_depth_and_color(depth_path, color_path):
    depth_raw = imageio.imread(depth_path).astype(np.float32)
    # 這裡假設 depth_raw 是 disparity，如果真的是 depth (米)，就不用 255−raw
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE
    color_image = imageio.imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image

# ---------- 取得 shift 參數（示意） ----------
def compute_shift_params(left_color, right_color):
    # 假設 analyze_displacement 已有實作，回傳你需要的值
    maxg, maxgx, ming, avgdx = analyze_displacement(left_color, right_color)
    shift = int(ming / 2)
    midx = int((maxgx - ming) / 2.15)
    return shift, midx

# ---------- 在 warp 前先左右平移 ----------
def shift_image_horizontal(img, shift_amount):
    h, w = img.shape[:2]
    shift_abs = abs(2*shift_amount)
    new_w = w + shift_abs  # 擴展寬度

    if (new_w%2) == 1:
        new_w += 1
    if (shift_abs%2) == 1:
        shift_abs += 1

    # 建立對應大小的空白圖
    if img.ndim == 2:  # 灰階圖（例如深度圖）
        shifted = np.ones((h, new_w), dtype=img.dtype) * 2  # 預設深度值為 2
    else:  # 彩色圖
        shifted = np.zeros((h, new_w, img.shape[2]), dtype=img.dtype)

    if shift_amount < 0:
        # 向右平移：放在右偏的位置
        shifted[:, 0:w] = img
    elif shift_amount > 0:
        # 向左平移：放在最左邊，裁掉 img 的左邊部分
        shifted[:, shift_abs:w+shift_abs] = img
    else:
        shifted[:, :w] = img

    return shifted

# ---------- 畫素向前投影 + Z‐buffer ----------
def forward_warp_zbuffer(depth_map: np.ndarray, color_img: np.ndarray,
                         midcenter_x: float, midcenter_z: float,
                         threshold=0.3, constant=1):
    """
    把 depth_map + color_img forward‐warp 到同一參考平面，
    並在寫入 warped_depth 時做 Z‐buffer，warped_depth=nan 表示空洞。
    """

    h, w = depth_map.shape
    warped_depth = np.full((h, w), np.nan, dtype=np.float32)
    warped_color = np.zeros((h, w, 3), dtype=np.uint8)

    ys, xs = np.indices((h, w))
    flat_y = ys.ravel()
    flat_x = xs.ravel()
    flat_d = depth_map.ravel()
    flat_c = color_img.reshape(-1, 3)

    # 2) 有效深度篩選
    valid = (flat_d > 1e-2) & (flat_d < 1.5)
    flat_y = flat_y[valid]
    flat_x = flat_x[valid]
    flat_d = flat_d[valid]
    flat_c = flat_c[valid]

    # 3) 計算水平偏移量 (可改為 float，再做小數插值)
    is_shift = flat_d < threshold
    shifts = np.zeros_like(flat_x)
    if np.any(is_shift):
        shiftx = midcenter_x * (1.0 - flat_d[is_shift]) / (1.0 - midcenter_z)
        shifts[is_shift] = np.floor(constant * shiftx).astype(int)
    new_x = flat_x + shifts
    new_y = flat_y.copy()

    # 4) 邊界過濾
    inside = (new_x >= 0) & (new_x < w)
    flat_y = flat_y[inside]
    new_y = new_y[inside]
    new_x = new_x[inside]
    flat_d = flat_d[inside]
    flat_c = flat_c[inside]

    # 5) Z‐buffer：對每個 (new_y, new_x) 只保留最小 depth
    for y0, x0, d0, c0 in zip(new_y, new_x, flat_d, flat_c):
        curd = warped_depth[y0, x0]
        if np.isnan(curd) or (d0 < curd):
            warped_depth[y0, x0] = d0
            warped_color[y0, x0] = c0

    return warped_depth, warped_color

# ---------- 合併左右事後比較深度 ----------
def fuse_left_right(left_d, left_c, right_d, right_c):
    h, w = left_d.shape
    final_d = np.full((h, w), np.nan, dtype=np.float32)
    final_c = np.zeros((h, w, 3), dtype=np.uint8)

    # 這邊直接用向量化作法
    # 用 boolean mask 區分左只有/右只有/都存在
    maskL = ~np.isnan(left_d)
    maskR = ~np.isnan(right_d)

    # 只有左有、只有右有、左右都有
    both = maskL & maskR
    onlyL = maskL & (~maskR)
    onlyR = maskR & (~maskL)

    # 只有左：直接複製
    final_d[onlyL] = left_d[onlyL]
    final_c[onlyL] = left_c[onlyL]

    # 只有右：直接複製
    final_d[onlyR] = right_d[onlyR]
    final_c[onlyR] = right_c[onlyR]

    # 左右都有：取 depth 較小的那一方（最近）
    idx_both = np.argwhere(both)
    for (y, x) in idx_both:
        if left_d[y, x] < right_d[y, x]:
            final_d[y, x] = left_d[y, x]
            final_c[y, x] = left_c[y, x]
        else:
            final_d[y, x] = right_d[y, x]
            final_c[y, x] = right_c[y, x]

    return final_d, final_c

# ---------- 用 OpenCV inpaint 填洞 ----------
def inpaint_depth_color(depth, color):
    # depth 裡 nan→0，mask=1
    mask = np.isnan(depth).astype(np.uint8)
    tmp_depth = depth.copy()
    tmp_depth[mask == 1] = 0
    # Telea 演算法，半徑選 3
    depth_f = cv2.inpaint(tmp_depth, mask, 5, cv2.INPAINT_TELEA)
    depth[mask == 1] = depth_f[mask == 1]

    # face color 孔洞（純 0 RGB）
    colormask = (np.all(color == 0, axis=2)).astype(np.uint8)
    color_f = cv2.inpaint(color, colormask, 3, cv2.INPAINT_TELEA)
    color[colormask == 1] = color_f[colormask == 1]

    return depth, color

# ---------- 顯示結果 ----------
def show_result(left_c, right_c, merged_c):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.title("Left Warped"); plt.imshow(left_c); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Right Warped"); plt.imshow(right_c); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Fused"); plt.imshow(merged_c); plt.axis('off')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # 1) 載入左右深度、顏色
    left_depth, left_color = load_depth_and_color("output/disparity_left.jpg", "image/bbox_left_left.jpg")
    right_depth, right_color = load_depth_and_color("output/disparity_right.jpg", "image/bbox_right_right.jpg")

    # 2) 計算 shift 參數
    shift_px, midx = compute_shift_params(left_color, right_color)

    # 3) 把左右影像水平位移
    left_color_shifted = shift_image_horizontal(left_color, -shift_px)
    right_color_shifted = shift_image_horizontal(right_color, shift_px)
    left_depth_shifted = shift_image_horizontal(left_depth, -shift_px)
    right_depth_shifted = shift_image_horizontal(right_depth, shift_px)

    # 4) forward warp + z‐buffer
    left_wd, left_wc = forward_warp_zbuffer(left_depth_shifted, left_color_shifted, midx, 0, threshold=0.5, constant=-1)
    right_wd, right_wc = forward_warp_zbuffer(right_depth_shifted, right_color_shifted, midx, 0, threshold=0.5, constant=1)

    # 5) 合併左右
    merged_d, merged_c = fuse_left_right(left_wd, left_wc, right_wd, right_wc)

    # 6) 填補孔洞
    merged_d, merged_c = inpaint_depth_color(merged_d, merged_c)

    # 7) 顯示
    show_result(left_wc, right_wc, merged_c)
