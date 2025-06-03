import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D
from def_orb import analyze_displacement
import cv2
import time

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 載入深度和顏色圖像 ----------
def load_depth_and_color(depth_path, color_path):
    depth_raw = imageio.imread(depth_path).astype(np.float32)
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE
    color_image = imageio.imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image, depth_raw

# ---------- 左右平移 ----------
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

# ---------- 前向投影為 stack ----------
def forward_warp_to_stack(depth_map, color_img, baseline, focal_length):
    h, w = depth_map.shape
    depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if  1.5 > d > 1e-2:
                if 0 <= x < w:
                    depth_stack[y][x].append(d)
                    color_stack[y][x].append(color_img[y, x])

    filled_depth = median_filter(depth_map, size=3)

    for y in range(h):
        for x in range(w):
            if not depth_stack[y][x]:
                if filled_depth[y, x]!= 2.0:
                    depth_stack[y][x].append(filled_depth[y, x])
                    color_stack[y][x].append(color_img[y, x])

    return depth_stack, color_stack

# ---------- 深度平移深度圖 ----------
def shift_points_in_stack(depth_stack, color_stack, midcenter_x, midcenter_z, threshold=0.3, constant=1,time=1):
    h, w = len(depth_stack), len(depth_stack[0])
    new_depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    new_color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(0 if constant > 0 else h - 1, h if constant > 0 else -1, int(constant)):
        for x in range(0 if constant > 0 else w - 1, w if constant > 0 else -1, int(constant)):
            for d, c in zip(depth_stack[y][x], color_stack[y][x]):
                if d < threshold:
                    # 計算平移後的 x 座標
                    shift_x = midcenter_x * (1 - d) / (1 - midcenter_z)
                    new_x = x + int(constant * time * shift_x)

                    # 如果超出邊界則跳過，不進行平移
                    if 0 <= new_x < w:
                        new_depth_stack[y][new_x].append(d)
                        new_color_stack[y][new_x].append(c)
                    else:
                        continue  # 超出邊界直接跳過
                else:
                    # 深度夠大則不平移，保留原位
                    new_depth_stack[y][x].append(d)
                    new_color_stack[y][x].append(c)

    return new_depth_stack, new_color_stack

# ---------- 合併左右堆疊 ----------
def merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color):
    h, w = len(left_shifted_depth), len(left_shifted_depth[0])
    merged_depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    merged_color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            # 取出兩張圖在這個 pixel 的資料
            left_depths = left_shifted_depth[y][x]
            right_depths = right_shifted_depth[y][x]
            left_colors = left_shifted_color[y][x]
            right_colors = right_shifted_color[y][x]

            # 合併兩個 list
            if left_depths:
                merged_depth_stack[y][x].extend(left_depths)
                merged_color_stack[y][x].extend(left_colors)
            if right_depths:
                merged_depth_stack[y][x].extend(right_depths)
                merged_color_stack[y][x].extend(right_colors)

    return merged_depth_stack, merged_color_stack

# ---------- 深度圖補洞 ----------
def inpaint_stack_median(depth_stack, color_stack, kernel_size=3, max_iter=10):
    for i in range(max_iter):
        filled = 0
        h, w = len(depth_stack), len(depth_stack[0])
        pad = kernel_size // 2
        new_depth_stack = [[list(p) for p in row] for row in depth_stack]
        new_color_stack = [[list(p) for p in row] for row in color_stack]

        for y in range(pad, h - pad):
            for x in range(pad, w - pad):
                # 將兩者都沒資料的才補洞
                if not depth_stack[y][x] or not color_stack[y][x]:
                    neighbor_depths = []
                    neighbor_colors = []

                    for dy in range(-pad, pad + 1):
                        for dx in range(-pad, pad + 1):
                            ny, nx = y + dy, x + dx
                            if depth_stack[ny][nx] and color_stack[ny][nx]:
                                neighbor_depths.extend(depth_stack[ny][nx])
                                neighbor_colors.extend(color_stack[ny][nx])

                    if neighbor_depths and neighbor_colors:
                        # 同時補洞
                        median_depth = float(np.median(neighbor_depths))
                        median_color = np.median(np.array(neighbor_colors), axis=0).astype(np.uint8).tolist()
                        new_depth_stack[y][x] = [median_depth]
                        new_color_stack[y][x] = [median_color]
                        filled += 1
                    else:
                        # 如果只有一個是空，另外一個也填預設值保同步
                        if not depth_stack[y][x]:
                            new_depth_stack[y][x] = [2.0] # 預設深度
                        if not color_stack[y][x]:
                            new_color_stack[y][x] = [[0,0,0]] # 預設顏色

        depth_stack = new_depth_stack
        color_stack = new_color_stack
        print(f"[第 {i+1} 輪] 補了 {filled} 個點")
        if filled == 0:
            break

    return depth_stack, color_stack

# ---------- 顯示圖像視覺結果 ----------
def show_virtual_views(left_stack_color, merged_stack_color, right_stack_color, merged_stack_depth):
    def frontmost_color(stack_color, stack_depth):
        h, w = len(stack_color), len(stack_color[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                colors = stack_color[y][x]
                depths = stack_depth[y][x] if stack_depth and stack_depth[y][x] else [1] * len(colors)
                if len(colors) > 0:
                    min_idx = np.argmin(depths)
                    if min_idx >= len(colors):
                        min_idx = 0
                    img[y, x] = np.array(colors[min_idx], dtype=np.uint8)
        return img


    # left_img = frontmost_color(left_stack_color, left_stack_depth)
    merged_img = frontmost_color(merged_stack_color, merged_stack_depth)
    # right_img = frontmost_color(merged_stack_color, merged_stack_depth)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Warped Left View")
    plt.imshow(left_stack_color)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Merged Virtual View")
    plt.imshow(merged_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Warped Right View")
    plt.imshow(right_stack_color)
    plt.axis('off')


    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # 載入左右視角
    path = "./final/image/bbox"
    left_depth, left_color, _ = load_depth_and_color("./final/output/disparity_left.jpg", f"{path}_left_left.jpg")
    right_depth, right_color, _ = load_depth_and_color("./final/output/disparity_right.jpg", f"{path}_right_right.jpg")
    print(left_depth.shape)

    # 計算深度圖shift
    (max_group_mean, max_group_max, min_group_mean,avg_dx)= analyze_displacement(left_color, right_color)  # 假設 ImgShift 函數已經定義並返回位移量
    print(min_group_mean)
    shift = int(min_group_mean/2)
    midcenter_x = int(max_group_max-min_group_mean)/2.2 #int(abs(center_left_x - center_right_x) / 2)
    midcenter_z = 0
    print(midcenter_x,midcenter_z)

    # 平移圖像
    left_color_shifted = shift_image_horizontal(left_color, -shift)
    right_color_shifted = shift_image_horizontal(right_color, shift)
    left_depth_shifted = shift_image_horizontal(left_depth, -shift)
    right_depth_shifted = shift_image_horizontal(right_depth, shift)
    print(left_color_shifted.shape, right_color_shifted.shape)
    
    # warp 成深度圖
    left_stack_depth, left_stack_color = forward_warp_to_stack(left_depth_shifted, left_color_shifted, BASELINE, FOCAL_LENGTH)
    right_stack_depth, right_stack_color = forward_warp_to_stack(right_depth_shifted, right_color_shifted, BASELINE, FOCAL_LENGTH)

    # 平移深度圖
    a=time.time()
    left_shifted_depth, left_shifted_color = shift_points_in_stack(left_stack_depth, left_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=-1)
    right_shifted_depth, right_shifted_color = shift_points_in_stack(right_stack_depth, right_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=1)

    # 合併深度圖
    merged_depth_stack, merged_color_stack = merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color)

    merged_depth_stack, merged_color_stack = inpaint_stack_median(merged_depth_stack, merged_color_stack)
    a=time.time()
    merged_depth_shift, merged_color_shift = shift_points_in_stack(merged_depth_stack, merged_color_stack, midcenter_x, midcenter_z,threshold=0.5,constant=-1,time=1)
    
    # 補洞
    merged_depth_shift, merged_color_shift = inpaint_stack_median(merged_depth_shift, merged_color_stack)

    # merged_depth_shift, merged_color_shift = inpaint_stack_median(merged_depth_shift, merged_color_shift)
    # 顯示虛擬視角圖像
    print("⚠️ 顯示虛擬視角圖像")

    print(time.time()-a)
    show_virtual_views(left_color_shifted, merged_color_shift, right_color_shifted ,merged_depth_shift)
    
    # print("⚠️ 顯示點雲")
    # show_point_cloud_from_stack(merged_depth_stack, merged_color_stack, downsample=DOWNSAMPLE)
