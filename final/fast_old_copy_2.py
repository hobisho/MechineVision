import cv2
import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from def_orb import analyze_displacement

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 載入深度和顏色圖像 ----------
@time_wrapper
def load_depth_and_color(depth_path, color_path):
    depth_raw = imageio.imread(depth_path).astype(np.float32)
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE
    color_image = imageio.imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image, depth_raw

# ---------- 在 warp 前先左右平移 ----------
@time_wrapper
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
@time_wrapper
def forward_warp_to_stack(depth_map, color_img):
    print('depth_map shape:', depth_map.shape)
    print('color_img shape:', color_img.shape)
    
    h, w = depth_map.shape
    depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if 2.0> d > 1e-2:
                if 0 <= x <= w:
                    depth_stack[y][x].append(d)
                    color_stack[y][x].append(color_img[y, x])

    filled_depth = median_filter(depth_map, size=3)

    for y in range(h):
        for x in range(w):
            if not depth_stack[y][x]:
                # if filled_depth[y, x] <= 1.9:
                depth_stack[y][x].append(filled_depth[y, x])
                color_stack[y][x].append(color_img[y, x])

    return depth_stack, color_stack

@time_wrapper
def fast_forward_warp_to_stack(depth_map, color_img):
    h = min(depth_map.shape[0], color_img.shape[0])
    w = min(depth_map.shape[1], color_img.shape[1])
    depth_map = depth_map[:h, :w]
    color_img = color_img[:h, :w, ...]  # ... 保留3通道

    depth_img = np.zeros((h, w), dtype=np.float32)
    color_img_out = np.zeros((h, w, color_img.shape[2]), dtype=color_img.dtype)

    mask = (depth_map > 1e-2) & (depth_map < 2.0)
    depth_img[mask] = depth_map[mask]
    color_img_out[mask] = color_img[mask]

    # 用 median_filter 補洞
    filled_depth = median_filter(depth_img, size=3)
    fill_mask = (depth_img == 0)
    depth_img[fill_mask] = filled_depth[fill_mask]
    color_img_out[fill_mask] = color_img[fill_mask]

    return depth_img, color_img_out


# ---------- 深度平移深度圖 ----------
@time_wrapper
def shift_points_in_stack(depth_stack, color_stack, midcenter_x, midcenter_z, threshold=0.3, constant=1,time = 1):
    h, w = len(depth_stack), len(depth_stack[0])
    new_depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    new_color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(0 if constant > 0 else h - 1, h if constant > 0 else -1, constant):
        for x in range(0 if constant > 0 else w - 1, w if constant > 0 else -1, constant):
            for d, c in zip(depth_stack[y][x], color_stack[y][x]):
                if d < threshold:
                    shift_x = midcenter_x *(1-d)/(1-midcenter_z)
                    new_x = x + int(constant * time *  shift_x)
                    if 0 <= new_x < w:
                        new_depth_stack[y][new_x].append(d)
                        new_color_stack[y][new_x].append(c)
                    else:
                        continue  # 超出邊界直接跳過
                else:
                    new_depth_stack[y][x].append(d)
                    new_color_stack[y][x].append(c)
    
    return new_depth_stack, new_color_stack

@time_wrapper
def fast_shift_points_in_stack(depth_map, color_img, midcenter_x, midcenter_z, threshold=0.3, constant=1, time=1):
    h, w = depth_map.shape
    new_depth = np.full((h, w), 2.0, dtype=depth_map.dtype)  # 2.0為填充值
    new_color = np.zeros((h, w, 3), dtype=color_img.dtype)
    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d < threshold:
                shift_x = midcenter_x * (1 - d) / (1 - midcenter_z)
                new_x = x + int(constant * time * shift_x)
                if 0 <= new_x < w:
                    new_depth[y, new_x] = d
                    new_color[y, new_x] = color_img[y, x]
            else:
                new_depth[y, x] = d
                new_color[y, x] = color_img[y, x]
    return new_depth, new_color

# ---------- 合併左右堆疊 ----------
@time_wrapper
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

@time_wrapper
def fast_merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color):
    # left_shifted_depth/left_shifted_color/... shape: (h, w, N)
    # 輸出 shape: (h, w, N*2)

    left_depth_exp = np.expand_dims(left_shifted_depth, axis=2)    # (h, w, 1)
    right_depth_exp = np.expand_dims(right_shifted_depth, axis=2)  # (h, w, 1)
    merged_depth_stack = np.concatenate((left_depth_exp, right_depth_exp), axis=2)  # (h, w, 2)

    left_color_exp = np.expand_dims(left_shifted_color, axis=2)    # (h, w, 1, 3)
    right_color_exp = np.expand_dims(right_shifted_color, axis=2)  # (h, w, 1, 3)
    merged_color_stack = np.concatenate((left_color_exp, right_color_exp), axis=2)  # (h, w, 2, 3)
    
    return merged_depth_stack, merged_color_stack


# ---------- 深度圖補洞 ----------
@time_wrapper
def inpaint_stack_median(depth_stack, color_stack, kernel_size=3, max_iter=10):
    for i in range(max_iter):
        filled = 0
        h, w = len(depth_stack), len(depth_stack[0])
        pad = kernel_size // 2
        new_depth_stack = [[list(p) for p in row] for row in depth_stack]
        new_color_stack = [[list(p) for p in row] for row in color_stack]

        for y in range(pad, h - pad):
            for x in range(pad, w - pad):
                if not depth_stack[y][x]:
                    neighbor_depths = []
                    neighbor_colors = []

                    for dy in range(-pad, pad + 1):
                        for dx in range(-pad, pad + 1):
                            ny, nx = y + dy, x + dx
                            if depth_stack[ny][nx]:
                                neighbor_depths.extend(depth_stack[ny][nx])
                                neighbor_colors.extend(color_stack[ny][nx])

                    if neighbor_depths:
                        median_depth = float(np.median(neighbor_depths))
                        median_color = np.median(np.array(neighbor_colors), axis=0).astype(np.uint8).tolist()
                        new_depth_stack[y][x].append(median_depth)
                        new_color_stack[y][x].append(median_color)
                        filled += 1

        depth_stack = new_depth_stack
        color_stack = new_color_stack
        print(f"[第 {i+1} 輪] 補了 {filled} 個點")
        if filled == 0:
            break

    return depth_stack, color_stack

@time_wrapper
def fast_inpaint_stack_median_ndarray(depth_stack, color_stack, kernel_size=3, max_iter=10):
    mask = (depth_stack != 2.0)
    filled = 0
    depth_filled = depth_stack.copy()
    color_filled = color_stack.copy()
    for i in range(max_iter):
        missing = ~np.any(mask, axis=2)  # (h, w)
        if not np.any(missing):
            print(f"第 {i+1} 輪 無需補洞")
            break

        # median filter inpaint
        median_depth = median_filter(
            np.where(mask, depth_filled, np.nan),
            size=(kernel_size, kernel_size, depth_filled.shape[2]),  # 例如 (3, 3, 2)
            mode='nearest'
        )
        # For color (h, w, N, 3)
        median_color = median_filter(
            np.where(mask[..., None], color_filled, np.nan),
            size=(kernel_size, kernel_size, 1, 1),
            mode='nearest'
        )

        for y in range(depth_filled.shape[0]):
            for x in range(depth_filled.shape[1]):
                if missing[y, x]:
                    depth_filled[y, x, 0] = np.nanmedian(median_depth[y, x])
                    color_filled[y, x, 0] = np.nanmedian(median_color[y, x], axis=0)
                    mask[y, x, 0] = True
                    filled += 1

        print(f"[第 {i+1} 輪] 補了 {filled} 個點")
        if filled == 0:
            break

    return depth_filled, color_filled



# ---------- 顯示圖像視覺結果 ----------
@time_wrapper
def show_virtual_views(left_stack_color, right_stack_color, merged_stack_color, right_stack_depth=None):
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


    left_img = frontmost_color(left_stack_color, None)
    right_img = frontmost_color(right_stack_color, right_stack_depth)
    merged_img = frontmost_color(merged_stack_color, None)
    
    def stack_to_min_depth_img(stack_depth):
        h, w = len(stack_depth), len(stack_depth[0])
        img = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if stack_depth[y][x]:  # 如果該 pixel 有資料
                    img[y, x] = 255 - min(stack_depth[y][x])*255
                else:
                    img[y, x] = 0  # 沒資料可用時設為 0 或你要的預設值
        return img


    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Warped Left View")
    plt.imshow(left_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Merged Virtual View")
    plt.imshow(right_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Warped Right View")
    plt.imshow(merged_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    a = stack_to_min_depth_img(right_stack_depth)
    img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("color.jpg", img_rgb)
    cv2.imwrite("depth.jpg", a)


if __name__ == "__main__":
    # 載入左右視角
    path = "./final/image/bbox"
    left_depth, left_color, _ = load_depth_and_color("./final/output/disparity_left.jpg", f"{path}_left_left.jpg")
    right_depth, right_color, _ = load_depth_and_color("./final/output/disparity_right.jpg", f"{path}_right_right.jpg")
    print(left_depth.shape)

    # 計算深度圖shift
    (max_group_mean, max_group_max, min_group_mean,avg_dx) = analyze_displacement(left_color, right_color)  # 假設 ImgShift 函數已經定義並返回位移量
    print(min_group_mean)
    shift = int(min_group_mean/2)
    midcenter_x = int(max_group_max-min_group_mean)/2.2#int(abs(center_left_x - center_right_x) / 2)
    midcenter_z = 0
    print(midcenter_x,midcenter_z)

    # 平移圖像
    left_color_shifted = shift_image_horizontal(left_color, -shift)
    right_color_shifted = shift_image_horizontal(right_color, shift)
    left_depth_shifted = shift_image_horizontal(left_depth, -shift)
    right_depth_shifted = shift_image_horizontal(right_depth, shift)
    print(right_color_shifted.shape, left_color_shifted.shape)
    
    # warp 成深度圖
    left_stack_depth, left_stack_color = forward_warp_to_stack(left_depth_shifted, left_color_shifted)
    right_stack_depth, right_stack_color = forward_warp_to_stack(right_depth_shifted, right_color_shifted)
    
    fast_left_stack_depth, fast_left_stack_color = fast_forward_warp_to_stack(left_depth_shifted, left_color_shifted)
    fast_right_stack_depth, fast_right_stack_color = fast_forward_warp_to_stack(right_depth_shifted, right_color_shifted)

    # 平移深度圖
    left_shifted_depth, left_shifted_color = shift_points_in_stack(left_stack_depth, left_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=-1)
    right_shifted_depth, right_shifted_color = shift_points_in_stack(right_stack_depth, right_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=1)

    fast_left_shifted_depth, fast_left_shifted_color = fast_shift_points_in_stack(fast_left_stack_depth, fast_left_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=-1)
    fast_right_shifted_depth, fast_right_shifted_color = fast_shift_points_in_stack(fast_right_stack_depth, fast_right_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=1)
    
    # 合併深度圖
    merged_depth_stack1, merged_color_stack1 = merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color)
    fast_merged_depth_stack2, fast_merged_color_stack2 = fast_merge_stacks(fast_left_shifted_depth, fast_left_shifted_color, fast_right_shifted_depth, fast_right_shifted_color)

    # 補洞
    merged_depth_stack, merged_color_stack = inpaint_stack_median(merged_depth_stack1, merged_color_stack1)
    fast_merged_depth_stack, fast_merged_color_stack = fast_inpaint_stack_median_ndarray(fast_merged_depth_stack2, fast_merged_color_stack2)
    
    merged_depth_stack, merged_color_stack = shift_points_in_stack(merged_depth_stack, merged_color_stack, midcenter_x, midcenter_z,threshold=0.5,constant=-1,time = 1)

    # merged_depth_stack, merged_color_stack = inpaint_stack_median(merged_depth_stack, merged_color_stack)
    # 顯示虛擬視角圖像
    print("⚠️ 顯示虛擬視角圖像")
    show_virtual_views(left_stack_color, merged_color_stack, right_stack_color,merged_depth_stack)
    print(merged_depth_stack[964][1494],merged_color_stack[964][1494])  # 顯示特定點的顏色和深度值
#1494 964
    # print("⚠️ 顯示點雲")
    # show_point_cloud_from_stack(merged_depth_stack, merged_color_stack, downsample=DOWNSAMPLE)