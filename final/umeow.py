import cv2
import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter, binary_dilation

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 計時器 ----------
def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# ---------- 修剪照片大小 ----------
@njit
def shrink_nan_array(depth_arr, color_arr):
    mask = ~np.isnan(depth_arr)  # True 表示不是 NaN
    valid_counts = np.sum(mask, axis=2)  # 每個 [h, w] 有幾個非 NaN
    max_valid = np.max(valid_counts)     # 所有像素中最大的非 NaN 數量

    # 回傳裁切後的陣列
    return depth_arr[:, :, :max_valid], color_arr[:, :, :max_valid, :]

# ---------- 左右深度圖計算 ----------
@time_wrapper
def depth(imgLL, imgLR, imgRL, imgRR, target_width):
    # 左到右視差圖
    Disparity = 16 * 10
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=Disparity,
        blockSize=9,
        P1=4 * 3 * 9 ** 2,
        P2=64 * 2 * 9 ** 2,
        disp12MaxDiff=1,
        speckleWindowSize=200,
        speckleRange=32
    )
    disparity_left = stereo_left.compute(imgLL, imgLR).astype(np.float32) / 16.0
    disparity_left[disparity_left < 0] = 0
    disp_normalized_left = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_8u_left = np.uint8(disp_normalized_left)

    # 右到左視差圖
    stereo_right = cv2.StereoSGBM_create(
        minDisparity=-Disparity,
        numDisparities=Disparity,
        blockSize=9,
        P1=4 * 3 * 9 ** 2,
        P2=64 * 2 * 9 ** 2,
        disp12MaxDiff=1,
        speckleWindowSize=200,
        speckleRange=32
    )
    disparity_right = stereo_right.compute(imgRR, imgRL).astype(np.float32) / 16.0
    disparity_right[disparity_right < -Disparity] = 0  # 過濾不合理值
    disp_normalized_right = -cv2.normalize(disparity_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_8u_right = np.uint8(disp_normalized_right)
    
    w, h = disp_8u_left.shape
    
    d = target_width / w 
    
    disp_resize_left = cv2.resize(disp_8u_left, (int(round(d*h)), int(round(d*w))))
    disp_resize_right = cv2.resize(disp_8u_right, (int(round(d*h)), int(round(d*w))))
    
    return disp_resize_left, disp_resize_right

# ----------  中間50%的平均值 ---------- 
def middle_50_percent_mean(data):
        data_sorted = np.sort(data)
        n = len(data_sorted)
        q1 = int(n * 0.25)
        q3 = int(n * 0.75)
        return np.mean(data_sorted[q1:q3])

# ---------- 平均平移值計算 ---------- 
@time_wrapper
def analyze_displacement(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img2, None)
    kp2, des2 = orb.detectAndCompute(img1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    dx_list = []
    dy_list = []
    filtered_matches = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        if abs(dy) <= 5:
            dx_list.append(dx)
            dy_list.append(dy)
            filtered_matches.append(m)

    avg_dx = middle_50_percent_mean(dx_list)

    dx_array = np.array(dx_list).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(dx_array)
    labels = kmeans.labels_
    dx_array = dx_array.flatten()

    groups = {}
    counts = {}
    group_values = {}
    for i in range(2):
        group_dx = dx_array[labels == i]
        groups[i] = np.mean(group_dx)
        counts[i] = len(group_dx)
        group_values[i] = group_dx

    # 最大群
    max_count_group_id = max(counts, key=counts.get)
    max_group_mean = groups[max_count_group_id]
    max_group_max = np.max(group_values[max_count_group_id])
    # 最小群
    min_count_group_id = min(counts, key=counts.get)
    min_group_mean = groups[min_count_group_id]

    return (abs(max_group_mean) if max_group_mean is not None else None,
            abs(max_group_max) if max_group_max is not None else None,
            abs(min_group_mean) if min_group_mean is not None else None,
            abs(avg_dx))

# ---------- 載入深度和顏色圖像 ----------
@time_wrapper
def load_depth_and_color(depth_raw, color_path):
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE
    color_image = imageio.imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image

# ---------- 在平移背景 ----------
@njit
def shift_image_background(img, shift_amount):
    h, w = img.shape[:2]
    shift_abs = abs(2*shift_amount)
    new_w = w + shift_abs  # 加寬

    if (new_w%2) == 1:
        new_w += 1
    if (shift_abs%2) == 1:
        shift_abs += 1

    # 空圖
    if img.ndim == 2:  # 灰階圖
        shifted = np.ones((h, new_w), dtype=img.dtype) * 2  
    else:  # 彩色圖
        shifted = np.zeros((h, new_w, img.shape[2]), dtype=img.dtype)

    if shift_amount < 0:
        # 向右平移
        shifted[:, 0:w] = img
    elif shift_amount > 0:
        # 向左平移
        shifted[:, shift_abs:w+shift_abs] = img
    else:
        shifted[:, :w] = img

    return shifted

# ---------- warping ----------
@time_wrapper
def warping(depth_map: np.ndarray, color_img: np.ndarray):
    """
    - depth_map: np.float32, shape = (h, w)
    - color_img: np.uint8,  shape = (h, w, 3)
    
    - depth_stack: shape = (h, w, 1), dtype=float32
    - color_stack: shape = (h, w, 1, 3), dtype=uint8
    """

    # 1) 檢查維度是否正確
    assert depth_map.ndim == 2, f"深度圖必須是 2D array，但你傳入了 shape = {depth_map.shape}"
    assert color_img.ndim == 3 and color_img.shape[2] == 3, \
        f"彩色影像必須是 (h, w, 3)，但你傳入了 shape = {color_img.shape}"

    h_d, w_d = depth_map.shape
    h_c, w_c, _ = color_img.shape
    assert (h_d, w_d) == (h_c, w_c), \
        f"深度圖 shape = {(h_d, w_d)}，彩色影像 shape = {(h_c, w_c)}，兩者必須相同。"

    # 2) 初始化：同原本維度
    depth_stack = np.full((h_d, w_d, 1), np.nan, dtype=np.float32)
    color_stack = np.zeros((h_d, w_d, 1, 3), dtype=np.uint8)

    # 3) 先做中值濾波，得到 filled_depth (h, w)
    filled_depth = median_filter(depth_map, size=3)

    # 4) 建立布林遮罩：
    valid_mask = (depth_map > 1e-2) & (depth_map < 1.5)
    fallback_mask = (~valid_mask) & (filled_depth != 2.0)

    # 5) 針對 valid_mask 一次性批量賦值
    depth_stack[..., 0][valid_mask] = depth_map[valid_mask]
    color_stack[..., 0, :][valid_mask] = color_img[valid_mask]

    # 6) 針對 fallback_mask 一次性批量賦值
    depth_stack[..., 0][fallback_mask] = filled_depth[fallback_mask]
    color_stack[..., 0, :][fallback_mask] = color_img[fallback_mask]

    return depth_stack, color_stack

# ---------- 深度平移深度圖 ----------
@time_wrapper
def shift_image_object(depth_stack, color_stack, midcenter_x, midcenter_z, threshold=0.3, constant=1, time=1):
    h, w, c = depth_stack.shape

    # 找非 NaN的像素
    valid_mask = ~np.isnan(depth_stack)
    y_idx, x_idx, z_idx = np.where(valid_mask)

    d_vals = np.empty(len(y_idx), dtype=np.float32)
    for i in range(len(y_idx)):
        d_vals[i] = depth_stack[y_idx[i], x_idx[i], z_idx[i]]

    c_vals = np.empty((len(y_idx), 3), dtype=np.uint8)
    for i in range(len(y_idx)):
        c_vals[i] = color_stack[y_idx[i], x_idx[i], z_idx[i]]


    new_w = w + int(midcenter_x)
    max_c = c * 2

    # 建立新陣列
    new_depth = np.full((h, new_w, max_c), np.nan, dtype=np.float32)
    new_color = np.zeros((h, new_w, max_c, 3), dtype=np.uint8)
    index_mask = np.zeros((h, new_w), dtype=np.int32)

    # 判斷哪些需要平移
    need_shift = d_vals < threshold
    shift_x = midcenter_x * (1 - d_vals[need_shift]) / (1 - midcenter_z)
    new_x = x_idx[need_shift] + (constant * time * shift_x).astype(int)

    # 範圍是否超過
    in_bounds = (new_x >= 0) & (new_x < new_w)
    y_s, x_s, z_s = y_idx[need_shift][in_bounds], new_x[in_bounds], z_idx[need_shift][in_bounds]
    d_s, c_s = d_vals[need_shift][in_bounds], c_vals[need_shift][in_bounds]

    # 寫入
    for y, x, d, c in zip(y_s, x_s, d_s, c_s):
        idx = index_mask[y, x]
        if idx < max_c:
            new_depth[y, x, idx] = d
            new_color[y, x, idx] = c
            index_mask[y, x] += 1

    # 不平移的直接寫入
    no_shift_mask = ~need_shift
    y_ns, x_ns, z_ns = y_idx[no_shift_mask], x_idx[no_shift_mask], z_idx[no_shift_mask]
    d_ns, c_ns = d_vals[no_shift_mask], c_vals[no_shift_mask]

    for y, x, d, c in zip(y_ns, x_ns, d_ns, c_ns):
        idx = index_mask[y, x]
        if idx < max_c:
            new_depth[y, x, idx] = d
            new_color[y, x, idx] = c
            index_mask[y, x] += 1

    return shrink_nan_array(new_depth, new_color)

# ---------- 合併左右堆疊 ----------
@time_wrapper
def merge_stacks_numpy(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color):
    """
        left_shifted_depth  : np.ndarray, shape = (h, w, c)
        left_shifted_color  : np.ndarray, shape = (h, w, c, 3)
        right_shifted_depth : np.ndarray, shape = (h, w, c)
        right_shifted_color : np.ndarray, shape = (h, w, c, 3)

        merged_depth_sorted : np.ndarray, shape = (h, w, 2c)
        merged_color_sorted : np.ndarray, shape = (h, w, 2c, 3)
    """
    # 合併深度與顏色
    merged_depth = np.concatenate([left_shifted_depth, right_shifted_depth], axis=2)
    merged_color = np.concatenate([left_shifted_color, right_shifted_color], axis=2)

    # 將 NaN 深度替換成極大值進行排序
    big = np.finfo(np.float32).max
    depth_key = np.where(np.isnan(merged_depth), big, merged_depth)

    # 每個像素在 z 軸排序
    idx = np.argsort(depth_key, axis=2)

    # 重排深度與顏色
    merged_depth_sorted = np.take_along_axis(merged_depth, idx, axis=2)
    idx_expanded = idx[..., np.newaxis]
    merged_color_sorted = np.take_along_axis(merged_color, idx_expanded, axis=2)

    return shrink_nan_array(merged_depth_sorted, merged_color_sorted)

# ---------- 深度圖補洞 ----------
@time_wrapper
def inpainting(depth_stack, color_stack, kernel_size=3, max_iter=10):
    # 找非 NaN 的地方
    valid_mask = ~np.isnan(depth_stack[..., 0])

    # 計算補丁範圍的 padding 數量
    pad = kernel_size // 2

    # 計算 padding 後的深度圖形狀
    padded_shape = (
        depth_stack.shape[0] + 2 * pad,
        depth_stack.shape[1] + 2 * pad,
        depth_stack.shape[2]
    )

    # 創建遮罩
    structure = np.ones((kernel_size, kernel_size, 1), dtype=bool)

    for i in range(max_iter):
        # 找出要補的點
        invalid_mask = np.isnan(depth_stack[..., 0])
        has_valid_neighbors = binary_dilation(valid_mask, structure=structure[..., 0]) & invalid_mask

        # 如果沒有需要補的點，提前結束
        if not np.any(has_valid_neighbors):
            break

        # 找座標 (y, x)
        fill_coords = np.argwhere(has_valid_neighbors)

        # 建立 padding 後的深度與顏色圖
        padded_depth = np.full(padded_shape, np.nan, dtype=np.float32)
        padded_color = np.zeros(padded_shape + (3,), dtype=np.uint8)

        # 將原始堆疊資料放入 padding 中央
        padded_depth[pad:-pad, pad:-pad] = depth_stack
        padded_color[pad:-pad, pad:-pad] = color_stack

        for y, x in fill_coords:
            # 擷取以 (y, x) 為中心的區域
            depth_roi = padded_depth[y:y+kernel_size, x:x+kernel_size]
            color_roi = padded_color[y:y+kernel_size, x:x+kernel_size]

            # 擷取該區域中有效的深度值
            valid_depths = depth_roi[~np.isnan(depth_roi)]

            # 將顏色轉成一維陣列
            valid_colors = color_roi.reshape(-1, 3)
            valid_colors = valid_colors[~np.isnan(depth_roi.reshape(-1))]

            # 若此區域中有有效深度值
            if len(valid_depths) > 0:
                # 計算深度與顏色的中位數
                median_depth = np.median(valid_depths)
                median_color = np.median(valid_colors, axis=0).astype(np.uint8)

                # 將結果填入最前層
                depth_stack[y, x, 0] = median_depth
                color_stack[y, x, 0] = median_color

                # 更新該點為有效像素
                valid_mask[y, x] = True

    return depth_stack, color_stack

# ---------- Warping to image ----------
def warping_to_img(stack_color, stack_depth=None):
    h, w = stack_color.shape[:2]

    if stack_depth is None:
        return stack_color[..., 0, :].copy()

    depth = stack_depth
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)

    yy, xx = np.indices((h, w))
    
    front_color = stack_color[yy, xx, 0]
    
    return front_color


# ---------- 主函數 ----------
if __name__ == "__main__":
    template = cv2.imread('image/bbox_left_left.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 計算深度圖 
    imgLL = cv2.imread(f"image/box_left_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgLR = cv2.imread(f"image/box_left_right.jpg", cv2.IMREAD_GRAYSCALE)
    imgRL = cv2.imread(f"image/box_right_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgRR = cv2.imread(f"image/box_right_right.jpg", cv2.IMREAD_GRAYSCALE)
    print(template.shape)

    depth_left, depth_right = depth(imgLL, imgLR, imgRL, imgRR, target_width=template.shape[0])
    
    # 載入左右視角
    left_depth, left_color = load_depth_and_color(depth_left, f"image/bbox_left_left.jpg")
    right_depth, right_color = load_depth_and_color(depth_right, f"image/bbox_right_right.jpg")

    # 計算深度圖shift
    (max_group_mean, max_group_max, min_group_mean,avg_dx) = analyze_displacement(left_color, right_color)  
    shift = int(min_group_mean / 2)
    midcenter_x = int(max_group_max - min_group_mean) / 2.3 #int(abs(center_left_x - center_right_x) / 2)
    midcenter_z = 0

    # 平移圖像
    left_color_shifted = shift_image_background(left_color, -shift)
    right_color_shifted = shift_image_background(right_color, shift)
    left_depth_shifted = shift_image_background(left_depth, -shift)
    right_depth_shifted = shift_image_background(right_depth, shift)
    
    # warp 成深度圖
    fast_left_stack_depth, fast_left_stack_color = warping(left_depth_shifted, left_color_shifted)
    fast_right_stack_depth, fast_right_stack_color = warping(right_depth_shifted, right_color_shifted)

    # 平移深度圖
    fast_left_shifted_depth, fast_left_shifted_color = shift_image_object(fast_left_stack_depth, fast_left_stack_color, midcenter_x, midcenter_z, threshold=0.5, constant=-1)
    fast_right_shifted_depth, fast_right_shifted_color = shift_image_object(fast_right_stack_depth, fast_right_stack_color, midcenter_x, midcenter_z, threshold=0.5, constant=1)

    # 合併深度圖

    fast_merged_depth_stack1, fast_merged_color_stack1 = merge_stacks_numpy(fast_left_shifted_depth, fast_left_shifted_color, fast_right_shifted_depth, fast_right_shifted_color)

    # 補洞

    mid_merged_depth_stack, mid_merged_color_stack = inpainting(fast_merged_depth_stack1, fast_merged_color_stack1)

    # 顯示中央圖片
    # times = 0.5  #(-1~1)中央向左右移動的比例
    # fast_merged_depth_stack, fast_merged_color_stack = shift_image_object(mid_merged_depth_stack, mid_merged_color_stack, midcenter_x, midcenter_z,threshold=0.5, constant= 1,time = times)
    # fast_merged_depth_stack, fast_merged_color_stack = inpainting(fast_merged_depth_stack, fast_merged_color_stack)
    # show_img = warping_to_img(fast_merged_color_stack, fast_merged_depth_stack)
    # plt.imshow(show_img)
    # plt.show()
    

    # 製作gif
    gif = []
    # 第二次偏移
    for t in range (-20,21):
        print(t)
        times = (t/20) 
        shift_time = int(2 * shift * ((t+20)/40))
        print(shift_time)
        
        fast_merged_depth_stack, fast_merged_color_stack = shift_image_object(mid_merged_depth_stack, mid_merged_color_stack, midcenter_x, midcenter_z,threshold=0.5, constant= 1,time = times)

        fast_merged_depth_stack, fast_merged_color_stack = inpainting(fast_merged_depth_stack, fast_merged_color_stack)
            
        merged_img = warping_to_img(fast_merged_depth_stack, fast_merged_color_stack)

        # print(len(merged_img),len(merged_img[0]))
        out = merged_img[: ,shift_time:int(shift_time+template.shape[1]), :]

        img = Image.fromarray(out)

        gif.append(img)

    k = len(gif)
    for i in range(k-1):
        gif.append(gif[k-1-i])

    gif[0].save(
    "rsynced_square.gif",
    save_all=True,
    append_images=gif[1:],
    duration=100, 
    loop=0
)