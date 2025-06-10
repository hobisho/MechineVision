import cv2
import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter, binary_dilation

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@njit
def shrink_nan_array(depth_arr, color_arr):
    # arr.shape = (h, w, c)
    mask = ~np.isnan(depth_arr)  # True 表示不是 NaN
    valid_counts = np.sum(mask, axis=2)  # 每個 [h, w] 有幾個非 NaN
    max_valid = np.max(valid_counts)     # 所有像素中最大的非 NaN 數量

    # 回傳裁切後的陣列
    return depth_arr[:, :, :max_valid], color_arr[:, :, :max_valid, :]

@time_wrapper
def depth(imgLL, imgLR, imgRL, imgRR, target_width):
    # 左到右視差圖（已確認可行）
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

    # 右到左視差圖（調整參數）
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
    
    d = target_width / w # 假設 d 是高度與寬度的比例
    
    disp_resize_left = cv2.resize(disp_8u_left, (int(round(d*h)), int(round(d*w))))
    disp_resize_right = cv2.resize(disp_8u_right, (int(round(d*h)), int(round(d*w))))
    
    return disp_resize_left, disp_resize_right

def middle_50_percent_mean(data):
        data_sorted = np.sort(data)
        n = len(data_sorted)
        q1 = int(n * 0.25)
        q3 = int(n * 0.75)
        return np.mean(data_sorted[q1:q3])
    
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

    if len(dx_list) == 0:
        print("⚠️ 過濾後沒有匹配點")
        return None, None, None, None, None, None, None

    avg_dx = middle_50_percent_mean(dx_list)

    dx_array = np.array(dx_list).reshape(-1, 1)
    if len(dx_array) < 5:
        max_group_mean = None
        max_group_max = None
        min_group_mean = None
    else:
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

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

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

# ---------- 在 warp 前先左右平移 ----------
@njit
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
def forward_warp_to_stack_numpy(depth_map: np.ndarray, color_img: np.ndarray):
    """
    - depth_map: np.float32, shape = (h, w)
    - color_img: np.uint8,  shape = (h, w, 3)
    回傳：
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
    #    valid_mask：原 depth_map 落在 (1e-2, 1.5) 的那些 (y, x)
    #    fallback_mask：不屬於 valid_mask，且 filled_depth != 2.0
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
def shift_points_in_stack2(depth_stack, color_stack, midcenter_x, midcenter_z, threshold=0.3, constant=1, time=1):
    h, w, c = depth_stack.shape

    # 取得所有有效索引 (非 NaN)
    valid_mask = ~np.isnan(depth_stack)
    y_idx, x_idx, z_idx = np.where(valid_mask)

    d_vals = np.empty(len(y_idx), dtype=np.float32)
    for i in range(len(y_idx)):
        d_vals[i] = depth_stack[y_idx[i], x_idx[i], z_idx[i]]

    c_vals = np.empty((len(y_idx), 3), dtype=np.uint8)
    for i in range(len(y_idx)):
        c_vals[i] = color_stack[y_idx[i], x_idx[i], z_idx[i]]


    # 預估最大 channel 數（多一點以避免溢位）
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

    # 範圍合法檢查
    in_bounds = (new_x >= 0) & (new_x < new_w)
    y_s, x_s, z_s = y_idx[need_shift][in_bounds], new_x[in_bounds], z_idx[need_shift][in_bounds]
    d_s, c_s = d_vals[need_shift][in_bounds], c_vals[need_shift][in_bounds]

    # 平移後寫入
    for y, x, d, c in zip(y_s, x_s, d_s, c_s):
        idx = index_mask[y, x]
        if idx < max_c:
            new_depth[y, x, idx] = d
            new_color[y, x, idx] = c
            index_mask[y, x] += 1

    # 不平移的直接寫入原位置
    no_shift_mask = ~need_shift
    y_ns, x_ns, z_ns = y_idx[no_shift_mask], x_idx[no_shift_mask], z_idx[no_shift_mask]
    d_ns, c_ns = d_vals[no_shift_mask], c_vals[no_shift_mask]

    for y, x, d, c in zip(y_ns, x_ns, d_ns, c_ns):
        idx = index_mask[y, x]
        if idx < max_c:
            new_depth[y, x, idx] = d
            new_color[y, x, idx] = c
            index_mask[y, x] += 1

    # 最後裁剪多餘的第三維
    return shrink_nan_array(new_depth, new_color)

# ---------- 合併左右堆疊 ----------
@time_wrapper
def merge_stacks_numpy(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color):
    """
    把左右兩個已經 shift 過的 depth/color stacks 合併，並在 z 軸（第三維）做排序，NaN 排在最後。
    輸入:
        left_shifted_depth:  (h, w, c)
        left_shifted_color:  (h, w, c, 3)
        right_shifted_depth: (h, w, c)
        right_shifted_color: (h, w, c, 3)
    回傳:
        merged_depth_sorted: (h, w, 2*c)
        merged_color_sorted: (h, w, 2*c, 3)
    """
    # 1) 先水平拼接兩邊的 depth / color，形狀變成 (h, w, 2*c) / (h, w, 2*c, 3)
    merged_depth = np.concatenate([left_shifted_depth, right_shifted_depth], axis=2)      # (h, w, 2*c)
    merged_color = np.concatenate([left_shifted_color, right_shifted_color], axis=2)      # (h, w, 2*c, 3)

    # 2) 為了讓 np.argsort 將 NaN 排在後面，我們先把 NaN 替換成一個極大值。
    #    這裡使用 float32 的最大值 (np.finfo(np.float32).max) 作為 NaN 的代理 key。
    #    只要所有 real depth 都 < 這個值，就能保證排序後 NaN 一定跑到最後。
    big = np.finfo(np.float32).max
    depth_key = np.where(np.isnan(merged_depth), big, merged_depth)  # (h, w, 2*c), NaN→big

    # 3) 針對第三維做 argsort：對每個 (y, x) 的 2*c 個深度值排序，回傳排序後的 index
    #    這個 idx 的 shape 是 (h, w, 2*c)，每個 [y,x] 都是一個長度為 2*c 的排列索引，
    #    使得 depth_key[y, x, idx[y,x,:]] 是從小到大（NaN/大值在最後）。
    idx = np.argsort(depth_key, axis=2)  # (h, w, 2*c)

    # 4) 利用 take_along_axis，一步把深度與顏色依照 idx 重排
    #    - 對深度陣列：merged_depth_sorted[y,x,:] = merged_depth[y,x, idx[y,x,:]]
    merged_depth_sorted = np.take_along_axis(merged_depth, idx, axis=2)  # (h, w, 2*c)

    #    - 對顏色陣列：要先把 idx 擴充成 (h, w, 2*c, 1)，這樣才能沿著 z 軸把最後的 RGB 數值也一起帶過去
    idx_expanded = idx[..., np.newaxis]  # (h, w, 2*c, 1)
    merged_color_sorted = np.take_along_axis(merged_color, idx_expanded, axis=2)  # (h, w, 2*c, 3)

    return shrink_nan_array(merged_depth_sorted, merged_color_sorted)

# ---------- 深度圖補洞 ----------
@time_wrapper
def inpaint_stack_median_numpy(depth_stack, color_stack, kernel_size=3, max_iter=10):
    # 创建有效点掩码（非NaN的点）
    valid_mask = ~np.isnan(depth_stack[..., 0])
    
    # 为深度和颜色创建填充后的数组
    pad = kernel_size // 2
    padded_shape = (depth_stack.shape[0] + 2 * pad, 
                   depth_stack.shape[1] + 2 * pad, 
                   depth_stack.shape[2])
    
    # 创建有效点计数的结构元素
    structure = np.ones((kernel_size, kernel_size, 1), dtype=bool)
    
    for i in range(max_iter):
        # 获取需要填充的位置（当前为NaN但周围有有效点）
        invalid_mask = np.isnan(depth_stack[..., 0])
        has_valid_neighbors = binary_dilation(valid_mask, structure=structure[..., 0]) & invalid_mask
        
        # 如果没有需要填充的点，提前退出
        if not np.any(has_valid_neighbors):
            # print(f"[第 {i+1} 輪] 沒有需要補的點")
            break
        
        # 获取需要填充的坐标
        fill_coords = np.argwhere(has_valid_neighbors)
        # print(f"[第 {i+1} 輪] 需要補 {len(fill_coords)} 個點")
        
        # 创建深度和颜色的扩展视图
        padded_depth = np.full(padded_shape, np.nan, dtype=np.float32)
        padded_color = np.zeros(padded_shape + (3,), dtype=np.uint8)
        
        padded_depth[pad:-pad, pad:-pad] = depth_stack
        padded_color[pad:-pad, pad:-pad] = color_stack
        
        # 为每个需要填充的点计算邻域中值
        for y, x in fill_coords:
            # 获取邻域数据
            depth_roi = padded_depth[y:y+kernel_size, x:x+kernel_size]
            color_roi = padded_color[y:y+kernel_size, x:x+kernel_size]
            
            # 获取所有有效深度值
            valid_depths = depth_roi[~np.isnan(depth_roi)]
            
            # 获取所有有效颜色值
            valid_colors = color_roi.reshape(-1, 3)
            valid_colors = valid_colors[~np.isnan(depth_roi.reshape(-1))]
            
            if len(valid_depths) > 0:
                # 计算中值
                median_depth = np.median(valid_depths)
                median_color = np.median(valid_colors, axis=0).astype(np.uint8)
                
                # 填充到第一个通道
                depth_stack[y, x, 0] = median_depth
                color_stack[y, x, 0] = median_color
                
                # 更新有效点掩码
                valid_mask[y, x] = True
        
    return depth_stack, color_stack

# ---------- 顯示圖像視覺結果 ----------
def frontmost_color_numpy(stack_depth,stack_color):
    """
    參數:
    - stack_color: shape = (h, w, c, 3)，每個 (y,x) 都是一個 c 長度的顏色堆疊 (RGB)
    - stack_depth: None 或 shape = (h, w, c)，每個 (y,x) 都是一個 c 長度的深度堆疊 (float)，其中可能包含 np.nan。
    
    行為（向量化版）：
    - 如果 stack_depth is None，直接回傳 stack_color[..., 0, :]；
    - 否則把所有 np.nan 視為極大值，對 axis=2 做 argmin，得到索引 idx (h, w)，
        再用 idx 去從 stack_color 抽出對應的 RGB 值。
    - 這樣：如果該 (y,x) 的深度全為 NaN，同樣會因為「所有深度＋極大值」下 argmin 結果 = 0 而取第 0 個元素。
    """
    h, w = stack_color.shape[:2]

    # case 1: 沒有深度資訊，或想統一直接拿第一個通道
    if stack_depth is None:
        return stack_color[..., 0, :].copy()  # (h, w, 3)

    # 確認深度為浮點陣列
    depth = stack_depth
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)

    # 用 advanced-indexing 一次抽出最前端的顏色 (h, w, 3)
    # 先生成 (h, w) 的 y, x 座標網格
    yy, xx = np.indices((h, w))
    
    front_color = stack_color[yy, xx, 0]  # shape = (h, w, 3)
    
    return front_color

def frontmost_depth_numpy(stack_depth):
    """
    參數:
    - stack_depth: shape = (h, w, c)，每個 (y,x) 都是一個 c 長度的深度堆疊 (float)，其中可能包含 np.nan。
    
    行為（向量化版）：
    - 把所有 np.nan 視為極大值，對 axis=2 做 argmin，得到索引 idx (h, w)，
        再用 idx 去從 stack_depth 抽出對應的深度值。
    - 這樣：如果該 (y,x) 的深度全為 NaN，同樣會因為「所有深度＋極大值」下 argmin 結果 = 0 而取第 0 個元素。
    """
    h, w = stack_depth.shape[:2]

    # 確認深度為浮點陣列
    depth = stack_depth
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)

    # 用 advanced-indexing 一次抽出最前端的深度 (h, w)
    yy, xx = np.indices((h, w))
    
    front_depth = depth[yy, xx, 0]  # shape = (h, w)
    
    return front_depth

def show_virtual_views_numpy(merged_stack_color, merge_stack_depth):
    # 先算出每張「最前端顏色/深度」的圖
    merged_img = frontmost_color_numpy(merged_stack_color, merge_stack_depth)
    
    print(merge_stack_depth[1010, 1484])

    # merged_depth_img = frontmost_depth_numpy(merge_stack_depth)

    return merged_img


if __name__ == "__main__":
    template = cv2.imread('image/bbox_left_left.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 計算深度圖 
    imgLL = cv2.imread(f"image/box_left_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgLR = cv2.imread(f"image/box_left_right.jpg", cv2.IMREAD_GRAYSCALE)
    imgRL = cv2.imread(f"image/box_right_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgRR = cv2.imread(f"image/box_right_right.jpg", cv2.IMREAD_GRAYSCALE)

    depth_left, depth_right = depth(imgLL, imgLR, imgRL, imgRR, target_width=template.shape[0])
    
    # 載入左右視角
    left_depth, left_color = load_depth_and_color(depth_left, f"image/bbox_left_left.jpg")
    right_depth, right_color = load_depth_and_color(depth_right, f"image/bbox_right_right.jpg")

    # 計算深度圖shift
    (max_group_mean, max_group_max, min_group_mean,avg_dx) = analyze_displacement(left_color, right_color)  # 假設 ImgShift 函數已經定義並返回位移量
    shift = int(min_group_mean / 2)
    midcenter_x = int(max_group_max - min_group_mean) / 2.4 #int(abs(center_left_x - center_right_x) / 2)
    midcenter_z = 0

    # 平移圖像
    left_color_shifted = shift_image_horizontal(left_color, -shift)
    right_color_shifted = shift_image_horizontal(right_color, shift)
    left_depth_shifted = shift_image_horizontal(left_depth, -shift)
    right_depth_shifted = shift_image_horizontal(right_depth, shift)
    
    # warp 成深度圖
    
    fast_left_stack_depth, fast_left_stack_color = forward_warp_to_stack_numpy(left_depth_shifted, left_color_shifted)
    fast_right_stack_depth, fast_right_stack_color = forward_warp_to_stack_numpy(right_depth_shifted, right_color_shifted)

    # 平移深度圖
    
    fast_left_shifted_depth, fast_left_shifted_color = shift_points_in_stack2(fast_left_stack_depth, fast_left_stack_color, midcenter_x, midcenter_z, threshold=0.5, constant=-1)
    fast_right_shifted_depth, fast_right_shifted_color = shift_points_in_stack2(fast_right_stack_depth, fast_right_stack_color, midcenter_x, midcenter_z, threshold=0.5, constant=1)

    # 合併深度圖

    fast_merged_depth_stack1, fast_merged_color_stack1 = merge_stacks_numpy(fast_left_shifted_depth, fast_left_shifted_color, fast_right_shifted_depth, fast_right_shifted_color)

    # 補洞

    mid_merged_depth_stack, mid_merged_color_stack = inpaint_stack_median_numpy(fast_merged_depth_stack1, fast_merged_color_stack1)

    # show_virtual_views_numpy(fast_left_stack_color, fast_right_stack_color, fast_merged_color_stack, fast_left_stack_depth, fast_right_stack_depth, fast_merged_depth_stack)
    
    gif = []
    # 第二次偏移
    for t in range (0,11):
        print(t)
        times = (t/10) 
        if t >= 0:
            constant = 1
        else:
            constant = -1
        
        fast_merged_depth_stack, fast_merged_color_stack = shift_points_in_stack2(mid_merged_depth_stack, mid_merged_color_stack, midcenter_x, midcenter_z,threshold=0.5, constant= 1,time = times)

        fast_merged_depth_stack, fast_merged_color_stack = inpaint_stack_median_numpy(fast_merged_depth_stack, fast_merged_color_stack)
            
        merged_img = frontmost_color_numpy(fast_merged_depth_stack, fast_merged_color_stack)

        img = Image.fromarray(merged_img)

        gif.append(img)

    gif[0].save(
    "synced_square.gif",
    save_all=True,
    append_images=gif[1:],
    duration=50,  # 每幀 50 毫秒，控制同步播放速度
    loop=0
)