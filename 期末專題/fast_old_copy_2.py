import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from def_orb import analyze_displacement
from scipy.ndimage import median_filter, binary_dilation

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

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
    valid_mask = (depth_map > 1e-3) & (depth_map < 1.5)
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
def shift_points_in_stack_numpy(depth_stack, color_stack,
                                     midcenter_x, midcenter_z,
                                     threshold=0.1, constant=1, time=1):
    """
    Vectorized 版本，最終 new_depth_stack/new_color_stack 的 shape
    與舊版一樣都是 (h, w, c) 和 (h, w, c, 3)，
    但內部演算法邏輯和舊版 for-loop 1:1 等價，跑起來更快。
    """
    h, w, c = depth_stack.shape[:3]
    # 1) 分配輸出，注意還是 c 而非 c*2
    new_depth_stack = np.full((h, w, c), np.nan, dtype=np.float32)
    new_color_stack = np.zeros((h, w, c, 3), dtype=np.uint8)

    # 2) 找出所有非 NaN 的 (y, x, idx)
    valid_mask = ~np.isnan(depth_stack)   # shape (h, w, c)
    y_idx, x_idx, z_idx = np.where(valid_mask)            # (N,) 向量
    depth_vals = depth_stack[y_idx, x_idx, z_idx]          # (N,)
    color_vals = color_stack[y_idx, x_idx, z_idx]          # (N,3)

    # 3) 先算哪些要 shift、哪些不用
    is_shift = depth_vals < threshold
    new_x = x_idx.copy()
    if np.any(is_shift):
        shift_x = midcenter_x * (1.0 - depth_vals[is_shift]) / (1.0 - midcenter_z)
        moved = x_idx[is_shift] + np.floor(constant * time * shift_x).astype(int)
        new_x[is_shift] = moved

    # 4) 濾掉超出 [0, w) 的那部分
    inside = (new_x >= 0) & (new_x < w)
    if not np.all(inside):
        y_idx = y_idx[inside]
        x_idx = x_idx[inside]
        new_x = new_x[inside]
        depth_vals = depth_vals[inside]
        color_vals = color_vals[inside]
        is_shift = is_shift[inside]

    # 5) 這裡我們要把所有最終「會寫進去 new_depth_stack」的點做一次排序與分組：
    #    group_by key = y*w + final_x
    keys = y_idx * w + new_x
    order = np.argsort(keys)
    keys_s = keys[order]
    y_s = y_idx[order]
    x_s = new_x[order]
    depth_s = depth_vals[order]
    color_s = color_vals[order]

    # 6) 計算每組(grid cell)裡的「第一個空的 channel rank」
    M = len(keys_s)
    ranks = np.zeros(M, dtype=np.int32)
    if M > 0:
        ranks[0] = 0
        same_grp = keys_s[1:] == keys_s[:-1]
        for i in range(1, M):
            if same_grp[i - 1]:
                ranks[i] = ranks[i - 1] + 1
            else:
                ranks[i] = 0

    # 7) 如果 ranks[i] >= c，就表示那個 cell 已經被塞滿 c 個 depth，必須跳過
    good = ranks < c
    y_f = y_s[good]
    x_f = x_s[good]
    rank_f = ranks[good]
    depth_f = depth_s[good]
    color_f = color_s[good]  # shape (K, 3)

    # 8) 最後一次性寫入 (h, w, c)、不超出索引
    new_depth_stack[y_f, x_f, rank_f] = depth_f
    new_color_stack[y_f, x_f, rank_f] = color_f

    return new_depth_stack, new_color_stack



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
    # 取尺寸
    h, w, c = left_shifted_depth.shape

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

    return merged_depth_sorted, merged_color_sorted

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
            print(f"[第 {i+1} 輪] 沒有需要補的點")
            break
        
        # 获取需要填充的坐标
        fill_coords = np.argwhere(has_valid_neighbors)
        print(f"[第 {i+1} 輪] 需要補 {len(fill_coords)} 個點")
        
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
def frontmost_color_numpy(stack_color, stack_depth=None):
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

    # 作為「NaN 代理」的極大值：取相同 dtype 下的最大浮點
    big = np.finfo(depth.dtype).max

    # 把 nan 都換成 big，讓它們在排序/argmin 時排在最末端
    depth_key = np.where(np.isnan(depth), big, depth)  # shape = (h, w, c)

    # 在 z 軸 (axis=2) 取 argmin，得到 shape = (h, w) 的 index，
    # 對於「全 NaN」那一組，因為全被替換成 big，argmin 自然回傳 0
    idx = np.argmin(depth_key, axis=2)  # (h, w), 類型 int64

    # 用 advanced-indexing 一次抽出最前端的顏色 (h, w, 3)
    # 先生成 (h, w) 的 y, x 座標網格
    yy, xx = np.indices((h, w))
    
    front_color = stack_color[yy, xx, idx]  # shape = (h, w, 3)
    
    return front_color

def show_virtual_views_numpy(left_stack_color, right_stack_color, merged_stack_color, right_stack_depth, merge_stack_depth):
    left_img = frontmost_color_numpy(left_stack_color)
    right_img = frontmost_color_numpy(right_stack_color, right_stack_depth)
    merged_img = frontmost_color_numpy(merged_stack_color, merge_stack_depth)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Warped Left View")
    plt.imshow(left_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Merged Virtual View")
    plt.imshow(merged_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Warped Right View")
    plt.imshow(right_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 載入左右視角
    path = "image/bbox"
    left_depth, left_color, _ = load_depth_and_color("output/disparity_left.jpg", f"{path}_left_left.jpg")
    right_depth, right_color, _ = load_depth_and_color("output/disparity_right.jpg", f"{path}_right_right.jpg")

    # 計算深度圖shift
    (max_group_mean, max_group_max, min_group_mean,avg_dx) = analyze_displacement(left_color, right_color)  # 假設 ImgShift 函數已經定義並返回位移量
    shift = int(min_group_mean / 2)
    midcenter_x = int(max_group_max - min_group_mean) / 2.15 #int(abs(center_left_x - center_right_x) / 2)
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
    
    fast_left_shifted_depth, fast_left_shifted_color = shift_points_in_stack_numpy(fast_left_stack_depth, fast_left_stack_color, midcenter_x, midcenter_z, threshold=0.65, constant=-1)
    fast_right_shifted_depth, fast_right_shifted_color = shift_points_in_stack_numpy(fast_right_stack_depth, fast_right_stack_color, midcenter_x, midcenter_z, threshold=0.85, constant=1)

    # 合併深度圖

    fast_merged_depth_stack1, fast_merged_color_stack1 = merge_stacks_numpy(fast_left_shifted_depth, fast_left_shifted_color, fast_right_shifted_depth, fast_right_shifted_color)

    # 補洞

    fast_merged_depth_stack, fast_merged_color_stack = inpaint_stack_median_numpy(fast_merged_depth_stack1, fast_merged_color_stack1)
    
    # fast_merged_depth_stack, fast_merged_color_stack = shift_points_in_stack_numpy(fast_merged_depth_stack, fast_merged_color_stack, midcenter_x, midcenter_z,threshold=1, constant=-1)

    # fast_merged_depth_stack, fast_merged_color_stack = inpaint_stack_median_numpy(fast_merged_depth_stack, fast_merged_color_stack)
    
    show_virtual_views_numpy(fast_left_stack_color, fast_right_stack_color, fast_merged_color_stack, fast_right_stack_depth, fast_merged_depth_stack)
