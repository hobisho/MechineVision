import numpy as np
import cv2
from scipy.ndimage import median_filter
from tqdm import tqdm

def forward_warp(depth_map, baseline, focal_length, shift, img_shape):
    """
    Forward warp depth map to virtual view using disparity = bf / depth
    shift > 0 for left-to-virtual, shift < 0 for right-to-virtual
    """
    depth_map = (255.0 - depth_map) / 255.0 
    h, w = depth_map.shape
    warped_depth = np.zeros(img_shape, dtype=np.float32)
    map = np.zeros(img_shape, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 0:
                disparity = (baseline * focal_length) / d
                x_virtual = int(round(x + shift * disparity))
                if 0 <= x_virtual < w:
                    if warped_depth[y, x_virtual] == 0 or warped_depth[y, x_virtual] > d:
                        warped_depth[y, x_virtual] = d
                        map[y, x_virtual] = 1

    # 中值濾波填補空洞
    warped_depth = median_filter(warped_depth, size=3)
    return warped_depth, map

def warp_triangle(img, depth, tri, shift, baseline, focal_length, target_img):
    """
    將原始圖像中的一個三角形，透過對應深度值進行映射
    """
    pts_src = np.array(tri, dtype=np.float32)
    pts_dst = []
    for (x, y) in pts_src:
        d = depth[int(y), int(x)]
        if d <= 0:
            return  # skip invalid triangle
        disparity = (baseline * focal_length) / d
        x_virtual = x + shift * disparity
        pts_dst.append([x_virtual, y])
    
    pts_dst = np.array(pts_dst, dtype=np.float32)

    # 計算仿射變換
    M = cv2.getAffineTransform(pts_src, pts_dst)
    
    # 在三角形內建立mask
    mask = np.zeros_like(target_img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_dst), (1, 1, 1))

    # warp
    warped = cv2.warpAffine(img, M, (target_img.shape[1], target_img.shape[0]))
    target_img[mask > 0] = warped[mask > 0]

def merge_views(TL, TR, maskL, maskR, alpha):
    """
    合併左右圖像：權重混合 + 空洞處理
    """
    merged = np.zeros_like(TL, dtype=np.uint8)
    h, w, _ = TL.shape
    for y in range(h):
        for x in range(w):
            inL = maskL[y, x]
            inR = maskR[y, x]
            if inL and inR:
                merged[y, x] = (1 - alpha) * TL[y, x] + alpha * TR[y, x]
            elif inL:
                merged[y, x] = TL[y, x]
            elif inR:
                merged[y, x] = TR[y, x]
            else:
                merged[y, x] = 0  # 空洞，留待補全
    return merged

def dibr_render(left_img, right_img, left_depth, right_depth, baseline, focal_length, alpha):
    """
    主函數：生成虛擬視角圖像
    """
    h, w = left_img.shape[:2]

    # 左右深度圖 Forward warp
    TL_depth, maskL = forward_warp(left_depth, baseline, focal_length, shift=0, img_shape=(h, w))
    TR_depth, maskR = forward_warp(right_depth, baseline, focal_length, shift=0, img_shape=(h, w))
    print(" Forward warp")

    # 初始化渲染圖像
    TL = np.zeros_like(left_img)
    TR = np.zeros_like(right_img)
    print("渲染圖像")

    # 三角形插值（示例：使用小三角形網格）
    grid_size = 5
    for y in tqdm(range(0, h - grid_size, grid_size)):
        for x in range(0, w - grid_size, grid_size):
            tri1 = [(x, y), (x + grid_size, y), (x, y + grid_size)]
            tri2 = [(x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]
            warp_triangle(left_img, left_depth, tri1, shift=-alpha, baseline=baseline, focal_length=focal_length, target_img=TL)
            warp_triangle(left_img, left_depth, tri2, shift=-alpha, baseline=baseline, focal_length=focal_length, target_img=TL)
            warp_triangle(right_img, right_depth, tri1, shift=1 - alpha, baseline=baseline, focal_length=focal_length, target_img=TR)
            warp_triangle(right_img, right_depth, tri2, shift=1 - alpha, baseline=baseline, focal_length=focal_length, target_img=TR)

    # 合併左右映射圖
    merged = merge_views(TL, TR, maskL, maskR, alpha)

    return merged


if __name__ == "__main__":
    left_img = cv2.imread("left.jpg")
    right_img = cv2.imread("right.jpg")
    left_depth = cv2.imread("disparity_left.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    right_depth = cv2.imread("disparity_right.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

    baseline = 0.1  # meters
    focal_length = 700  # pixels
    alpha = 0.5  # 虛擬視角在兩者之間

    virtual_view = dibr_render(left_img, right_img, left_depth, right_depth, baseline, focal_length, alpha)
    cv2.imwrite("virtual_view.png", virtual_view)
