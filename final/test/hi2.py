import cv2
import numpy as np
import math

def index(m, n):
    if 0 <= m < n:
        return m
    elif m < 0:
        return 0
    elif m >= n:
        return n - 1
    return 0

def obtain_new_disp_map(ref_disp, value):
    height, width = ref_disp.shape
    dst_disp = np.zeros((height, width), dtype=np.float32)

    for j in range(height):
        for i in range(width):
            disp = int(ref_disp[j, i])
            new_disp = disp * value
            inew = int(i - new_disp)
            inew = index(inew, width)
            if new_disp >= 255:
                new_disp = 0
            dst_disp[j, inew] = new_disp
    return dst_disp

def insert_depth_32f(depth):
    height, width = depth.shape
    integral_map = np.zeros((height, width), dtype=np.float64)
    pts_map = np.zeros((height, width), dtype=np.int32)

    mask = depth > 1e-3
    integral_map[mask] = depth[mask]
    pts_map[mask] = 1

    # Integral image calculation
    integral_map = integral_map.cumsum(axis=1).cumsum(axis=0)
    pts_map = pts_map.cumsum(axis=1).cumsum(axis=0)

    dWnd = 2
    while dWnd > 1:
        wnd = int(dWnd)
        dWnd /= 2
        for i in range(height):
            for j in range(width):
                left = max(0, j - wnd - 1)
                right = min(j + wnd, width - 1)
                top = max(0, i - wnd - 1)
                bot = min(i + wnd, height - 1)

                def get_val(arr, x, y):
                    return arr[y, x]

                pts_cnt = pts_map[bot, right] + pts_map[top, left] - pts_map[bot, left] - pts_map[top, right]
                sum_gray = integral_map[bot, right] + integral_map[top, left] - integral_map[bot, left] - integral_map[top, right]

                if pts_cnt > 0:
                    depth[i, j] = sum_gray / pts_cnt

        s = wnd // 2 * 2 + 1
        if s > 201:
            s = 201
        depth[:] = cv2.GaussianBlur(depth, (s, s), s, s)

def adjust_contrast(disp):
    minval = np.min(disp)
    maxval = np.max(disp)
    disp = ((disp - minval) / (maxval - minval) * 255).astype(np.uint8)
    return disp

def main():
    src_img_l = cv2.imread("right.jpg")
    disp_l = cv2.imread("disparity_8bit.png", 0)
    disp_l = (disp_l / 4).astype(np.uint8)  # 1/4 resolution

    img_height, img_width = src_img_l.shape[:2]

    dst_img_l = np.zeros_like(src_img_l)
    dst_new_disp_img = np.zeros((img_height, img_width), dtype=np.float32)
    save_disp = np.zeros((img_height, img_width), dtype=np.uint8)

    # Obtain new disparity map
    dst_new_disp_img = obtain_new_disp_map(disp_l, 1)
    insert_depth_32f(dst_new_disp_img)

    p_new_disp_data = dst_new_disp_img

    for j in range(img_height):
        for i in range(img_width):
            disp = p_new_disp_data[j, i]
            id_ = i + disp
            id0 = math.floor(id_)
            id1 = math.floor(id_ + 1)

            weight1 = 1 - (id_ - id0)
            weight2 = id_ - id0

            id0 = index(id0, img_width)
            id1 = index(id1, img_width)

            save_disp[j, i] = np.clip(disp, 0, 255)

            # Interpolation for color image
            dst_img_l[j, i, 0] = weight1 * src_img_l[j, id0, 0] + weight2 * src_img_l[j, id1, 0]
            dst_img_l[j, i, 1] = weight1 * src_img_l[j, id0, 1] + weight2 * src_img_l[j, id1, 1]
            dst_img_l[j, i, 2] = weight1 * src_img_l[j, id0, 2] + weight2 * src_img_l[j, id1, 2]

    save_disp = adjust_contrast(save_disp)

    cv2.imwrite("save_disp.png", save_disp)
    cv2.imwrite("img_syn_backward.png", dst_img_l)

if __name__ == "__main__":
    main()