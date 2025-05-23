import cv2
import numpy as np

def warp_to_middle_view(left_img, right_img, disp_left, disp_right, alpha=0.5, background=None):
    H, W = disp_left.shape
    middle_img = np.zeros_like(left_img)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # 左視角warp
    for y in range(H):
        for x in range(W):
            d = disp_left[y, x]
            if d <= 0:
                continue
            x_mid = int(round(x - alpha * d))
            if 0 <= x_mid < W:
                middle_img[y, x_mid] = background[y, x]
                # weight_map[y, x_mid] += 1.0

    # # 右視角warp
    # for y in range(H):
    #     for x in range(W):
    #         d = disp_right[y, x]
    #         if d <= 0:
    #             continue
    #         x_mid = int(round(x + (1 - alpha) * d))
    #         if 0 <= x_mid < W:
    #             if weight_map[y, x_mid] == 0:
    #                 middle_img[y, x_mid] = right_img[y, x]
    #                 weight_map[y, x_mid] += 1.0
    #             else:
    #                 # 平均融合
    #                 middle_img[y, x_mid] = ((middle_img[y, x_mid].astype(np.float32) * weight_map[y, x_mid]) + right_img[y, x].astype(np.float32)) / (weight_map[y, x_mid] + 1)
    #                 weight_map[y, x_mid] += 1.0

    # 空洞填補: 沒有投影的地方用背景圖補
    # for i in range (middle_img.shape[0]):
    #     for j in range (middle_img.shape[1]):
    #         if weight_map[i][j] == 0:
    #             middle_img[i][j] = background[i][j]

    return middle_img.astype(np.uint8)

def main():
    # 讀取圖像
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    background = cv2.imread('background.jpg')

    if left_img is None or right_img is None or background is None:
        print("Error: Could not load one or more input images.")
        return

    # 讀取視差
    disp_left = cv2.imread('disparity_left.png', cv2.IMREAD_GRAYSCALE)
    disp_right = cv2.imread('disparity_right.png', cv2.IMREAD_GRAYSCALE)

    # 產生中間視角圖
    alpha = 0.5  # 中間視角，你可以調整 0~1
    middle_view = warp_to_middle_view(left_img, right_img, disp_left, disp_right, alpha, background)

    # 存檔
    cv2.imwrite('middle_view.jpg', middle_view)
    print("Saved middle view image to 'middle_view.png'")

if __name__ == '__main__':
    main()
