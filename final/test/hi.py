import cv2
import numpy as np

# === 載入左右影像（彩色） + 計算視差（用灰階） ===
imgL_color = cv2.imread("left.jpg", cv2.IMREAD_COLOR)
imgR_color = cv2.imread("right.jpg", cv2.IMREAD_COLOR)

imgL_gray = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)

# 計算視差圖（用 OpenCV SGBM）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0
disparity[disparity < 0] = 0  # 去除無效視差

# === 3D-Warping 變換左/右圖到虛擬視角（支援彩色） ===
def warp_image_color(image, disparity, alpha):
    """
    彩色版本 warp，image shape: (h,w,3)
    disparity shape: (h,w)
    """
    h, w = disparity.shape
    warped = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            d = disparity[y, x]
            if d > 0:
                shift = int(round((0.5 - alpha) * d))
                new_x = x + shift
                if 0 <= new_x < w:
                    warped[y, new_x] = image[y, x]
                    mask[y, new_x] = 1
    return warped, mask

# alpha = 0.5 表示左右平均中間視角
virtualL, maskL = warp_image_color(imgL_color, disparity, alpha=0.5)
virtualR, maskR = warp_image_color(imgR_color, -disparity, alpha=0.5)

# === Blending 左右虛擬圖 ===
blended = np.zeros_like(virtualL)
blended_mask = maskL + maskR

h, w = blended_mask.shape
for y in range(h):
    for x in range(w):
        if blended_mask[y, x] == 2:
            blended[y, x] = ((virtualL[y, x].astype(np.uint16) + virtualR[y, x].astype(np.uint16)) // 2).astype(np.uint8)
        elif maskL[y, x]:
            blended[y, x] = virtualL[y, x]
        elif maskR[y, x]:
            blended[y, x] = virtualR[y, x]

# === 輸出還沒填補的融合圖 ===
cv2.imwrite("virtual_view_unfilled.jpg", blended)

# === Inpainting 處理空洞（彩色）===
holes = (blended_mask == 0).astype(np.uint8) * 255
blended_inpaint = cv2.inpaint(blended, holes, 3, cv2.INPAINT_TELEA)

# === 顯示與儲存填補後結果 ===
cv2.imwrite("virtual_view_color.jpg", blended_inpaint)
cv2.imshow("Virtual View Color", blended_inpaint)
cv2.waitKey(0)
cv2.destroyAllWindows()

