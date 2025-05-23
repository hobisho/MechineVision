import cv2
import numpy as np

left_img_path = "left.jpg"
right_img_path = "right.jpg"

imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
    raise ValueError("無法讀取左或右影像，請確認檔案路徑是否正確")

# 左到右視差圖（已確認可行）
stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 6,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disparity_left = stereo_left.compute(imgL, imgR).astype(np.float32) / 16.0
disparity_left[disparity_left < 0] = 0
disp_normalized_left = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_8u_left = np.uint8(disp_normalized_left)

# 右到左視差圖（調整參數）
stereo_right = cv2.StereoSGBM_create(
    minDisparity=-16 * 6,  # 允許負視差
    numDisparities=16 * 6,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disparity_right = stereo_right.compute(imgR, imgL).astype(np.float32) / 16.0
disparity_right[disparity_right < -16 * 6] = 0  # 過濾不合理值
disp_normalized_right = -cv2.normalize(disparity_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_8u_right = np.uint8(disp_normalized_right)
disp_8u_left = cv2.medianBlur(disp_8u_left, 9)  # 平滑視差圖，去除雜訊
disp_8u_right = cv2.medianBlur(disp_8u_right, 9)  # 平滑視差圖，去除雜訊
kernel = np.ones((3, 3), np.uint8)
disp_8u_right = cv2.erode(disp_8u_right, kernel, iterations=5)
disp_8u_left = cv2.erode(disp_8u_left, kernel, iterations=5)
disp_8u_right = cv2.dilate(disp_8u_right, kernel, iterations=15)
disp_8u_left = cv2.dilate(disp_8u_left, kernel, iterations=15)
disp_8u_right = cv2.erode(disp_8u_right, kernel, iterations=10)
disp_8u_left = cv2.erode(disp_8u_left, kernel, iterations=10)

# 顯示和儲存結果
# cv2.imshow("Left-to-Right Disparity", disp_8u_left)
# cv2.imshow("Right-to-Left Disparity", disp_8u_right)
cv2.imwrite("disparity_left.png", disp_8u_left)
cv2.imwrite("disparity_right.png", disp_8u_right)
cv2.waitKey(0)
cv2.destroyAllWindows()