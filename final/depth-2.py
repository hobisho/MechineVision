import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- 參數設定 ----------
BASELINE = 0.3           # 拍照時相機左右移動的距離（公尺），請自行量尺
FOCAL_LENGTH = 1500.0    # 根據照片解析度估算，約為寬度的0.75倍
DEPTH_SCALE = 5.0        # 依實際深度對應距離自行調整
DOWNSAMPLE = 4           # 點雲下採樣倍率

def ErodeDilate(img):
    medianBlur_img = cv2.medianBlur(img, 9)
    kernel = np.ones((3, 3), np.uint8)
    erode_img = cv2.erode(medianBlur_img, kernel, iterations=5)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=15)
    finish_img = cv2.erode(dilate_img, kernel, iterations=10)
    return finish_img

def compute_disparity(img_left, img_right):
    window_size = 1
    min_disp = 16
    num_disp = 192-min_disp
    blockSize = window_size
    uniquenessRatio = 0
    speckleRange = 50
    speckleWindowSize = 0
    disp12MaxDiff = 250
    P1 = 600
    P2 = 2400
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = P1,
        P2 = P2
    )
    disp = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    return disp

if __name__ == "__main__":
    window_size = 1
    min_disp = 16
    num_disp = 192-min_disp
    blockSize = window_size
    uniquenessRatio = 0
    speckleRange = 50
    speckleWindowSize = 0
    disp12MaxDiff = 250
    
    path = "./final/image/bbox"
    imgLL = cv2.imread(f"{path}_left_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgLR = cv2.imread(f"{path}_left_right.jpg", cv2.IMREAD_GRAYSCALE)
    imgRL = cv2.imread(f"{path}_right_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgRR = cv2.imread(f"{path}_right_right.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 計算視差圖
    disp_left = compute_disparity(imgLL, imgRR)
    
    # 適當做形態學處理（去雜點、連續性）
    disp_left = ErodeDilate(disp_left)
    
    # 儲存與顯示
    cv2.imwrite("./final/output/disparity_left.jpg", (disp_left-min_disp)/num_disp)
    plt.imshow(disp_left, cmap='gray')
    plt.title("Disparity Map")
    plt.axis('off')
    plt.show()
