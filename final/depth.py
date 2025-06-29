import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------- 參數設定 ----------
BASELINE = 0.3           # 你拍照時的左右鏡頭或移動距離（公尺），如果是手機左右移拍，請自行量尺
FOCAL_LENGTH = 1500.0    # 你照片如果接近1920x1080，填1500，大約是照片寬的0.75倍（手機大多是1000~2000）
DEPTH_SCALE = 5.0        # 先預設1.0，之後用實物距離再微調
DOWNSAMPLE = 4

def ErodeDilate(img):
    medianBlur_img = cv2.medianBlur(img, 9)  # 平滑視差圖，去除雜訊
    kernel = np.ones((3, 3), np.uint8)
    erode_img = cv2.erode(medianBlur_img, kernel, iterations=5)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=15)
    finish_img = cv2.erode(dilate_img, kernel, iterations=10)
    return finish_img


def depth(imgLL, imgLR, imgRL, imgRR):
    # 左到右視差圖（已確認可行）
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=64 * 2 * 5 ** 2,
        disp12MaxDiff=1,
        speckleWindowSize=200,
        speckleRange=32
    )
    disparity_left = stereo_left.compute(imgLL, imgLR).astype(np.float32) / 16.0
    disparity_left[disparity_left < 0] = 0
    disp_normalized_left = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_8u_left = np.uint8(disp_normalized_left)

    # 右到左視差圖（調整參數）
    Disparity = 16 * 10
    stereo_right = cv2.StereoSGBM_create(
        minDisparity=-Disparity,
        numDisparities=Disparity,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=64 * 2 * 5 ** 2,
        disp12MaxDiff=1,
        speckleWindowSize=200,
        speckleRange=32
    )
    disparity_right = stereo_right.compute(imgRR, imgRL).astype(np.float32) / 16.0
    disparity_right[disparity_right < -Disparity] = 0  # 過濾不合理值
    disp_normalized_right = -cv2.normalize(disparity_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_8u_right = np.uint8(disp_normalized_right)
    
    d = 0.3664
    w,h=disp_8u_left.shape
    disp_resize_left = cv2.resize(disp_8u_left, (int(d*h), int(d*w)))
    disp_resize_right = cv2.resize(disp_8u_right, (int(d*h), int(d*w)))
    cv2.imwrite(f"./final/output/disparity_left.jpg", disp_resize_left)
    cv2.imwrite(f"./final/output/disparity_right.jpg", disp_resize_right)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Disparity Left")
    plt.imshow(disp_resize_left, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Disparity Right")
    plt.imshow(disp_resize_right, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "./final/image/box"
    imgLL = cv2.imread(f"{path}_left_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgLR = cv2.imread(f"{path}_left_right.jpg", cv2.IMREAD_GRAYSCALE)
    imgRL = cv2.imread(f"{path}_right_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgRR = cv2.imread(f"{path}_right_right.jpg", cv2.IMREAD_GRAYSCALE)

    depth(imgLL, imgLR, imgRL, imgRR)
