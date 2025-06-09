import cv2
import numpy as np

def ErodeDilate(img):
    medianBlur_img = cv2.medianBlur(img, 5)  # 平滑視差圖，去除雜訊
    kernel = np.ones((3, 3), np.uint8)
    morph_img = cv2.morphologyEx(medianBlur_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph_img


def depth(imgLL, imgLR, imgRL, imgRR):
    # 左到右視差圖（已確認可行）
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
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
    disp_resize_left = cv2.resize(disp_8u_left, (int(round(d*h)), int(round(d*w))))
    disp_resize_right = cv2.resize(disp_8u_right, (int(round(d*h)), int(round(d*w))))
    cv2.imwrite(f"output/disparity_left.jpg", disp_resize_left)
    cv2.imwrite(f"output/disparity_right.jpg", disp_resize_right)

if __name__ == "__main__":
    path = "image/box"
    imgLL = cv2.imread(f"{path}_left_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgLR = cv2.imread(f"{path}_left_right.jpg", cv2.IMREAD_GRAYSCALE)
    imgRL = cv2.imread(f"{path}_right_left.jpg", cv2.IMREAD_GRAYSCALE)
    imgRR = cv2.imread(f"{path}_right_right.jpg", cv2.IMREAD_GRAYSCALE)

    depth(imgLL, imgLR, imgRL, imgRR)
