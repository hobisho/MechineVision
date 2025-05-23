import cv2
import numpy as np
from test.inpainter import inpaint  

def warp_image(image, disparity, alpha=0.5):
    h, w = image.shape
    warped = np.zeros_like(image)
    mask = np.zeros_like(image, dtype=np.uint8)

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

def main():
    left_img_path = "left.jpg"
    right_img_path = "right.jpg"

    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        raise ValueError("無法讀取左或右影像，請確認檔案路徑是否正確")

    window_size = 5
    min_disp = 0
    num_disp = 16 * 6

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity[disparity < 0] = 0

    virtualL, maskL = warp_image(imgL, disparity, alpha=0.5)
    virtualR, maskR = warp_image(imgR, -disparity, alpha=0.5)

    blended = np.zeros_like(virtualL)
    blended_mask = maskL + maskR

    for y in range(blended.shape[0]):
        for x in range(blended.shape[1]):
            if blended_mask[y, x] == 2:
                blended[y, x] = (virtualL[y, x] + virtualR[y, x]) // 2
            elif maskL[y, x]:
                blended[y, x] = virtualL[y, x]
            elif maskR[y, x]:
                blended[y, x] = virtualR[y, x]

    holes = (blended_mask == 0).astype(np.uint8)
    blended_inpaint = inpaint(blended, holes)

    cv2.imwrite("virtual_view.jpg", blended_inpaint)
    cv2.imshow("Virtual View", blended_inpaint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
