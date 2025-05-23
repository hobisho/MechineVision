import cv2
import numpy as np

# 輸入影像路徑（請替換成你的檔名）
left_img_path = 'left.jpg'
right_img_path = 'right.jpg'
backgroung_left_img_path = 'backgroung_left.jpg'
backgroung_right_img_path = 'backgroung_right.jpg'

# 載入影像
img_left = cv2.imread(left_img_path)
img_right = cv2.imread(right_img_path)
img_backgroung_left = cv2.imread(backgroung_left_img_path)
img_backgroung_right = cv2.imread(backgroung_right_img_path)

# 視差估計參數
window_size = 5
min_disp = 0
num_disp = 16 * 6  # 16 的倍數

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=9,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# 灰階轉換
gray_left = cv2.cvtColor(img_backgroung_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_backgroung_right, cv2.COLOR_BGR2GRAY)

# 視差圖
disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
disparity = cv2.medianBlur(disparity, 5)  # 平滑視差圖，去除雜訊


h, w = disparity.shape
center_from_left = np.zeros_like(img_left)
center_from_right = np.zeros_like(img_right)

# 建立中央視角圖（左圖向右移一半視差，右圖向左移一半視差）
for y in range(h):
    for x in range(w):
        d = disparity[y, x]
        if d > 0:
            xl = int(x - d / 2)
            xr = int(x + d / 2)
            if 0 <= xl < w:
                center_from_left[y, xl] = img_left[y, x]
            if 0 <= xr < w:
                center_from_right[y, xr] = img_right[y, x]

# 融合兩張中央圖（左右）
center_fused = cv2.addWeighted(center_from_left, 0.5, center_from_right, 0.5, 0)

background = cv2.imread('background.jpg')

threshold = 10
mask = np.all(center_fused < threshold, axis=2)  # shape=(h,w), True代表黑色區域

center_fused[mask] = background[mask]

# 建立遮罩（找出黑區域，即空洞）
mask = cv2.cvtColor(center_fused, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)[1]

# 修補破洞：inpainting
inpainted = cv2.inpaint(center_fused, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 儲存與顯示
cv2.imwrite("center_from_left.png", center_from_left)
cv2.imwrite("center_from_right.png", center_from_right)
cv2.imwrite("center_view_inpainted.png", inpainted)
cv2.imwrite("center_fused.png", center_fused)
cv2.imwrite("disparity.png", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
