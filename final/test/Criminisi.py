import cv2
import numpy as np
from tqdm import tqdm
import time

# === 參數設定 ===
left_img_path = "left.jpg"
right_img_path = "right.jpg"

# === 讀取灰階影像 ===
imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
    raise ValueError("無法讀取左或右影像，請確認檔案路徑是否正確")

# === SGBM 視差參數設定 ===
window_size = 5
min_disp = 0
num_disp = 16 * 6  # 必須是16的倍數

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

# === 計算視差圖，顯示進度條 ===
print("計算視差圖中...")
disparity_raw = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# 模擬計算進度條（真實計算是一次完成，這裡演示用）
for _ in tqdm(range(100), desc="視差計算進度"):
    time.sleep(0.01)

# 過濾不合理視差
disparity = disparity_raw.copy()
disparity[disparity < 0] = 0

# === 建立缺失區域遮罩（視差=0處） ===
mask = np.uint8(disparity == 0) * 255  # 0~255遮罩

# === 進行 Inpainting 修補視差圖缺失區域 ===
print("Inpainting 修補中...")
# 這裡用 TELEA 演算法，另一個選擇是 Navier-Stokes 演算法
disparity_inpaint = cv2.inpaint(disparity.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)

# 模擬 Inpainting 進度條
for _ in tqdm(range(100), desc="Inpainting進度"):
    time.sleep(0.01)

# === 超分辨率提升 ===
# 示意用雙三次插值將視差圖放大2倍
print("進行超分辨率提升...")
scale_factor = 2
height, width = disparity_inpaint.shape
new_size = (width * scale_factor, height * scale_factor)

# 使用雙三次插值
disparity_sr = cv2.resize(disparity_inpaint, new_size, interpolation=cv2.INTER_CUBIC)

for _ in tqdm(range(100), desc="超分辨率進度"):
    time.sleep(0.01)

# === 歸一化結果方便顯示 ===
disp_normalized = cv2.normalize(disparity_sr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_8u = np.uint8(disp_normalized)

# === 顯示結果 ===
cv2.imshow("Disparity Super-Resolved and Inpainted", disp_8u)
cv2.imwrite("disparity_sr_inpainted.png", disp_8u)
cv2.waitKey(0)
cv2.destroyAllWindows()
