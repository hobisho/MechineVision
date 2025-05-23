import numpy as np
import cv2

# 載入膨脹後的區塊圖（每個像素的 label）
labels = np.load('./final/output/dilated_labels.npy')

# 載入破碎的視差圖（需你換成你的視差檔案）
disparity = cv2.imread('final\output\disparity_norm.png', cv2.IMREAD_UNCHANGED).astype(np.float32)

# 建立一張新的視差圖，用來放修補結果
filled_disparity = disparity.copy()

# 遍歷每個區塊，做區域內的填補
for label in np.unique(labels):
    mask = (labels == label)
    valid_values = disparity[mask & (disparity > 0)]
    if len(valid_values) > 0:
        fill_value = np.median(valid_values)  # 也可以用 np.mean
        filled_disparity[mask & (disparity == 0)] = fill_value

# 儲存結果
cv2.imwrite('disparity_filled.png', filled_disparity.astype(np.uint8))
