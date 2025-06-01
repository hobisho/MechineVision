import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

path = "./final/image/2image/tv"
imgL = cv2.imread(f"{path}_left.jpg", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(f"{path}_right.jpg", cv2.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
    print("影像讀取失敗，請確認檔名與路徑")
    exit()

# 建立StereoBM物件
numDisparities = 16*20  # 必須是16的倍數，設定視差範圍
blockSize = 15         # 區塊大小，調整影響結果平滑度與細節

stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# 計算視差圖（16位元固定點，需除以16）
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# 將視差值限制範圍（避免負值影響）
disparity[disparity < 0] = 0

# 正規化視差到0~255
disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_norm = np.uint8(disparity_norm)

# 顯示視差圖
cv2.imshow('Disparity Map (0~255)', disparity_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 或用matplotlib顯示
plt.imshow(disparity_norm, cmap='gray')
plt.title('Disparity Map (0~255)')
plt.axis('off')
plt.show()
