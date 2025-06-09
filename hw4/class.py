import cv2
import numpy as np

img = cv2.imread('photo.jpg') 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 找葡萄
_, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

mask = np.zeros(binary.shape, dtype=np.uint8)
for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 50000:
        mask[labels == i] = 255

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_mask = np.zeros_like(mask)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

result = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if filled_mask[i, j] == 255:
            result[i, j] = img[i, j]

# remove 白色再做一次面積大小塞選
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, white_binary = cv2.threshold(gray_result, 200, 255, cv2.THRESH_BINARY)

num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(white_binary, connectivity=8)

white_rm = np.zeros_like(white_binary)
for i in range(1, num_labels2):
    area = stats2[i, cv2.CC_STAT_AREA]
    if area >= 250:
        white_rm[labels2 == i] = 255

final_result = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if white_rm[i, j] == 0:
            final_result[i, j] = result[i, j]


# 顯示結果
cv2.imwrite('grapes.png', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
