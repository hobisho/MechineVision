import cv2
import numpy as np

img = cv2.imread('photo.jpg') 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# 找區域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

#刪掉小區域
mask = np.zeros(binary.shape, dtype=np.uint8)
for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 50000:
        mask[labels == i] = 255

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_mask = np.zeros_like(mask)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

result = np.zeros_like(img)
result[filled_mask == 255] = img[filled_mask == 255]

cv2.imwrite('filled_mask.jpg', result)
cv2.imshow('Result Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
