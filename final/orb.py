import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀入圖片（轉灰階效果更好）
path = "./final/image/bbox"
img1 = cv2.imread(f"{path}_left_left.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"{path}_right_right.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 特徵匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 取匹配點的位移量，並過濾 dy 超過 5 的點
dx_list = []
dy_list = []
filtered_matches = []
for m in matches:
    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    if abs(dy) <= 5:  # 過濾條件
        dx_list.append(dx)
        dy_list.append(dy)
        filtered_matches.append(m)

# 將位移排序後取中間 50%
def middle_50_percent_mean(data):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    q1 = int(n * 0.1)
    q3 = int(n * 0.2)
    return np.mean(data_sorted[q1:q3])

if len(dx_list) == 0:
    print("⚠️ 過濾後沒有匹配點")
    avg_dx = 0
    avg_dy = 0
else:
    avg_dx = middle_50_percent_mean(dx_list)
    avg_dy = middle_50_percent_mean(dy_list)

print(f"📐 過濾後中間 50% 平均位移量：Δx = {avg_dx:.2f}, Δy = {avg_dy:.2f}")

# 可視化過濾後匹配點（最多前 50 個）
filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches[:], None, flags=2)
plt.imshow(img_match)
plt.title("Filtered Matches (|Δy| ≤ 5)")
plt.axis("off")
plt.show()
