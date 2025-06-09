import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from random import randint

def func(gray):
    color_i = 0
    # # 可選：先模糊降低雜訊
    # blur = cv2.GaussianBlur(gray, (3, 3), 1.4)

    # Canny 邊緣檢測
    edges = cv2.Canny(gray, 100, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾掉面積太小的輪廓
    min_area = 100  # 調整這個閾值來過濾小區塊
    filtered = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(filtered, [cnt], -1, (
                255 if color_i == 0 else 0,
                255 if color_i == 1 else 0,
                255 if color_i == 2 else 0,
            ), thickness=cv2.FILLED)  # 填滿輪廓區域
            
            color_i += 1
            color_i %= 3
    
    return filtered

# 讀入圖片（轉灰階效果更好）
path = "image/bbox"
img1 = cv2.imread(f"{path}_left_left.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"{path}_right_right.jpg", cv2.IMREAD_GRAYSCALE)

# ORB 特徵點偵測與描述子
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img2, None)
kp2, des2 = orb.detectAndCompute(img1, None)

# 特徵匹配（使用 Hamming 距離 + crossCheck）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 過濾 dy 超過 5 的點
dx_list = []
dy_list = []
filtered_matches = []
for m in matches:
    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    if abs(dy) <= 5:
        dx_list.append(dx)
        dy_list.append(dy)
        filtered_matches.append(m)

# 計算中間 50% 平均
def middle_50_percent_mean(data):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    q1 = int(n * 0.25)
    q3 = int(n * 0.75)
    return np.mean(data_sorted[q1:q3])

if len(dx_list) == 0:
    print("⚠️ 過濾後沒有匹配點")
    avg_dx = 0
    avg_dy = 0
else:
    avg_dx = middle_50_percent_mean(dx_list)
    avg_dy = middle_50_percent_mean(dy_list)

print(f"📐 過濾後中間 50% 平均位移量：Δx = {avg_dx:.2f}, Δy = {avg_dy:.2f}")

# ====== KMeans 分群以找出最近的群主群組 ======
dx_array = np.array(dx_list).reshape(-1, 1)

if len(dx_array) < 5:
    print("⚠️ 匹配點太少，無法分群")
else:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dx_array)
    labels = kmeans.labels_
    dx_array = dx_array.flatten()

    # 計算每群 Δx 平均值
    groups = {}
    for i in range(2):
        group_dx = dx_array[labels == i]
        groups[i] = {
            "values": group_dx,
            "mean": np.mean(group_dx),
            "std": np.std(group_dx),
            "size": len(group_dx)
        }

    # ✅ 改成挑 Δx 最小（數值最小的群，不是變異數）
    min_mean_group_id = min(groups, key=lambda k: groups[k]["mean"])
    min_group = groups[min_mean_group_id]

    print(f"🎯 最集中群的資料數：{min_group['size']}")
    print(f"📏 最集中群 Δx 平均：{min_group['mean']:.2f}")
    print(f"📊 最集中群 Δx 標準差：{min_group['std']:.2f}")

# ====== 匹配點視覺化 ======
filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(func(img2), kp1, func(img1), kp2, filtered_matches[:], None, flags=2)
plt.figure(figsize=(12, 6))
plt.imshow(img_match)
plt.title("Filtered Matches (|Δy| ≤ 5)")
plt.axis("off")
plt.show()
