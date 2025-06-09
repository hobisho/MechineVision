import cv2
import numpy as np
from sklearn.cluster import KMeans

def analyze_displacement(img1, img2):
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img2, None)
    kp2, des2 = orb.detectAndCompute(img1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

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

    def middle_50_percent_mean(data):
        data_sorted = np.sort(data)
        n = len(data_sorted)
        q1 = int(n * 0.25)
        q3 = int(n * 0.75)
        return np.mean(data_sorted[q1:q3])

    if len(dx_list) == 0:
        print("⚠️ 過濾後沒有匹配點")
        return None, None, None, None, None, None, None

    avg_dx = middle_50_percent_mean(dx_list)

    dx_array = np.array(dx_list).reshape(-1, 1)
    if len(dx_array) < 5:
        max_group_mean = None
        max_group_max = None
        max_group_min = None
        min_group_mean = None
        min_group_max = None
        min_group_min = None
    else:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dx_array)
        labels = kmeans.labels_
        dx_array = dx_array.flatten()

        groups = {}
        counts = {}
        group_values = {}
        for i in range(2):
            group_dx = dx_array[labels == i]
            groups[i] = np.mean(group_dx)
            counts[i] = len(group_dx)
            group_values[i] = group_dx

        # 最大群
        max_count_group_id = max(counts, key=counts.get)
        max_group_mean = groups[max_count_group_id]
        max_group_max = np.max(group_values[max_count_group_id])
        # 最小群
        min_count_group_id = min(counts, key=counts.get)
        min_group_mean = groups[min_count_group_id]

    return (abs(max_group_mean) if max_group_mean is not None else None,
            abs(max_group_max) if max_group_max is not None else None,
            abs(min_group_mean) if min_group_mean is not None else None,
            abs(avg_dx))

# 範例用法
if __name__ == "__main__":
    left_path = "./final/image/bbox_left_left.jpg"
    right_path = "./final/image/bbox_right_right.jpg"
    img1 = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    (max_group_mean, max_group_max, min_group_mean,
     avg_dx) = analyze_displacement(img1, img2)
    print(f"最大群 Δx 平均 (絕對值): {max_group_mean}")
    print(f"最大群 Δx 最大值 (絕對值): {max_group_max}")
    print(f"最小群 Δx 平均 (絕對值): {min_group_mean}")
    print(f"整體中間 50% 平均 Δx (絕對值): {avg_dx}")

    print("analyze_displacement done")
    print(f"shift:{min_group_mean}")
    print(f"max_shift:{((max_group_mean+max_group_max)/2-min_group_mean)/2}")
