import cv2
import numpy as np
from sklearn.cluster import KMeans

def analyze_displacement(img1, img2):

    orb = cv2.ORB_create(nfeatures=1000)
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
    else:
        avg_dx = middle_50_percent_mean(dx_list)
        avg_dy = middle_50_percent_mean(dy_list)

    dx_array = np.array(dx_list).reshape(-1, 1)
    if len(dx_array) < 5:
        cluster_mean = None
    else:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dx_array)
        labels = kmeans.labels_
        dx_array = dx_array.flatten()

        groups = {}
        for i in range(2):
            group_dx = dx_array[labels == i]
            groups[i] = np.mean(group_dx)

        min_mean_group_id = min(groups, key=groups.get)
        cluster_mean = groups[min_mean_group_id]

    # 回傳絕對值，去掉負號
    return abs(cluster_mean) if cluster_mean is not None else None, abs(avg_dx)

# 範例用法
if __name__ == "__main__":
    left_path = "./final/image/box_left_left.jpg"
    right_path = "./final/image/box_right_right.jpg"
    img1 = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    cluster_mean, avg_dx = analyze_displacement(img1, img2)
    print(f"最集中群 Δx 平均 (絕對值): {cluster_mean}")
    print(f"整體中間 50% 平均 Δx (絕對值): {avg_dx}")
