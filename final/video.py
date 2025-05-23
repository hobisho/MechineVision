import numpy as np
import open3d as o3d
import cv2

def interpolate_extrinsics(extrinsic_start, extrinsic_end, steps):
    """線性插值兩個4x4外參矩陣（只針對平移部分插值）"""
    extrinsics = []
    for i in range(steps):
        alpha = i / (steps - 1)
        # 線性插值旋轉會有問題，這裡假設旋轉固定，平移線性插值
        R = extrinsic_start[:3, :3]  # 固定旋轉
        t_start = extrinsic_start[:3, 3]
        t_end = extrinsic_end[:3, 3]
        t = (1 - alpha) * t_start + alpha * t_end

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        extrinsics.append(extrinsic)
    return extrinsics

# 你先定義左右視角外參
extrinsic_left = ...  # 4x4 numpy array
extrinsic_right = ... # 4x4 numpy array

# 固定相機內參
intrinsic = ...  # 3x3 numpy array

width, height = 640, 480
steps = 30

# 生成插值外參序列
extrinsics = interpolate_extrinsics(extrinsic_left, extrinsic_right, steps)

# 讀入點雲 pcd (Open3D格式)
pcd = o3d.io.read_point_cloud('point_cloud.ply')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output.mp4', fourcc, 15, (width, height))

for extrinsic in extrinsics:
    img = project_point_cloud_to_image(pcd, intrinsic, extrinsic, width, height)
    video_writer.write(img)

video_writer.release()
print("影片已儲存為 output.mp4")
