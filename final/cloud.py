import numpy as np
import matplotlib.pyplot as plt

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 儲存點雲為 PLY ----------
def save_point_cloud_as_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r*255)} {int(g*255)} {int(b*255)}\n")


# ---------- 顯示點雲 ----------
def show_point_cloud_from_stack(depth_stack, color_stack, focal_length=FOCAL_LENGTH, downsample=4):
    h, w = len(depth_stack), len(depth_stack[0])
    points = []
    colors = []

    for y in range(0, h, downsample):
        for x in range(0, w, downsample):
            for d, c in zip(depth_stack[y][x], color_stack[y][x]):
                if  d > 1e-3: #isinstance(d, (float, int)) and
                    z = d
                    x3d = (x - w / 2) * z / focal_length
                    y3d = (y - h / 2) * z / focal_length
                    points.append([x3d, -y3d, z])
                    colors.append(np.array(c) / 255.0)

    if not points:
        print("⚠️ 沒有有效的點雲資料")
        return

    points = np.array(points)
    colors = np.array(colors)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)
    # ax.view_init(elev=90, azim=-90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title("3D Point Cloud from Warp Stack")
    plt.tight_layout()
    plt.show()
    save_point_cloud_as_ply("point_cloud.ply", points, colors)

if __name__ == "__main__":
    # 模擬深度和顏色堆疊數據
    depth_stack = np.random.rand(480, 640, 10) * 10  # 假設深度值在0~10公尺之間
    color_stack = np.random.randint(0, 256, (480, 640, 10, 3))  # 隨機顏色

    # 顯示點雲
    show_point_cloud_from_stack(depth_stack, color_stack, downsample=DOWNSAMPLE)
