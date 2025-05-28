import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D
from background import ImgShift
import cv2

# ---------- 參數設定 ----------
BASELINE = 0.1           # 雙目基線（公尺）
FOCAL_LENGTH = 525.0     # 相機焦距（像素）
DEPTH_SCALE = 1.0        # 0~255 深度值映射至幾公尺
DOWNSAMPLE = 4           # 點雲下採樣倍率

# ---------- 前向投影為 stack ----------
def forward_warp_to_stack(depth_map, color_img, baseline, focal_length, shift):
    h, w = depth_map.shape
    depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 1e-3:
                disparity = (baseline * focal_length) / d
                x_virtual = int(round(x + shift * disparity))
                if 0 <= x_virtual < w:
                    depth_stack[y][x_virtual].append(d)
                    color_stack[y][x_virtual].append(color_img[y, x])

    filled_depth = median_filter(depth_map, size=3)
    for y in range(h):
        for x in range(w):
            if not depth_stack[y][x]:
                depth_stack[y][x].append(filled_depth[y, x])
                color_stack[y][x].append(color_img[y, x])

    return depth_stack, color_stack

def load_depth_and_color(depth_path, color_path):
    depth_raw = imageio.imread(depth_path).astype(np.float32)
    depth_map = (255.0 - depth_raw) / 255.0 * DEPTH_SCALE
    color_image = imageio.imread(color_path)
    if color_image.ndim == 2:
        color_image = np.stack([color_image] * 3, axis=-1)
    elif color_image.shape[2] == 4:
        color_image = color_image[:, :, :3]
    return depth_map, color_image, depth_raw

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

# ---------- 顯示圖像視覺結果 ----------
def show_virtual_views(left_stack_color, right_stack_color, merged_stack_color, left_stack_depth=None, right_stack_depth=None, merged_stack_depth=None):
    def frontmost_color(stack_color, stack_depth):
        h, w = len(stack_color), len(stack_color[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                colors = stack_color[y][x]
                depths = stack_depth[y][x] if stack_depth and stack_depth[y][x] else [1] * len(colors)  # 使用深度，若為 None 或空則設為 1
                if colors:
                    min_idx = np.argmin(depths)
                    img[y, x] = np.array(colors[min_idx], dtype=np.uint8)
        return img

    left_img = frontmost_color(left_stack_color, left_stack_depth)
    right_img = frontmost_color(right_stack_color, right_stack_depth)
    merged_img = frontmost_color(merged_stack_color, merged_stack_depth)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Warped Left View")
    plt.imshow(left_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Warped Right View")
    plt.imshow(right_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Merged Virtual View")
    plt.imshow(merged_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ---------- 計算重心 ----------
def compute_centroid_from_stack(depth_stack, threshold=0.5):
    h, w = len(depth_stack), len(depth_stack[0])
    xs, ys ,zs= [], [], []

    for y in range(h):
        for x in range(w):
            for d in depth_stack[y][x]:
                if  0.3 < d < threshold:
                    xs.append(x)
                    ys.append(y)
                    zs.append(d)

    if not xs:
        print("⚠️ 找不到深度 > {:.2f} 的有效點".format(threshold))
        return None

    # 計算重心
    print(f"✅ 重心位置 (X, Y, Z)：{np.mean(xs).astype(int),np.mean(ys).astype(int),np.mean(zs)}")

    return np.mean(xs).astype(int),np.mean(zs)

# ---------- 深度堆疊對齊 ----------
def shift_points_in_stack(depth_stack, color_stack, midcenter_x, midcenter_z, threshold=0.3, constant=1):
    h, w = len(depth_stack), len(depth_stack[0])
    new_depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    new_color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            for d, c in zip(depth_stack[y][x], color_stack[y][x]):
                if d < threshold:
                    shift_x = midcenter_x *(1-d)/(1-midcenter_z)
                    new_x = x + int(constant * shift_x)
                    if 0 <= new_x < w:
                        new_depth_stack[y][new_x].append(d)
                        new_color_stack[y][new_x].append(c)
                else:
                    new_depth_stack[y][x].append(d)
                    new_color_stack[y][x].append(c)
    
    return new_depth_stack, new_color_stack


def merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color, h, w):
    merged_depth_stack = [[[] for _ in range(w)] for _ in range(h)]
    merged_color_stack = [[[] for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            left_depths = left_shifted_depth[y][x]
            right_depths = right_shifted_depth[y][x]
            left_colors = left_shifted_color[y][x]
            right_colors = right_shifted_color[y][x]

            if not left_depths and not right_depths:
                # 空洞，兩邊都沒點，留空待後續補洞
                continue

            elif left_depths and not right_depths:
                # 只有左邊有點，全部保留
                merged_depth_stack[y][x].extend(left_depths)
                merged_color_stack[y][x].extend(left_colors)

            elif right_depths and not left_depths:
                # 只有右邊有點，全部保留
                merged_depth_stack[y][x].extend(right_depths)
                merged_color_stack[y][x].extend(right_colors)

            else:
                # 兩邊都有點，針對每對深度點做加權合併
                # 這裡假設左右兩邊點數相同且一一對應（可根據需求調整）
                n = min(len(left_depths), len(right_depths))
                for i in range(n):
                    ld = left_depths[i]
                    rd = right_depths[i]
                    lc = np.array(left_colors[i])
                    rc = np.array(right_colors[i])

                    # 計算權重，深度越小權重越大 (防止除零加小常數)
                    w_left = 1.0 / ((ld + 1e-5) ** 3)
                    w_right = 1.0 / ((rd + 1e-5) ** 3)

                    w_sum = w_left + w_right
                    w_left /= w_sum
                    w_right /= w_sum

                    merged_depth = (ld * w_left + rd * w_right)
                    merged_color = (lc * w_left + rc * w_right).astype(np.uint8)

                    merged_depth_stack[y][x].append(merged_depth)
                    merged_color_stack[y][x].append(merged_color.tolist())

                # 若兩邊點數不等，保留多出來的點
                if len(left_depths) > n:
                    for i in range(n, len(left_depths)):
                        merged_depth_stack[y][x].append(left_depths[i])
                        merged_color_stack[y][x].append(left_colors[i])

                if len(right_depths) > n:
                    for i in range(n, len(right_depths)):
                        merged_depth_stack[y][x].append(right_depths[i])
                        merged_color_stack[y][x].append(right_colors[i])

    return merged_depth_stack, merged_color_stack

# ---------- 在 warp 前先左右平移 ----------
def shift_image_horizontal(img, shift_amount):
    h, w = img.shape[:2]

    if len(img.shape) == 2:  # 灰階或單通道（如 depth）
        shifted = np.zeros_like(img)
    else:  # 彩色圖
        shifted = np.zeros_like(img)

    if shift_amount > 0:
        shifted[:, shift_amount:] = img[:, :w - shift_amount]
    elif shift_amount < 0:
        shifted[:, :w + shift_amount] = img[:, -shift_amount:]
    else:
        shifted = img.copy()

    return shifted



if __name__ == "__main__":
    # 載入左右視角
    path = "./final/image/bbox"
    left_depth, left_color, _ = load_depth_and_color("./final/output/disparity_left.jpg", f"{path}_left_left.jpg")
    right_depth, right_color, _ = load_depth_and_color("./final/output/disparity_right.jpg", f"{path}_right_right.jpg")
    print(left_depth.shape)
    
    shift ,error= ImgShift(left_color, right_color)  # 假設 ImgShift 函數已經定義並返回位移量
    print(shift)
    shift = int(102/2)
    left_color_shifted = shift_image_horizontal(left_color, shift)
    right_color_shifted = shift_image_horizontal(right_color, -shift)
    left_depth_shifted = shift_image_horizontal(left_depth, shift)
    right_depth_shifted = shift_image_horizontal(right_depth, -shift)
    cv2.imshow("f",right_color_shifted)
    cv2.imshow("hi",left_color_shifted)
    cv2.waitKey(0)

    # warp 成 stack
    # left_stack_depth, left_stack_color = forward_warp_to_stack(left_depth_shifted, left_color_shifted, BASELINE, FOCAL_LENGTH, shift=0)
    # right_stack_depth, right_stack_color = forward_warp_to_stack(right_depth_shifted, right_color_shifted, BASELINE, FOCAL_LENGTH, shift=0)

    # center_left_x,center_left_z = compute_centroid_from_stack(left_stack_depth, threshold=0.4)
    # center_right_x,center_right_z = compute_centroid_from_stack(right_stack_depth, threshold=0.4)
    # midcenter_x = int(abs(center_left_x - center_right_x) / 2)
    # midcenter_z = (center_left_z + center_right_z) / 2
    # print(midcenter_x)

    # left_shifted_depth, left_shifted_color = shift_points_in_stack(left_stack_depth, left_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=-1)
    # right_shifted_depth, right_shifted_color = shift_points_in_stack(right_stack_depth, right_stack_color, midcenter_x, midcenter_z,threshold=0.5,constant=1)

    # # 合併 stack
    # h, w = left_depth.shape
    # merged_depth_stack, merged_color_stack = merge_stacks(left_shifted_depth, left_shifted_color, right_shifted_depth, right_shifted_color, h, w)

    # # 顯示虛擬視角圖像
    # print("⚠️ 顯示虛擬視角圖像")
    # show_virtual_views(left_stack_color, right_stack_color, merged_color_stack)

    # print("⚠️ 顯示點雲")
    # print(merged_depth_stack[0][0])
    # show_point_cloud_from_stack(merged_depth_stack, merged_color_stack, downsample=DOWNSAMPLE)
