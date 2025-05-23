import cv2
import numpy as np

def load_images(color_path, depth_path):
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)
    return color, depth

def dibr_synthesize(color_img, depth_map, fx=525.0, baseline=0.05):
    h, w = depth_map.shape
    virtual_img = np.zeros_like(color_img)

    # 建立像素座標網格
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # 計算位移
    z = depth_map.copy()
    z[z == 0] = 1  # 避免除以零
    dx = (baseline * fx) / z
    x_virtual = (x_coords + dx).astype(np.int32)

    # 過濾在畫面內的像素
    valid = (x_virtual >= 0) & (x_virtual < w)
    y_valid = y_coords[valid]
    x_valid = x_coords[valid]
    x_virtual_valid = x_virtual[valid]

    virtual_img[y_valid, x_virtual_valid] = color_img[y_valid, x_valid]

    return virtual_img

if __name__ == "__main__":
    color_img, depth_map = load_images("color.png", "depth.png")
    virtual_view = dibr_synthesize(color_img, depth_map, fx=525.0, baseline=0.05)

    cv2.imwrite("virtual_view.png", virtual_view)
    cv2.imshow("Virtual View", virtual_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
