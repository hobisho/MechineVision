import numpy as np
import cv2
from PIL import Image

def compute_disparity_and_remove_bg(img_path, num_disparities=96, block_size=11, disparity_threshold=30,name=""):
    # 讀取圖片
    image = Image.open(img_path)
    width, height = image.size

    # 分割左右影像
    left_image = image.crop((0, 0, width // 2, height))
    right_image = image.crop((width // 2, 0, width, height))

    # 轉為 NumPy 陣列
    left_np = np.array(left_image, dtype=np.uint8)
    right_np = np.array(right_image, dtype=np.uint8)

    # 轉為灰階
    left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_np, cv2.COLOR_RGB2GRAY)

    # **調整 numDisparities & blockSize**
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray)

    # **視差正規化**
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)
    
    
    # 構建一個3x3的核
    kernel = np.ones((3, 3), np.uint8)
    
    # 進行膨脹操作
    disparity_norm = cv2.dilate(disparity_norm, kernel, iterations=9)

    # 進行侵蝕操作
    disparity_norm = cv2.erode(disparity_norm, kernel, iterations=21)
    
    disparity_norm = cv2.dilate(disparity_norm, kernel, iterations=3)
    

    # **建立遮罩**
    mask = disparity_norm > disparity_threshold  # 深度大於閾值的保留

    # **應用遮罩**
    foreground = left_np.copy()
    foreground[~mask] = [0, 0, 0]  # 設為黑色背景，這裡仍然保留了 RGB 格式

    # 轉換回 PIL 圖片並保存
    Image.fromarray(foreground).save(f"{name}output.png")
    Image.fromarray(left_np).save(f"{name}output_left.png")
    Image.fromarray(right_np).save(f"{name}output_right.png")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # 轉換為 0 和 255 顏色的二值圖
    mask_image.save(f"{name}mask.png")

    # 顯示結果
    cv2.imshow("Disparity Map", disparity_norm)
    cv2.imshow("Masked Foreground", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name ="k"
    compute_disparity_and_remove_bg(f"./double_version/{name}.jpg", num_disparities=16*4, block_size=27, disparity_threshold=190,name=f"output/{name}_")


# ug    16*8  13  70  11  35  11
# k      16*8  21  70  11  35  11
# k2    16*8  21  70  11  35  11
# ku    16*8  21  70  11  35  11
# t       16*8  11  35  11  35  11
# bu    16*4  17  35  11  35  11
