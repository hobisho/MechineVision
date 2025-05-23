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
    print( disparity_norm )
    # disparity_norm = disparity_norm > disparity_threshold  # 深度大於閾值的保留

    disparity_norm = np.uint8(disparity_norm)
    # 構建一個3x3的核
    kernel = np.ones((3, 3), np.uint8)
    
    # 進行膨脹操作
    disparity_norm = cv2.dilate(disparity_norm, kernel, iterations=9)

    # 進行侵蝕操作
    disparity_norm = cv2.erode(disparity_norm, kernel, iterations=21)
    
    mask = cv2.dilate(disparity_norm, kernel, iterations=3)
    

    # **建立遮罩**
    

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
    name ="ug"
    compute_disparity_and_remove_bg(f"./double_version/{name}.jpg", num_disparities=16*5, block_size=9, disparity_threshold=1,name=f"output/{name}_")


# ug    16*5  9 150  9 21 3
# k      16*4  27 190  9 21 3
# k2    
# ku    
# t       
# bu    16*3  13  150  11  33  7
