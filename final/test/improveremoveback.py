import numpy as np
import cv2
from PIL import Image

def compute_disparity_and_remove_bg(left_path, right_path, num_disparities=96, block_size=11, disparity_threshold=30, name=""):
    # 讀取左右圖片
    left_image = Image.open(left_path)
    right_image = Image.open(right_path)

    # 確保大小一致
    if left_image.size != right_image.size:
        print("左右圖片尺寸不同，請確認輸入圖片一致。")
        return

    # 轉為 NumPy 陣列
    left_np = np.array(left_image, dtype=np.uint8)
    right_np = np.array(right_image, dtype=np.uint8)

    # 轉為灰階
    left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_np, cv2.COLOR_RGB2GRAY)

    # 建立視差計算器
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray)

    # 正規化視差圖
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)

    # 濾波（先中值濾波，再膨脹 + 侵蝕）
    disparity_norm = cv2.medianBlur(disparity_norm, 9)  # 中值濾波器，核大小5
    kernel = np.ones((3, 3), np.uint8)
    # disparity_norm = cv2.dilate(disparity_norm, kernel, iterations=9)
    disparity_norm = cv2.erode(disparity_norm, kernel, iterations=7)
    disparity_norm = cv2.dilate(disparity_norm, kernel, iterations=30)

    # 建立遮罩（只保留近距離區域）
    mask = disparity_norm > disparity_threshold

    # 應用遮罩到左圖
    foreground = left_np.copy()
    foreground[~mask] = [0, 0, 100]  # 背景變色（可改為 [0,0,0] 全黑）

    # 儲存圖片
    Image.fromarray(disparity_norm).save(f"{name}disparity_norm.png")
    Image.fromarray(foreground).save(f"{name}output.png")
    Image.fromarray(left_np).save(f"{name}left.png")
    Image.fromarray(right_np).save(f"{name}right.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(f"{name}mask.png")

    # 顯示結果（如有需要）
    # cv2.imshow("Disparity", disparity_norm)
    # cv2.imshow("Foreground", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 修改為你的左右圖檔路徑
    left_img_path = "./left.jpg"
    right_img_path = "./right.jpg"

    compute_disparity_and_remove_bg(
        left_path=left_img_path,
        right_path=right_img_path,
        num_disparities=16*4,
        block_size=31,
        disparity_threshold=50,
        name="./final/output/"
    )
