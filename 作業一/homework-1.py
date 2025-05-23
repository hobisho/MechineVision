import numpy as np
import cv2
from PIL import Image

def split_image(image_path):
    # 讀取圖片
    image = Image.open(image_path)
    width, height = image.size

    # 分割圖片為左右兩部分
    left_image = image.crop((0, 0, width // 2, height))
    right_image = image.crop((width // 2, 0, width, height))

    left_np = np.array(left_image, dtype=np.uint8)
    right_np = np.array(right_image, dtype=np.uint8)

    left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_np, cv2.COLOR_RGB2GRAY)

    # 顯示左右影像
    # cv2.imshow('Left Image', left_gray)
    # cv2.imshow('Right Image', right_gray)

    # 創建視差計算對象
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
    disparity = stereo.compute(left_gray, right_gray)

    # # 正規化視差圖以便顯示
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)

    cv2.imshow('Disparity Map', disparity_norm)
    cv2.imwrite('output.jpg', disparity_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    split_image("./double_version/b.jpg")  # 確保路徑正確
