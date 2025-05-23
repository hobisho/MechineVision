import cv2
import numpy as np

# 讀取彩色圖片
def read_image_mask(imgname):
    
    image = cv2.imread(f'{imgname}.jpg')

    # 將圖像轉成灰階
    mask = cv2.imread(f'disparity_{imgname}.png', cv2.IMREAD_GRAYSCALE)

    # 執行二值化
    threshold_value = 40
    _, binary_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)

    # 將 binary mask 擴展成 3 通道，與彩色圖同大小
    ibinary_mask = cv2.bitwise_not(binary_mask)

    ibinary_mask_3ch = cv2.merge([ibinary_mask, ibinary_mask, ibinary_mask])
    binary_mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

    # 將原圖與二值化 mask 相乘（遮罩）
    imasked_image = cv2.bitwise_and(image, ibinary_mask_3ch)
    masked_image = cv2.bitwise_and(image, binary_mask_3ch)

    # 顯示與儲存結果
    # cv2.imshow('Masked', ibinary_mask)
    cv2.imwrite(f'backgroung_{imgname}.jpg', masked_image)
    cv2.imwrite(f'object_{imgname}.jpg', masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":  
    read_image_mask("left")
    read_image_mask("right")