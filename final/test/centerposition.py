import cv2
import numpy as np

def dismove(img, threshold=128):
    # 確保是灰階圖
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    height, width = gray.shape

    for y in range(height):
        for x in range(width):
            if gray[y, x] > threshold:
                return (x, y)  # 回傳 (x, y) 座標

    return None  # 沒有找到符合條件的像素

if __name__ == "__main__":
    img1 = cv2.imread("backgroung_left.jpg")
    pos1 = dismove(img1)
    img2 = cv2.imread("backgroung_right.jpg")
    pos2 = dismove(img2)
    print(f"第一個大於128的像素位置：{pos1}{pos2}{img1.shape}")

