import cv2
import numpy as np
import matplotlib.pyplot as plt

def mse(a, b):
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)

def find_first_similar_column(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("兩張圖片尺寸不一樣，無法比較")

    height, width = img1.shape[:2]
    best_error = 5000

    for x in range(width-2):
        col1 = img1[0:int(height/2), img1.shape[1] - 3:img1.shape[1] - 1]
        col2 = img2[0:int(height/2), x : x+2]
        error = mse(col1, col2)
        if error <  best_error:
            best_x = x
            best_error = error

    background_shift= img1.shape[1] - best_x
    return background_shift, best_error

if __name__ == "__main__":
    path = "./final/image/tv"
    img1 = cv2.imread(f"{path}_left.jpg")
    img2 = cv2.imread(f"{path}_right.jpg")

    pos, error = find_first_similar_column(img1, img2)
    if pos >= 0:
        print(f"背景位移: {pos}, MSE誤差: {error}")
    