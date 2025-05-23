import cv2
import numpy as np

def mse(a, b):
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)

def find_first_similar_column(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("兩張圖片尺寸不一樣，無法比較")

    height, width = img1.shape[:2]
    best_error = 5000

    for x in range(width):
        col1 = img1[:, x]
        col2 = img2[:, img1.shape[1]-1]
        error = mse(col1, col2)
        # print(error)
        if error <  best_error:
            best_x = x
            best_error = error

    return best_x, best_error

if __name__ == "__main__":
    img1 = cv2.imread("left.jpg")
    img2 = cv2.imread("right.jpg")
    print(img1.shape, img2.shape)

    pos, error = find_first_similar_column(img1, img2)
    if pos >= 0:
        print(f"第一個相似列位置: {pos}, MSE誤差: {error}")
    else:
        print("找不到相似的列")