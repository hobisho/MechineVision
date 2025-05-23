import cv2
import numpy as np

def read_image_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0

    return img, mask

def overlapping_width(mask1, mask2):
    overlap = np.logical_and(mask1, mask2)
    cols = np.any(overlap, axis=0)
    if not np.any(cols):
        return 0
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(cols[::-1])
    width = right - left + 1
    return width

def combine_with_zero_fill(img1, mask1, img2, mask2):
    overlap_width = overlapping_width(mask1, mask2)

    new_width = img1.shape[1] - overlap_width + img2.shape[1]
    print(img1.shape[1], overlap_width, img2.shape[1], new_width)
    new_height = max(img1.shape[0], img2.shape[0])
    combined = np.zeros((new_height, new_width, img1.shape[2]), dtype=img1.dtype)

    # 放 img1 到左邊
    combined[:img1.shape[0], :img1.shape[1]] = img1

    cv2.imshow("background.jpg", combined)
    cv2.waitKey(0)

    start_x = img1.shape[1] - overlap_width

    # 放 img2 到重疊右側開始的位置
    combined[:img2.shape[0], start_x:start_x + img2.shape[1]] = img2
    cv2.imshow("background.jpg", combined)
    cv2.waitKey(0)

    # 在重疊區域做黑色(0)像素補齊
    overlap_start = start_x
    overlap_end = start_x + overlap_width

    img1_overlap_x = slice(img1.shape[1] - overlap_width, img1.shape[1])
    img2_overlap_x = slice(0, overlap_width)

    for y in range(new_height):
        for c in range(img1.shape[2]):
            img1_pixels = img1[y, img1_overlap_x, c] if y < img1.shape[0] else np.zeros(overlap_width, dtype=img1.dtype)
            img2_pixels = img2[y, img2_overlap_x, c] if y < img2.shape[0] else np.zeros(overlap_width, dtype=img2.dtype)
            combined_pixels = combined[y, overlap_start:overlap_end, c]

            for i in range(overlap_width):
                if combined_pixels[i] == 0:
                    if img1_pixels[i] != 0:
                        combined_pixels[i] = img1_pixels[i]
                    elif img2_pixels[i] != 0:
                        combined_pixels[i] = img2_pixels[i]

            combined[y, overlap_start:overlap_end, c] = combined_pixels

    return combined,a

# def pde_inpainting(img):
#     # 先找黑洞區域 mask：灰階中值為0的地方
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     mask = (gray == 0).astype(np.uint8) * 255  # mask要是 0 或 255

#     if np.count_nonzero(mask) == 0:
#         print("沒有要修補的區域")
#         return img

#     # 確保輸入是 8-bit 三通道
#     if img.dtype != np.uint8:
#         img_8u = cv2.convertScaleAbs(img)
#     else:
#         img_8u = img

#     # 使用 OpenCV PDE inpainting，方法：Telea 或 Navier-Stokes
#     inpainted = cv2.inpaint(img_8u, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#     return inpainted

if __name__ == "__main__":
    img1, mask1 = read_image_mask("backgroung_left.jpg")
    img2, mask2 = read_image_mask("backgroung_right.jpg")
    mask1 = cv2.imread(f'disparity_left.png', cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(f'disparity_right.png', cv2.IMREAD_GRAYSCALE)

    combined,a = combine_with_zero_fill(img1, mask1, img2, mask2)

    
    cv2.imwrite("background.jpg", combined)
    cv2.waitKey(0)
