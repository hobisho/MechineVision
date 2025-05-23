import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def DFT(img):
    img_gray = np.float32(img)
    f = np.fft.fft2(img_gray)  # fft轉換
    fshift = np.fft.fftshift(f)  # shift到中間
    return fshift

def IDFT(img):
    f_ishift = np.fft.ifftshift(img)  # shift回角落
    orignal_img = np.fft.ifft2(f_ishift)  # 反fft轉換
    real_orignal_img = np.real(orignal_img)
    round_orignal_img = np.round(real_orignal_img)
    return round_orignal_img

def Convolution(img, kernel):# 時域卷積
    img_convolved = signal.convolve2d(img, kernel, mode='same', boundary='wrap') 
    return img_convolved

def DFTKernel(img, kernel):# 頻域濾波器轉換
    mid_height = (img.shape[0] - kernel.shape[0]) // 2
    mid_width = (img.shape[1] - kernel.shape[1]) // 2
    padded_kernel = np.pad(kernel, ((mid_height, img.shape[0] - kernel.shape[0] - mid_height),
                                    (mid_width, img.shape[1] - kernel.shape[1] - mid_width)),
                           mode='constant', constant_values=0)
    padded_kernel = np.fft.ifftshift(padded_kernel)
    kernel_dft = DFT(padded_kernel)
    return kernel_dft

def NormalizeImage(img):  # 正規化影像
    img = 20 * np.log(np.abs(img) + 1e-5)
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype(np.uint8)

def AbsErrorAvg(img1, img2):
    diff = np.abs(img1 - img2)
    error_avg = np.sum(diff) / (img1.shape[0] * img1.shape[1])
    return error_avg

def kernal(size, D):
    kernal = np.zeros((size, size), dtype=np.float32)
    for x in range(-(size//2), size//2+1):
        for y in range(-(size//2), size//2+1):
            normalization = -1 / (np.pi * D**4)
            gaussian = (1 - (x**2 + y**2) / (2 * D**2)) * np.exp(-(x**2 + y**2) / (2 * D**2))
            kernal[x + size//2][y + size//2] = normalization * gaussian
    return kernal


if __name__ == "__main__":
    img_name = "13"
    kernel = kernal(31, 1.5) 

    img = cv2.imread(f"{img_name}.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 頻域處理
    img_dft = DFT(img_gray)
    kernel_dft = DFTKernel(img_gray, kernel)
    dft_results = kernel_dft * img_dft
    timedomain_results = IDFT(dft_results)
    timedomain_results_normal = cv2.normalize(timedomain_results, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 時域處理
    Convolution_img = Convolution(img_gray, kernel)
    Convolution_img_normal = cv2.normalize(Convolution_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 比較誤差
    diff = AbsErrorAvg(timedomain_results, Convolution_img)
    print("絕對平均誤差", diff)

    image_list = {
        'Original': img_gray,
        'DFT': NormalizeImage(img_dft),
        'DFT_kernel': NormalizeImage(kernel_dft),
        'DFT X DFT_kernel': NormalizeImage(dft_results),
        'Convolution': Convolution_img_normal,
        'Frequency IDFT Results': timedomain_results_normal
    }
    # 儲存圖片
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    for title, image in image_list.items():
        filename = f"{output_dir}/{img_name}_{title.replace(' ', '_')}.png"
        cv2.imwrite(filename, image)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, image) in zip(axs.ravel(), image_list.items()):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for i in range(len(image_list), 6):
        axs.ravel()[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results.png")
    plt.show()
