import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def DFT(img):
    img_gray = np.float32(img)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return fshift

def IDFT(img):
    f_ishift = np.fft.ifftshift(img)
    img_reconstructed = np.fft.ifft2(f_ishift)
    img_reconstructed = np.real(img_reconstructed)
    img_reconstructed = np.round(img_reconstructed)
    return img_reconstructed

def Convolution(img, kernel):
    img_convolved = signal.convolve2d(img, kernel, mode='same', boundary='wrap') 
    return img_convolved

def DFTKernel(img,kernel):
    img_shape = img.shape
    pad_height = (img_shape[0] - kernel.shape[0]) // 2
    pad_width = (img_shape[1] - kernel.shape[1]) // 2
    padded_kernel = np.pad(kernel, ((pad_height, pad_height if img_shape[0] % 2 == 1 else pad_height + 1), (pad_width, pad_width if img_shape[0] % 2 == 1 else pad_width + 1)), mode='constant', constant_values=0)
    img_gray = np.float32(padded_kernel)
    kernel_dft = np.fft.fft2(img_gray)
    return kernel_dft

def NormalizeImage(img):
    img = 20 * np.log(np.abs(img) + 1e-5)
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype(np.uint8)

def calcu_error_avg(img1, img2):
    diff = np.abs(img1 - img2)
    error_avg = np.sum(diff) / (img1.shape[0] * img1.shape[1])
    return error_avg

def kernal(size, D):
    kernal = np.zeros((size, size), dtype=np.float32)
    for x in range(-(size//2),size//2+1):
        for y in range(-(size//2),size//2+1):
            normalization = -1 / (np.pi * D**4)
            gaussian = (1 - (x**2 + y**2) / (2 * D**2)) * np.exp(-(x**2 + y**2) / (2 * D**2))
            kernal[x+size//2][y+size//2] = normalization * gaussian
    return kernal

# 主程式
img_path = "jerry.png"
kernel = kernal(3, 0.9) 
print(kernel)

img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 頻域處理
img_dft = DFT(img_gray)
kernel_dft = DFTKernel(img_gray, kernel)
dft_results = kernel_dft * img_dft
timedomain_results = IDFT(dft_results)
timedomain_results_normal = cv2.normalize(timedomain_results, None, 0, 255, cv2.NORM_MINMAX)

# 空間卷積
Convolution_img = Convolution(img_gray, kernel)
Convolution_img_normal = cv2.normalize(Convolution_img, None, 0, 255, cv2.NORM_MINMAX)

# 比較誤差
diff = calcu_error_avg(timedomain_results, Convolution_img)
print("絕對平均誤差", diff)

# 顯示結果（改用 plt）
image_list = {
    'Original': img_gray,
    'DFT': NormalizeImage(img_dft),
    'DFT_kernel': NormalizeImage(kernel_dft),
    'DFT X DFT_kernel': NormalizeImage(dft_results),
    'Convolution': Convolution_img_normal,
    'Frequency IDFT Results': timedomain_results_normal
}

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for ax, (title, image) in zip(axs.ravel(), image_list.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for i in range(len(image_list), 6):
    axs.ravel()[i].axis('off')

plt.tight_layout()
plt.savefig("機器視覺/results.png")
plt.show()
