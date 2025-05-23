from    scipy   import signal 
import  numpy   as     np
import  cv2
import matplotlib.pyplot as plt


def generate_log_filter(size, sigma):
    """
    生成拉普拉斯高斯濾波器 (LoG filter)。
    """
    # 確保濾波器尺寸為奇數
    if size % 2 == 0:
        raise ValueError("濾波器尺寸必須是奇數")

    # 創建 (x, y) 坐標網格
    half_size = size // 2
    x, y = np.meshgrid(np.arange(-half_size, half_size + 1),
                       np.arange(-half_size, half_size + 1))

    # 計算 LoG 濾波器的值
    normalization = -1 / (np.pi * sigma**4)
    gaussian = (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    log_filter = normalization * gaussian

    return np.round(log_filter, 2).astype(np.float32)

def padding_kernel(kernel, img_shape):
    """
    將濾波器填充到與圖像大小相同。
    """
    pad_height = (img_shape[0] - kernel.shape[0]) // 2
    pad_width = (img_shape[1] - kernel.shape[1]) // 2
    padded_kernel = np.pad(kernel, ((pad_height, pad_height if img_shape[0] % 2 == 1 else pad_height + 1), (pad_width, pad_width if img_shape[0] % 2 == 1 else pad_width + 1)), mode='constant', constant_values=0)
    padded_kernel = np.fft.ifftshift(padded_kernel)
    return padded_kernel

def fft_convolution(img, kernel):
    """
    使用快速傅立葉變換 (FFT) 進行卷積操作。
    """
    fft_img = np.fft.fft2(img)  # 計算圖像的傅立葉變換
    fft_kernel = np.fft.fft2(kernel)  # 計算濾波器的傅立葉變換
    fshift = (fft_img * fft_kernel)  # 頻域中的卷積
    img_freq = np.fft.ifft2(fshift)  # 傅立葉逆變換回空間域
    img_freq = np.fft.ifftshift(np.real(img_freq))  # 移位並取實部

    # 將 img_freq 正規化到 0-255 範圍
    img_freq = cv2.normalize(img_freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 處理 fft_img 以便可視化
    fft_img_shifted = np.fft.fftshift(fft_img)
    fft_img_magnitude = np.abs(fft_img_shifted)
    fft_img_log = 20 * np.log(fft_img_magnitude + 1e-5)
    fft_img = cv2.normalize(fft_img_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 處理 fft_kernel 以便可視化
    fft_kernel_magnitude = np.abs(fft_kernel)
    fft_kernel_log = 20 * np.log(fft_kernel_magnitude + 1e-5)
    fft_kernel = cv2.normalize(fft_kernel_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 處理 fshift 以便可視化
    fshift_magnitude = np.abs(fshift)
    fshift_log = 20 * np.log(fshift_magnitude + 1e-5)  # 避免 log(0)
    fshift = cv2.normalize(fshift_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_freq, fft_img, fft_kernel, fshift

def convolve2d(image, kernel):
    """
    使用 scipy.signal.convolve2d 進行二維卷積操作。
    """
    img_real = signal.convolve2d(image, kernel, mode='same', boundary='wrap')
    return cv2.normalize(img_real, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def calcu_error_avg(img1, img2):
    """
    計算兩張圖像之間的平均絕對誤差。
    """
    if img1.shape != img2.shape:
        raise ValueError("兩張圖像必須具有相同的尺寸。")
    
    diff = np.abs(img1 - img2)  # 計算像素差異的絕對值
    error_avg = np.sum(diff) / (img1.shape[0] * img1.shape[1])  # 計算平均誤差
    
    return error_avg

if __name__ == "__main__":
    file_path = r"jerry.PNG"  # 圖像路徑
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 0)  # 讀取灰度圖像
    kernel = generate_log_filter(size=3, sigma=0.9)  # 生成 LoG 濾波器
    print(kernel)
    padded_kernel = padding_kernel(kernel, img.shape)  # 將濾波器填充到與圖像相同大小
    img_freq, fft_img, fft_kernel, fshift = fft_convolution(img, padded_kernel)  # 使用 FFT 進行卷積
   
    img_real = convolve2d(img, kernel)  # 使用時域卷積
    
    diff = calcu_error_avg(img_freq, img_real)  # 計算兩種卷積結果的平均誤差
    print("絕對平均誤差", diff.astype(np.float32))
    images = {
        "Original Image": img,
        "FFT Image": fft_img,
        "FFT Kernel": fft_kernel,
        "Frequency Domain Result": img_freq,
        "Spatial Domain Result": img_real,
        "Frequency Shift Product": fshift
    }

    # 顯示所有圖像
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, image) in zip(axs.ravel(), images.items()):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("results_fft_vs_spatial.png")
    plt.show()