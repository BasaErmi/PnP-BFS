import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d
from model.GSDRUNet import GSDRUNet_denoise

def denoiser(m_bar, picsize, method, noise_level):
    """
    对单帧图像进行去噪处理。
    参数:
      frame: 2D numpy 数组，原始图像
      method: 去噪方法，可以选择 'BM3D' 或 'GSDRUNet'
      sigma: 噪声等级
    返回:
      去噪后的图像（2D numpy 数组）
    """
    m = picsize[0]
    if method == 'False':
        return m_bar.reshape(-1,1)

    # 重构为二维图像，注意这里使用 Fortran 顺序与原来投影保持一致
    frame = m_bar.reshape(picsize, order='F')

    if method == 'BM3D':
        # 使用 BM3D 去噪
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max - frame_min > 0:
            frame_norm = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame_norm = frame.copy().astype(np.float32)
        denoised_norm = bm3d(frame_norm, sigma_psd=noise_level + 0.01)
        # 恢复到原始灰度范围
        denoised_frame = denoised_norm * (frame_max - frame_min) + frame_min
    elif method == 'GSDRUNet':
        # 使用 GSDRUNet 去噪
        denoised_frame = GSDRUNet_denoise(frame, sigma=noise_level + 0.009)
    else:
        # 找不到方法
        assert False, "Unknown denoising method"
    # 将去噪后的图像展平并保存
    return denoised_frame.flatten(order='F').reshape(-1, 1)


def display(matrix, image_height, image_width, num_images, image_index, title="Matrix Image"):
    matrix = np.array(matrix)
    if num_images == 1:
        image = matrix.reshape(image_height, image_width, order="F")
    else:
        reshape_matrix = matrix.reshape(image_height, image_width, num_images, order="F")
        image = reshape_matrix[:, :, image_index]
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")  # 以灰度图显示
    plt.title(title)
    plt.axis("off")
    plt.show()

def lp_thresholding_matrix(Y, lam, p, tol=1e-12, max_newton_iter=1000):
    """
    求解: min_X lam * ||X||_p^p + 1/2 * ||X - Y||_F^2，
    对每个元素分别进行求解。
    当 p==1 时采用软阈值操作，否则采用牛顿法。
    返回 X 及牛顿法迭代次数总和。
    """
    iter_total = 0
    X = np.zeros_like(Y)
    if p == 1:
        X = np.sign(Y) * np.maximum(np.abs(Y) - lam, 0)
    else:
        B = np.abs(Y)
        tau_val = (2 * lam * (1 - p)) ** (1 / (2 - p)) + lam * p * (2 * lam * (1 - p)) ** ((p - 1) / (2 - p))
        index = B > tau_val
        if np.sum(index) > 0:
            X_new = B[index].copy()
            Y_sub = B[index].copy()
            diff = np.inf
            newton_iter = 0
            while diff > tol and newton_iter < max_newton_iter:
                gX = X_new - Y_sub + lam * p * X_new ** (p - 1)
                hX = 1 + lam * p * (p - 1) * X_new ** (p - 2)
                X_prev = X_new.copy()
                X_new = X_new - gX / hX
                diff = np.max(np.abs(X_new - X_prev))
                newton_iter += 1
            iter_total += newton_iter
            X[index] = X_new * np.sign(Y[index])
        # 如果 index 为空，则 X 中对应部分已经是零，无需更新
    return X, iter_total
