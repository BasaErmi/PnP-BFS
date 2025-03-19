import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import time
from bm3d import bm3d
import matplotlib.pyplot as plt
from torch.cuda import seed_all

np.random.seed(42)

def denoise_frame(frame, method, weight, p , noise_level):
    """
    对单帧图像进行去噪处理。
    参数:
      frame: 2D numpy 数组，原始图像
      method: 去噪方法，可以选择 'TV' 或 'l1'
      weight: 去噪权重
    返回:
      去噪后的图像（2D numpy 数组）
    """
    if method == 'BM3D':
        # 使用 BM3D 去噪。确保安装了 bm3d 库，例如通过 pip install bm3d
        # BM3D 要求图像为浮点型并归一化到 [0,1]
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max - frame_min > 0:
            frame_norm = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame_norm = frame.copy().astype(np.float32)
        # weight 作为噪声标准差 sigma，通常取值范围在 0.01~0.1 之间（根据实际噪声水平调整）
        denoised_norm = bm3d(frame_norm, sigma_psd=noise_level)
        # 恢复到原始灰度范围
        denoised_frame = denoised_norm * (frame_max - frame_min) + frame_min
        return denoised_frame
    else:
        # 默认直接返回原图
        return frame


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

def run_me():
    """
    主函数：加载数据、设置参数、调用修改后的 ADMM 方法并显示结果
    """
    # 加载数据（确保 hall.mat 文件中包含 data, label, groundtruth, picture_size 等变量）
    dataset_name = 'Bootstrap'
    mat_contents = sio.loadmat(f'data/{dataset_name}.mat')
    D = mat_contents['data']
    picture_size = tuple(mat_contents.get('picture_size', [int(np.sqrt(D.shape[0]))] * 2))
    # 模拟D被高斯噪声污染,每张图片的噪声都固定一样
    np.random.seed(42)
    noise_level = 0.05

    noise = np.zeros(picture_size[0])
    noise += noise_level * np.random.randn(*noise.shape)

    noise_D = np.zeros(D.shape)
    for i in range(D.shape[1]):
        D_frame = D[:, i].reshape(picture_size[0], order="F")
        D_frame_noised = D_frame + noise
        noise_D[:, i] = D_frame_noised.flatten(order="F")
    D = noise_D
    label = mat_contents.get('label', np.arange(D.shape[1])).flatten() - 1 # 匹配matlab的1-based index
    D_frame = D[:, label].reshape(picture_size[0], order="F")


    # 显示结果（根据实际数据可能需要调整 reshape 操作）
    # for i in range(10):
    #     D_frame = D[:, i].reshape(picture_size[0], order="F")
    #     display(D_frame, *picture_size[0], 1, i, title="noised")

    D_frame = D[:, label].reshape(picture_size[0], order="F")
    Denoised_D_frame = denoise_frame(D_frame, method='BM3D', weight=0.5, p=0.5, noise_level=noise_level)
    display(D_frame, *picture_size[0], 1, label, title="noised")
    display(Denoised_D_frame, *picture_size[0], 1, 0, title="denoised")


if __name__ == "__main__":
    run_me()
