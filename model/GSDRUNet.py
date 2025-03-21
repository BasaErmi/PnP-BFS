import torch
import numpy as np
import deepinv.models
import torchvision.transforms as transforms
from PIL import Image


def GSDRUNet_denoise(image_np: np.ndarray, sigma: float, model_path="model/weights/GSDRUNet_grayscale.ckpt"):
    """
    使用 GSDRUNet 进行图像去噪。

    参数：
        image_np (np.ndarray) : 输入的灰度图像 (H, W) 或 (H, W, 1)，像素范围 [0,1]。
        sigma (float)         : 噪声水平，通常是 0~1 之间的小数，例如 25/255。
        model_path (str)      : 预训练权重文件路径，默认为 "weights/GSDRUNet_grayscale.ckpt"。

    返回：
        denoised_np (np.ndarray) : 去噪后的灰度图像 (H, W)，像素范围 [0,1]。
    """

    # 检查输入是否是 2D（H, W），如果是，转换为 3D（H, W, 1）
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)  # 变成 (H, W, 1)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 GSDRUNet
    denoiser = deepinv.models.GSDRUNet(
        alpha=1.0,           # 松弛参数（默认 1.0）
        in_channels=1,       # 灰度图输入通道数
        out_channels=1,      # 灰度图输出通道数
        nb=2,                # model 的 block 数量（默认 2）
        pretrained=None,     # 预训练权重
        device=device
    ).to(device)

    # 加载预训练权重
    checkpoints = torch.load(model_path, map_location=device)
    state_dict = checkpoints["state_dict"]

    # 过滤掉无关 key
    filtered_state_dict = {k: v for k, v in state_dict.items() if "data_range" not in k}

    # 加载到模型
    denoiser.load_state_dict(filtered_state_dict, strict=False)

    # 归一化转换 (numpy -> tensor)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 PyTorch 张量 (C, H, W)
    ])

    # 处理输入图像
    noisy_tensor = transform(image_np).unsqueeze(0).to(device)  # (1, 1, H, W)

    # 噪声水平 (需要转换为 PyTorch tensor)
    sigma_tensor = torch.tensor([sigma]).to(device)

    # 运行去噪
    with torch.no_grad():  # 不计算梯度，加速推理
        denoised_tensor = denoiser(noisy_tensor, sigma_tensor)

    # 还原为 numpy 格式 (H, W)
    denoised_np = denoised_tensor.squeeze().cpu().numpy()

    return denoised_np  # 返回去噪后的图像
