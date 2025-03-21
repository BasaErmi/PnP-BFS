import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import time
from prettytable import PrettyTable
from PnP_admm import PnP_admm
from utils import *

def background_extraction(opts, noise_level=0, dataset_name='Hall'):
    """
    主函数：加载数据、设置参数、调用修改后的 ADMM 方法并显示结果
    """
    # 加载数据（确保 hall.mat 文件中包含 data, label, groundtruth, picture_size 等变量）
    mat_contents = sio.loadmat(f'data/{dataset_name}.mat')
    D = mat_contents['data']
    label = mat_contents.get('label', np.arange(D.shape[1])).flatten() - 1 # 匹配matlab的1-based index
    groundtruth = mat_contents.get('groundtruth', np.zeros_like(D)).flatten(order='F')
    picture_size = tuple(mat_contents.get('picture_size', [int(np.sqrt(D.shape[0]))] * 2))

    # 模拟D被高斯噪声污染,每张图片的噪声都固定一样
    np.random.seed(42)

    opts['picsize'] = picture_size[0]
    noise_type = opts['noise_type']
    if noise_type == 'random':
        D = D + noise_level * np.random.randn(*D.shape)
    else:
        # noise = np.zeros(picture_size[0])
        # noise += noise_level * np.random.randn(*noise.shape)

        noise_D = np.zeros(D.shape)
        for i in range(D.shape[1]):
            noise = np.zeros(picture_size[0])
            noise += noise_level * np.random.randn(*noise.shape)
            D_frame = D[:, i].reshape(picture_size[0], order="F")
            D_frame_noised = D_frame + noise
            noise_D[:, i] = D_frame_noised.flatten(order="F")
        D = noise_D



    mu = opts['mu']
    p = opts['p']

    start_time = time.time()
    l_out, S_out, Z_out,Iter_out = PnP_admm(D, mu, p, opts, noise_level)
    elapsed_time = time.time() - start_time

    # 目标函数值
    f_val = 0.5 * norm(D - Z_out, 'fro') ** 2 + mu * (norm(S_out.flatten(), p) ** p)

    # 计算稀疏率
    S_mask = np.abs(S_out) > 1e-3
    spr = np.sum(S_out != 0) / D.size

    # 计算F-measure
    foreground = S_mask[:, label]
    # 展示foreground中不等于0的坐标
    ind_f = np.where(foreground == 1)[0]
    ind_g = np.where(groundtruth == 1)[0]
    ind_correct = np.intersect1d(ind_f, ind_g)
    precision = len(ind_correct) / (len(ind_f))
    recall = len(ind_correct) / (len(ind_g))
    F = 2 * precision * recall / (precision + recall)


    # 创建表格
    table = PrettyTable()
    table.field_names = ["Dataset", "denoiser", "p", "mu", "tau", "rho", "Iter", "Outertol", "Time(s)", "spr", "F-measure", "f_val"]
    table.add_row([dataset_name, opts["denoise_method"], p, f"{mu:.0e}", opts['tau'], opts['rho'],Iter_out, f"{opts['Outertol']:.0e}", f"{elapsed_time:.2f}",
                   f"{spr:.4f}", f"{F:.4f}", f"{f_val:.4f}"])

    # 打印表格
    print(table)

    # 记录实验结果到本地 txt 文件
    log_file = "experiment_results.txt"
    with open(log_file, "a") as f:
        f.write(f"================ Experiment {time.strftime('%Y-%m-%d %H:%M:%S')} ================\n")
        f.write(table.get_string())  # 直接保存表格
        f.write("\n==========================================================================\n\n")

    # 存储结果
    result = {
        'l_out': l_out,
        'S_out': S_out,
        'Z_out': Z_out,
        'Iter_out': Iter_out,
        'elapsed_time': elapsed_time,
        'spr': spr,
        'f_val': f_val,
    }

    # sio.savemat('result_denoise_with_separation_0.01.mat', result)

    # 显示结果（根据实际数据可能需要调整 reshape 操作）
    display(D, *picture_size[0], D.shape[1], label, title=f"Original \n noise_level={noise_level}")
    display(l_out, *picture_size[0], 1, -1, title=f"Background for {opts['denoise_method']} \n denoise_level={noise_level} Outertol = {opts['Outertol']}")
    display(S_out, *picture_size[0], D.shape[1], label, title=f"Foreground for {opts['denoise_method']} \n denoise_level={noise_level} Outertol = {opts['Outertol']}")

    if opts['denoise_method'] == 'False':
        l_out_denoised = denoiser(l_out, picsize=opts['picsize'],method="GSDRUNet", noise_level=noise_level)
        display(l_out_denoised, *picture_size[0], 1, -1, title="Denoised Background")

