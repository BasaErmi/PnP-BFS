import numpy as np
import scipy.io as sio
from mpmath import zeros
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
from bm3d import bm3d
from numpy.ma import reshape


def denoiser(frame, method):
    """
    对单帧图像进行去噪处理。
    参数:
      frame: 2D numpy 数组，原始图像
      method: 去噪方法，可以选择 'TV' 或 'l1'
      weight: 去噪权重
    返回:
      去噪后的图像（2D numpy 数组）
    """
    # if method == 'L1':
    #     denoised_frame = lp_thresholding_matrix(frame, weight, p)[0]
    #     return denoised_frame
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
        denoised_norm = bm3d(frame_norm, sigma_psd=0.01)
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
    plt.colorbar()
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


def admm_new(H, mu, p, opts):
    """
    针对如下优化问题的 ADMM 求解：
        min_{ell, S, Z}  0.5*||H - A(Z)||_F^2 + Psi(ell) + Phi(S)
        s.t.            S + ell * 1^T = Z
    其中我们假设 A 为单位算子。

    输入:
      H   : 数据矩阵，形状 (m, n)
      mu  : S 正则化参数
      p   : S 上的 lp 惩罚参数（0 < p <= 1）
      opts: 包含算法参数的字典，其中应包含：
             tau, rho, upbd, tol, maxiter, display, displayfreq 等
    输出:
      ell : 优化得到的 ell 向量，形状 (m, 1)
      S   : 优化得到的 S 矩阵，形状 (m, n)
      Z   : 优化得到的 Z 矩阵，形状 (m, n)
      k   : 迭代次数
    """

    succ_chg = np.inf

    tau = opts.get('tau', 0.8)
    rho = opts.get('rho', 1.0)
    Outertol = opts.get('tol', 1e-4)
    InnerTol = opts.get('InnerTol', 5e-3)
    maxiter = opts.get('maxiter', 1000)
    display = opts.get('display', 1)
    disp_freq = opts.get('displayfreq', 10)
    picsize = opts.get('picsize', None)
    heuristic = opts.get('heuristic', 1)
    identity = opts.get('identity', 1)
    blurring = opts.get('blurring', 0)
    succ_chgk = succ_chg  # 用于记录上一次迭代的变化幅度，判断收敛

    m, n = H.shape

    if identity == 1:
        lambda_max = 1
        lambda_min = 1
    elif blurring == 1:
        if not all(k in opts for k in ['lambda_max', 'lambda_min', 'Ac', 'Ar', 'picsize']):
            raise ValueError("For blurring case, lambda_max, lambda_min, Ac, Ar, picsize must be provided.")
        lambda_max = opts['lambda_max']
        lambda_min = opts['lambda_min']
        Ac = opts['Ac']
        Ar = opts['Ar']
        picsize = opts['picsize']
        # SVD分解（这里只计算右奇异向量，实际根据需要调整）
        Uc, Sc, Vc = np.linalg.svd(Ac, full_matrices=False)
        Ur, Sr, Vr = np.linalg.svd(opts['Ar'], full_matrices=False)
    else:
        raise ValueError("identity 和 blurring 选项设置错误")

    # rho_bound 的计算
    term1 = lambda_max / tau
    term2 = lambda_max * tau
    term3 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * lambda_max ** 2 / tau)
    term4 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * tau ** 2 * lambda_max ** 2 / (1 + tau - tau ** 2))
    rho_bound = max([term1, term2, term3, term4])

    # 初始 rho 及自适应参数
    # if heuristic:
    #     sigma1 = opts.get('sigma1', 0.6)
    #     sigma2 = opts.get('sigma2', 0.3)
    #     rho_beta = opts.get('rho_beta', 1.1)
    #     rho = sigma1 * rho_bound
    #     rho_max = 1.01 * rho_bound
    #     count = 0
    # else:
    #     rho = 1.01 * rho_bound

    # 初始化变量
    # ell 为 m×1 向量（每个元素受限于 |ell_i|<=upbd）
    ell = np.zeros((m, 1))
    S = np.zeros((m, n))
    Z = S + np.tile(ell, (1, n))  # 初始满足 S + ell*1^T = Z
    Lambda = np.zeros((m, n))  # 对偶变量

    for k in range(1, maxiter + 1):
        ell_old = ell.copy()
        S_old = S.copy()
        Z_old = Z.copy()
        Lambda_old = Lambda.copy()

        # -----------------------------
        # ell 更新：
        # 令 M^k = Z - S - Lambda/rho, 计算各列的平均值 m_bar
        M = Z - S - Lambda / rho
        m_bar = np.mean(M, axis=1)  # m_bar shape (m,)
        # ell = m_bar.reshape(m,1)
        # print(np.tile(ell, (1, n)).shape)
        # 使用去噪器对 m_bar 进行去噪，即计算其proximal map
        # 重构为二维图像，注意这里使用 Fortran 顺序与原来投影保持一致
        m_reshape = m_bar.reshape(picsize, order='F')
        # # 对图像进行去噪
        m_denoised = denoiser(m_reshape, method='BM3D')
        # # 将去噪后的图像展平并保存
        ell = m_denoised.flatten(order='F').reshape(m, 1)


        # -----------------------------
        # S 更新：
        # S 子问题为：min_S  Phi(S) + (rho/2)*||S - (Z - ell*1^T - Lambda/rho)||_F^2

        S_tmp = Z - np.tile(ell, (1, n)) - Lambda / rho
        S, iter_Newton = lp_thresholding_matrix(S_tmp, mu / rho, p)

        # -----------------------------
        # Z 更新：
        # Z 子问题： min_Z  0.5*||H - Z||_F^2 + (rho/2)*||Z - (S + ell*1^T + Lambda/rho)||_F^2
        # 对于 A 为单位算子，其闭式解为：
        Z = (H + rho * (S + np.tile(ell, (1, n))) + Lambda) / (1 + rho)

        # -----------------------------
        # 对偶变量更新：
        Lambda = Lambda + tau * rho * (S + np.tile(ell, (1, n)) - Z)

        # -----------------------------
        # 计算收敛准则
        succ_chg = (norm(ell - ell_old, 'fro') + norm(S - S_old, 'fro') + norm(Z - Z_old, 'fro'))
        fnorm = (norm(ell, 'fro') + norm(S, 'fro') + norm(Z, 'fro') + 1)
        relchg = succ_chg / fnorm

        succ_chg_2nd = norm(S - S_old, 'fro') + norm(Lambda - Lambda_old, 'fro')
        fnorm_2nd = norm(S, 'fro') + norm(Lambda, 'fro') + 1
        relchg_2nd = succ_chg_2nd / fnorm_2nd

        # -----------------------------
        # heuristic 更新 rho
        # if heuristic:
        #     if (0.99 * succ_chgk - succ_chg) < 0:
        #         count += 1
        #     if count >= np.ceil(sigma2 * k) or fnorm > 1e10:
        #         rho = min(rho * rho_beta, rho_max)

        if display and k % disp_freq == 0:
            obj_val = 0.5 * norm(H - Z, 'fro') ** 2 + mu * (norm(S.flatten(), p) ** p)
            print("{:4d} | relchg: {:0.8e} | obj_val: {:0.8e} | iter_Newton: {:d} | relchg_2nd: {:0.8e} | rho: {:g}".format(k, relchg, obj_val,
                                                                                          iter_Newton, relchg_2nd, rho))

        if relchg < Outertol:
            if relchg_2nd < InnerTol:
                break

    return ell, S, Z, k


def run_me():
    """
    主函数：加载数据、设置参数、调用修改后的 ADMM 方法并显示结果
    """
    # 加载数据（确保 hall.mat 文件中包含 data, label, groundtruth, picture_size 等变量）
    mat_contents = sio.loadmat('hall.mat')
    D = mat_contents['data']
    label = mat_contents.get('label', np.arange(D.shape[1])).flatten() - 1 # 匹配matlab的1-based index
    groundtruth = mat_contents.get('groundtruth', np.zeros_like(D)).flatten(order='F')
    picture_size = tuple(mat_contents.get('picture_size', [int(np.sqrt(D.shape[0]))] * 2))

    mu = 1e-2
    p = 0.54

    opts = {
        'tau': 0.8,
        'rho': 1.0,  # 惩罚参数
        'Outtol': 1e-4,
        'InnerTol': 5e-3,
        'maxiter': 2000,
        'upbd': 1,
        'identity': 1,  # 若为1，则采用单位算子；若采用模糊算子则设为0并设置 blurring=1
        'blurring': 0,
        'display': 1,
        'displayfreq': 1,
        'picsize': (picture_size[0]),  # 图像大小
        'heuristic': 1,
        'sigma1': 0.6,
        'sigma2': 0.3,
        'rho_beta': 1.1,
        # 若 blurring==1，需添加 'Ac', 'Ar', 'picsize' 参数
    }

    start_time = time.time()
    l_out, S_out, Z_out,Iter_out = admm_new(D, mu, p, opts)
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

    print("\n$p$ & $mu$ & $tau$ & Iter & Time(s) & spr & F-measure &f_val")
    print("{:g}  &  {:.0e}  & {:.1f} & {:d} & {:.2f} & {:.4f} & {:.4f} & {:.4f}".format(
        p, mu, opts['tau'], Iter_out, elapsed_time, spr, F, f_val))

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

    sio.savemat('result_denoise_with_separation_0.01.mat', result)

    # 显示结果（根据实际数据可能需要调整 reshape 操作）
    display(D, *picture_size[0], D.shape[1], label, title="Original")
    display(l_out, *picture_size[0], 1, -1, title="Background")
    display(S_out, *picture_size[0], D.shape[1], label, title="Foreground")


if __name__ == "__main__":
    run_me()
