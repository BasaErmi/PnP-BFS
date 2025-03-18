import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from bm3d import bm3d

def denoise_frame(frame, method, weight, p):
    """
    对单帧图像进行去噪处理。
    参数:
      frame: 2D numpy 数组，原始图像
      method: 去噪方法，可以选择 'TV' 或 'l1'
      weight: 去噪权重
    返回:
      去噪后的图像（2D numpy 数组）
    """
    if method == 'L1':
        denoised_frame = lp_thresholding_matrix(frame, weight, p)[0]
        return denoised_frame
    elif method == 'BM3D':
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


def projection_Omega(X, upbd):
    """
    投影算子：对于每一行，取所有列的平均值后再进行截断（阈值 upbd）。
    """
    m, n = X.shape
    X_mean = np.mean(X, axis=1, keepdims=True)
    L_proj = np.tile(X_mean, (1, n))
    L_proj = np.sign(L_proj) * np.minimum(np.abs(L_proj), upbd)
    return L_proj


def admm_lp(D, mu, p, opts):
    """
    修改后的 ADMM 算法，目标函数为
      min_{L,S,Z,Y} 0.5*||H-Z||_F^2 + Ψ(L) + ψ_σ(Y) + μ*||S||_p^p
      s.t. L+S=Z,  L=Y,
    其中 Ψ 和 ψ_σ 均为指示函数，要求 L, Y ∈ Ω (即每一行取平均后再截断)。

    参数 opts 包含：
      tau: 对偶步长
      rho: 惩罚参数 (若未提供，则默认为1)
      upbd: 上界，用于 L 和 Y 的投影
      tol: 外部收敛容限
      maxiter: 最大迭代次数
      identity: 若为1则为单位（无模糊）情况，否则为模糊情况
      blurring: 若为1则进入模糊情况，此时还需提供 'Ac', 'Ar', 'picsize'
      display, displayfreq: 输出信息控制
    返回 L, S, Z, Y 及迭代次数。
    """
    tau = opts.get('tau', 0.8)
    rho = opts.get('rho', 1.0)
    upbd = opts.get('upbd', 1)
    OuterTol = opts.get('tol', 1e-4)
    maxiter = opts.get('maxiter', 1000)
    identity = opts.get('identity', 1)
    blurring = opts.get('blurring', 0)
    display = opts.get('display', 1)
    displayfreq = opts.get('displayfreq', 1)

    beta = 0.01  # Y 的去噪参数
    # 对于无模糊情况，将 H 设为 D；若为模糊，则 H 依然为 D，但 Z 更新中使用模糊算子
    if identity == 1:
        H = D.copy()
    elif blurring == 1:
        if not all(k in opts for k in ['Ac', 'Ar', 'picsize']):
            raise ValueError("blurring==1 时，必须提供 Ac, Ar, picsize 参数。")
        H = D.copy()
        Ac = opts['Ac']
        Ar = opts['Ar']
        picsize = opts['picsize']
        Uc, Sc, Vc = np.linalg.svd(Ac, full_matrices=False)
        Ur, Sr, Vr = np.linalg.svd(Ar, full_matrices=False)
    else:
        raise ValueError("identity 与 blurring 参数设置错误")

    m, n = D.shape

    # 初始化变量：L, S, Z, Y, 对偶变量 Lambda, Gamma
    if all(k in opts for k in ['L0', 'S0', 'Z0', 'Y0', 'Lambda0', 'Gamma0']):
        L = opts['L0']
        S = opts['S0']
        Z = opts['Z0']
        Y = opts['Y0']
        Lambda = opts['Lambda0']
        Gamma = opts['Gamma0']
    else:
        L = projection_Omega(np.tile(np.mean(D, axis=1, keepdims=True), (1, n)), upbd)
        S = np.zeros((m, n))
        Z = L.copy()
        Y = L.copy()
        Lambda = np.zeros((m, n))
        Gamma = np.zeros((m, n))

    succ_chg = np.inf
    iter_Newton = 0  # 初始化 S 子问题中牛顿法迭代次数

    for iter in range(1, maxiter + 1):
        L_old = L.copy()
        S_old = S.copy()
        Z_old = Z.copy()
        Y_old = Y.copy()
        Lambda_old = Lambda.copy()
        Gamma_old = Gamma.copy()

        # L 更新：先计算 U, V，再投影到 Ω 上
        U = Z - S - Lambda / rho
        V = Y - Gamma / rho
        L = projection_Omega((U + V) / 2, upbd)

        # S 更新：中心项为 Z - L - (Lambda/rho)
        S_tmp = Z - L - Lambda / rho
        S, iter_Newton = lp_thresholding_matrix(S_tmp, mu / rho, p)

        # Z 更新
        if identity == 1:
            Z = (H + rho * (L + S) + Lambda) / (1 + rho)
        elif blurring == 1:
            Z_tmp1 = rho * (L + S) + Lambda
            Sig = rho + (np.diag(Sc) ** 2) * (np.diag(Sr) ** 2).reshape(1, -1)  # 注意维度匹配
            Z = np.zeros_like(D)
            for iz in range(n):
                Z_tmp2 = (Ac.T @ H[:, iz].reshape(picsize) @ Ar) + Z_tmp1[:, iz].reshape(picsize)
                temp = Vc.T @ Z_tmp2 @ Vr
                temp = temp / Sig  # elementwise 除法
                Z_tmp3 = Vc @ temp @ Vr.T
                Z[:, iz] = Z_tmp3.flatten()

        # Y 更新：先计算 L + Gamma/rho，然后对每一列重构成图像进行去噪，再展平回来
        # Y = projection_Omega(L + Gamma / rho, upbd)

        tempY = L + Gamma / rho
        # Y = lp_thresholding_matrix(tempY, beta / rho, 1)[0]    # 使用l1范数去噪

        # Y = denoise_frame(tempY, method='BM3D', weight=0.01, p=1)

        picsize = opts.get('picsize', None)
        Y_new = np.zeros_like(tempY)
        # 对每一列进行去噪
        for i in range(tempY.shape[1]):
            # 重构为二维图像，注意这里使用 Fortran 顺序与原来投影保持一致
            frame = tempY[:, i].reshape(picsize, order='F')
            # 对图像进行去噪
            denoised_frame = denoise_frame(frame, method='BM3D', weight=beta / rho, p=1)
            # 将去噪后的图像展平并保存
            Y_new[:, i] = denoised_frame.flatten(order='F')
        Y = Y_new

        # 对偶变量更新
        Lambda = Lambda + tau * rho * (L + S - Z)
        Gamma = Gamma + tau * rho * (L - Y)

        # 计算收敛准则（可将 L, S, Z, Y 的变化累计）
        succ_chg = (norm(L - L_old, 'fro') + norm(S - S_old, 'fro') +
                    norm(Z - Z_old, 'fro') + norm(Y - Y_old, 'fro'))
        fnorm = norm(L, 'fro') + norm(S, 'fro') + norm(Z, 'fro') + norm(Y, 'fro') + 1
        relchg = succ_chg / fnorm

        # 计算目标函数值（Ψ 和 ψ_σ 为指示函数，当 L, Y 在 Ω 内时取0）
        obj_val = 0.5 * norm(H - Z, 'fro') ** 2 + mu * (norm(S.flatten(), p) ** p)

        if display and iter % displayfreq == 0:
            print("{:4d} | relchg: {:0.8e} | iter_Newton: {:4d} | obj_val: {:0.8e} | rho: {:g}".format(
                iter, relchg, iter_Newton, obj_val, rho))

        if iter > 20:
            break

    return L, S, Z, Y, iter


def run_me():
    """
    主函数：加载数据、设置参数、调用修改后的 ADMM 方法并显示结果
    """
    # 加载数据（确保 hall.mat 文件中包含 data, label, groundtruth, picture_size 等变量）
    mat_contents = sio.loadmat('hall.mat')
    D = mat_contents['data'][:, :20]  # 取前10帧
    # label = mat_contents.get('label', np.arange(D.shape[1])).flatten() - 1 # 匹配matlab的1-based index
    label = 5
    groundtruth = mat_contents.get('groundtruth', np.zeros_like(D)).flatten(order='F')
    picture_size = tuple(mat_contents.get('picture_size', [int(np.sqrt(D.shape[0]))] * 2))

    mu = 1e-2
    p = 0.5

    opts = {
        'tau': 0.8,
        'rho': 1.0,  # 惩罚参数
        'tol': 1e-4,
        'maxiter': 2000,
        'upbd': 1,
        'identity': 1,  # 若为1，则采用单位算子；若采用模糊算子则设为0并设置 blurring=1
        'blurring': 0,
        'display': 1,
        'displayfreq': 1,
        'picsize': (picture_size[0]),  # 图像大小
        # 若 blurring==1，需添加 'Ac', 'Ar', 'picsize' 参数
    }

    start_time = time.time()
    L_out, S_out, Z_out, Y_out, Iter_out = admm_lp(D, mu, p, opts)
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
        'L_out': L_out,
        'S_out': S_out,
        'Z_out': Z_out,
        'Y_out': Y_out,
        'Iter_out': Iter_out,
        'elapsed_time': elapsed_time,
        'spr': spr,
        'f_val': f_val,
    }

    sio.savemat('result_denoise_with_separation_0.01.mat', result)

    # 显示结果（根据实际数据可能需要调整 reshape 操作）
    display(D, *picture_size[0], D.shape[1], label, title="Original")
    display(L_out, *picture_size[0], D.shape[1], label, title="Background")
    display(S_out, *picture_size[0], D.shape[1], label, title="Foreground")


if __name__ == "__main__":
    run_me()
