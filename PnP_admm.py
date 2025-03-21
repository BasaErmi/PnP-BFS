import numpy as np
import scipy.io as sio
from numpy.linalg import norm
from utils import *

def PnP_admm(H, mu, p, opts, noise_level):
    """
    针对如下优化问题的 ADMM 求解：
        min_{ell, S, Z}  0.5*||H - A(Z)||_F^2 + Psi(ell) + Phi(S)
        s.t.            S + ell * 1^T = Z
    其中我们假设 A 为单位算子。

    输入:
      H   : 数据矩阵，形状 (m, n)
      mu  : S 正则化参数
      p   : S 上的 lp 惩罚参数（0 < p <= 1）
      opts: 包含算法参数的字典
    输出:
      ell : 优化得到的 ell 向量，形状 (m, 1)
      S   : 优化得到的 S 矩阵，形状 (m, n)
      Z   : 优化得到的 Z 矩阵，形状 (m, n)
      k   : 迭代次数
    """
    succ_chg = np.inf

    tau = opts.get('tau')
    rho = opts.get('rho')
    Outertol = opts.get('Outertol')
    InnerTol = opts.get('InnerTol')
    maxiter = opts.get('maxiter')
    display = opts.get('display')
    disp_freq = opts.get('displayfreq')
    picsize = opts.get('picsize')
    heuristic = opts.get('heuristic')
    identity = opts.get('identity')
    blurring = opts.get('blurring')
    succ_chgk = succ_chg  # 用于记录上一次迭代的变化幅度，判断收敛
    denoise_method = opts.get('denoise_method')

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
    # term1 = lambda_max / tau
    # term2 = lambda_max * tau
    # term3 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * lambda_max ** 2 / tau)
    # term4 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * tau ** 2 * lambda_max ** 2 / (1 + tau - tau ** 2))
    # rho_bound = max([term1, term2, term3, term4])

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

        # # 使用去噪器对 m_bar 进行去噪，即计算其proximal map
        # # 对图像进行去噪
        ell = denoiser(m_bar,picsize, method=denoise_method, noise_level=noise_level)
        # print(ell.shape)

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