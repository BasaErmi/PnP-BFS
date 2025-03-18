import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt


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
    Solve: min_X lam * ||X||_p^p + 1/2 * ||X - Y||_F^2,
    elementwise.
    When p==1, use soft thresholding.
    Otherwise, use Newton method for each element.
    Returns X and total number of Newton iterations.
    """
    iter_total = 0
    X = np.zeros_like(Y)
    if p == 1:
        X = np.sign(Y) * np.maximum(np.abs(Y) - lam, 0)
    else:
        B = np.abs(Y)
        # threshold tau as in MATLAB code
        tau_val = (2 * lam * (1 - p)) ** (1 / (2 - p)) + lam * p * (2 * lam * (1 - p)) ** ((p - 1) / (2 - p))
        index = B > tau_val
        # for elements above tau, use Newton iteration
        X_new = B[index].copy()
        Y_sub = B[index].copy()
        # Newton iteration
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
    return X, iter_total


def get_hz(Z, Ac, Ar, picsize):
    """
    Compute HZ in blurring case.
    Z: (num_pixels, n) matrix, each column转成图像大小 picsize
    Ac, Ar: blurring operator matrices
    picsize: tuple (height, width)
    """
    n = Z.shape[1]
    HZ = []
    for k in range(n):
        Zk = Z[:, k].reshape(picsize)
        hz = Ac @ Zk @ Ar.T
        HZ.append(hz.flatten())
    HZ = np.column_stack(HZ)
    return HZ


def admm_lp(D, mu, p, opts):
    """
    ADMM 求解背景前景分解问题：
    min mu*||S||_p^p + 1/2 * ||D - A(L+S)||_F^2
    s.t. L in C, 其中 C = {L | 每列相同且 ||L||_∞ <= upbd }
    其中 opts 为字典，包含各参数：
      tau, upbd, tol, InnerTol, maxiter, identity, blurring, heuristic, display, displayfreq
      对于 blurring 情况，还需提供 lambda_max, lambda_min, Ac, Ar, picsize
    返回：L_sol, S_sol, Iter_sol
    """
    # 参数设置
    tau = opts.get('tau', 0.8)
    upbd = opts.get('upbd', 1)
    OuterTol = opts.get('tol', 1e-4)
    InnerTol = opts.get('InnerTol', 5e-3)
    maxiter = opts.get('maxiter', 1000)
    identity = opts.get('identity', 1)
    blurring = opts.get('blurring', 0)
    heuristic = opts.get('heuristic', 1)
    display = opts.get('display', 1)
    displayfreq = opts.get('displayfreq', 1)

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

    # beta_bound 的计算
    term1 = lambda_max / tau
    term2 = lambda_max * tau
    term3 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * lambda_max ** 2 / tau)
    term4 = -0.5 * lambda_min + 0.5 * np.sqrt(lambda_min ** 2 + 8 * tau ** 2 * lambda_max ** 2 / (1 + tau - tau ** 2))
    beta_bound = max([term1, term2, term3, term4])

    # 初始 beta 及自适应参数
    if heuristic:
        sigma1 = opts.get('sigma1', 0.6)
        sigma2 = opts.get('sigma2', 0.3)
        rho_beta = opts.get('rho_beta', 1.1)
        beta = sigma1 * beta_bound
        beta_max = 1.01 * beta_bound
        count = 0
    else:
        beta = 1.01 * beta_bound

    m, n = D.shape

    # 初始化变量
    if all(k in opts for k in ['L0', 'S0', 'Z0', 'Lambda0']):
        L = opts['L0']
        S = opts['S0']
        Z = opts['Z0']
        Lambda = opts['Lambda0']
    else:
        kappa = 1
        # 需要保证初始矩阵满足约束为秩1矩阵。求每行的平均值作为一列然后代替所有列。
        PcD = np.tile(np.mean(kappa * D, axis=1, keepdims=True), (1, n))
        PcD = np.sign(PcD) * np.minimum(np.abs(PcD), upbd) #截断超过上限1的数据（因为为图像数据）
        L = PcD.copy()
        S = np.zeros((m, n))
        Z = L.copy()
        if identity == 1:
            Lambda = D - Z
        elif blurring == 1:
            Lambda = np.zeros_like(D)
            for il in range(n):
                tmp = D[:, il].reshape(picsize) - (Ac @ Z[:, il].reshape(picsize) @ Ar.T)
                # 根据MATLAB代码，计算 Lambda 的列
                Lam_tmp = Ac.T @ tmp @ Ar
                Lambda[:, il] = Lam_tmp.flatten()

    if display:
        print("\n------------ 3-blocks ADMM for nonconvex background model ---------------")
        print("p = {:.1f},   tau = {:.1f}".format(p, tau))
        print(
            "iter   |   relerr         |   iter_Newton   |   f_val         |   Phi_value     |  Gap_Phi        |  beta")

    succ_chg = np.inf

    # 主循环
    for iter in range(1, maxiter + 1):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()
        Lambdak = Lambda.copy()
        succ_chgk = succ_chg # 用于记录上一次迭代的变化幅度，判断收敛

        # L 更新：先计算临时变量，即找出离Z + Lambda / beta - S的L值
        Tmp1 = Z + Lambda / beta
        L_tmp = Tmp1 - S
        # 然后满足\in \Omega 的约束,相当于投影到约束上
        # 保证秩1约束
        L_mean = np.mean(L_tmp, axis=1, keepdims=True)
        L = np.tile(L_mean, (1, n))
        L = np.sign(L) * np.minimum(np.abs(L), upbd)

        # S 更新
        S_tmp = Tmp1 - L #先满足F范数内的约束
        S, iter_Newton = lp_thresholding_matrix(S_tmp, mu / beta, p)

        # Z 更新
        if identity == 1:
            Z = (D + beta * (L + S) - Lambda) / (1 + beta)
        elif blurring == 1:
            Z_tmp1 = beta * (L + S) - Lambda
            Sig = beta + (np.diag(Sc) ** 2) * (np.diag(Sr) ** 2).reshape(1, -1)  # 注意维度匹配
            Z = np.zeros_like(D)
            for iz in range(n):
                Z_tmp2 = (Ac.T @ D[:, iz].reshape(picsize) @ Ar) + Z_tmp1[:, iz].reshape(picsize)
                # 对 Z_tmp2 进行 SVD（或直接使用 Vc,Vr）：
                # 这里简单按 MATLAB 逻辑使用 Vc 和 Vr 来调整
                temp = Vc.T @ Z_tmp2 @ Vr
                temp = temp / Sig  # elementwise除法
                Z_tmp3 = Vc @ temp @ Vr.T
                Z[:, iz] = Z_tmp3.flatten()
        # 更新乘子
        Lambda = Lambda - tau * beta * (L + S - Z)

        # 计算相对误差
        succ_chg = norm(L - Lk, 'fro') + norm(Z - Zk, 'fro')
        fnorm = norm(L, 'fro') + norm(Z, 'fro') + 1
        relchg = succ_chg / fnorm

        if relchg < OuterTol:
            succ_chg_2nd = norm(S - Sk, 'fro') + norm(Lambda - Lambdak, 'fro')
            fnorm_2nd = norm(S, 'fro') + norm(Lambda, 'fro') + 1
            relchg_2nd = succ_chg_2nd / fnorm_2nd
            if relchg_2nd < InnerTol:
                if display:
                    if identity == 1:
                        Phi_valk = mu * (norm(Sk.flatten(), p) ** p) + 0.5 * norm(D.flatten() - Zk.flatten()) ** 2 - \
                                   np.dot(Lambdak.flatten(), (Lk + Sk - Zk).flatten()) + \
                                   (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                            (Lk + Sk - Zk).flatten()) ** 2
                        Phi_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - Z.flatten()) ** 2 - \
                                  np.dot(Lambda.flatten(), (L + S - Z).flatten()) + \
                                  (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                            (L + S - Z).flatten()) ** 2
                        Gap_Phi = Phi_valk - Phi_val
                        f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(
                            D.flatten() - L.flatten() - S.flatten()) ** 2
                    elif blurring == 1:
                        HZk = get_hz(Zk, Ac, Ar, picsize)
                        HZ = get_hz(Z, Ac, Ar, picsize)
                        Phi_valk = mu * (norm(Sk.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HZk.flatten()) ** 2 - \
                                   np.dot(Lambdak.flatten(), (Lk + Sk - Zk).flatten()) + \
                                   (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                            (Lk + Sk - Zk).flatten()) ** 2
                        Phi_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HZ.flatten()) ** 2 - \
                                  np.dot(Lambda.flatten(), (L + S - Z).flatten()) + \
                                  (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                            (L + S - Z).flatten()) ** 2
                        Gap_Phi = Phi_valk - Phi_val
                        HLS = get_hz(L + S, Ac, Ar, picsize)
                        f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HLS.flatten()) ** 2
                    print(
                        "{:4d} | {:0.8e} | {:4d} | {:0.8e} | {:0.8e} | {:0.8e} | {:g}".format(iter, relchg, iter_Newton,
                                                                                              f_val, Phi_val, Gap_Phi,
                                                                                              beta))
                break

        if display and iter % displayfreq == 0:
            if identity == 1:
                Phi_valk = mu * (norm(Sk.flatten(), p) ** p) + 0.5 * norm(D.flatten() - Zk.flatten()) ** 2 - \
                           np.dot(Lambdak.flatten(), (Lk + Sk - Zk).flatten()) + \
                           (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                    (Lk + Sk - Zk).flatten()) ** 2
                Phi_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - Z.flatten()) ** 2 - \
                          np.dot(Lambda.flatten(), (L + S - Z).flatten()) + \
                          (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                    (L + S - Z).flatten()) ** 2
                Gap_Phi = Phi_valk - Phi_val
                f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - L.flatten() - S.flatten()) ** 2
            elif blurring == 1:
                HZk = get_hz(Zk, Ac, Ar, picsize)
                HZ = get_hz(Z, Ac, Ar, picsize)
                Phi_valk = mu * (norm(Sk.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HZk.flatten()) ** 2 - \
                           np.dot(Lambdak.flatten(), (Lk + Sk - Zk).flatten()) + \
                           (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                    (Lk + Sk - Zk).flatten()) ** 2
                Phi_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HZ.flatten()) ** 2 - \
                          np.dot(Lambda.flatten(), (L + S - Z).flatten()) + \
                          (beta / 2 + max(1 - tau, (tau - 1) * tau ** 2 / (1 + tau - tau ** 2))) * norm(
                    (L + S - Z).flatten()) ** 2
                Gap_Phi = Phi_valk - Phi_val
                HLS = get_hz(L + S, Ac, Ar, picsize)
                f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HLS.flatten()) ** 2
            print(
                "{:4d} | {:0.8e} | {:4d} | {:0.8e} | {:0.8e} | {:0.8e} | {:g}".format(iter, relchg, iter_Newton, f_val,
                                                                                      Phi_val, Gap_Phi, beta))

        # heuristic 更新 beta
        if heuristic:
            if (0.99 * succ_chgk - succ_chg) < 0:
                count += 1
            if count >= np.ceil(sigma2 * iter) or fnorm > 1e10:
                beta = min(beta * rho_beta, beta_max)

    return L, S, iter


def run_me():
    """
    主函数：加载数据、设置参数、调用 ADMM 或 PALM 方法并显示结果
    """
    # 加载数据（需提前准备 hall.mat 文件，文件中包含变量 data, label, groundtruth, picture_size 等）
    mat_contents = sio.loadmat('hall.mat')
    D = mat_contents['data']

    # 其它变量根据 hall.mat 的内容进行设置，例如 label, groundtruth, picture_size
    label = mat_contents.get('label', np.arange(D.shape[1])).flatten() - 1
    groundtruth = mat_contents.get('groundtruth', np.zeros_like(D)).flatten(order='F')
    picture_size = tuple(mat_contents.get('picture_size', [np.sqrt(D.shape[0]).astype(int)] * 2))


    mu = 1e-2
    p = 0.5

    opts = {
        'tau': 0.8,
        'tol': 1e-4,
        'InnerTol': 5e-3,
        'maxiter': 2000,
        'identity': 1,  # 如果需要使用模糊算子，则置为0，并设置 blurring=1 以及相应参数
        'blurring': 0,
        'heuristic': 1,
        'display': 1,
        'displayfreq': 1,
        # 若 blurring==1，需添加以下参数：
        # 'lambda_max': lambda_max, 'lambda_min': lambda_min,
        # 'Ac': Ac, 'Ar': Ar, 'picsize': picture_size
    }

    # 选择使用 PALM 方法（也可调用 admm_lp(D, mu, p, opts)）
    start_time = time.time()
    L_out, S_out, Iter_out = admm_lp(D, mu, p, opts)
    elapsed_time = time.time() - start_time

    if opts['identity'] == 1:
        f_val = mu * (norm(S_out.flatten(), p) ** p) + 0.5 * norm(D.flatten() - L_out.flatten() - S_out.flatten()) ** 2
    else:
        # 对于 blurring 情况
        HLS = get_hz(L_out + S_out, opts['Ac'], opts['Ar'], picture_size)
        f_val = mu * (norm(S_out.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HLS.flatten()) ** 2

    # 计算稀疏率
    S_mask = np.abs(S_out) > 1e-3
    spr = np.sum(S_out != 0) / D.size

    # 计算前景检测评价（需要 label 和 groundtruth 信息）
    # 此处简单示例，具体计算方式可能需要根据实际 groundtruth 修改
    foreground = S_mask[:, label]
    ind_f = np.where(foreground.flatten() == 1)[0]
    ind_g = np.where(groundtruth == 1)[0]
    ind_correct = np.intersect1d(ind_f, ind_g)
    precision = len(ind_correct) / (len(ind_f) + 1e-8)
    recall = len(ind_correct) / (len(ind_g) + 1e-8)
    F = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n$p$ & $mu$ & $tau$ & Iter & Time(s) & spr & F-measure & f_val")
    print("{:g}  &  {:.0e}  & {:.1f} & {:d} & {:.2f} & {:.4f} & {:.4f} & {:.4f}".format(
        p, mu, opts['tau'], Iter_out, elapsed_time, spr, F, f_val))

    # 显示结果（需要根据具体数据 reshape 操作）
    display(D, *picture_size[0], D.shape[1], label, title="Original")
    display(L_out, *picture_size[0], D.shape[1], label, title="Background")
    display(S_out, *picture_size[0], D.shape[1], label, title="Foreground")


if __name__ == "__main__":
    run_me()



def palm_lp(D, mu, p, opts):
    """
    PALM 方法求解背景前景分解问题：
    min mu*||S||_p^p + 1/2*||D - A(L+S)||_F^2,  L in C（背景一致性约束）
    参数 opts 同 admm_lp，区分 identity 和 blurring 两种情况
    返回 L_sol, S_sol, Iter_sol
    """
    upbd = opts.get('upbd', 1)
    Tol = opts.get('tol', 1e-4)
    maxiter = opts.get('maxiter', 1000)
    identity = opts.get('identity', 1)
    blurring = opts.get('blurring', 0)
    display = opts.get('display', 1)
    displayfreq = opts.get('displayfreq', 1)

    if identity == 1:
        lambda_max = 1
    elif blurring == 1:
        if not all(k in opts for k in ['lambda_max', 'lambda_min', 'Ac', 'Ar', 'picsize']):
            raise ValueError("For blurring case, lambda_max, lambda_min, Ac, Ar, picsize must be provided.")
        lambda_max = opts['lambda_max']
        picsize = opts['picsize']
        Ac = opts['Ac']
        Ar = opts['Ar']
    else:
        raise ValueError("identity 和 blurring 选项设置错误")

    m, n = D.shape
    c = lambda_max / 0.99
    d = lambda_max / 0.99

    # 初始化
    if all(k in opts for k in ['L0', 'S0']):
        L = opts['L0']
        S = opts['S0']
    else:
        L = np.zeros((m, n))
        S = np.zeros((m, n))

    if display:
        print("\n------------ PALM for nonconvex background model with p = {:.1f} ---------------------".format(p))
        print("iter   |   relerr         |   iter_Newton   |   f_val")

    for iter in range(1, maxiter + 1):
        Lk = L.copy()
        Sk = S.copy()

        if identity == 1:
            # L 更新
            L_tmp = L - (1 / c) * ((L + S) - D)
            L_mean = np.mean(L_tmp, axis=1, keepdims=True)
            L = np.tile(L_mean, (1, n))
            L = np.sign(L) * np.minimum(np.abs(L), upbd)
            # S 更新
            S_tmp = S - (1 / d) * ((L + S) - D)
            S, iter_Newton = lp_thresholding_matrix(S_tmp, mu / d, p)
        elif blurring == 1:
            # L 更新
            L1 = []
            for il in range(n):
                l_tmp = Ac.T @ (Ac @ (L[:, il] + S[:, il]).reshape(picsize) @ Ar.T - D[:, il].reshape(picsize)) @ Ar
                L1.append(l_tmp.flatten())
            L1 = np.column_stack(L1)
            L2 = L - (1 / c) * L1
            L_mean = np.mean(L2, axis=1, keepdims=True)
            L = np.tile(L_mean, (1, n))
            L = np.sign(L) * np.minimum(np.abs(L), upbd)
            # S 更新
            S1 = []
            for is_ in range(n):
                s_tmp = Ac.T @ (Ac @ (L[:, is_] + S[:, is_]).reshape(picsize) @ Ar.T - D[:, is_].reshape(picsize)) @ Ar
                S1.append(s_tmp.flatten())
            S1 = np.column_stack(S1)
            S2 = S - (1 / d) * S1
            S, iter_Newton = lp_thresholding_matrix(S2, mu / d, p)

        succ_chg = norm(L - Lk, 'fro') + norm(S - Sk, 'fro')
        fnorm = norm(L, 'fro') + norm(S, 'fro') + 1
        relchg = succ_chg / fnorm

        if relchg < Tol:
            if display:
                if identity == 1:
                    f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - L.flatten() - S.flatten()) ** 2
                elif blurring == 1:
                    HLS = get_hz(L + S, Ac, Ar, picsize)
                    f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HLS.flatten()) ** 2
                print(" {:4d} | {:0.8e} | {:4d} | {:0.8e}".format(iter, relchg, iter_Newton, f_val))
            break

        if display and iter % displayfreq == 0:
            if identity == 1:
                f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - L.flatten() - S.flatten()) ** 2
            elif blurring == 1:
                HLS = get_hz(L + S, Ac, Ar, picsize)
                f_val = mu * (norm(S.flatten(), p) ** p) + 0.5 * norm(D.flatten() - HLS.flatten()) ** 2
            print(" {:4d} | {:0.8e} | {:4d} | {:0.8e}".format(iter, relchg, iter_Newton, f_val))

    return L, S, iter
