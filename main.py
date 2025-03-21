from background_extraction import background_extraction

if __name__ == "__main__":
    opts = {
        'tau': 0.8,
        'rho': 1,  # 惩罚参数
        'Outertol': 1e-4,
        'InnerTol': 5e-3,
        'maxiter': 200,
        'upbd': 1,
        'identity': 1,  # 若为1，则采用单位算子；若采用模糊算子则设为0并设置 blurring=1
        'blurring': 0,
        'display': 1,
        'displayfreq': 1,
        'picsize': None,  # 图像大小
        'heuristic': 1,
        'sigma1': 0.6,
        'sigma2': 0.3,
        'rho_beta': 1.1,
        'mu': 1e-2,  # S 惩罚参数
        'p': 0.5,  # S 上的 lp 正则参数
        'denoise_method': 'GSDRUNet',  # 是否进行去噪/具体去噪器
        'noise_type': 'fixed',
        # 若 blurring==1，需添加 'Ac', 'Ar', 'picsize' 参数
    }
    background_extraction(opts, noise_level=0.001, dataset_name='hall')
