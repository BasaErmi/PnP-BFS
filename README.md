# A Provably Convergent Plug-and-Play Alternating Direction Method of Multipliers with Deep Image Prior for Background/Foreground Separation

## Modeling and Algorithm Iteration Framework

This repository implements image denoising and background extraction algorithms based on deep learning and traditional optimization methods. It primarily implements a Plug-and-Play Alternating Direction Method of Multipliers (ADMM) for solving the following optimization problem:

![image](https://github.com/user-attachments/assets/b1cf70d1-943b-4e50-bacb-18085d83f2f8)


The corresponding ADMM algorithm is:

![image](https://github.com/user-attachments/assets/ef1f8fd6-6d9c-4e56-bea1-9c17e85f375d)


The detailed iteration scheme is as follows:

![image](https://github.com/user-attachments/assets/8e1190f6-0bff-4207-93c5-7287467aa37a)


## **Running Example**

Download pretrained checkpoint [GS-DRUNet pretrained ckpts](https://plmbox.math.cnrs.fr/f/04318d36824443a6bf8d/?dl=1) for grayscale denoising and save it as `ProvablePnP-ADMM-BFS/model/weights/GSDRUNet_grayscale.ckpt`

Simply run `main.py` to test the background extraction functionality:

```bash
python main.py
```

Check the algorithm’s iterative process and denoising effect based on the output.

You can modify the `opts` parameters as needed to adjust the penalty parameters, convergence criteria, and other settings in the algorithm.

```python
    opts = {
        'tau': 0.8,
        'rho': 1,  # penalty parameter
        'Outertol': 1e-4,
        'InnerTol': 5e-3,
        'maxiter': 200,
        'upbd': 1,
        'identity': 1,  # if 1, use the identity operator; if using a blur operator, set to 0 and configure 'blurring' as 1
        'blurring': 0,
        'display': 1,
        'displayfreq': 1,
        'picsize': None,  # image size
        'heuristic': 1,
        'sigma1': 0.6,
        'sigma2': 0.3,
        'rho_beta': 1.1,
        'mu': 1e-2,  # penalty parameter for S
        'p': 0.5,  # lp regularization parameter for S
        'denoise_method': 'GSDRUNet',  # choose the denoiser (e.g., BM3D or GSDRUNet), or set to False for no denoising prior
        'noise_type': 'fixed',
        # If blurring==1, additional parameters 'Ac', 'Ar', and 'picsize' are required
    }
```

- In the `denoise_method` parameter, you can select either `BM3D` or `GSDRUNet` for different denoisers, or choose `False` to indicate that no denoising prior is applied.

Running the code will yield results similar to those obtained with the `hall.mat` dataset.
![image](https://github.com/user-attachments/assets/7ba588c7-b2e7-44e6-8c50-5d3a2ffb2b77)

![image](https://github.com/user-attachments/assets/b2a33f00-310b-4dc3-a9f4-cfa555017ef3)

![image](https://github.com/user-attachments/assets/4fd8bde6-b57f-42df-a8db-9c9d5c91a1e6)

![image](https://github.com/user-attachments/assets/e186f2bc-b114-42b2-b4f4-3202f15c2b1c)


**--------More features and details are under development; code is still being written------------**

