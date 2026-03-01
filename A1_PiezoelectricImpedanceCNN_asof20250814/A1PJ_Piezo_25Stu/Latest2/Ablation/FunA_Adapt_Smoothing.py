import numpy as np
from scipy.ndimage import gaussian_filter1d

def adaptive_smooth(data, base_sigma=10, window=21, alpha=25):  # 局部方差自适应平滑函数
    # 如果是一维数据，转为二维方便处理
    if data.ndim == 1:
        data = data[:, np.newaxis]

    rows, cols = data.shape
    smoothed = np.zeros_like(data)
    half = window // 2

    for col in range(cols):
        sig = data[:, col]
        for i in range(rows):
            left = max(0, i - half)
            right = min(rows, i + half + 1)
            local_std = np.std(sig[left:right])
            sigma = base_sigma / (1 + alpha * local_std)
            smoothed[i, col] = gaussian_filter1d(sig, sigma=sigma, mode='reflect')[i]

    return smoothed.squeeze()  # 局部方差自适应平滑函数 # 局部方差自适应平滑函数
