# -*- coding: utf-8 -*-
"""合成地震数据 + 断层标签生成（模拟 FaultSeg3D 的思路）。

用于 MD loss 训练 FaultNet。生成方式参考 Wu et al. 2019 (FaultSeg3D)：
  - 随机起伏的反射层（模拟地层）
  - 随机断层（走向/断距）错断反射层
  - 加噪声
  - 标签：断层面体素 = 1，其余 = 0

注意：本生成器是"可运行的教学实现"，参数为参考值；要复现论文指标
应使用 FaultSeg3D 官方公开的 200 对合成数据（Google Drive）。
"""
import numpy as np


def make_one(rng, shape=(128, 128, 128), n_layers=8, n_faults=2,
             noise=0.08, fold_amp=3.0):
    """生成一对 (seismic, fault_label)。

    返回:
        seismic: float32, shape (D, H, W)，归一化振幅（约 N(0,1)）
        label:   float32, shape (D, H, W)，0/1
    """
    D, H, W = shape
    h = np.arange(H, dtype=np.float32)
    w = np.arange(W, dtype=np.float32)

    # 构造褶皱：低频正弦扰动各层深度
    fold = (fold_amp * np.sin(2 * np.pi * w[None, :] / W * 2)
            + fold_amp * 0.6 * np.cos(2 * np.pi * h[:, None] / H * 1.5))
    fold = fold.astype(np.float32)          # (H, W)

    layer_z = np.linspace(8, D - 8, n_layers, dtype=np.float32)

    # 断层参数：(走向角, 断距, 断层面水平位置 w0/h0)
    faults = []
    for _ in range(n_faults):
        strike = rng.uniform(0, np.pi)
        dip = rng.uniform(3.0, 10.0)
        w0 = W / 2 + rng.uniform(-0.2, 0.2) * W
        h0 = H / 2 + rng.uniform(-0.2, 0.2) * H
        faults.append((strike, dip, w0, h0))

    # 断层面平面函数：水平坐标 -> 标量（>0 一侧下移，=0 即断层面）
    def plane(strike, w0, h0):
        return ((np.cos(strike) * (w[None, :] - w0)
                 + np.sin(strike) * (h[:, None] - h0)) / max(W, H))

    # 反射系数体：把每层写到其错断后的深度
    reflect = np.zeros((D, H, W), dtype=np.float32)
    for lz in layer_z:
        depth = lz + fold
        for (strike, dip, w0, h0) in faults:
            depth = depth + np.where(plane(strike, w0, h0) > 0, dip, 0.0).astype(np.float32)
        depth = np.clip(np.floor(depth).astype(np.int64), 0, D - 1)
        reflect[depth, np.arange(H)[:, None], np.arange(W)[None, :]] = 1.0

    # 短子波沿深度方向卷积，模拟地震道
    wl = np.array([-0.2, 0.6, 1.0, 0.6, -0.2], dtype=np.float32)
    seismic = np.apply_along_axis(
        lambda tr: np.convolve(tr, wl, mode="same"), 0, reflect)
    seismic = seismic.astype(np.float32)

    # 加噪声
    seismic = seismic + rng.normal(0, noise, size=seismic.shape).astype(np.float32)

    # 标签：断层面（plane 接近 0 的带）在整个深度范围标记为断层
    label = np.zeros((D, H, W), dtype=np.float32)
    for (strike, dip, w0, h0) in faults:
        band = np.abs(plane(strike, w0, h0)) < (2.0 / max(W, H))   # (H, W) bool
        label[:, band] = 1.0

    # 归一化
    seismic = (seismic - seismic.mean()) / (seismic.std() + 1e-6)
    return seismic.astype(np.float32), label.astype(np.float32)


def make_batch(n, rng, shape=(128, 128, 128)):
    """批量生成。返回 (seismic_batch, label_batch)，shape (n,1,D,H,W)。"""
    ss, ll = [], []
    for _ in range(n):
        s, l = make_one(rng, shape)
        ss.append(s[None])
        ll.append(l[None])
    return np.stack(ss), np.stack(ll)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    s, l = make_one(rng, (96, 96, 96), n_faults=2)
    print(f"生成耗时 {time.time()-t0:.2f}s")
    print("seismic", s.shape, s.dtype, f"[{s.min():.2f},{s.max():.2f}]")
    print("label 1 占比", (l > 0).mean() * 100, "%")
