# -*- coding: utf-8 -*-
"""FaultSeg3D 合成数据加载器 + 稀疏 2D 标签构建。

数据格式（来自 FaultSeg3D utils.py）：
  - 文件: <id>.dat，np.fromfile(..., dtype=np.float32)，reshape (128,128,128)
  - 预处理: z-score 归一化 + np.transpose（完全反转轴序，n3n2n1 → n1n2n3）
  - 目录: data/train/{seis,fault}/0..199.dat, data/validation/{seis,fault}/0..19.dat

稀疏标签（FaultNet 论文 III-A）：
  - 完整 3D 标签中，断层=1，背景=0
  - 只保留少量 inline 切片为"已标注"，其余位置标 -1（未标注）
"""
import os

import numpy as np


def load_dat(path, shape=(128, 128, 128), normalize=True):
    """读一个 .dat 体数据，复刻 FaultSeg3D 预处理。"""
    arr = np.fromfile(path, dtype=np.float32).reshape(shape)
    if normalize:
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)     # z-score
    arr = np.transpose(arr)                                # n3n2n1 -> n1n2n3
    return arr.astype(np.float32)


class FaultSegDataset:
    """加载 FaultSeg3D 训练/验证集。

    每个样本返回 (seismic, label3d)：
      seismic: (1, D, H, W) float32，已归一化
      label3d: (D, H, W) int，0/1
    """

    def __init__(self, root, split="train", n=None):
        self.root = root
        self.seis_dir = os.path.join(root, split, "seis")
        self.fault_dir = os.path.join(root, split, "fault")
        files = [f for f in os.listdir(self.seis_dir) if f.endswith(".dat")]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        self.files = files[:n] if n else files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        s = load_dat(os.path.join(self.seis_dir, name))
        f = load_dat(os.path.join(self.fault_dir, name), normalize=False)
        return s[None], (f > 0).astype(np.int64)


def make_sparse_label(label3d, frac=0.033, axis=0, rng=None):
    """从完整 3D 标签抽取稀疏 2D 切片，模拟"人工标注"。

    参数:
        label3d: (D, H, W) 0/1
        frac: 标注切片比例（论文 3.3%）
        axis: 沿哪个轴抽切片（0=inline/时间轴）
    返回:
        y: (D, H, W)，标注切片保留 0/1，其余 -1
    """
    rng = rng or np.random.default_rng()
    D, H, W = label3d.shape
    n_label = max(1, int(round(D * frac)))
    idxs = rng.choice(D, size=n_label, replace=False)

    y = np.full_like(label3d, -1, dtype=np.int64)
    for i in idxs:
        if axis == 0:
            y[i] = label3d[i]
        elif axis == 1:
            y[:, i] = label3d[:, i]
        else:
            y[:, :, i] = label3d[:, :, i]
    return y


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    if not os.path.isdir(root):
        print(f"目录不存在: {root}。请先把 FaultSeg3D 数据放到 data/ 下")
        sys.exit(1)
    ds = FaultSegDataset(root, "train")
    print(f"训练样本数: {len(ds)}")
    s, f = ds[0]
    print("seismic", s.shape, s.dtype, f"[{s.min():.2f},{s.max():.2f}]")
    print("label 断层占比", (f > 0).mean() * 100, "%")
    y = make_sparse_label(f, frac=0.033, rng=np.random.default_rng(0))
    labeled = (y >= 0).mean() * 100
    print(f"稀疏标签: 已标注占比 {labeled:.2f}%，其余为 -1")
