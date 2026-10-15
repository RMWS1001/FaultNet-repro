# -*- coding: utf-8 -*-
"""FaultNet 训练脚本（从零训练，MD loss + 稀疏 2D 标签）。

用法:
    python train.py --gamma 0.5 --epochs 100 --shape 128 128 128

说明:
  - 网络结构见 network.py（已用官方权重验证对齐）
  - MD loss 见 md_loss.py（论文 Equation 13/14 精确实现）
  - 合成数据见 synthetic.py（教学实现，参考 FaultSeg3D；复现论文指标请用官方 200 对数据）
"""
import argparse
import time

import numpy as np
import torch
from torch import optim

from network import FaultNet
from md_loss import MDLoss, build_label
from synthetic import make_one


def sparse_labels(label3d, frac=0.033, rng=None):
    """从完整 3D 标签抽取稀疏 2D 切片作为"人工标注"。

    只保留约 frac 比例的 inline 切片为已标注，其余位置 -1。
    返回 shape (D,H,W) 的标签（1/0/-1）。
    """
    D, H, W = label3d.shape
    n_label = max(1, int(round(D * frac)))
    idxs = rng.choice(D, size=n_label, replace=False)
    return build_label(torch.from_numpy(label3d), {"inline": idxs.tolist()}).numpy()


def dice_score(pred_bin, label_bin):
    inter = (pred_bin * label_bin).sum()
    return (2 * inter / (pred_bin.sum() + label_bin.sum() + 1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20, help="每 epoch 的迭代步数")
    ap.add_argument("--shape", type=int, nargs="+", default=[128, 128, 128])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-frac", type=float, default=0.033, help="标注切片比例")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="FaultNet_trained.pt")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"设备: {dev}")

    model = FaultNet().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_par/1e6:.3f} M")

    criterion = MDLoss(gamma=args.gamma, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    shape = tuple(args.shape)
    assert all(s % 16 == 0 for s in shape), "shape 各维需能被 16 整除"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        tot_dice = 0.0
        for step in range(args.steps):
            # 1) 在线生成一对合成数据
            seismic, label3d = make_one(rng, shape)
            y = sparse_labels(label3d, frac=args.label_frac, rng=rng)

            # 2) 转 tensor（加 batch/通道维）
            x = torch.from_numpy(seismic)[None, None].to(dev)   # (1,1,D,H,W)
            y = torch.from_numpy(y)[None, None].to(dev)          # (1,1,D,H,W)，含 -1

            # 3) 前向 + 损失
            pred = model(x)                                      # (1,1,D,H,W)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4) 监控：用完整 3D 标签算 Dice（仅监控，不参与训练）
            with torch.no_grad():
                pred_bin = (pred > 0.5).float()
                full = torch.from_numpy(label3d)[None, None].to(dev)
                d = dice_score(pred_bin, full)
            tot_loss += loss.item()
            tot_dice += d

        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  loss={tot_loss/args.steps:.4f}  "
              f"dice(监控)={tot_dice/args.steps:.4f}  {dt:.1f}s")

    torch.save(model.state_dict(), args.out)
    print(f"已保存模型到 {args.out}")


if __name__ == "__main__":
    main()
