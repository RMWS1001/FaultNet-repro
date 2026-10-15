# -*- coding: utf-8 -*-
"""FaultNet 论文复现训练脚本（Group-4：纯合成数据）。

复现论文（arXiv:2110.05319, DOI 10.1109/TGRS.2022.3196810）实验设置：
  - 数据: FaultSeg3D 合成数据 200 train + 20 val（128^3），
          上采样到 256^3/384^3 后随机截 128^3，扩到 600 train + 120 val
  - 增强: 随机旋转 0/90/180/270 度（沿时间轴）
  - 阶段1: Dice loss 预训练 200 epochs（Adam lr=1e-3, batch=10）
  - 阶段2: MD loss(γ) 微调 200 epochs
  - 指标: IOU（阈值 0.5）
  - 目标(Group-4, Fault-Net): γ=0.5 -> 67.53, γ=0.6 -> 67.59, γ=0.7 -> 67.33

用法:
  python train_faultnet.py --gamma 0.7 --data FaultNet_rebuild/data
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import FaultNet
from md_loss import md_loss


# ---------------- 数据 ----------------
def load_all(root, split):
    """加载某个 split 的所有 (seismic, label)，返回 CPU 上的 torch 张量。"""
    from dataset import FaultSegDataset
    ds = FaultSegDataset(root, split)
    ss, ll = [], []
    for i in range(len(ds)):
        s, l = ds[i]                 # s: (1,D,H,W), l: (D,H,W)
        ss.append(s[0])              # (D,H,W)
        ll.append(l)
    # (N, 1, D, H, W)
    return torch.stack(ss).unsqueeze(1), torch.stack(ll).unsqueeze(1).float()


def make_batch_gpu(seismic, labels, bs, dev, rng, crop=128):
    """GPU 上采样 + 随机截取 + 随机旋转，返回一个 batch（都在 dev 上）。"""
    N = seismic.shape[0]
    idxs = rng.integers(0, N, bs)
    s = seismic[idxs].to(dev)        # (bs,1,128,128,128)
    l = labels[idxs].to(dev)

    # 上采样到 256 或 384
    scale = int(rng.choice([2, 3]))
    s = F.interpolate(s, scale_factor=scale, mode="trilinear", align_corners=False)
    l = F.interpolate(l, scale_factor=scale, mode="nearest")

    # 随机截取 128^3
    D, H, W = s.shape[2:]
    d0 = int(rng.integers(0, D - crop + 1))
    h0 = int(rng.integers(0, H - crop + 1))
    w0 = int(rng.integers(0, W - crop + 1))
    s = s[:, :, d0:d0 + crop, h0:h0 + crop, w0:w0 + crop]
    l = l[:, :, d0:d0 + crop, h0:h0 + crop, w0:w0 + crop]

    # 随机旋转（沿时间轴，即 dims=(3,4) 平面）
    k = int(rng.integers(0, 4))
    s = torch.rot90(s, k, dims=(3, 4))
    l = torch.rot90(l, k, dims=(3, 4))

    return s, l


# ---------------- 损失 / 指标 ----------------
def dice_loss(y_hat, y, eps=1e-6):
    num = (y_hat * y).sum(dim=(1, 2, 3, 4))
    den = (0.5 * y_hat + 0.5 * y).sum(dim=(1, 2, 3, 4))
    return (1 - num / (den + eps)).mean()


def iou_score(y_hat, y, thr=0.5):
    pred = (y_hat > thr).float()
    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    return tp / (tp + fp + fn + 1e-6) * 100


# ---------------- 训练 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--epochs-pre", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"设备: {dev}, gamma={args.gamma}, batch={args.batch_size}", flush=True)

    # 加载数据（一次性到 CPU 内存）
    print("加载数据...", flush=True)
    t0 = time.time()
    tr_s, tr_l = load_all(args.data, "train")
    va_s, va_l = load_all(args.data, "validation")
    print(f"train {tuple(tr_s.shape)}, validation {tuple(va_s.shape)}, "
          f"用时 {time.time()-t0:.0f}s", flush=True)

    model = FaultNet().to(dev)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.3f} M", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda"))

    out = args.out or f"FaultNet_Gamma{args.gamma}.pt"
    n_batches = 600 // args.batch_size   # 等效 600 样本 / epoch

    def run_epoch(loss_fn, phase):
        model.train()
        tot = 0.0
        for _ in range(n_batches):
            xs, ys = make_batch_gpu(tr_s, tr_l, args.batch_size, dev, rng)
            optimizer.zero_grad()
            with autocast("cuda", enabled=(dev.type == "cuda")):
                pred = model(xs)
                loss = loss_fn(pred, ys)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item()
        return tot / n_batches

    def validate():
        model.eval()
        ious = []
        with torch.no_grad():
            for i in range(0, va_s.shape[0], args.batch_size):
                xs = va_s[i:i + args.batch_size].to(dev)
                ys = va_l[i:i + args.batch_size].to(dev)
                with autocast("cuda", enabled=(dev.type == "cuda")):
                    pred = model(xs)
                ious.append(iou_score(pred, ys))
        return np.mean(ious)

    print(f"阶段1: Dice loss 预训练 {args.epochs_pre} epochs", flush=True)
    t0 = time.time()
    for e in range(1, args.epochs_pre + 1):
        l = run_epoch(dice_loss, "pre")
        if e % 50 == 0 or e == 1:
            print(f"  [pre] epoch {e}: loss={l:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"阶段2: MD loss (gamma={args.gamma}) 微调 {args.epochs} epochs", flush=True)
    best = 0.0
    for e in range(1, args.epochs + 1):
        l = run_epoch(lambda p, y: md_loss(p, y, gamma=args.gamma).mean(), "ft")
        if e % args.val_every == 0 or e == 1:
            iou = validate()
            best = max(best, iou)
            print(f"  [ft] epoch {e}: loss={l:.4f}  val_IOU={iou:.2f}%  "
                  f"(best={best:.2f}%)  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), out)
    target = {0.5: 67.53, 0.6: 67.59, 0.7: 67.33}.get(args.gamma, "?")
    print(f"完成，best val_IOU={best:.2f}%（论文 Group-4 γ={args.gamma} 目标 {target}%），"
          f"已保存 {out}", flush=True)


if __name__ == "__main__":
    main()
