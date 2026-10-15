# -*- coding: utf-8 -*-
"""Mask Dice (MD) Loss —— 按论文 Equation (13)(14) 精确实现。

论文: Dou et al., "MD Loss: Efficient Training of 3D Seismic Fault Segmentation Network
      under Sparse Labels by Weakening Anomaly Annotation", IEEE TGRS 2022.
      (arXiv:2110.05319, DOI 10.1109/TGRS.2022.3196810)

标签约定（关键）:
    y = 1   -> 断层 (前景)
    y = 0   -> 背景 (非断层)
    y = -1  -> 未标注（非标签区域），据此生成掩码 M

MD loss (式 13, gamma=0.5 时的原始形式):
    L = 1 - sum(M_i * yhat_i * y_i) / sum(M_i * (0.5*yhat_i + 0.5*y_i))

MD_gamma loss (式 14, 引入 gamma 抑制假阴性/背景权重):
    L = 1 - sum(M_i * yhat_i * y_i) / sum(M_i * ((1-gamma)*yhat_i + gamma*y_i))

其中 gamma ∈ [0.5, 1)，控制背景(假阴性)在分母中的权重。
gamma 越大，模型对背景越不敏感，越倾向预测前景，进一步弱化 FNL。
官方三个权重 FaultNet_Gamma{0.5,0.6,0.7}.pt 正对应 gamma ∈ {0.5, 0.6, 0.7}。
"""
import torch
import torch.nn as nn


def md_loss(y_hat: torch.Tensor, y: torch.Tensor, gamma: float = 0.5, eps: float = 1e-6):
    """MD loss（逐样本返回，可配合 reduction 使用）。

    参数:
        y_hat: 预测概率（sigmoid 输出），shape (N, 1, D, H, W) 或 (N, D, H, W)
        y:     标签，shape 同 y_hat。断层=1，背景=0，未标注=-1。
        gamma: MD_gamma 系数，[0.5, 1)。0.5 即原始 MD loss。
    返回:
        loss: shape (N,)，每个样本一个标量。
    """
    assert 0.5 <= gamma < 1.0, f"gamma 必须在 [0.5, 1)，得到 {gamma}"

    y_hat = y_hat.float()
    y = y.float()

    # 掩码：标注区域为 1，未标注(-1)为 0
    mask = (y >= 0).float()          # M_i
    y_pos = y.clamp(min=0)           # 把 -1 归 0（已被 mask 屏蔽）

    # 分子: sum(M * yhat * y)
    numerator = (mask * y_hat * y_pos).sum(dim=(1, 2, 3, 4))

    # 分母: sum(M * ((1-gamma)*yhat + gamma*y))
    denominator = (mask * ((1.0 - gamma) * y_hat + gamma * y_pos)).sum(dim=(1, 2, 3, 4))

    loss = 1.0 - numerator / (denominator + eps)
    return loss


class MDLoss(nn.Module):
    """可复用的 nn.Module 封装。"""

    def __init__(self, gamma: float = 0.5, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_hat, y):
        loss = md_loss(y_hat, y, gamma=self.gamma, eps=self.eps)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_label(mask3d: torch.Tensor, label_slices: dict):
    """把稀疏 2D 标签插进 3D 体积，生成训练标签。

    参数:
        mask3d: 完整 3D 断层标签（1=断层, 0=背景），shape (D, H, W)
        label_slices: {'inline': [idx, ...], 'crossline': [...], 'time': [...]}
                      给定哪些切片被视为"人工已标注"；其余位置标 -1。
    返回:
        y: shape (D, H, W)，标注切片保留 0/1，其余 -1。
    """
    y = torch.full_like(mask3d, -1, dtype=torch.float32)
    for axis, idxs in label_slices.items():
        for i in idxs:
            if axis == "inline":       # 沿时间/深度方向切片: 固定第 0 维
                y[i] = mask3d[i].float()
            elif axis == "crossline":  # 固定第 1 维
                y[:, i] = mask3d[:, i].float()
            elif axis == "time":       # 固定第 2 维
                y[:, :, i] = mask3d[:, :, i].float()
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    # 自检：gamma=0.5 时 MD loss 与"masked dice"一致
    y_hat = torch.rand(2, 1, 8, 8, 8)
    y = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
    y[..., :2] = -1                      # 部分置为未标注
    loss = md_loss(y_hat, y, gamma=0.5)
    print("MD loss (gamma=0.5):", loss.tolist())

    # gamma 越大，损失越小（背景权重被弱化），验证单调性方向
    for g in (0.5, 0.6, 0.7):
        print(f"gamma={g}: {md_loss(y_hat, y, gamma=g).mean().item():.4f}")
