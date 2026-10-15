# -*- coding: utf-8 -*-
"""FaultNet 网络结构 —— 从官方 TorchScript 权重反向工程重建。

重建依据（见 repro_check/arch/*.txt）：
  - 模块树 / 参数表 / 各子模块 forward 代码
  - 91 个卷积的 stride/padding/dilation/groups
验证方式：加载官方权重后，前向输出与官方 torch.jit 模型逐元素一致（见 verify.py）。

整体结构（输入 D,H,W，通道 1）：
  stem:   Conv(1→8, s2) → BasicBlock → Conv(8→8, s2)   # 下采样 4x
  layer1: Bottleneck(8→64)                              # 保持 H/4
  HRNet 主干: 分支 [8@H/4, 16@H/8] → [8@H/4, 16@H/8, 24@H/16]
  融合:    三个分支 1x1 到 8ch → cat → MCF(24→8) → x2 up
           cat(conv1 输出 8ch) → MCF(16→8) → 2x BasicBlock → x2 up
  head:    BasicBlock(8) → Conv(8→1, 1x1) → sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """两个 3x3x3 卷积 + 残差，通道不变。官方无 bias、无 stride、无 downsample。"""

    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU6(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈块：1x1 → 3x3 → 1x1，带 1x1 下采样旁路。官方无 bias。"""

    def __init__(self, in_planes, planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU6(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm3d(out_planes),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out


class SegSEBlock(nn.Module):
    """空间/通道注意力门：3x3 空洞卷积(通道减半) → 1x1 → sigmoid。官方带 bias。"""

    def __init__(self, in_ch, mid_ch):
        super().__init__()
        # dilation=2, padding=2 保持分辨率不变（等效感受野 5x5x5）
        self.dila_conv = nn.Conv3d(in_ch, mid_ch, 3, padding=2, dilation=2, bias=True)
        self.conv1 = nn.Conv3d(mid_ch, in_ch, 1, bias=True)

    def forward(self, x):
        out = self.dila_conv(x)
        out = self.conv1(out)
        return torch.sigmoid(out)


class MCF_Block(nn.Module):
    """多尺度压缩融合块（MultiScale Compression Fusion）。

    forward: expansion_conv → bn → ReLU6 → norm_conv → ×(segse 门控)
             → zoom_conv + skip_conv(shortcut)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expansion_conv = nn.Conv3d(in_ch, in_ch, 1, bias=True)
        self.skip_conv = nn.Conv3d(in_ch, out_ch, 1, bias=True)
        self.zoom_conv = nn.Conv3d(in_ch, out_ch, 1, bias=True)
        self.bn = nn.BatchNorm3d(in_ch)
        self.norm_conv = nn.Conv3d(in_ch, in_ch, 3, padding=1, bias=True)
        self.segse_block = SegSEBlock(in_ch, in_ch // 2)

    def forward(self, x):
        out = self.expansion_conv(x)
        out = self.bn(out)
        out = F.hardtanh(out, 0.0, 6.0)          # ReLU6
        out = self.norm_conv(out)
        out = out * self.segse_block(out)         # 门控
        out = self.zoom_conv(out) + self.skip_conv(x)
        return out


class HighResolutionModule(nn.Module):
    """HRNet 多分辨率模块：多分支并行 + 跨分支融合（一次融合）。"""

    def __init__(self, num_branches, channels, num_blocks=2):
        super().__init__()
        self._check_branches(num_branches, channels)
        self.num_branches = num_branches
        self.branches = self._make_branches(num_branches, channels, num_blocks)
        self.fuse_layers = self._make_fuse_layers(num_branches, channels)
        self.relu = nn.ReLU6(inplace=False)

    @staticmethod
    def _check_branches(nb, ch):
        assert nb == len(ch)

    def _make_one_branch(self, channels, num_blocks):
        return nn.Sequential(*[BasicBlock(channels) for _ in range(num_blocks)])

    def _make_branches(self, nb, ch, num_blocks):
        branches = [self._make_one_branch(ch[i], num_blocks) for i in range(nb)]
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, nb, ch):
        fuse_layers = []
        for i in range(nb):                      # 目标分支 i
            fuse_layer = []
            for j in range(nb):                  # 源分支 j
                if j > i:                        # 低分辨率 → 高分辨率：1x1 升/降通道，forward 里再上采样
                    fuse_layer.append(nn.Sequential(nn.Conv3d(ch[j], ch[i], 1, bias=False),
                                                    nn.BatchNorm3d(ch[i])))
                elif j < i:                      # 高分辨率 → 低分辨率：若干 stride2 卷积下采样
                    down = []
                    for k in range(i - j):
                        if k == i - j - 1:       # 最后一段不加 ReLU（与官方一致）
                            down.append(nn.Sequential(nn.Conv3d(ch[j], ch[i], 3, stride=2, padding=1, bias=False),
                                                      nn.BatchNorm3d(ch[i])))
                        else:                    # 中间段加 ReLU6（跨多级下采样时）
                            down.append(nn.Sequential(nn.Conv3d(ch[j], ch[j], 3, stride=2, padding=1, bias=False),
                                                      nn.BatchNorm3d(ch[j]),
                                                      nn.ReLU6(inplace=False)))
                    fuse_layer.append(nn.Sequential(*down))
                else:                            # 自环：占位（官方无参数，用空模块）
                    fuse_layer.append(nn.Module())
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        # 分支各自前向
        branch_outs = [self.branches[i](x[i]) for i in range(self.num_branches)]
        # 融合
        fused = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                if j == i:
                    continue
                src = branch_outs[j]
                if j > i:
                    src = self.fuse_layers[i][j](src)
                    src = F.interpolate(src, size=branch_outs[i].shape[2:],
                                        mode="trilinear", align_corners=True)
                else:
                    src = self.fuse_layers[i][j](src)
                y = src if y is None else y + src
            y = y + branch_outs[i]
            fused.append(self.relu(y))
        return tuple(fused)


class FaultNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ---- stem（下采样 4x）----
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU6(inplace=False),
        )
        self.conv1_0 = BasicBlock(8)
        self.conv1_1 = nn.Conv3d(8, 8, 3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU6(inplace=False)

        # ---- layer1：Bottleneck 8→64 ----
        self.layer1 = nn.Sequential(Bottleneck(8, 8, 64))

        # ---- transition1：64 → [8, 16]（分支 0 保分辨率，分支 1 下采样 2x）----
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv3d(64, 8, 3, padding=1, bias=False),
                          nn.BatchNorm3d(8), nn.ReLU6(inplace=False)),
            # 新增分支：两层嵌套 Sequential（HRNet 原版 _make_transition_layer）
            nn.Sequential(nn.Sequential(nn.Conv3d(64, 16, 3, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(16), nn.ReLU6(inplace=False))),
        ])

        # ---- stage2：2 分支 [8, 16] × 2 次 ----
        self.stage2 = nn.Sequential(
            HighResolutionModule(2, [8, 16]),
            HighResolutionModule(2, [8, 16]),
        )

        # ---- transition2：16 → 24（新增分支 2，下采样 2x）----
        self.transition2 = nn.ModuleList()
        self.transition2.append(nn.Module())     # 占位 index 0
        self.transition2.append(nn.Module())     # 占位 index 1
        self.transition2.append(
            # 新增分支：两层嵌套 Sequential
            nn.Sequential(nn.Sequential(nn.Conv3d(16, 24, 3, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(24), nn.ReLU6(inplace=False)))
        )

        # ---- stage3：3 分支 [8, 16, 24] × 2 次 ----
        self.stage3 = nn.Sequential(
            HighResolutionModule(3, [8, 16, 24]),
            HighResolutionModule(3, [8, 16, 24]),
        )

        # ---- 多分辨率融合到单分支 8ch ----
        self.fuse1 = nn.Conv3d(8, 8, 1, bias=True)
        self.fuse2 = nn.Conv3d(16, 8, 1, bias=True)
        self.fuse3 = nn.Conv3d(24, 8, 1, bias=True)

        self.fuse_layer1 = MCF_Block(24, 8)
        self.fuse_layer2 = MCF_Block(16, 8)

        self.layer2 = nn.Sequential(BasicBlock(8), BasicBlock(8))

        # ---- head ----
        self.last_layer = nn.Sequential(
            BasicBlock(8),
            nn.Conv3d(8, 1, 1, bias=True),
        )

    def forward(self, x):
        # stem
        x = self.conv1(x)                        # 8ch @ H/2
        x_skip = self.conv1_0(x)                 # 8ch @ H/2（skip 来源，cat 用）
        x = self.conv1_1(x_skip)                 # 8ch @ H/4
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.layer1(x)                       # 64ch @ H/4

        # 主干
        x_list = [self.transition1[0](x),        # 8ch  @ H/4
                  self.transition1[1](x)]        # 16ch @ H/8
        x_list = list(self.stage2(x_list))
        x3 = self.transition2[2](x_list[1])      # 24ch @ H/16
        x_list = list(self.stage3([x_list[0], x_list[1], x3]))

        # 融合
        f1 = self.fuse1(x_list[0])               # 8ch @ H/4
        f2 = self.fuse2(x_list[1])               # 8ch @ H/8
        f3 = self.fuse3(x_list[2])               # 8ch @ H/16
        f2 = F.interpolate(f2, size=f1.shape[2:], mode="trilinear", align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode="trilinear", align_corners=True)
        fused = torch.cat([f1, f2, f3], dim=1)   # 24ch @ H/4

        out = self.fuse_layer1(fused)            # 8ch @ H/4
        out = F.interpolate(out, scale_factor=2, mode="trilinear", align_corners=True)  # H/2
        out = torch.cat([out, x_skip], dim=1)    # 16ch @ H/2
        out = self.fuse_layer2(out)              # 8ch @ H/2
        out = self.layer2(out)
        out = F.interpolate(out, scale_factor=2, mode="trilinear", align_corners=True)  # H
        out = self.last_layer(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    m = FaultNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n} ({n/1e6:.3f} M)")
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = m(x)
    print("输入", tuple(x.shape), "→ 输出", tuple(y.shape))
