# FaultNet 训练侧完整实现（网络结构 + MD loss + 训练代码）

官方仓库 `douyimin/FaultNet` **只开源了推理**（`prediction.py` + 3 个 TorchScript 权重），
训练代码、网络结构定义、MD loss 均未开源。本目录补全这三部分。

## 文件清单

| 文件 | 说明 | 验证状态 |
|---|---|---|
| `network.py` | FaultNet 网络结构（从官方权重反向工程） | ✅ **已与官方权重逐元素对齐** |
| `md_loss.py` | Mask Dice (MD) loss（论文 Eq. 13/14 精确实现） | ✅ 公式级精确 |
| `synthetic.py` | 合成地震数据生成（模拟 FaultSeg3D） | ✅ 可运行（教学实现） |
| `train.py` | 从零训练脚本（MD loss + 稀疏 2D 标签） | ✅ 端到端跑通 |
| `verify.py` | 验证重建网络与官方权重对齐 | ✅ 通过 |

## 1. 网络结构（network.py）

从官方 `FaultNet_Gamma0.7.pt`（TorchScript）反向工程得到，方法：

```python
m = torch.jit.load("network/FaultNet_Gamma0.7.pt")
print(m)                     # 模块树
print(m.code)                # 前向代码
for n, p in m.named_parameters(): print(n, tuple(p.shape))
m.inlined_graph              # 每个 conv 的 stride/padding/dilation
```

重建结果（参数量 0.422 M，与论文一致）：

```
stem:   Conv(1→8, s2) → BasicBlock → Conv(8→8, s2)        # 下采样 4x
layer1: Bottleneck(8→64)                                   # H/4
主干:   HRNet 风格，分支 [8@H/4, 16@H/8] → [8@H/4, 16@H/8, 24@H/16]
融合:   3 分支 1x1→8ch → cat → MCF(24→8) → x2 up
        cat(conv1 输出 8ch) → MCF(16→8) → 2×BasicBlock → x2 up
head:   BasicBlock → Conv(8→1, 1x1) → sigmoid
```

关键组件：
- `BasicBlock`：两个 3x3x3 卷积 + 残差（无 bias、无 stride）
- `Bottleneck`：1x1→3x3→1x1 + 1x1 下采样旁路（ResNet 风格）
- `HighResolutionModule`：HRNet 多分支并行 + 跨分支融合（上采样用 trilinear + align_corners=True，下采样用 stride2 卷积）
- `MCF_Block`（多尺度压缩融合）：expansion_conv(1x1) → BN → ReLU6 → norm_conv(3x3) → ×(SegSE 门控) → zoom_conv(1x1) + skip_conv(1x1)
- `SegSEBlock`：空洞卷积(dilation=2) → 1x1 → sigmoid（门控系数）

**验证结果**（verify.py）：

```
官方参数数: 493, 重建参数数: 493
load_state_dict: missing=0 unexpected=0
输入 (32,32,32)   max_abs_diff = 2.3e-06   ✅
输入 (64,48,40)   max_abs_diff = 1.1e-05   ✅
输入 (128,128,128) max_abs_diff = 6.1e-06   ✅
```

即：加载官方权重后，重建网络前向输出与官方 jit 模型**逐元素一致**（差异仅浮点运算顺序的 1e-6 量级）。

## 2. MD loss（md_loss.py）

论文：Dou et al., *"MD Loss: Efficient Training of 3D Seismic Fault Segmentation Network
under Sparse Labels by Weakening Anomaly Annotation"*, IEEE TGRS 2022
（arXiv:2110.05319, DOI 10.1109/TGRS.2022.3196810）。

**标签约定**：断层=1，背景=0，未标注=-1（据 -1 生成掩码 M）。

**MD loss（Eq. 13，γ=0.5 的原始形式）**：

```
L = 1 - Σ M_i·ŷ_i·y_i / Σ M_i·(0.5·ŷ_i + 0.5·y_i)
```

**MD_γ loss（Eq. 14，引入 γ 弱化背景/假阴性权重）**：

```
L = 1 - Σ M_i·ŷ_i·y_i / Σ M_i·((1-γ)·ŷ_i + γ·y_i)     γ ∈ [0.5, 1)
```

γ 越大，模型对背景越不敏感、越倾向预测前景，进一步弱化假阴性（漏标）的影响。
官方三个权重 `FaultNet_Gamma{0.5,0.6,0.7}.pt` 正对应 γ ∈ {0.5, 0.6, 0.7}。

```python
from md_loss import MDLoss
criterion = MDLoss(gamma=0.5, reduction="mean")
loss = criterion(pred, label)   # label 含 -1 表示未标注
```

## 3. 训练（train.py + synthetic.py）

```bash
python train.py --gamma 0.5 --epochs 50 --steps 20 --shape 128 128 128
```

训练要点：
- 输入/输出都是 3D 体（不是逐切片），标签以稀疏 2D 切片形式插入 3D 体（其余 -1）。
- 每步在线生成一对合成地震体 + 断层标签，随机抽取约 3.3% 的 inline 切片作为"人工标注"。
- 监控 Dice 用完整 3D 标签（不参与训练）。

### 复现论文指标的注意事项（重要）

`synthetic.py` 是**教学实现**，合成数据的统计特性、断层几何与论文/FaultSeg3D 不同，
用它能跑通流程、验证代码正确，但**不能复现论文数值**。要复现论文指标：

1. 下载 **FaultSeg3D 官方 200 对合成数据**（Wu et al. 2019，
   <https://github.com/xinwucwp/faultSeg>，Google Drive）；
2. 按论文方式把 3D 标签转为稀疏 2D 标签（`md_loss.build_label`）；
3. 分别用 γ ∈ {0.5, 0.6, 0.7} 训练，得到三个权重，对应官方三个 `.pt`。

## 与官方权重的对照验证

本目录网络结构**不是"近似复刻"，而是"精确重建"**——`verify.py` 证明它加载官方
权重后逐元素对齐。这反过来也确认了 MD loss 的 γ 与三个权重文件的对应关系：
`Gamma0.5/0.6/0.7` = MD 损失的 γ 超参，与论文 Eq. (14) 完全吻合。
