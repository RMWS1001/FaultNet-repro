# FaultNet 复现项目

**FaultNet**：数据驱动的三维地震断层分割网络（`MD loss` + 高分辨率多尺度融合），
论文 *"MD Loss: Efficient Training of 3D Seismic Fault Segmentation Network under
Sparse Labels by Weakening Anomaly Annotation"*, IEEE TGRS 2022
（[arXiv:2110.05319](https://arxiv.org/abs/2110.05319), DOI 10.1109/TGRS.2022.3196810）。

> 官方仓库 [douyimin/FaultNet](https://github.com/douyimin/FaultNet) **只开源了推理**，
> 训练代码、网络结构定义、MD loss 均未开源。本项目补齐了这三部分。

## 目录结构

```
.
├── prediction.py              官方推理入口（命令行）
├── utils.py                   官方推理工具（已修复 np.int 兼容性）
├── network/                   官方 3 个预训练权重（TorchScript）
│   ├── FaultNet_Gamma0.5.pt
│   ├── FaultNet_Gamma0.6.pt
│   └── FaultNet_Gamma0.7.pt
└── training/                  训练代码（本项目补齐）
    ├── network.py             ★ 网络结构（从官方权重反向工程，已验证逐元素对齐）
    ├── md_loss.py             ★ Mask Dice (MD) loss（论文 Eq.13/14）
    ├── dataset.py             FaultSeg3D 数据加载 + 稀疏标签
    ├── train_faultnet.py      两阶段训练脚本（Dice 预训练 + MD 微调）
    ├── run_all.py             一键训练 γ=0.5/0.6/0.7
    ├── verify.py              网络结构与官方权重对齐验证
    ├── train.py + synthetic.py  教学演示（自造合成数据）
    ├── 训练指南.md            训练详细说明（复现指标）
    └── README.md              网络结构反向工程 + MD loss 公式细节
```

## 一、推理（开箱即用）

### 环境

```bash
pip install numpy torch segyio opencv-python matplotlib
# GPU 版 torch 按显卡选，例如 RTX 50 系：
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 依赖 `numpy` 建议 **< 1.24**，或直接装最新版——本项目已把 `utils.py` 里的
> `np.int` 修复为 `np.int64`，新老版本 numpy 都能跑。

### 运行

```bash
# .npy 输入（形状须为 (iline, xline, tline)）
python prediction.py --input data/F3.npy --gamma 0.7

# .segy/.sgy 输入
python prediction.py --input data/my_survey.sgy --iline 189 --xline 193 --gamma 0.7
```

结果输出到 `output/<文件名>/{iline,xline,tline}/*.png`（左=地震剖面，右=断层概率叠色）。

**关键参数**

| 参数 | 说明 |
|---|---|
| `--gamma` | 只能取 0.5/0.6/0.7，对应三个权重 |
| `--infer_size` | 分块推理（每个数需是 16 的倍数）。⚠️ 官方分块实现有重叠累加缺陷，**建议整卷跑**（不加此参数） |
| `--save_fault_cuboid` | 保存完整概率体 |

**注意**：模型在 GPU 上以 FP16 推理（自动），CPU 上以 FP32 推理，两者结果有 ~1e-6 量级差异，肉眼无差。

测试数据（F3、Kerry3D）：<https://drive.google.com/drive/folders/1LEHd2VO9TZTOjrMuAQ7I446OfYDgcdWo>

---

## 二、训练（复现论文指标）

详见 [training/训练指南.md](training/训练指南.md)。三步：

```bash
cd training
pip install torch numpy

# 1. 下载 FaultSeg3D 合成数据（约 3.5 GB，需代理）
gdown --folder "https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8" -O data

# 2. 一键三次训练（γ=0.5/0.6/0.7）
python run_all.py data
```

复现目标（论文 Group-4，纯合成数据，验证 IOU%）：γ=0.5 → 67.53，γ=0.6 → 67.59，γ=0.7 → 67.33。

> 复现范围：论文 Group-1/2/3 依赖未公开的 Shengli 油田场数据，**只有 Group-4（纯合成数据）可完整复现**。

---

## 关于网络结构（关键）

`training/network.py` 是**从官方 TorchScript 权重精确反向工程**得到的，不是近似复刻。
运行 `python training/verify.py` 可验证：加载官方权重后，重建网络前向输出与官方模型
**逐元素一致**（max diff ~1e-6，493 个参数完全对齐）。

## 引用

```bibtex
@article{dou2022md,
  title={MD loss: Efficient training of 3-D seismic fault segmentation network under sparse labels by weakening anomaly annotation},
  author={Dou, Yimin and Li, Kewen and Zhu, Jianbing and Li, Timing and Tan, Shaoquan and Huang, Zongchao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60}, pages={1--14}, year={2022}, publisher={IEEE}
}
```

## License

沿用官方 [Apache-2.0](LICENSE)。
