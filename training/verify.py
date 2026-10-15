# -*- coding: utf-8 -*-
"""验证重建网络与官方权重的对齐程度：
  1) state_dict key 是否一致
  2) 加载官方权重后，前向输出是否逐元素一致
"""
import os
import sys
import numpy as np
import torch

# 让同目录的 network.py 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import FaultNet  # noqa: E402

# 官方权重在仓库根目录的 network/ 下
_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "network",
                  "FaultNet_Gamma0.7.pt")
official = torch.jit.load(os.path.normpath(_W), map_location="cpu")
rebuild = FaultNet()

off_sd = official.state_dict()
reb_sd = rebuild.state_dict()

off_keys = set(off_sd.keys())
reb_keys = set(reb_sd.keys())

print(f"官方参数数: {len(off_keys)}, 重建参数数: {len(reb_keys)}")
print("\n--- 仅官方有、重建缺失的 key ---")
for k in sorted(off_keys - reb_keys):
    print("  MISS:", k)
print("--- 仅重建有、官方没有的 key ---")
for k in sorted(reb_keys - off_keys):
    print("  EXTRA:", k)

# shape 不匹配的
mismatch = [k for k in (off_keys & reb_keys) if off_sd[k].shape != reb_sd[k].shape]
print(f"\n--- shape 不匹配的 key ({len(mismatch)}) ---")
for k in mismatch:
    print(f"  {k}: official {tuple(off_sd[k].shape)} vs rebuild {tuple(reb_sd[k].shape)}")

# 拷贝权重
missing, unexpected = rebuild.load_state_dict(off_sd, strict=False)
print(f"\nload_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

# 数值对齐
torch.manual_seed(0)
for shape in [(32, 32, 32), (64, 48, 40), (128, 128, 128)]:
    x = torch.randn(1, 1, *shape)
    with torch.no_grad():
        official.eval(); rebuild.eval()
        y_o = official(x)
        y_r = rebuild(x)
    diff = (y_o - y_r).abs()
    print(f"\n输入 {shape}: 输出 {tuple(y_o.shape)}")
    print(f"  max_abs_diff = {diff.max().item():.3e}  mean_abs_diff = {diff.mean().item():.3e}")
    print(f"  官方输出范围 [{y_o.min():.4f}, {y_o.max():.4f}]  重建 [{y_r.min():.4f}, {y_r.max():.4f}]")
    if diff.max().item() < 1e-4:
        print("  ✅ 对齐")
    else:
        print("  ❌ 未对齐")
