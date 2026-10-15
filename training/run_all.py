# -*- coding: utf-8 -*-
"""一键跑三次训练（γ=0.5/0.6/0.7），对应官方三个权重。

前置：FaultSeg3D 数据已下载到 data/ 目录。
用法:
  python run_all.py --data FaultNet_rebuild/data
"""
import os
import subprocess
import sys


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "train_faultnet.py")

    for gamma in (0.5, 0.6, 0.7):
        print(f"\n{'='*60}\n开始训练 γ={gamma}\n{'='*60}", flush=True)
        cmd = [sys.executable, script, "--gamma", str(gamma), "--data", data,
               "--out", os.path.join(here, f"FaultNet_Gamma{gamma}.pt")]
        r = subprocess.call(cmd)
        if r != 0:
            print(f"γ={gamma} 训练失败，退出码 {r}", flush=True)
            break


if __name__ == "__main__":
    main()
