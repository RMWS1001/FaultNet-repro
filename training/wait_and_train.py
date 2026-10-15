# -*- coding: utf-8 -*-
"""等待数据下载完成，然后自动跑三次训练（γ=0.5/0.6/0.7）。

用法: python wait_and_train.py FaultNet_rebuild/data
"""
import os
import subprocess
import sys
import time

NEED = {
    "train/seis": 200,
    "train/fault": 200,
    "validation/seis": 20,
    "validation/fault": 20,
}


def count_dat(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".dat")])


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    here = os.path.dirname(os.path.abspath(__file__))

    print("等待数据下载完成...", flush=True)
    while True:
        ready = all(count_dat(os.path.join(data, k)) >= v for k, v in NEED.items())
        if ready:
            break
        status = "  ".join(f"{k}={count_dat(os.path.join(data,k))}/{v}" for k, v in NEED.items())
        print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
        time.sleep(120)

    print("数据齐全，开始三次训练", flush=True)
    script = os.path.join(here, "train_faultnet.py")
    for gamma in (0.5, 0.6, 0.7):
        print(f"\n{'='*60}\n训练 γ={gamma}\n{'='*60}", flush=True)
        cmd = [sys.executable, script, "--gamma", str(gamma), "--data", data,
               "--out", os.path.join(here, f"FaultNet_Gamma{gamma}.pt")]
        r = subprocess.call(cmd)
        if r != 0:
            print(f"γ={gamma} 失败，退出码 {r}", flush=True)
            break
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
