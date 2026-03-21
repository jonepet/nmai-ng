#!/usr/bin/env python3
"""Print GPU status in a clean table. Runs on remote machine."""
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--format=csv,noheader",
     "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"],
    capture_output=True, text=True
)
for line in result.stdout.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 6:
        idx, name, util, mem_used, mem_total, temp = parts[:6]
        util = util.replace(" %", "")
        mem_used = mem_used.replace(" MiB", "")
        mem_total = mem_total.replace(" MiB", "")
        print(f"    [{idx}] {name:<25s} {util:>3s}% util  {mem_used:>5s} / {mem_total:>5s} MiB  {temp}°C")
