#!/usr/bin/env python3
"""Tail a training log and display live iter/loss/rate/ETA. Zero CPU when idle."""
import re, subprocess, sys, time
from datetime import datetime, timedelta

log = sys.argv[1] if len(sys.argv) > 1 else "outputs/truck-3k-run3.log"
proc = subprocess.Popen(["tail", "-f", "-n", "+1", log], stdout=subprocess.PIPE, text=True)

seen: list[tuple[int, int, float, float]] = []  # (iter, total, loss, ts)

for raw in proc.stdout:
    line = raw.strip()
    m = re.match(r"iter (\d+)/(\d+)\s+loss=([\d.eE+\-]+)", line)
    if not m:
        print(line, flush=True)
        continue

    it, total, loss = int(m[1]), int(m[2]), float(m[3])
    seen.append((it, total, loss, time.time()))

    if len(seen) >= 2:
        dt = seen[-1][3] - seen[-2][3]
        d_it = seen[-1][0] - seen[-2][0]
        iters_per_min = d_it / dt * 60
        remaining = total - it
        eta = datetime.now() + timedelta(seconds=remaining / (d_it / dt))
        eta_str = eta.strftime("%b %d %H:%M")
        rem_sec = remaining / (d_it / dt)
        rem_h, rem_m = int(rem_sec // 3600), int((rem_sec % 3600) // 60)
        print(f"iter {it}/{total}  loss={loss:.4f}  {iters_per_min:.1f} it/min  ETA {eta_str} (in {rem_h}h {rem_m}m)", flush=True)
    else:
        print(f"iter {it}/{total}  loss={loss:.4f}  (need 2 pts for ETA)", flush=True)
