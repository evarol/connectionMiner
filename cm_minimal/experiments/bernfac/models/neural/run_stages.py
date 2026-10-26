"""Staged validation-set selection on the full data (resumable; each config writes results/<name>.json).

Every fit gets the same wall-clock budget (BUDGET seconds of optimisation, plus the tol / max_outer
stopping rules), so selection compares configurations at equal compute. Decisions use val_nll only.

Stage 1  affine pg_gem, widths (64,64), d=10, lambda_output in {1, 10, 30}
Stage 2  affine, best lambda_output, d in {5, 20}
Stage 3  affine, best (d, lambda_output), widths (128,128)
Stage 4  nonlinear head at the best affine (d, widths, lambda_output), warm-started from that affine fit
Stage 5  second restart (seed 2) of the best affine configuration; restart chosen by training F
Stage 6  direct_bce for the best affine architecture, same initialisation as its pg_gem restart 0
"""
from __future__ import annotations

import json
import sys
import time

import run_full as rf

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0
T0 = time.time()


def val(name):
    return json.loads((rf.RES / f"{name}.json").read_text())["val_nll"]


def run(name, **kw):
    print(f"[{(time.time() - T0) / 3600:.2f} h] {name}", flush=True)
    r = rf.run_config(name, max_seconds=BUDGET, **kw)
    rf.build_table()
    return r["val_nll"]


def nm(opt="pg", head="affine", d=10, w=64, lv=10, r=0):
    lvs = f"{lv:g}"
    return f"{opt}_{head}_d{d}_w{w}_lv{lvs}_r{r}"


# stage 1
s1 = {lv: run(nm(lv=lv)) for lv in (1, 10, 30)}
lv = min(s1, key=s1.get)
# stage 2
s2 = {10: s1[lv]}
for d in (5, 20):
    s2[d] = run(nm(d=d, lv=lv))
d = min(s2, key=s2.get)
# stage 3
s3 = {64: s2[d], 128: run(nm(d=d, w=128, lv=lv))}
w = min(s3, key=s3.get)
best_aff = nm(d=d, w=w, lv=lv)
print(f"best affine: {best_aff} val {s3[w]:.5f}", flush=True)
# stage 4: nonlinear head
run(nm(head="nonlinear", d=d, w=w, lv=lv) + "_warm", warm=best_aff)
# stage 5: second restart
run(nm(d=d, w=w, lv=lv, r=1))
# stage 6: direct BCE, matched initialisation (same seed as restart 0)
run(nm(opt="direct", d=d, w=w, lv=lv))
print(f"all stages done in {(time.time() - T0) / 3600:.2f} h", flush=True)
