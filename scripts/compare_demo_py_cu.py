#!/usr/bin/env python3
import argparse
import csv
import math


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_qid = {int(r["query_id"]): r for r in rows}
    return by_qid


def to_float(v):
    if v is None or v == "" or v.lower() == "none":
        return None
    return float(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python_csv", required=True)
    p.add_argument("--cuda_csv", required=True)
    p.add_argument("--fms_exact", action="store_true", default=True)
    p.add_argument("--mps_rtol", type=float, default=1e-4)
    p.add_argument("--mps_atol", type=float, default=1e-4)
    args = p.parse_args()

    py = load_csv(args.python_csv)
    cu = load_csv(args.cuda_csv)

    if set(py.keys()) != set(cu.keys()):
        missing_py = sorted(set(cu.keys()) - set(py.keys()))
        missing_cu = sorted(set(py.keys()) - set(cu.keys()))
        raise SystemExit(f"query_id mismatch, missing_py={missing_py[:5]} missing_cu={missing_cu[:5]}")

    baseline_mismatch = 0
    neug_mismatch = 0
    max_mps_abs = 0.0
    max_mps_rel = 0.0

    for qid in sorted(py.keys()):
        rp = py[qid]
        rc = cu[qid]
        if int(rp["baseline_fms"]) != int(rc["baseline_fms"]):
            baseline_mismatch += 1
        if int(rp["neugn_fms"]) != int(rc["neugn_fms"]):
            neug_mismatch += 1

        for key in ["baseline_mps", "neugn_mps"]:
            vp = to_float(rp.get(key))
            vc = to_float(rc.get(key))
            if vp is None or vc is None:
                continue
            abs_e = abs(vp - vc)
            rel_e = abs_e / max(abs(vp), 1e-12)
            max_mps_abs = max(max_mps_abs, abs_e)
            max_mps_rel = max(max_mps_rel, rel_e)

    print(f"baseline_fms mismatches: {baseline_mismatch}")
    print(f"neugn_fms mismatches: {neug_mismatch}")
    print(f"max MPS abs error: {max_mps_abs:.8e}")
    print(f"max MPS rel error: {max_mps_rel:.8e}")

    mps_ok = (max_mps_abs <= args.mps_atol) or (max_mps_rel <= args.mps_rtol)
    ok = baseline_mismatch == 0 and mps_ok
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
