import argparse
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compare two float32 output binaries.")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    return p.parse_args()


def load_shape(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        raise ValueError("Empty shape file")
    return tuple(int(x) for x in txt.split(",") if x)


def main():
    args = parse_args()
    shape = load_shape(args.shape)

    a = np.fromfile(args.a, dtype=np.float32)
    b = np.fromfile(args.b, dtype=np.float32)

    expected = int(np.prod(shape))
    if a.size != expected:
        raise RuntimeError(f"--a size mismatch: got {a.size}, expected {expected}")
    if b.size != expected:
        raise RuntimeError(f"--b size mismatch: got {b.size}, expected {expected}")

    a = a.reshape(shape)
    b = b.reshape(shape)

    abs_err = np.abs(a - b)
    max_abs = float(abs_err.max())

    denom = np.maximum(np.abs(a), 1e-12)
    rel_err = abs_err / denom
    max_rel = float(rel_err.max())

    ok = np.allclose(a, b, atol=args.atol, rtol=args.rtol)

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    argmax_a = int(np.argmax(a_flat))
    argmax_b = int(np.argmax(b_flat))

    top5_a_idx = np.argsort(-a_flat)[:5]
    top5_b_idx = np.argsort(-b_flat)[:5]

    print(f"shape={shape}")
    print(f"max_abs_error={max_abs:.8e}")
    print(f"max_rel_error={max_rel:.8e}")
    print(f"argmax_a={argmax_a} argmax_b={argmax_b}")
    print("top5_a:", [(int(i), float(a_flat[i])) for i in top5_a_idx])
    print("top5_b:", [(int(i), float(b_flat[i])) for i in top5_b_idx])
    print("PASS" if ok else "FAIL")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
