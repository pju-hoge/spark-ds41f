"""Can the Top-2 branch pay off if its 256-dim confidence weight is fitted?

The kernel swaps the draft to the runner-up when

    (second_val - best_val) + W * <w_conf, W1[second] - W1[best]> > 0

with `w_conf` fixed to the last 256 dims of `confidence_head.proj.weight`. The
offline simulation showed that shipped weight rescues about as many drafts as it
breaks. This script fits `w_conf` on half the samples (logistic loss, positives =
"runner-up is the committed token while the draft is wrong") and reports the net
rescue/break on the other half, for a grid of margins and weights.

    python3 fit_top2_wconf.py pairs.npz --w1 ... --w2 ... --out wconf_fit.npy
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.pairs)
    prev = d["prev"].astype(np.int64)
    target = d["target"].astype(np.int64)
    base_ids = d["base_ids"].astype(np.int64)
    base_vals = d["base_vals"].astype(np.float32)
    w1 = np.load(args.w1).astype(np.float32)
    w2 = np.load(args.w2).astype(np.float32)

    post = base_vals + np.einsum("nkr,nr->nk", w2[base_ids], w1[prev])
    order = np.argsort(-post, axis=1)
    best_id = np.take_along_axis(base_ids, order[:, :1], axis=1)[:, 0]
    second_id = np.take_along_axis(base_ids, order[:, 1:2], axis=1)[:, 0]
    best_val = np.take_along_axis(post, order[:, :1], axis=1)[:, 0]
    second_val = np.take_along_axis(post, order[:, 1:2], axis=1)[:, 0]
    margin = best_val - second_val
    best_ok = best_id == target
    second_ok = second_id == target

    x = w1[second_id] - w1[best_id]           # [N, 256]
    gap = second_val - best_val               # negative by construction
    pos = (~best_ok) & second_ok              # swapping here is a rescue
    neg = best_ok                             # swapping here is a break
    usable = pos | neg
    print(f"samples {len(x):,}  positives {pos.sum():,}  negatives {neg.sum():,}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(x))
    half = len(x) // 2
    train, test = idx[:half], idx[half:]

    w = np.zeros(x.shape[1], dtype=np.float32)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    y = pos.astype(np.float32)
    for epoch in range(args.epochs):
        batch = train[usable[train]]
        rng.shuffle(batch)
        for start in range(0, len(batch), 4096):
            b = batch[start:start + 4096]
            logit = x[b] @ w
            p = 1 / (1 + np.exp(-logit))
            g = (p - y[b]) @ x[b] / len(b) + args.l2 * w
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w -= args.lr * (m / (1 - 0.9 ** (epoch + 1))) / (np.sqrt(v / (1 - 0.999 ** (epoch + 1))) + 1e-8)
    np.save(args.out, w)
    print(f"fitted w_conf: norm {np.linalg.norm(w):.3f}")

    def net(mask, wconf, m_thresh, weight):
        sel = mask & (margin <= m_thresh)
        fire = sel & (gap + weight * (x @ wconf) > 0)
        rescued = int((fire & pos).sum())
        broken = int((fire & neg).sum())
        return rescued, broken, rescued - broken

    test_mask = np.zeros(len(x), dtype=bool)
    test_mask[test] = True
    print("\n  margin  weight  rescued  broken     net     net%/all")
    for m_thresh in (0.25, 0.5, 0.8, 1.0, 1.5):
        for weight in (0.5, 1.0, 2.0, 4.0):
            rescued, broken, n = net(test_mask, w, m_thresh, weight)
            print(f"  {m_thresh:6.2f}  {weight:6.2f}  {rescued:8d}  {broken:6d}  {n:+7d}  "
                  f"{n / max(1, test_mask.sum()) * 100:+7.3f}%")


if __name__ == "__main__":
    main()
