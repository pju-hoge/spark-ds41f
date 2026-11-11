"""Simulate the DSpark Top-2 branch offline on captured draft samples.

The kernel builds, per shard, (best_val, best_idx, second_val, second_idx) from
the *post-Markov* logits and, when the top-2 margin is small, may swap the draft
to the runner-up:

    take_second  <=>  second_val + W * <w_conf, W1[second] - W1[best]> > best_val

with `w_conf` = the last 256 dims of `confidence_head.proj.weight`.

We have the *pre*-bias top-32 per sample (patch 0050's `dl`), the previous token
and the committed token, so the post-bias values can be reconstructed exactly
for those candidates:  val_post[y] = val_pre[y] + W2[y] @ W1[prev].

This script reports, for a grid of (margin, weight), how many drafts the rule
would rescue (best wrong, second == committed) versus break (best already
correct, swap makes it wrong) -- i.e. the net gain -- before touching the engine.

    python3 sim_top2_branch.py pairs.npz --w1 w1.npy --w2 w2.npy --wconf wconf.npy
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w2", required=True)
    ap.add_argument("--wconf", required=True, help="[256] markov slice of confidence_head.proj.weight")
    ap.add_argument("--margins", default="0.25,0.5,0.8,1.0,1.5,2.0")
    ap.add_argument("--weights", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.pairs)
    prev = d["prev"].astype(np.int64)
    target = d["target"].astype(np.int64)
    base_ids = d["base_ids"].astype(np.int64)      # [N, k] pre-bias top-k
    base_vals = d["base_vals"].astype(np.float32)  # [N, k]
    if args.limit:
        prev, target, base_ids, base_vals = (
            prev[: args.limit], target[: args.limit],
            base_ids[: args.limit], base_vals[: args.limit],
        )
    w1 = np.load(args.w1).astype(np.float32)
    w2 = np.load(args.w2).astype(np.float32)
    wconf = np.load(args.wconf).astype(np.float32).reshape(-1)
    assert wconf.shape[0] == w1.shape[1], (wconf.shape, w1.shape)

    n, k = base_ids.shape
    # post-bias values for every candidate of every sample
    bias = np.einsum("nkr,nr->nk", w2[base_ids], w1[prev])  # [N,k]
    post = base_vals + bias

    order = np.argsort(-post, axis=1)
    best_pos, second_pos = order[:, 0], order[:, 1]
    best_val = np.take_along_axis(post, best_pos[:, None], axis=1)[:, 0]
    second_val = np.take_along_axis(post, second_pos[:, None], axis=1)[:, 0]
    best_id = np.take_along_axis(base_ids, best_pos[:, None], axis=1)[:, 0]
    second_id = np.take_along_axis(base_ids, second_pos[:, None], axis=1)[:, 0]
    margin = best_val - second_val

    best_ok = best_id == target
    second_ok = second_id == target
    # the branch can only act where the draft (best) is wrong and the runner-up is right
    pool = (~best_ok) & second_ok
    print(f"samples {n:,}  draft(top1) correct {best_ok.mean() * 100:.2f}%  "
          f"runner-up correct {second_ok.mean() * 100:.2f}%")
    print(f"rescue pool (top1 wrong & top2 right) {pool.sum():,} ({pool.mean() * 100:.2f}%)")
    print(f"margin: median {np.median(margin):.3f}  p10 {np.percentile(margin, 10):.3f}  "
          f"share<=1nat {np.mean(margin <= 1) * 100:.1f}%")

    crit = np.einsum("nr,nr->n", wconf[None, :], w1[second_id] - w1[best_id])
    print(f"confidence-term <w_conf, W1[2]-W1[1]>: mean {crit.mean():.4f} "
          f"std {crit.std():.4f}")

    print("\n  margin   weight   fired   rescued   broken    net   net_rate")
    best = None
    for m in [float(x) for x in args.margins.split(",")]:
        for w in [float(x) for x in args.weights.split(",")]:
            fire = margin <= m
            swap = fire & (second_val + w * crit > best_val)
            rescued = int((swap & pool).sum())
            broken = int((swap & best_ok).sum())
            net = rescued - broken
            rate = net / n * 100
            print(f"  {m:6.2f}  {w:6.2f}  {int(fire.sum()):7d}  {rescued:8d}  "
                  f"{broken:7d}  {net:6d}  {rate:+6.3f}%")
            if best is None or net > best[0]:
                best = (net, m, w, rescued, broken)
    if best:
        net, m, w, rescued, broken = best
        print(f"\nbest: margin {m} weight {w} -> rescued {rescued:,} broken {broken:,} "
              f"net {net:+,} ({net / n * 100:+.3f}% of samples)")


if __name__ == "__main__":
    main()
