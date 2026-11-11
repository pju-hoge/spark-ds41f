# Return to the qualified 64 KiB kernel (1019) with the memory saver active

Date: 2026-11-11. Site: `tp4-ring` on four DGX Spark nodes (192.168.1.96 rank 0,
.98 rank 1, .99 rank 2, .95 rank 3).

## What was wrong

Every node was running `7.0.0-1024-nvidia-64k` while
[config/kernel-trial.json](../../config/kernel-trial.json) pins
`7.0.0-1019-nvidia-64k`. The GRUB pin file
`/etc/default/grub.d/zz-spark-kernel-trial.cfg` had been rewritten to the 1024
entry on 2026-11-04 (a `.bak-20261104-64k` copy on each node still holds the 1019
entry). The memory saver's DKMS build is registered only for the pinned kernel
(`dkms status`: `dgx-spark-memory-saver/0.4.1, 7.0.0-1019-nvidia-64k`), and its
`dkms.conf` restricts the build to the kernels listed in `compatibility.json`.
Therefore, since 2026-11-04:

- `nvidia-uvm` was the stock module, not the packed one, and
- `/sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables` did not exist,

so the `tp4-ring` profile (10.5 GiB KV per rank) was served without the memory
saver that the profile is sized around. `doctor --live` reported this on all four
nodes as `memory-saver packing is not enabled in the loaded UVM module`.

## Why not 1025

The archive carries the whole 1025 set for this kernel
(`linux-image|modules|headers|tools-nvidia|modules-nvidia-580-open-7.0.0-1025-nvidia-64k`
at `7.0.0-1025.26~24.04.1`) and the 1025 `nvidia.ko`/`nvidia-uvm.ko` report
driver `580.178.04` — the same driver the pinned memory-saver revision patches.
But upstream's `compatibility.json` still lists only `7.0.0-1019-nvidia-64k` for
both 580.178.04 and 610.57.04, so a 1025 build means extending that list and
claiming a validation upstream has not made. Decision: return to the pinned
combination instead (master's call, 2026-11-11).

Note for later: install and verify a 1025 (or newer) memory saver only when
upstream lists that kernel, or when we are prepared to own the qualification
(build, `uvm_pack_sysmem_leaf_tables=Y`, CUDA validation, soak, and a
three-run bench on the same prompt set).

## What was done

Per node, with the cluster stopped (`bin/spark --cluster-config config/cluster.json
cluster stop --remove --parallel --apply` from rank 3's checkout — the OCI
checkout's `config/nodes-ring4.local.json` holds template addresses, so cluster
commands must run from a node):

1. `pin-1019.sh --apply` (kept outside the repository at `~/scripts/pin-1019.sh`;
   it refuses to change anything without `--apply`): rewrote
   `zz-spark-kernel-trial.cfg` to
   `gnulinux-advanced-$uuid>gnulinux-7.0.0-1019-nvidia-64k-advanced-$uuid`
   (backup `zz-spark-kernel-trial.cfg.bak-YYYYMMDD-HHMMSS`) and ran `update-grub`.
2. Reboot. Order: 192.168.1.99, then .98 and .95 together, then the head .96 last.
3. Verify: `uname -r` = `7.0.0-1019-nvidia-64k`, `getconf PAGESIZE` = 65536,
   `uvm_pack_sysmem_leaf_tables` = `Y`, driver `580.178.04`, `/swap-64k.img`
   active, no failed units.

The installed module is `srcversion 37A4D127AE60F1F3AE60A02`, signed by the
enrolled fleet MOK key — the value pinned in `config/kernel-trial.json`.

`cluster start --apply` reached API readiness at **+131.8 s** (2 m 24 s wall).
`doctor --live` from rank 3: `configuration OK; live cluster matches with 6
warnings`, with no kernel or memory-saver finding and no `nodes differ` line.

## Measured after the change

MemAvailable with the engine loaded: .96 24.3 GiB, .98 25.8 GiB, .99 24.4 GiB,
.95 15.5 GiB (rank 3 also serves `hy-mt2-goro` on :8012).

Single stream, temperature 0, 192 generated tokens, 1.5 k-token prompt,
`(generation_tokens - 1) / (wall - ttft)`, three runs after three warm-ups:
**96.2 tok/s median, TTFT 0.193 s** — consistent with the recorded c1 baseline
(81.4 tok/s) rather than the 203.8 tok/s in the 2026-11-10 trace-overhead note,
which used a different prompt and window and is not comparable.

Under a 40 s sustained load all four nodes held 2470-2535 MHz at 53-56 °C with
no throttle reason (`clocks_throttle_reasons.active` = 0), so the rate is not
thermal.

## Open findings

- **fan-control is not installed on any node** (`dgx_ec_fan_control` absent for
  both 1019 and 1024, no unit, no DKMS entry). `doctor --live` therefore
  recommends `scripts/install-fan-control`, and the same missing module also
  makes it recommend `sudo scripts/install-64k-kernel`. Ticks stayed at 53-56 °C
  without it, but the GB10 fan curve should be restored before long runs.
- The 4 KiB 1025 kernel (`linux-image-7.0.0-1025-nvidia`) remains installed and
  `/var/run/reboot-required` stays populated; it is the fallback entry and does
  not affect the pinned boot.

## Upstream merge (same session)

`upstream` (christopherowen/spark-ds41f) was 189 commits ahead of our `site`
branch and 19 behind. The merge changed only docs, site pages, release-test
scripts and `scripts/render_docs.py`; `patches/` and `bin/` are byte-identical,
so no image rebuild is involved. One conflict, in
`manifests/sources/2026-11-07-spark-ds41f-1107-source.json`: upstream added a
second `patch_head` field near the top of the `vllm` block. Resolution keeps our
643-patch head `112ad3d8`, tree `34995a8f` (the series is upstream's 641 plus the
two DSpark trace patches) and drops the duplicate key. Merge commit `8d861661`
on `site`.
