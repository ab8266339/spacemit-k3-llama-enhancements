# K3 decode synchronization and AIDMA

We tested Qwen3.6-35B-A3B with the same Q4_0 GGUF and the standalone A100 device used in the original comparison. These are measurements from a physical K3 with Bianbu 4.0.4, SpacemiT llama.cpp v0.2.5, and Spine runtime 0.7.2. The model used 128 uncached input tokens, 128 greedy output tokens, batch and physical batch 128, one slot and eight inference threads. Each server load ran six requests. The first was warm-up.

| Configuration | Median generated tokens/s |
|---|---:|
| Locally built stock backend, first control | 8.826 |
| Group small routing operations on one tile | 8.855 |
| Bounded-spin graph rendezvous with cooperative fallback | 8.839 |
| Stock backend, second control | 8.820 |
| MMZ tile schedule with RVV copies | 8.731 |
| Same tile schedule with at most one asynchronous DMA request outstanding | 8.696 |
| Quantize short Q4 GEMV activation rows in each tile's TCM | 8.894 |
| Repeat the private-row candidate in a fresh process | 8.866 |
| Reversed-order stock control immediately after the repeat | 8.908 |

Every completed configuration matched all six reference continuations, both short answer checks, and the twelve top-ten next-token records. The largest initial non-DMA gain reversed on independent repetition. The default launcher stays on the vendor kernel.

## What DMA accomplished

The [Spine 0.7.2 API](https://github.com/spacemit-com/spine-runtime/blob/0.7.2/docs/api-reference.md) provides device-reachable MMZ allocations and tile-owned TCM. We used eight 4 MiB MMZ blocks, initialized once, and kept heap pointers out of every DMA submission. In the copy-only probe, each worker rotated source offsets and alternated two scratch destinations. Both final destination buffers were compared bytewise per worker and trial. On eight simultaneous workers, the 69,632-byte case took 21.68 microseconds with the vendor RVV copy from ordinary DDR, 24.95 with RVV from MMZ, and 36.77 with DMA submission plus immediate completion wait.

The first full-model experiment mirrored selected repacked Q8 weights once at model load and pipelined transfers into alternating scratch buffers. All eight lanes submitted DMA requests, but the first full request exceeded 240 seconds. A later model load encountered memory pressure and the board was restarted. The precise runtime or driver cause was not established.

The bounded experiment capped individual transfers at 128 KiB, allowed one hardware request in flight across workers, and let busy lanes fall back to RVV. It retained overlap with independent GEMV work. Output bytes matched on four isolated `Q8_0` GEMV shapes and the full-model output checks passed. At K=1024, N=2048, the isolated operator took 0.1764 ms on the control, 0.1085 ms with the new RVV schedule, and 0.1166 ms with DMA. That shape alone did not predict the full-model result.

## Interpreting the hotspots

The original roughly 23% synchronization and 18% copy figures were user-space program-counter samples. In a new call-chain trace, 541 of 1,497 copy samples had a dense Q6_K matrix operation as caller, 356 had a row gather, and 304 had a dense Q8_0 matrix operation. On two native-build controls with nearly identical throughput, the named synchronization functions occupied 16.8–20.5% and RVV copies 17.7–21.8%. The bounded DMA run measured 21.4% and 18.4%. Moving work to the driver or to an inlined spin loop can change these sampling fractions without improving wall-clock generation.

[Per-round throughput, output hashes, quality probes, six candidate/control configurations and repeat runs, operator results and profiles](../../results/qwen36-decode-dma.json). Grok Build assisted with the source audit. No kernel change from this experiment was submitted upstream.
