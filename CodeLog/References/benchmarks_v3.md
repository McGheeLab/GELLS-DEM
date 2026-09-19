# GELS V3.0 — performance measurements

Machine: 2× Intel Xeon Gold 6148 (40 physical cores, 80 threads), 511 GB RAM, Windows 11,
Python 3.14, numba 0.67 (`omp` threading layer). All runs 2D circles, walls, solid fraction
0.62, R = 40 µm, monolayer cell coverage, default physics; `python -m gels.bench` (median of
5 steps after one warm-up step; render and metrics once).

Reproduce: `python -m gels.bench --mode 2D --N 1000,3000,10000 --steps 5 --path kernels --threads 8,40`
and `--path reference` for the pure-Python loops (`use_numba=False`).

## Reference (pure Python) vs compiled kernels — per step and per save

| N | cells | path | threads | step [s] | cell state [s] | forces [s] | render [s] | metrics [s] |
|---|---|---|---|---|---|---|---|---|
| 1 000 | 2 400 | reference | — | 0.251 | 0.033 | 0.178 | 30.9 | 0.26 |
| 3 000 | 7 200 | reference | — | 0.874 | 0.100 | 0.615 | 298 | 1.23 |
| 1 000 | 2 400 | kernels (Phase 6a–6e) | 8 | 0.0071 | 0.0001 | 0.0053 | 0.046 | 0.040 |
| 3 000 | 7 200 | kernels (Phase 6a–6e) | 8 | 0.0232 | 0.0002 | 0.0180 | 0.212 | 0.100 |
| 10 000 | 24 000 | kernels (Phase 6a–6e) | 8 | 0.0651 | 0.0006 | 0.0506 | 0.402 | 1.06 |
| 10 000 | 24 000 | kernels (Phase 6a–6e) | 1 | 0.0847 | 0.0025 | 0.0682 | 0.603 | 1.08 |
| 10 000 | 24 000 | kernels (Phase 6a–6e) | 40 | 0.1309 | 0.0007 | 0.0976 | 0.479 | 1.08 |

Speed-ups at N = 3 000: step 38×, render 1400×, metrics 12×. The reference render is the
quantity that made runs above a few thousand granules impractical (one full-grid stamp per
granule, grid ∝ domain size); the compiled renderer stamps bounding boxes in parallel.

The 40-thread rows are *slower* than 8 threads below N ≈ 10⁴: a step is ~15 small `prange`
regions and the fork/join cost of 40 OpenMP threads exceeds the work. `perf_threads = 0`
therefore means `auto_threads(N)` = clamp(N / 1000, 4, physical cores).

At N = 10⁴ (Phase 6a–6e, before the cell-list neighbour search) the per-step time was
dominated by serial glue — `cKDTree` pair queries, the Python conversion of clip planes to
lists and the `ContactSoA` assembly — which Phase 6g removed; the 6g rows are appended below
when measured.

## Packing (RSA + inflate-and-relax settle)

| N | reference [s] | kernels, tree settle (bit-identical) [s] | kernels, cell-list settle [s] |
|---|---|---|---|
| 1 000 | 7.5 | 7.4 (RSA grid; settle still tree-bound) | — |
| 3 000 | 20.2 | 15.7 | 8.1 |
| 100 000 | — | 611 | 155 (16 threads) / 184 (40 threads) |
| 5 000 (3D spheres) | — | — | 13.7 (16 threads) |
| 20 000 (3D spheres) | — | — | 62.2 (16 threads) |

The reference RSA scans every placed granule per attempt (O(N²)); the kernel path answers the
same inequality from a neighbour grid but keeps the scalar random draws (attempt counts are
data-dependent, so the stream cannot be batched), which is now the dominant packing cost.

## Phase 6g rows (cell-list neighbours, clip arrays, overlap kernel, strided Voronoi)

Kernel path after Phase 6g; packing still with the tree settle (the cell-list settle landed
after this run — see the packing table).

| N | cells | threads | step [s] | cell state [s] | forces [s] | of which contacts | of which bridging | render [s] | metrics [s] | granule·steps / s |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 000 | 24 000 | 8 | 0.0203 | 0.0004 | 0.0162 | 0.0081 | 0.0041 | 0.427 | 0.177 | 494 000 |
| 10 000 | 24 000 | 16 | 0.0188 | 0.0004 | 0.0145 | 0.0078 | 0.0027 | 0.310 | 0.132 | 531 000 |
| 100 000 | 240 000 | 8 | 0.1807 | 0.0049 | 0.1261 | 0.0495 | 0.0448 | 1.882 | 1.770 | 553 000 |

Against the reference at N = 3000 (0.874 s per step, 298 s per render) the N = 10⁴ step is
43× faster than the *smaller* reference system; a full 72 h run (144 steps of 0.5 h, 36 saves)
at N = 10⁵ costs ≈ 26 s of dynamics plus ≈ 36 × 3.7 s of rendering + metrics ≈ 2.5 min, so at
this size the save cadence, not the physics, sets the wall time. The N = 10⁵ packing took
611 s with the tree settle (dominated by one `cKDTree` per relaxation substep); the cell-list
settle (Phase 6f-2, default) removes that — see the packing table (155 s at 10⁵, of which the
RSA placement with its scalar random draws is now the larger part).

### Second run: N = 10⁵ at 16 / 40 threads (cell-list settle) and 3D spheres

`python -m gels.bench --mode 2D --N 100000 --steps 3 --path kernels --threads 16,40` and
`--mode 3D --N 5000,20000 --steps 3 --threads 16` (3D: Lx = Ly = Lz, solid fraction 0.62, R = 40 µm).

| mode | N | cells | threads | pack [s] | step [s] | cell state [s] | forces [s] | of which contacts | of which bridging | render [s] | metrics [s] | granule·steps / s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2D | 100 000 | 240 000 | 16 | 155.2 | 0.1433 | 0.0028 | 0.1066 | 0.0502 | 0.0288 | 1.777 | 1.799 | 698 000 |
| 2D | 100 000 | 240 000 | 40 | 183.7 | 0.2384 | 0.0028 | 0.1646 | 0.0837 | 0.0365 | 1.698 | 1.800 | 420 000 |
| 3D | 5 000 | 48 000 | 16 | 13.7 | 0.0331 | 0.0032 | 0.0216 | 0.0099 | 0.0072 | 0.651 | 0.337 | 151 000 |
| 3D | 20 000 | 192 000 | 16 | 62.2 | 0.1227 | 0.0130 | 0.0858 | 0.0348 | 0.0335 | 0.513 | 0.390 | 163 000 |

Even at N = 10⁵ the 40-thread step is slower than 16 threads (0.24 s vs 0.14 s): the pair pass is
memory-bound (≈ 20 GB/s plateau measured in Phase 0) and the second socket adds NUMA traffic rather
than bandwidth, so `auto_threads` stays capped by the physical-core count and 16 threads is the
practical sweet spot on this machine for one run. Several runs in parallel (e.g. 5 × 8 threads) use
the machine better than one run at 40. The 3D packing at solid fraction 0.62 is above the jammed
range for spheres and ends with the post-relax warning (max overlap 3.5–4.4 µm against a 2 µm
tolerance); production 3D runs should use ≈ 0.55.

## Showcase: nine full 72 h runs at once (`pipeline/run_showcase.py`, 2026-09-17)

All nine conditions ran concurrently as separate processes (thread budget 40 = physical cores);
144 steps of 0.5 h with a snapshot, render and metrics every 2 h (37 saves). Wall times are
therefore *contended* numbers — several runs shared the machine — not single-run optima.

| condition | mode | N | cells | threads | pack [s] | simulate 72 h [s] |
|---|---|---|---|---|---|---|
| binary f = 1/0 | 2D 5 × 5 mm | 3 183 | 7 759 | 4 | 33 | 46 |
| three species 1/0.5/0 | 2D 5 × 5 mm | 3 183 | 7 750 | 4 | 34 | 52 |
| five-step ladder | 2D 5 × 5 mm | 3 185 | 9 868 | 4 | 34 | 48 |
| low dose 0.2/0 | 2D 5 × 5 mm | 3 183 | 5 937 | 4 | 34 | 42 |
| uniform f = 0.5 | 2D 5 × 5 mm | 3 183 | 12 870 | 4 | 13 | 25 |
| uniform f = 1 | 2D 5 × 5 mm | 3 183 | 12 870 | 4 | 14 | 34 |
| shaped granules (superellipses) | 2D 5 × 5 mm | 3 183 | 8 391 | 4 | 19 | 65 |
| large 2D, three species | 2D 20 × 20 mm | 50 930 | 124 063 | 12–16 | 191 | 188 |
| 3D spheres, three species | 3D 1.6 mm cube | 8 026 | 75 638 | 8 | 86 | 91 |

Physics for all nine (≈ 78 000 granules, ≈ 260 000 cells, 72 simulated hours each) completed
within about six minutes of wall time. For comparison, the V2.7 Python engine needed 0.87 s per
step at N = 3 000 (≈ 2 min of dynamics per run) plus ≈ 5 min per render, i.e. ≈ 3 h per 5 mm run,
and could not have rendered the 20 mm run at all.

The per-run `viz2` suite is now by far the slow part: **31 min** for the 3 183-granule binary run
(seven runs plotting at once), against 79 s for its packing and dynamics. Module breakdown from
the output timestamps, 37 snapshots, 7 759 cells:

| viz2 module | time [s] | share |
|---|---|---|
| movies (5 GIFs × 40 frames) | 892 | 48 % |
| voronoi shape factors | 365 | 20 % |
| phase fractions (Voronoi local plots) | 288 | 15 % |
| percolation | 159 | 9 % |
| energy / stress | 149 | 8 % |
| scaffold map | 40 | 2 % |

Everything above redraws the full scene per snapshot in Python; the compiled kernels do not touch
it. Snapshots are independent, so each of those loops now runs in a process pool
(`viz2.parallel.pmap`, added the same day). Same run, same machine, idle, `python
viz2/postprocess.py -i results/showcase_v3/01_binary --workers N`:

| viz2 module | `--workers 1` [s] | `--workers 16` [s] | speed-up |
|---|---|---|---|
| scaffold map | 0.4 | 0.4 | — |
| voronoi shape factors | 253 | 50 | 5.1× |
| phase fractions | 253 | 40 | 6.3× |
| movies (5 GIFs × 37 frames) | 890 | 100 | 8.9× |
| energy / stress | 147 | 51 | 2.9× |
| void percolation | 160 | 72 | 2.2× |
| **whole suite** | **1709** | **318** | **5.4×** |

All 25 output files are byte-size identical between the two runs. The serial column is already
faster than the 1858 s measured before, because `run_all` now computes each snapshot's Voronoi
tessellation once and shares it instead of repeating it three times.

The two modules that gain least are the ones whose cost is in the parent: `void_clusters.png` and
the energy panels are large multi-panel PNGs drawn and encoded serially after the parallel part
finishes. Amdahl's law puts the suite's serial floor near 60 s for this run, so more than about
16 workers buys little. `--only scaffold phases` cuts post-processing to under a minute when only
the scaffold and composition figures are wanted; the 20 mm run uses that plus the
`LOCAL_VORONOI_MAX_GRANULES` guard.

`viz2/compare_runs.py` itself draws all nine runs, including
the 51 000-granule bed, in ≈ 2 min (24-frame timelapse included).

## 3D sweep: 21 conditions in a 2 mm cube (`--sweep sweep3d`, 2026-09-17/18)

Full factorial over functional granule size, inert granule size and functional share; 1.02 M
granules and 3.3 M cells in total; 144 steps of 0.5 h with a snapshot every 4 h (19 per run).
All 21 conditions ran concurrently under a 40-thread budget and finished in **8 h 10 min**
(29 396 s) of wall time. Cumulative step time across all runs:

| stage | cumulative | note |
|---|---|---|
| packing | 15.1 h | dominates; see the polydispersity note below |
| simulation (72 h each) | 3.6 h | the compiled engine — 6 % of the total |
| post-processing | 19.8 h | almost entirely the 3D movie module on the coarse packings |
| analysis (step 5) | 21.6 h | `analysis/` is unchanged since V1.7 and single-threaded |
| **total step time** | **60.0 h** | compressed into 8.2 h of wall time by running conditions concurrently |

The physics is no longer the cost. Two step-3 examples at opposite ends:

| condition | N | cells | pack [s] | simulate 72 h [s] |
|---|---|---|---|---|
| `func40_inert40_r75` (monodisperse 40 µm) | 127 479 | 309 126 | 1 578 | 782 |
| `func150_inert150_r50` (monodisperse 150 µm) | 2 417 | 55 705 | 24 | 65 |

**Packing degrades badly when the size ratio is large.** Compare two runs of the same sweep:

| condition | N | radii | pack [s] |
|---|---|---|---|
| `func40_inert40_r50` | 127 478 | 20 µm only | 1 722 |
| `func40_inert150_r50` | 64 948 | 20 + 75 µm | 10 914 |
| `func40_inert150_r75` | 96 213 | 20 + 75 µm | 17 270 |

Half the granules, six times the time. The linked-cell grid sizes its cells by the largest
bounding radius, so at a 3.75× radius ratio each cell holds tens of small granules and the pair
scan inside `settle_packing_3d` degenerates towards O(N · granules-per-cell). A size-class-aware
cell list (or a two-level grid) is the fix; it is the largest remaining scaling weakness in the
engine.

**Post-processing and analysis are now the bottleneck, not the engine.** The runs that skipped the
movie module (`--only scaffold phases`, applied above 20 000 granules) post-processed in 8–18 s;
the coarse packings that kept it spent 2 900–11 400 s, because a 3D movie frame draws every
granule and every cell of a z-slice and those runs have 55 000–155 000 cells despite few granules.
The threshold should key on cell count, not granule count.

Both of those line items were then traced to one routine: `analysis/coarse_grain.py::coarse_grain_field`
swept every grid cell against every contact in a Python loop (Ngrid³ × n_contacts = 4 × 10⁸ iterations
per call in 3D). It is used by the stress movie, the stress / strain maps and step 5. Vectorising it
made it 69–141× faster with agreement to 1e-15 (table in the CHANGELOG), so the sweep's 19.8 h of
plotting and 21.6 h of analysis are not representative of the current code — they measure the
pre-fix path.

