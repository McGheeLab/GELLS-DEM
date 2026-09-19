Written for: anyone running GELS simulations locally (primarily the maintainer).

# GELS local pipeline

Five numbered steps, run one at a time. Every step reads and writes one **run
directory** and records its progress in `<run_dir>/pipeline_state.json`, so you
can stop after any step, inspect what it produced, and pick up later.

```
step1_config.py  ->  step2_pack.py  ->  step3_simulate.py  ->  step4_postprocess.py  ->  step5_analysis.py
   params.json        snap_0000.npz      snap_NNNN.npz           viz2_plots/               analysis/
```

## Quick start

```bash
python pipeline/step0_new_setup.py -o setup.yaml          # commented template; edit it
python pipeline/step1_config.py  --setup setup.yaml --name my_run
python pipeline/step2_pack.py    -i results/my_run
python pipeline/step3_simulate.py -i results/my_run
python pipeline/step4_postprocess.py -i results/my_run
python pipeline/step5_analysis.py -i results/my_run
```

Legacy trial files and flat flags still work (`step1_config.py --trial Trials/DOE2_2D_0001.json`,
`--E_modulus 5`); they describe a two-species run (collagen-coated f = 1 / bare f = 0).

Each step prints the exact command for the next one when it finishes, and a
progress checklist:

```
  Progress for .../results/my_run:
    [x] 1. Configure          2026-09-16T12:16:43
    [x] 2. Generate packing   2026-09-16T12:16:50
    [ ] 3. Run simulation
    [ ] 4. Post-process
    [ ] 5. Analysis
```

## The steps

### Step 0 (optional) — Write a setup file

A run is described by a **sectioned YAML file**: `domain`, `boundary`, `granules`
(total solid fraction + a list of granule *species*), `cells`, `contact`, `dynamics`,
`packing`, `time`, `output`, `performance`. Each species has a name, a degree of
collagen-I functionalization `f ∈ [0,1]`, a share of the solid volume, a colour,
a radius distribution and optional shape moments and stiffness:

```yaml
granules:
  solid_fraction: 0.65
  species:
    - {name: collagen_full, functionalization: 1.0, volume_fraction: 0.20, color: "#CC2222",
       radius: {mean_um: 40, std_um: 5, min_um: 15}}
    - {name: collagen_half, functionalization: 0.5, volume_fraction: 0.40, color: "#2255CC",
       radius: {mean_um: 40, std_um: 5, min_um: 15}}
    - {name: bare,          functionalization: 0.0, volume_fraction: 0.40, color: "#22AA22",
       radius: {mean_um: 60, std_um: 8, min_um: 20}}
```

```bash
python pipeline/step0_new_setup.py -o setup.yaml                        # commented template
python pipeline/step0_new_setup.py --from-trial Trials/DOE2_2D_0001.json -o doe1.yaml
python pipeline/step0_new_setup.py --from-run results/my_run -o rerun.yaml
```

The template carries the recommended physical values for human dermal fibroblasts on
collagen-I microgels, each with a comment; provenance is in
`CodeLog/References/fibroblast_parameters.md`. Traction follows a saturating law in
absolute ligand density (`cells.ligand.sigma_max_per_um2`, `K_traction_per_um2`): with the
defaults a 20 %-coated granule is pulled at ~59 % of full force and keeps ~80 % of its cells.

### Step 1 — Configure

Resolves parameters and writes `<run_dir>/params.json` (what the engine reads) and
`<run_dir>/setup.yaml` (the sectioned form). Nothing is simulated, so it is cheap to
re-run while you settle on values. Precedence, lowest first:

1. `Params` dataclass defaults
2. a setup file (`--setup setup.yaml`) **or** a legacy Trial JSON
   (`--trial Trials/DOE2_2D_0001.json`, flat or nested format)
3. presets (`--preset pmma_well,fibroblast_realistic`, repeatable) — V3.1
4. dotted overrides (`--set granules.species[1].functionalization=0.3`, repeatable)
5. individual flat flags (`--E_modulus 5.0 --t_total 48`) — one exists for every
   `Params` field

```bash
python pipeline/step1_config.py --setup setup.yaml --name my_run
python pipeline/step1_config.py --setup setup.yaml --name f30 --set granules.species[1].functionalization=0.3 --set time.t_total_h=24
python pipeline/step1_config.py --setup well.yaml --preset inert_wall --name well_inert
python pipeline/step1_config.py --trial Trials/DOE2_2D_0001.json
python pipeline/step1_config.py --name stiff3d --mode 3D --E_modulus 20 --t_total 48
```

### Presets (V3.1)

A preset is a named bundle of dotted overrides, because the settings that matter come in
correlated groups: a PMMA well is a container shape, a boundary condition, a gravity setting,
a contact treatment and a packing mode that only make sense together.

```bash
python pipeline/step0_new_setup.py --list-presets
python pipeline/step0_new_setup.py --preset pmma_well,fibroblast_realistic -o well.yaml
```

| preset | what it sets |
|---|---|
| `pmma_well` | 2 mm cylindrical well, free top, gravity, PMMA contact, sedimenting packer |
| `pmma_dish_slice` | the same as a fast 2D dish slice (floor, two side walls, free top) |
| `functionalized_wall` | coated container + an immobile granule lining cells can bridge to |
| `inert_wall` | bare container (the default): the bed can detach from it |
| `fibroblast_realistic` | Hill contraction, 80 µm reach, rigid-substrate stall force, division |
| `hydrogel_box_legacy` | the V3.0 defaults, stated explicitly |

Applied names are recorded in `meta.presets`, and a setup written with `--preset` carries a
header listing every value that changed. Per-key documentation stays in `TEMPLATE_YAML`
(PyYAML cannot emit comments).

`--name X` puts output in `results/X`; `-o PATH` sets the directory outright.
`--seed` (default 42) is recorded and reused by later steps.

### Step 2 — Generate packing

RSA placement plus Lubachevsky–Stillinger inflate-and-relax, then cell seeding
and the `t = 0` force/metric evaluation. Writes `snapshots/snap_0000.npz`,
`history.json` and `metadata.json`, and reports granule counts, contacts,
coordination number Z, porosity and permeability.

Packing is the expensive non-dynamical part of a run, so isolating it means you
can inspect a packing before committing to a long simulation, and re-run the
dynamics against the identical initial condition.

### Step 3 — Run simulation

The expensive step. Restores snapshot 0000 and advances the overdamped
dynamics to `t_total`, saving a snapshot every `save_every_h` hours.

```bash
python pipeline/step3_simulate.py -i results/my_run
python pipeline/step3_simulate.py -i results/my_run --continue          # after an interruption
python pipeline/step3_simulate.py -i results/my_run --t_total 120 --continue  # extend a finished run
python pipeline/step3_simulate.py -i results/my_run --force             # restart from the packing
python pipeline/step3_simulate.py -i results/my_run --archive           # also write a .tar.gz
```

`--continue` resumes from the last snapshot on disk rather than starting over,
which also lets you extend a completed run to a later `t_total`.

#### Watching a run (V3.0)

```bash
python pipeline/step3_simulate.py -i results/my_run --live                 # viewer window alongside the run
python pipeline/step3_simulate.py -i results/my_run --live --live-every 5  # fixed cadence (default: adaptive)
python pipeline/step3_simulate.py -i results/my_run --threads 20           # numba threads for the kernels
python pipeline/live_view.py -i results/my_run                             # attach from another terminal
python pipeline/live_view.py -i results/my_run --replay --loop --fps 15    # play back a finished run
python pipeline/live_view.py -i results/my_run --headless                  # PNG per snapshot -> results/my_run/live/
```

The viewer is a separate process fed through a bounded queue, so the run never
waits for it (frames are dropped when the window falls behind). Keys: `space`
pause/resume, `n` single step, `x` stop the run gracefully (state is saved and
`--continue` resumes it), `q` detach (the run carries on), `c` colour by
species / functionalization / speed, `b` `l` `m` toggle cells / bridges /
metrics, `g` periodic ghost images, `[` `]` `0` move the z-slice (3D), `s`
screenshot to `results/my_run/live/`, `h` help. With `--no-live-hold` the
window closes when the run finishes; otherwise step 3 waits for you to close it.

### Step 4 — Post-process

Renders figures and movies. Two suites:

| `--suite` | Output | Contents |
|---|---|---|
| `viz2` (default) | `viz2_plots/` | scaffold maps, Voronoi, phase fractions, movies, energy/stress, void percolation, 3D scaffold maps |
| `viz` | `plots/` | V1.x suite: granule/field panels, timeseries, compaction, percolation, cells, surface stress, deformation |
| `both` | both dirs | |

```bash
python pipeline/step4_postprocess.py -i results/my_run --skip movies
python pipeline/step4_postprocess.py -i results/my_run --only scaffold phases
python pipeline/step4_postprocess.py -i results/my_run --suite both
python pipeline/step4_postprocess.py -i results/my_run --workers 16
```

The `viz2` suite draws every snapshot independently, so `--workers N` renders
them in a process pool (default: physical cores capped at 8; `--workers 1` is
the serial path). This matters: plotting is plain Python and matplotlib, not
compiled kernels, and on a 3 200-granule run it takes far longer than the
simulation it is drawing.

Since V2.6 the engine does not save phase-field grids (`save_fields=False`).
Both suites are driven here so the fields are rebuilt from particle positions
on demand, so figures are correct whether or not `fields/` exists.

### Step 5 — Analysis

Runs the `analysis/` package against the finished run and collects results into
`<run_dir>/analysis/summary.json`, with each module's plots in its own
subdirectory.

| Module | Produces |
|---|---|
| `mean_field` | two-zone compaction ODE fit: R², η_eff, σ₀, α |
| `coarse_grain` | stress tensor, strain rate, effective viscosity, Z(t) |
| `descriptors` | tissue architecture descriptors (BV/TV, thickness, Minkowski, correlation length) |
| `arch_distance` | architectural distance to each organ target, closest organ |

```bash
python pipeline/step5_analysis.py -i results/my_run
python pipeline/step5_analysis.py -i results/my_run --modules descriptors arch_distance
```

Modules run independently — one failing records its error in `summary.json`
and does not abort the others.

## Several conditions at once, compared on one figure

`pipeline/run_showcase.py` runs a table of conditions concurrently — each as its
own chain of steps 1–5 in a separate process with its own thread count — and
then draws them all together:

```bash
python pipeline/run_showcase.py --list              # the conditions and their expected N
python pipeline/run_showcase.py                     # all nine -> results/showcase_v3/
python pipeline/run_showcase.py --only 01_binary 02_three_species --t_total 24
python pipeline/run_showcase.py --compare-only      # redraw the comparison figures
```

Three condition sets are built in, chosen with `--sweep`:

```bash
python pipeline/run_showcase.py --sweep showcase           # the nine V3.0 demos (default)
python pipeline/run_showcase.py --sweep sweep3d            # 21-condition 3D sweep
python pipeline/run_showcase.py --sweep well               # V3.1 well calibration set
python pipeline/run_showcase.py --sweep well --list        # its condition table
```

`well` is the V3.1 calibration set: ten conditions that vary **one thing at a time** —
container (a closed-box control, a 2D dish slice, a 3D cylindrical well), bridge force model
(constant vs contractile), cell division, and wall chemistry (inert vs functionalized) — so
each can be fitted against a measured bed height separately. The 2D dish slices are ~1 900
granules and take about a minute each; the 3D well at 80 µm granules is ~10 000 granules.

To compare against measured data, pass a CSV of bed geometry vs time:

```bash
python viz2/compare_runs.py -i results/well_calibration -o results/well_calibration/comparison \
    --experiment measured.csv
```

with columns `t_h, bed_height_um[, bed_height_sd_um][, bed_diameter_um]`. The measured points
are overlaid on the bed-height and bed-radius panels, and `compare_bed_profiles.png` draws a
vertical section through the container axis at several times (it works on closed boxes too,
which is how a packing artefact shows up at a glance).

`sweep3d` is a full factorial in a 2 mm cube over functional granule size (mean diameter
40 / 80 / 150 µm), inert granule size (the same three) and functional share of the solid
(0.50 / 0.75 / 1.00), with cells seeded to 0.8 of each granule's surface area. The nine
100 %-functional cells collapse to three, giving 21 runs from 2 400 to 127 500 granules.
Output goes to `results/sweep3d/` and the setup files to `Trials/sweep3d/`.

The built-in table is the V3.0 showcase: a six-run functionalization ladder on
one 5 × 5 mm domain (binary, three species, five-step ladder, low dose, uniform
f = 0.5, uniform f = 1), shaped granules per species, a 20 × 20 mm bed of
~51 000 granules and a 1.6 mm 3D cube. Each condition is written as a commented
setup file in `Trials/showcase_v3/` (edit it, or run it alone with
`step1_config.py --setup`). A weighted scheduler keeps the sum of running
threads within `--budget` (default: physical cores) during the physics steps;
post-processing and analysis are single-threaded and overlap with the next
condition's physics. Several runs at 4–16 threads each use the workstation
better than one run at 40. The output folder gets a `README.md` describing
every condition and where the figures are, plus `showcase_manifest.json`.

The comparison itself is `viz2/compare_runs.py`, which works on any set of
finished runs:

```bash
python viz2/compare_runs.py -i results/showcase_v3 -o results/showcase_v3/comparison
python viz2/compare_runs.py -i results/run_a results/run_b --labels "stiff" "soft"
```

It writes `compare_scaffolds.png` (final scaffold of every run, same colour for
the same f, scale bars, one legend), `compare_timeseries.png` (bridging, bridge
force, connectivity, porosity, displacement — one line per run),
`compare_species.png`, `compare_dose_response.png` (per-species outcome against
that species' f), `compare_timelapse.gif` (all runs on a common time axis)
and `compare_summary.md/.csv` (sizes, composition, wall times, final metrics).
3D runs appear as their z-midplane slice. `--skip` takes any of `scaffolds
timeseries species dose summary timelapse`.

## Re-running steps

A completed step refuses to run again and tells you how to override:

```
  Step 2 already completed for this run (2026-09-16T12:16:50).
    n_granules: 11
    Z: 3.455

  Nothing to do. Re-run it anyway with:

      python pipeline/step2_pack.py -i results/my_run --force
```

`--force` on steps 2 and 3 also cleans up downstream state so you never resume
from a stale snapshot: re-packing clears all snapshots and history; re-running
the simulation drops everything after snapshot 0000 and truncates history back
to its `t = 0` entry.

Running a step before its prerequisite exits with status 1 and names the step
you need first.

## Run directory layout

```
results/my_run/
├── params.json            full Params (step 1; load_run-compatible)
├── metadata.json          version, git hash, seed, granule counts
├── pipeline_state.json    per-step completion and details
├── history.json / .csv    scalar metrics per timepoint
├── snapshots/
│   ├── snap_0000.npz      initial packing (step 2)
│   └── snap_NNNN.npz      simulated timepoints (step 3)
├── fields/                phase-field grids (only if save_fields=True)
├── live/                  viewer screenshots (s key) and --headless frames
├── viz2_plots/            step 4
├── showcase_log.txt       run_showcase.py only: the pipeline output of this condition
├── showcase.json          run_showcase.py only: label, thread count, per-step wall times
├── plots/                 step 4, --suite viz
└── analysis/
    ├── summary.json       step 5
    ├── mean_field/
    ├── coarse_grain/
    ├── descriptors/
    └── arch_distance/
```

## Notes

- **Reproducibility.** Steps 2 and 3 are joined by the engine's V2.6 resume
  mechanism, which restarts the random number stream at the resume point. A
  packing + simulate pair is therefore fully deterministic for a given seed,
  but not bit-for-bit identical to a single monolithic `run()` call.
- **Encoding.** The steps reconfigure stdout to UTF-8, because the engine
  prints characters (`ν`, `µ`, `→`) that a default Windows cp1252 console
  cannot encode.
- **Headless.** `MPLBACKEND=Agg` is set automatically and no pipeline script
  imports matplotlib at module level; figures are written to disk, never shown.
  The one window is the live viewer, which runs in its own process on TkAgg
  (`step3 --live`, `live_view.py`).
- **3D rendering.** The PyVista-backed 3D modules need a working `vtk`. See the
  troubleshooting note in the main README if VTK fails to load.
- **Performance (V3.0).** Steps 2 and 3 run on the compiled numba kernels by
  default (`use_numba: true`). The thread count is chosen from the granule
  count (`performance.threads: 0`); pass `--threads N` to step 3 to override.
  `performance.cells_backend: python` keeps the exact V2.7 cell/bridging code
  (slow, step-for-step comparable with older runs); `use_numba: false` runs
  the pure-Python reference everywhere. Snapshots are written by a background
  thread (`performance.async_io`), so a slow disk never stalls the run for
  more than `io_queue_depth` files. `python -m gels.bench` times the stages.
