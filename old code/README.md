Written for: the maintainer, when wondering why something in here no longer runs.

# Archived code

Everything in this folder is **retired**. It is kept for reference only and is
not part of the working tree. As of V2.7 the project runs locally through the
numbered steps in [`pipeline/`](../pipeline/README.md); HPC execution has been
dropped entirely.

**These scripts do not run as-is.** They import `new_dem_0` and `lsdem`, which
no longer exist at the repository root — the engine moved to `gels/engine.py`
and `gels/lsdem.py`. Imports here were deliberately left untouched so the files
remain a faithful record of the V2.6 state. To revive one, update its imports:

```python
from new_dem_0 import Params, run   ->   from gels.engine import Params, run
from lsdem import init_lsdem        ->   from gels.lsdem import init_lsdem
```

## Contents

| Path | Was | Replaced by |
|---|---|---|
| `hpc/` | SLURM templates, `Alex.json` cluster config, env setup, `sync_results.py`, `generate_hpc_scripts.py` | nothing — runs are local |
| `run_all_trials.py` | Batch trial runner (local / SLURM modes 1–4, incl. batch resume) | run `pipeline/` steps per trial |
| `run_hpc_headless.py` | Headless single-run CLI with trial-JSON loading | `pipeline/step1_config.py` + `step3_simulate.py`; its trial-JSON loader lives on in `pipeline/_common.py` |
| `run_analysis_pipeline.py` | V1.7 analysis orchestrator, hardcoded to `DOE_01`–`DOE_23` under `results/trials/trials` | `pipeline/step5_analysis.py`, which works on any single run |
| `reconstruct_history.py` | Rebuild `history.json` from snapshots after a crash | `step3_simulate.py --continue` |
| `test_hertz_clipping.py` | Standalone LS-DEM/JKR contact-clipping demo | — |

## Recovering the original files

The V2.6 tree is preserved in git:

```bash
git show origin/Version-2.4:new_dem_0.py > new_dem_0.py
git checkout origin/Version-2.4 -- hpc/
```
