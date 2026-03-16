"""
Run the full V1.7 mathematical analysis pipeline on DOE data and collect results.
Outputs: JSON summary + individual module results + plots.
"""
import os
import sys
import json
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCAN_DIR = 'results/trials/trials'
OUTDIR = 'results/analysis'
os.makedirs(OUTDIR, exist_ok=True)


def run_mean_field():
    """Run mean-field model fitting on all DOE runs."""
    print("\n" + "=" * 60)
    print("STEP 1: Mean-Field ODE Model Fitting")
    print("=" * 60)
    from analysis.mean_field_model import from_run, run_all as mf_run_all
    import matplotlib.pyplot as plt

    results = {}
    for i in range(1, 24):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(SCAN_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        try:
            model, fit_result, data = from_run(run_dir)
            p_obj = data['params']
            results[name] = {
                'R2': fit_result['R2'],
                'eta_eff': fit_result['eta_eff'],
                'sigma_0': fit_result['sigma_0'],
                'alpha': fit_result['alpha'],
                'phi_f_initial': float(data['phi_f'][0]),
                'phi_f_final': float(data['phi_f'][-1]),
                'phi_f_change': float(data['phi_f'][-1] - data['phi_f'][0]),
                'n_timepoints': len(data['time']),
                'E_modulus': getattr(p_obj, 'E_modulus', None),
                'n_cells_per_granule': getattr(p_obj, 'n_cells_per_granule', None),
            }
            print(f"  {name}: R²={fit_result['R2']:.4f}, η_eff={fit_result['eta_eff']:.2f}, "
                  f"Δφ_f={data['phi_f'][-1]-data['phi_f'][0]:.4f}")

            # Generate individual plot
            outdir_run = os.path.join(OUTDIR, 'mean_field', name)
            os.makedirs(outdir_run, exist_ok=True)
            try:
                mf_run_all(run_dir, outdir=outdir_run)
            except Exception as e:
                print(f"    Plot error: {e}")
            plt.close('all')
        except Exception as e:
            print(f"  {name}: ERROR -- {e}")
            results[name] = {'error': str(e)}

    # Save summary
    with open(os.path.join(OUTDIR, 'mean_field_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved mean-field results for {len(results)} runs")
    return results


def run_coarse_grain():
    """Run coarse-graining on all DOE runs."""
    print("\n" + "=" * 60)
    print("STEP 2: Coarse-Graining (Stress, Strain Rate, Viscosity)")
    print("=" * 60)
    from analysis.coarse_grain import extract_continuum_timeseries
    from new_dem_0 import load_run
    import matplotlib.pyplot as plt

    results = {}
    for i in range(1, 24):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(SCAN_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        try:
            hist, snaps, p, meta = load_run(run_dir)
            if not snaps:
                print(f"  {name}: no snapshots, skipping")
                continue

            ts = extract_continuum_timeseries(run_dir)
            eta_arr = ts['eta_eff']
            finite_eta = eta_arr[np.isfinite(eta_arr)]
            results[name] = {
                'n_timepoints': len(ts['time']),
                'pressure_initial': float(ts['pressure'][0]),
                'pressure_final': float(ts['pressure'][-1]),
                'von_mises_final': float(ts['von_mises'][-1]),
                'eta_eff_mean': float(np.mean(finite_eta)) if len(finite_eta) > 0 else None,
                'Z_final': float(ts['Z'][-1]),
                'Z_ff_final': float(ts['Z_ff'][-1]),
            }
            print(f"  {name}: P=[{results[name]['pressure_initial']:.2f} → {results[name]['pressure_final']:.2f}] kPa, "
                  f"Z={results[name]['Z_final']:.2f}")
            plt.close('all')
        except Exception as e:
            print(f"  {name}: ERROR -- {e}")
            results[name] = {'error': str(e)}

    with open(os.path.join(OUTDIR, 'coarse_grain_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved coarse-graining results for {len(results)} runs")
    return results


def run_dimensionless():
    """Run dimensionless analysis across all DOE runs."""
    print("\n" + "=" * 60)
    print("STEP 3: Dimensionless Analysis (β, Ca, Jamming)")
    print("=" * 60)
    from viz.dimensionless import load_doe_data, run_all as dim_run_all
    import matplotlib.pyplot as plt

    outdir = os.path.join(OUTDIR, 'dimensionless')
    os.makedirs(outdir, exist_ok=True)

    try:
        data = load_doe_data(SCAN_DIR)
        print(f"  Loaded {len(data['runs'])} DOE runs")
        print(f"  Factors: {data['factor_names']}")
        dim_run_all(data=data, outdir=outdir)
        plt.close('all')

        # Extract summary
        results = {}
        from viz.dimensionless import compute_dimensionless_groups
        for run in data['runs']:
            groups = compute_dimensionless_groups(run['params'])
            results[run['name']] = {
                'beta': groups.get('beta', None),
                'Ca': groups.get('Ca', None),
                'jamming_proximity': groups.get('jamming_proximity', None),
                'phi_ratio': groups.get('phi_ratio', None),
            }
        with open(os.path.join(OUTDIR, 'dimensionless_summary.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved dimensionless results")
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_tissue_descriptors():
    """Compute tissue descriptors for last snapshot of each DOE run."""
    print("\n" + "=" * 60)
    print("STEP 4: Tissue Architecture Descriptors")
    print("=" * 60)
    from analysis.tissue_descriptors import compute_descriptor_vector

    results = {}
    for i in range(1, 24):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(SCAN_DIR, name)
        fields_dir = os.path.join(run_dir, 'fields')
        params_path = os.path.join(run_dir, 'params.json')

        if not os.path.isdir(fields_dir):
            continue

        try:
            # Load last field snapshot
            field_files = sorted(f for f in os.listdir(fields_dir) if f.startswith('fields_'))
            if not field_files:
                continue
            last_field = os.path.join(fields_dir, field_files[-1])
            data = np.load(last_field)
            phi_f = data['phi_f']
            phi_i = data['phi_i']
            phi_v = data['phi_v']

            # Get dx from params
            with open(params_path) as f:
                params = json.load(f)
            Lx = params.get('Lx', 800.0)
            Ngrid = phi_f.shape[0]
            dx = Lx / Ngrid

            # Compute descriptors
            desc = compute_descriptor_vector(phi_f, phi_i, phi_v, dx, boundary_exclusion=0.2)
            # Convert numpy types to native Python
            desc_clean = {}
            for k, v in desc.items():
                if isinstance(v, (np.floating, np.integer)):
                    desc_clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    desc_clean[k] = v.tolist()
                else:
                    desc_clean[k] = v

            results[name] = desc_clean
            bvtv = desc.get('BV_TV', desc.get('bv_tv', 'N/A'))
            print(f"  {name}: BV/TV={bvtv:.3f}" if isinstance(bvtv, float) else f"  {name}: computed {len(desc)} descriptors")

        except Exception as e:
            print(f"  {name}: ERROR -- {e}")
            results[name] = {'error': str(e)}

    with open(os.path.join(OUTDIR, 'tissue_descriptors.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved tissue descriptors for {len(results)} runs")
    return results


def run_arch_distance():
    """Compute architectural distance to organ targets."""
    print("\n" + "=" * 60)
    print("STEP 5: Architectural Distance to Organ Targets")
    print("=" * 60)
    from analysis.arch_distance import run_all as arch_run_all
    import matplotlib.pyplot as plt

    outdir = os.path.join(OUTDIR, 'arch_distance')
    os.makedirs(outdir, exist_ok=True)

    try:
        arch_run_all(scan_dir=SCAN_DIR, outdir=outdir)
        plt.close('all')
        print("  Architectural distance analysis complete")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()


def main():
    print("GELLS-DEM V1.7 Mathematical Analysis Pipeline")
    print(f"Data: {SCAN_DIR}")
    print(f"Output: {OUTDIR}")

    mf_results = run_mean_field()
    cg_results = run_coarse_grain()
    dim_results = run_dimensionless()
    td_results = run_tissue_descriptors()
    run_arch_distance()

    # Compile master summary
    summary = {
        'n_runs_analyzed': 23,
        'mean_field': mf_results,
        'coarse_grain': cg_results,
        'dimensionless': dim_results,
        'tissue_descriptors': td_results,
    }
    with open(os.path.join(OUTDIR, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {OUTDIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
