"""
Headless runner for GELLS-DEM on HPC.

Runs the simulation and saves all figures to disk (no plt.show()).
Use this instead of `python3 new_dem_0.py` on HPC where there's no display.

Usage:
    python3 run_hpc_headless.py                          # default params
    python3 run_hpc_headless.py --t_total 48             # override params
    python3 run_hpc_headless.py --trial Trials/Trial1.json
    python3 run_hpc_headless.py --E_modulus 5.0 --t_total 96
"""

import matplotlib
matplotlib.use('Agg')  # must be before any other matplotlib imports

import argparse
import json
import os
import sys
from new_dem_0 import (
    Params, run, plot_granules, plot_fields,
    plot_timeseries, plot_composite, print_stiffness_info
)
import matplotlib.pyplot as plt


def load_trial_json(path: str) -> dict:
    """
    Load a Trial JSON and return a dict of Params field overrides.

    Supports two formats:
      - **Flat (V1.4+)**: Keys are Params field names directly.
        Detected by `"_format": "flat"` or any top-level Params key.
      - **Legacy (V1.3)**: Nested sections (domain, mechanics, shape, etc.).
        Mapped to Params fields via explicit translation.
    """
    with open(path) as f:
        t = json.load(f)

    # ── Detect format ──
    params_fields = {f for f in vars(Params()) if not f.startswith('_')}
    is_flat = (t.get("_format") == "flat" or
               any(k in params_fields for k in t if not k.startswith('_')))

    if is_flat:
        return _load_flat(t, params_fields)
    else:
        return _load_legacy(t)


def _load_flat(t: dict, params_fields: set) -> dict:
    """Load flat-format Trial JSON: keys map directly to Params fields."""
    overrides = {}
    for k, v in t.items():
        if k.startswith('_'):
            continue  # skip metadata keys (_name, _description, _format)
        if k in params_fields:
            overrides[k] = v
        else:
            print(f"  Warning: Trial JSON key '{k}' is not a Params field (ignored)")
    return overrides


def _load_legacy(t: dict) -> dict:
    """Load legacy nested Trial JSON format (V1.3 and earlier)."""
    overrides = {}

    # Domain
    if "domain" in t:
        d = t["domain"]
        if "side_length_um" in d:
            overrides["Lx"] = d["side_length_um"]
            overrides["Ly"] = d["side_length_um"]

    # Composition — convert functional_fraction to area fraction targets
    if "granule_ratio" in t:
        ff = t["granule_ratio"].get("functional_fraction", 0.5)
        packing = t.get("domain", {}).get("target_packing_fraction", 0.55)
        overrides["phi_f_target"] = ff * packing
        overrides["phi_i_target"] = (1.0 - ff) * packing

    # Functional granules
    if "functional_granules" in t:
        fg = t["functional_granules"]
        if "radius_mean_um" in fg:
            overrides["R_func_mean"] = fg["radius_mean_um"]
        if "radius_std_um" in fg:
            overrides["R_func_std"] = fg["radius_std_um"]

    # Inert granules
    if "inert_granules" in t:
        ig = t["inert_granules"]
        if "radius_mean_um" in ig:
            overrides["R_inert_mean"] = ig["radius_mean_um"]
        if "radius_std_um" in ig:
            overrides["R_inert_std"] = ig["radius_std_um"]

    # Cell properties
    if "cell_properties" in t:
        cp = t["cell_properties"]
        if "diameter_um" in cp:
            overrides["cell_diameter"] = cp["diameter_um"]
        if "attachment_area_fraction" in cp:
            overrides["cell_coverage"] = cp["attachment_area_fraction"]
        if "force_per_cell_nN" in cp:
            overrides["F_max_per_cell"] = cp["force_per_cell_nN"]
        if "max_bridge_gap_um" in cp:
            overrides["cell_sense_distance"] = cp["max_bridge_gap_um"]

    # Mechanics
    if "mechanics" in t:
        m = t["mechanics"]
        if "repulsion_stiffness" in m:
            overrides["E_modulus"] = m["repulsion_stiffness"]
        if "damping" in m:
            overrides["drag_scale"] = m["damping"]

    # Time
    if "time" in t:
        tm = t["time"]
        if "total_hours" in tm:
            overrides["t_total"] = tm["total_hours"]
        if "save_interval_hours" in tm:
            overrides["save_every_h"] = tm["save_interval_hours"]
        if "dt_max_hours" in tm:
            overrides["dt"] = tm["dt_max_hours"]

    # Granule shape (V1.3)
    if "shape" in t:
        s = t["shape"]
        if "enabled" in s:
            overrides["shape_enabled"] = bool(s["enabled"])
        for key in ("aspect_ratio_func_mean", "aspect_ratio_func_std",
                     "aspect_ratio_inert_mean", "aspect_ratio_inert_std",
                     "blockiness_func_mean", "blockiness_func_std",
                     "blockiness_inert_mean", "blockiness_inert_std",
                     "drag_scale_rot", "omega_max",
                     "aspect_ratio_c_func_mean", "aspect_ratio_c_func_std",
                     "aspect_ratio_c_inert_mean", "aspect_ratio_c_inert_std",
                     "blockiness_n2_func_mean", "blockiness_n2_func_std",
                     "blockiness_n2_inert_mean", "blockiness_n2_inert_std"):
            if key in s:
                overrides[key] = s[key]

    # Simulation mode (V1.4)
    if "mode" in t:
        overrides["mode"] = t["mode"]

    # 3D domain depth (V1.4)
    if "domain" in t:
        d = t["domain"]
        if "Lz_um" in d:
            overrides["Lz"] = d["Lz_um"]

    # 3D grid resolution (V1.4)
    if "output" in t:
        o = t["output"]
        if "Ngrid_3d" in o:
            overrides["Ngrid_3d"] = o["Ngrid_3d"]

    return overrides


def main():
    # Parse optional Params overrides from command line
    p = Params()
    parser = argparse.ArgumentParser(description="GELLS-DEM HPC runner")
    parser.add_argument("--trial", type=str, default=None,
                        help="Path to a Trial JSON config file")
    for field_name, field_val in vars(p).items():
        if isinstance(field_val, bool):
            # bool is a subclass of int, so check it first
            parser.add_argument(
                f"--{field_name}",
                type=lambda s: s.lower() in ('true', '1', 'yes'),
                default=None,
                metavar='BOOL',
            )
        elif isinstance(field_val, (int, float)):
            parser.add_argument(
                f"--{field_name}",
                type=type(field_val),
                default=None,
            )
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output figures")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Apply trial JSON first, then CLI overrides on top
    if args.trial:
        trial_overrides = load_trial_json(args.trial)
        for k, v in trial_overrides.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))

    # CLI overrides take precedence
    for field_name in vars(p):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(p, field_name, val)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Print config
    trial_label = f" (trial: {os.path.basename(args.trial)})" if args.trial else ""
    print("=" * 65)
    print(f"  GELLS-DEM: Headless HPC Run{trial_label}")
    print("=" * 65)
    print(f"  Output dir: {os.path.abspath(out_dir)}")
    mode = getattr(p, 'mode', '2D')
    if mode == '3D' or mode == '2D-slice':
        print(f"  Mode: {mode}")
        print(f"  Domain: {p.Lx:.0f} x {p.Ly:.0f} x {p.Lz:.0f} um")
    else:
        print(f"  Mode: 2D")
        print(f"  Domain: {p.Lx:.0f} x {p.Ly:.0f} um")
    print(f"  E_modulus={p.E_modulus} kPa, t_total={p.t_total} h, dt={p.dt} h")
    if hasattr(p, 'shape_enabled') and p.shape_enabled:
        print(f"  Shape: ON (AR_func={p.aspect_ratio_func_mean}±{p.aspect_ratio_func_std}, "
              f"block_func={p.blockiness_func_mean}±{p.blockiness_func_std})")
    print_stiffness_info(p)

    # Run simulation
    hist, snaps, p, gs = run(p, seed=args.seed)

    # Save figures
    fig1 = plot_granules(snaps, hist, p)
    fig1.savefig(os.path.join(out_dir, "granules.png"), dpi=150, bbox_inches='tight')

    fig2 = plot_fields(snaps, hist, p)
    fig2.savefig(os.path.join(out_dir, "fields.png"), dpi=150, bbox_inches='tight')

    fig3 = plot_timeseries(hist, p)
    fig3.savefig(os.path.join(out_dir, "timeseries.png"), dpi=150, bbox_inches='tight')

    fig4 = plot_composite(snaps, hist, p)
    fig4.savefig(os.path.join(out_dir, "composite.png"), dpi=150, bbox_inches='tight')

    # V1.4 visualization scripts
    import viz_compaction, viz_percolation, viz_movies, viz_phases
    viz_compaction.run_all(hist, snaps=snaps, outdir=out_dir)
    viz_percolation.run_all(hist, outdir=out_dir)
    viz_movies.run_all(snaps, hist, p, outdir=out_dir)
    viz_phases.run_all(hist, snaps=snaps, p=p, outdir=out_dir)

    plt.close('all')

    # Print summary
    h0, hf = hist[0], hist[-1]
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Functional: {h0['func_nc']} -> {hf['func_nc']} clusters")
    print(f"    Largest frac: {h0['func_lf']:.1%} -> {hf['func_lf']:.1%}")
    ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
          'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
    print(f"    Topology: {ft}")
    print(f"\n  Figures saved to {os.path.abspath(out_dir)}/")


if __name__ == '__main__':
    main()
