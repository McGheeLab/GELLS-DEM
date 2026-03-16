"""
Tissue Architecture Target Vectors for Native Organ Systems
============================================================
Defines reference descriptor vectors for native tissue architectures,
using the same descriptor keys as tissue_descriptors.py.

Each organ target includes mean values, standard deviations (natural
biological variability), a brief description, and key literature references.

IMPORTANT: These values are representative literature estimates. They should
be validated against the user's own histological / micro-CT measurements for
any quantitative design-of-experiments or optimization study. Organ
microarchitecture varies significantly with species, age, anatomical site,
disease state, and measurement methodology.

Units
-----
- Lengths: micrometres (um)
- Densities: per um or per um^3 as noted
- Permeability: um^2
- Dimensionless: BV/TV, porosity, SMI, DA, FA, tortuosity
"""

import numpy as np

# ======================================================================
# Descriptor key ordering (must match tissue_descriptors.py)
# ======================================================================

DESCRIPTOR_KEYS = [
    'bv_tv',                # bone volume / total volume (tissue fraction)
    'surface_density',      # tissue surface area per unit volume (um^-1)
    'specific_surface',     # surface area per tissue volume (um^-1)
    'tb_th',                # trabecular thickness / wall thickness (um)
    'tb_sp',                # trabecular separation / pore spacing (um)
    'tb_n',                 # trabecular number (1/um)
    'euler_characteristic', # Euler number (topology)
    'connectivity_density', # connections per unit volume (um^-3)
    'smi',                  # structure model index (0=plate, 3=rod, 4=sphere)
    'correlation_length',   # spatial autocorrelation length (um)
    'mean_chord_tissue',    # mean intercept length in tissue phase (um)
    'mean_chord_pore',      # mean intercept length in pore phase (um)
    'da',                   # degree of anisotropy (MIL eigenvalue ratio)
    'fa',                   # fractional anisotropy (0=isotropic, 1=anisotropic)
    'tortuosity',           # path length / straight-line distance
    'mean_pore_radius',     # average pore radius (um)
    'porosity',             # void fraction (1 - BV/TV)
    'permeability_KC',      # Kozeny-Carman permeability (um^2)
]


# ======================================================================
# Organ target definitions
# ======================================================================

ORGAN_TARGETS = {

    # ------------------------------------------------------------------
    # 1. Trabecular Bone
    # ------------------------------------------------------------------
    'trabecular_bone': {
        'mean': {
            'bv_tv':                0.25,
            'surface_density':      0.012,       # 12 mm^-1 = 0.012 um^-1
            'specific_surface':     0.048,       # surface_density / bv_tv
            'tb_th':                150.0,       # um
            'tb_sp':                600.0,       # um
            'tb_n':                 1.3e-3,      # 1/(Tb.Th + Tb.Sp) um^-1
            'euler_characteristic': -50.0,       # highly connected (negative)
            'connectivity_density': 5.0e-9,      # ~5 per mm^3 = 5e-9 per um^3
            'smi':                  1.5,         # mixed plates and rods
            'correlation_length':   300.0,       # um
            'mean_chord_tissue':    150.0,       # um, approx Tb.Th
            'mean_chord_pore':      450.0,       # um
            'da':                   2.0,         # moderately anisotropic
            'fa':                   0.4,
            'tortuosity':           1.5,
            'mean_pore_radius':     250.0,       # um
            'porosity':             0.75,
            'permeability_KC':      3.0e4,       # um^2 (order of magnitude)
        },
        'std': {
            'bv_tv':                0.08,        # CV ~32%
            'surface_density':      0.003,
            'specific_surface':     0.012,
            'tb_th':                40.0,
            'tb_sp':                150.0,
            'tb_n':                 3.0e-4,
            'euler_characteristic': 30.0,
            'connectivity_density': 2.0e-9,
            'smi':                  0.5,
            'correlation_length':   80.0,
            'mean_chord_tissue':    40.0,
            'mean_chord_pore':      120.0,
            'da':                   0.5,
            'fa':                   0.12,
            'tortuosity':           0.2,
            'mean_pore_radius':     70.0,
            'porosity':             0.08,
            'permeability_KC':      1.5e4,
        },
        'description': (
            'Cancellous bone: mixed plate-and-rod trabecular network. '
            'BV/TV ranges from ~5% (osteoporotic) to ~50% (subchondral). '
            'Representative value 25% for healthy iliac crest / vertebral body. '
            'Moderate anisotropy aligned with principal loading direction.'
        ),
        'references': [
            'Hildebrand & Ruegsegger 1999 (JBMR)',
            'Odgaard 1997 (Bone)',
            'Parfitt et al. 1987 (JBMR)',
        ],
    },

    # ------------------------------------------------------------------
    # 2. Lung Alveoli
    # ------------------------------------------------------------------
    'lung_alveoli': {
        'mean': {
            'bv_tv':                0.12,
            'surface_density':      0.050,       # very high surface area
            'specific_surface':     0.417,       # 0.050 / 0.12
            'tb_th':                10.0,        # thin alveolar walls (um)
            'tb_sp':                250.0,       # alveolar diameter (um)
            'tb_n':                 3.85e-3,     # 1/(10+250)
            'euler_characteristic': -500.0,      # extremely connected
            'connectivity_density': 1.0e-7,      # ~100 per mm^3
            'smi':                  3.0,         # thin walls / rod-like
            'correlation_length':   130.0,       # um
            'mean_chord_tissue':    10.0,        # um (very thin walls)
            'mean_chord_pore':      200.0,       # um
            'da':                   1.1,         # nearly isotropic
            'fa':                   0.08,
            'tortuosity':           1.1,         # open architecture
            'mean_pore_radius':     125.0,       # um
            'porosity':             0.88,
            'permeability_KC':      1.0e5,       # very permeable
        },
        'std': {
            'bv_tv':                0.03,
            'surface_density':      0.012,
            'specific_surface':     0.10,
            'tb_th':                3.0,
            'tb_sp':                60.0,
            'tb_n':                 1.0e-3,
            'euler_characteristic': 200.0,
            'connectivity_density': 5.0e-8,
            'smi':                  0.5,
            'correlation_length':   30.0,
            'mean_chord_tissue':    3.0,
            'mean_chord_pore':      50.0,
            'da':                   0.05,
            'fa':                   0.04,
            'tortuosity':           0.05,
            'mean_pore_radius':     30.0,
            'porosity':             0.03,
            'permeability_KC':      5.0e4,
        },
        'description': (
            'Pulmonary alveolar tissue: thin-walled, highly open architecture '
            'optimized for gas exchange. Extremely high surface-to-volume ratio. '
            'Nearly isotropic at the acinar level. Very high connectivity '
            '(Euler characteristic strongly negative). Low tortuosity.'
        ),
        'references': [
            'Ochs et al. 2004 (Am J Respir Crit Care Med)',
            'Hsia et al. 2010 (Compr Physiol)',
            'Weibel 2009 (Swiss Med Wkly)',
        ],
    },

    # ------------------------------------------------------------------
    # 3. Liver
    # ------------------------------------------------------------------
    'liver': {
        'mean': {
            'bv_tv':                0.85,
            'surface_density':      0.030,       # um^-1
            'specific_surface':     0.035,       # 0.030 / 0.85
            'tb_th':                20.0,        # hepatocyte cord thickness (um)
            'tb_sp':                8.0,         # sinusoidal diameter (um)
            'tb_n':                 3.57e-2,     # 1/(20+8)
            'euler_characteristic': -200.0,      # well-connected lobular network
            'connectivity_density': 5.0e-8,
            'smi':                  0.0,         # plate-like hepatocyte cords
            'correlation_length':   500.0,       # lobular repeat distance (um)
            'mean_chord_tissue':    20.0,        # um
            'mean_chord_pore':      8.0,         # um (sinusoidal lumen)
            'da':                   1.2,         # mild anisotropy (radial lobules)
            'fa':                   0.12,
            'tortuosity':           1.3,
            'mean_pore_radius':     4.0,         # um (sinusoid radius)
            'porosity':             0.15,
            'permeability_KC':      5.0,         # very low (dense + small pores)
        },
        'std': {
            'bv_tv':                0.05,
            'surface_density':      0.008,
            'specific_surface':     0.010,
            'tb_th':                5.0,
            'tb_sp':                2.0,
            'tb_n':                 8.0e-3,
            'euler_characteristic': 80.0,
            'connectivity_density': 2.0e-8,
            'smi':                  0.3,
            'correlation_length':   100.0,
            'mean_chord_tissue':    5.0,
            'mean_chord_pore':      2.0,
            'da':                   0.1,
            'fa':                   0.06,
            'tortuosity':           0.1,
            'mean_pore_radius':     1.0,
            'porosity':             0.05,
            'permeability_KC':      2.0,
        },
        'description': (
            'Hepatic parenchyma: dense tissue with plate-like hepatocyte cords '
            'separated by narrow sinusoidal channels (~8 um diameter). '
            'Lobular architecture produces an oscillatory spatial correlation '
            'function with period ~500 um. Low porosity, very low permeability.'
        ),
        'references': [
            'Debbaut et al. 2014 (J Anat)',
            'Teutsch 2005 (Anat Rec)',
        ],
    },

    # ------------------------------------------------------------------
    # 4. Kidney Cortex
    # ------------------------------------------------------------------
    'kidney_cortex': {
        'mean': {
            'bv_tv':                0.80,
            'surface_density':      0.025,
            'specific_surface':     0.031,       # 0.025 / 0.80
            'tb_th':                45.0,        # tubule outer diameter (um)
            'tb_sp':                30.0,        # interstitial space (um)
            'tb_n':                 1.33e-2,     # 1/(45+30)
            'euler_characteristic': -100.0,
            'connectivity_density': 3.0e-8,
            'smi':                  2.0,         # mixed tubular + glomerular
            'correlation_length':   200.0,       # um (nephron repeat)
            'mean_chord_tissue':    45.0,        # um
            'mean_chord_pore':      30.0,        # um
            'da':                   3.0,         # highly anisotropic (radial)
            'fa':                   0.6,
            'tortuosity':           2.0,         # convoluted tubules
            'mean_pore_radius':     15.0,        # um
            'porosity':             0.20,
            'permeability_KC':      50.0,        # um^2
        },
        'std': {
            'bv_tv':                0.05,
            'surface_density':      0.006,
            'specific_surface':     0.008,
            'tb_th':                10.0,
            'tb_sp':                8.0,
            'tb_n':                 3.0e-3,
            'euler_characteristic': 40.0,
            'connectivity_density': 1.0e-8,
            'smi':                  0.5,
            'correlation_length':   50.0,
            'mean_chord_tissue':    10.0,
            'mean_chord_pore':      8.0,
            'da':                   0.6,
            'fa':                   0.12,
            'tortuosity':           0.3,
            'mean_pore_radius':     4.0,
            'porosity':             0.05,
            'permeability_KC':      20.0,
        },
        'description': (
            'Renal cortex: densely packed convoluted tubules (~45 um diameter) '
            'with narrow interstitial spaces. Strongly anisotropic due to radial '
            'organization of collecting ducts and vasa recta. High tortuosity '
            'from tubular convolution.'
        ),
        'references': [
            'Layton 2014 (Math Biosci)',
            'Kriz & Kaissling 2008 (Seldin & Giebisch)',
        ],
    },

    # ------------------------------------------------------------------
    # 5. Cardiac Muscle
    # ------------------------------------------------------------------
    'cardiac_muscle': {
        'mean': {
            'bv_tv':                0.88,
            'surface_density':      0.035,
            'specific_surface':     0.040,       # 0.035 / 0.88
            'tb_th':                15.0,        # myocyte diameter (um)
            'tb_sp':                22.0,        # capillary spacing (um)
            'tb_n':                 2.70e-2,     # 1/(15+22)
            'euler_characteristic': -80.0,
            'connectivity_density': 2.0e-8,
            'smi':                  3.0,         # rod/fiber-like myocytes
            'correlation_length':   100.0,       # um (fiber bundle spacing)
            'mean_chord_tissue':    15.0,        # um
            'mean_chord_pore':      22.0,        # um
            'da':                   4.0,         # very anisotropic (aligned fibers)
            'fa':                   0.7,
            'tortuosity':           1.4,
            'mean_pore_radius':     3.0,         # um (capillary radius)
            'porosity':             0.12,
            'permeability_KC':      2.0,         # very low (dense + tiny capillaries)
        },
        'std': {
            'bv_tv':                0.04,
            'surface_density':      0.008,
            'specific_surface':     0.010,
            'tb_th':                3.0,
            'tb_sp':                5.0,
            'tb_n':                 6.0e-3,
            'euler_characteristic': 30.0,
            'connectivity_density': 8.0e-9,
            'smi':                  0.4,
            'correlation_length':   25.0,
            'mean_chord_tissue':    3.0,
            'mean_chord_pore':      5.0,
            'da':                   0.8,
            'fa':                   0.12,
            'tortuosity':           0.15,
            'mean_pore_radius':     0.8,
            'porosity':             0.04,
            'permeability_KC':      1.0,
        },
        'description': (
            'Myocardium: aligned cardiomyocyte fibers (~15 um diameter) with '
            'dense capillary network (~22 um spacing). Highly anisotropic fiber '
            'architecture with transmural rotation of fiber angle. Rod-like SMI. '
            'Very low porosity and permeability.'
        ),
        'references': [
            'LeGrice et al. 2001 (Am J Physiol Heart Circ Physiol)',
            'Sands et al. 2005 (Microsc Res Tech)',
        ],
    },

    # ------------------------------------------------------------------
    # 6. Pancreatic Islet
    # ------------------------------------------------------------------
    'pancreatic_islet': {
        'mean': {
            'bv_tv':                0.65,
            'surface_density':      0.008,
            'specific_surface':     0.012,       # 0.008 / 0.65
            'tb_th':                200.0,       # islet diameter (um)
            'tb_sp':                100.0,       # exocrine spacing (um)
            'tb_n':                 3.33e-3,     # 1/(200+100)
            'euler_characteristic': 20.0,        # isolated islands (positive chi)
            'connectivity_density': 1.0e-9,      # low connectivity (discrete clusters)
            'smi':                  4.0,         # spherical clusters
            'correlation_length':   250.0,       # um (inter-islet distance)
            'mean_chord_tissue':    200.0,       # um (islet scale)
            'mean_chord_pore':      100.0,       # um
            'da':                   1.1,         # isotropic (random islet distribution)
            'fa':                   0.08,
            'tortuosity':           1.6,
            'mean_pore_radius':     50.0,        # um
            'porosity':             0.35,
            'permeability_KC':      500.0,       # um^2
        },
        'std': {
            'bv_tv':                0.10,
            'surface_density':      0.002,
            'specific_surface':     0.003,
            'tb_th':                60.0,
            'tb_sp':                30.0,
            'tb_n':                 1.0e-3,
            'euler_characteristic': 10.0,
            'connectivity_density': 5.0e-10,
            'smi':                  0.5,
            'correlation_length':   60.0,
            'mean_chord_tissue':    60.0,
            'mean_chord_pore':      30.0,
            'da':                   0.05,
            'fa':                   0.04,
            'tortuosity':           0.2,
            'mean_pore_radius':     15.0,
            'porosity':             0.10,
            'permeability_KC':      200.0,
        },
        'description': (
            'Pancreatic islets of Langerhans: discrete spheroidal cell clusters '
            '(~100-300 um diameter) embedded in exocrine parenchyma. '
            'Island-in-a-sea topology yields positive Euler characteristic and '
            'high SMI (~4). Isotropic spatial distribution. Dense internal '
            'vasculature within each islet.'
        ),
        'references': [
            'Cabrera et al. 2006 (PNAS)',
            'Brissova et al. 2005 (J Histochem Cytochem)',
        ],
    },

    # ------------------------------------------------------------------
    # 7. Intestinal Mucosa
    # ------------------------------------------------------------------
    'intestinal_mucosa': {
        'mean': {
            'bv_tv':                0.50,
            'surface_density':      0.015,
            'specific_surface':     0.030,       # 0.015 / 0.50
            'tb_th':                100.0,       # villus diameter (um)
            'tb_sp':                80.0,        # crypt spacing (um)
            'tb_n':                 5.56e-3,     # 1/(100+80)
            'euler_characteristic': -30.0,       # connected crypt network
            'connectivity_density': 1.0e-8,
            'smi':                  3.0,         # finger-like villi (rod-like)
            'correlation_length':   150.0,       # um (villus repeat distance)
            'mean_chord_tissue':    100.0,       # um
            'mean_chord_pore':      80.0,        # um
            'da':                   2.5,         # anisotropic (apical-basal axis)
            'fa':                   0.5,
            'tortuosity':           1.7,
            'mean_pore_radius':     40.0,        # um
            'porosity':             0.50,
            'permeability_KC':      800.0,       # um^2
        },
        'std': {
            'bv_tv':                0.08,
            'surface_density':      0.004,
            'specific_surface':     0.008,
            'tb_th':                25.0,
            'tb_sp':                20.0,
            'tb_n':                 1.5e-3,
            'euler_characteristic': 15.0,
            'connectivity_density': 4.0e-9,
            'smi':                  0.5,
            'correlation_length':   35.0,
            'mean_chord_tissue':    25.0,
            'mean_chord_pore':      20.0,
            'da':                   0.5,
            'fa':                   0.12,
            'tortuosity':           0.2,
            'mean_pore_radius':     10.0,
            'porosity':             0.08,
            'permeability_KC':      300.0,
        },
        'description': (
            'Small intestinal mucosa: finger-like villi (~100 um diameter, '
            '~500 um tall) projecting from a connected crypt base. Strongly '
            'anisotropic along the apical-basal axis. Rod-like SMI from villus '
            'projections. Moderate porosity with connected luminal space.'
        ),
        'references': [
            'Helander & Fandriks 2014 (Scand J Gastroenterol)',
            'Marsh 1992 (Gut)',
        ],
    },
}


# ======================================================================
# Public API
# ======================================================================

def get_target(organ_name):
    """Return the target dict for the named organ.

    Parameters
    ----------
    organ_name : str
        Key in ORGAN_TARGETS (e.g. 'trabecular_bone', 'lung_alveoli').

    Returns
    -------
    dict
        Target dict with keys 'mean', 'std', 'description', 'references'.

    Raises
    ------
    KeyError
        If the organ name is not found.
    """
    if organ_name not in ORGAN_TARGETS:
        available = ', '.join(sorted(ORGAN_TARGETS.keys()))
        raise KeyError(
            f"Unknown organ '{organ_name}'. Available: {available}")
    return ORGAN_TARGETS[organ_name]


def list_organs():
    """Return sorted list of available organ names.

    Returns
    -------
    list of str
    """
    return sorted(ORGAN_TARGETS.keys())


def get_all_targets():
    """Return the full ORGAN_TARGETS dictionary.

    Returns
    -------
    dict
        organ_name -> target dict
    """
    return ORGAN_TARGETS


def descriptor_keys():
    """Return the ordered list of descriptor keys used in comparisons.

    Returns
    -------
    list of str
    """
    return list(DESCRIPTOR_KEYS)


def target_vector(organ_name, keys=None):
    """Return (mean_array, std_array) as numpy arrays for the given organ.

    Parameters
    ----------
    organ_name : str
        Key in ORGAN_TARGETS.
    keys : list of str or None
        Descriptor keys to include. Defaults to DESCRIPTOR_KEYS.

    Returns
    -------
    mean_arr : np.ndarray
        Mean descriptor values.
    std_arr : np.ndarray
        Standard deviation values.
    """
    target = get_target(organ_name)
    if keys is None:
        keys = DESCRIPTOR_KEYS
    mean_vals = []
    std_vals = []
    for k in keys:
        mean_vals.append(target['mean'].get(k, np.nan))
        std_vals.append(target['std'].get(k, np.nan))
    return np.array(mean_vals), np.array(std_vals)


def _normalize_across_organs(keys=None):
    """Compute min/max across all organs for normalization to [0, 1].

    Parameters
    ----------
    keys : list of str or None
        Descriptor keys to normalize. Defaults to DESCRIPTOR_KEYS.

    Returns
    -------
    mins : np.ndarray
    maxs : np.ndarray
    """
    if keys is None:
        keys = DESCRIPTOR_KEYS
    all_means = []
    for name in ORGAN_TARGETS:
        mean_arr, _ = target_vector(name, keys)
        all_means.append(mean_arr)
    all_means = np.array(all_means)
    mins = np.nanmin(all_means, axis=0)
    maxs = np.nanmax(all_means, axis=0)
    return mins, maxs


def plot_organ_profiles(organs=None, outdir=None):
    """Radar/spider chart comparing organ architecture profiles.

    Each spoke is a descriptor, normalized to [0, 1] across all organs.
    Only descriptors present in all selected organs are plotted.

    Parameters
    ----------
    organs : list of str or None
        Organ names to plot. Defaults to all organs.
    outdir : str or None
        Output directory for saved figure. If None, uses current directory.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if organs is None:
        organs = list_organs()
    if outdir is None:
        outdir = '.'

    keys = DESCRIPTOR_KEYS
    mins, maxs = _normalize_across_organs(keys)
    span = maxs - mins
    span[span == 0] = 1.0  # avoid division by zero

    # Abbreviate labels for radar chart
    short_labels = [k.replace('_', '\n') for k in keys]

    n_keys = len(keys)
    angles = np.linspace(0, 2 * np.pi, n_keys, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(organs)))
    for organ, color in zip(organs, colors):
        mean_arr, _ = target_vector(organ, keys)
        normed = (mean_arr - mins) / span
        normed = np.nan_to_num(normed, nan=0.0)
        values = normed.tolist() + [normed[0]]
        ax.plot(angles, values, 'o-', label=organ.replace('_', ' ').title(),
                color=color, linewidth=1.5, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title('Organ Architecture Profiles (normalized)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=8)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'organ_profiles_radar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_organ_table(outdir=None):
    """Generate a formatted table of all organs x descriptors saved as PNG.

    Parameters
    ----------
    outdir : str or None
        Output directory. If None, uses current directory.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if outdir is None:
        outdir = '.'

    organs = list_organs()
    keys = DESCRIPTOR_KEYS

    # Build table data
    col_labels = [k.replace('_', ' ') for k in keys]
    row_labels = [o.replace('_', ' ').title() for o in organs]
    cell_text = []
    for organ in organs:
        target = get_target(organ)
        row = []
        for k in keys:
            val = target['mean'].get(k, float('nan'))
            if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
                row.append(f'{val:.2e}')
            else:
                row.append(f'{val:.3g}')
        cell_text.append(row)

    fig, ax = plt.subplots(
        figsize=(max(18, len(keys) * 1.0), len(organs) * 0.6 + 1.5))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(keys)):
        table[0, j].set_facecolor('#d4e6f1')
        table[0, j].set_text_props(fontweight='bold', fontsize=6)
    for i in range(len(organs)):
        table[i + 1, -1].set_text_props(fontweight='bold')

    ax.set_title('Native Organ Architecture Descriptors',
                 fontsize=12, fontweight='bold', pad=10)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'organ_descriptors_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Module-level convenience
# ======================================================================

import os

if __name__ == '__main__':
    print("Organ Targets Module")
    print("=" * 50)
    print(f"Available organs: {list_organs()}")
    print(f"Descriptor keys ({len(DESCRIPTOR_KEYS)}): {DESCRIPTOR_KEYS}")
    print()
    for name in list_organs():
        target = get_target(name)
        print(f"--- {name.replace('_', ' ').title()} ---")
        print(f"  {target['description']}")
        print(f"  References: {target['references']}")
        print()
    # Generate plots
    plot_organ_profiles()
    plot_organ_table()
