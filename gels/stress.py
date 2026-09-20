"""
gels/stress.py -- the Love-Weber (virial) stress tensor.  V3.7
================================================================

Every mechanical number the engine reported before V3.7 was a **force**:
``F_mean``, ``F_max``, ``F_func_mean``. A force is extensive -- it grows with
the number of granules and with their size -- so it cannot be compared with a
measurement, and two runs at different N cannot be compared with each other.
A stress can. This module turns the contact list the force evaluation already
built into the Cauchy stress of the assembly, in kPa.

The decomposition is the point
------------------------------
``P_contact`` is what the granular skeleton carries; ``P_active`` is what the
cells are generating. ``stress_active_frac`` is the fraction of the bed's
stress that is cell-derived -- the question this model exists to answer, and
one that ``F_mean`` (which sums the two indiscriminately) cannot address.

Sign convention: COMPRESSION POSITIVE
-------------------------------------
This is the granular-physics convention (and robotsim's ``P_vir``), not the
solid-mechanics tension-positive one. Stated once here and pinned by
``tests/test_stress.py``:

    sigma_ab = (1/V) sum_int  f_a . l_b ,   f = the force on body A,
                                            l = x_A - x_B

Summed **once per interaction**, with the *same* body playing A in both
factors. Get that pairing right and the signs follow from the physics with no
further convention:

* a repulsive contact has ``f`` along ``+n`` on ``j`` and ``l = x_j - x_i``
  also along ``+n``, so ``tr(sigma) > 0`` -- compression;
* a **contracting bridge** pulls its host ``gi`` toward its target ``gj``, so
  ``f`` is along ``+n`` on ``gi`` while ``l = x_gi - x_gj`` is along ``-n``,
  and ``tr(sigma) < 0`` -- tension.

That second sign is not decorative. A cell bridge puts the bed in tension; it
RELIEVES the compression the contacts carry. ``analysis/coarse_grain.py`` used
``l = x_gj - x_gi`` for the active term while using ``l = x_j - x_i`` for the
contact term, so its active stress came out with the wrong sign and cell
contraction read as *adding* to the compression (V3.7 bug fix).

Pair contacts only, deliberately
--------------------------------
Wall contacts are applied inline in the gathers and never enter the contact
list (the same fact V3.6 had to work around for the semi-implicit stiffness).
They are absent here too, and correctly so: for a self-equilibrated assembly
the internal (pair) virial *is* the Cauchy stress, and the wall reactions are
the external tractions that stress balances. Adding them would double-count
the boundary condition.

Trace
-----
``P = tr(sigma) / D`` with **D = 2 in two dimensions**. ``coarse_grain``
divided by 3 unconditionally, so every 2D pressure it ever reported was 2/3 of
what it should have been.

Leaf module: numpy only at import time. The engine helpers it needs
(``domain_volume``, ``bed_surface``, ``boundary_geometry``) are imported
lazily inside the functions, exactly as ``gels/laguerre.py`` does, so the
engine, the pipeline and the tests all run this one implementation.
"""

import numpy as np

__all__ = ['virial_stress', 'stress_volume', 'record_stress', 'stress_metrics',
           'STRESS_KEYS']

# Every key this module can emit. `compute_metrics` asserts the two twins agree
# on the key SET, so the dict must be the same shape on every call -- including
# the N == 0 and no-contacts cases.
STRESS_KEYS = (
    'P_vir', 'P_contact', 'P_active', 'stress_von_mises', 'stress_dev',
    'stress_active_frac', 'stress_volume', 'stress_n_contacts',
    'stress_n_bridges',
    # V3.7 stress-force-fabric (Rothenburg & Bathurst 1989) and the
    # small-strain validity diagnostic that says whether to believe any of it.
    'stress_q_over_p', 'fabric_a_c', 'fabric_a_fn', 'fabric_a_n',
    'sff_q_over_p', 'sff_closure', 'stress_patch_p95', 'stress_patch_max',
)

#: |deviator| -> Rothenburg-Bathurst anisotropy coefficient. 2D: 4/sqrt(2).
#: 3D: 15/(2 sqrt(3/2)). Both follow from the second-order spherical-harmonic
#: expansion; both are CHECKED against the directly computed q/p in
#: `tests/test_stress.py` rather than taken on trust.
_ANIS_K = {2: 4.0 / np.sqrt(2.0), 3: 15.0 / (2.0 * np.sqrt(1.5))}


def _columns(contacts, names):
    """Pull named columns off a ContactSoA or a reference list-of-dicts.

    Missing names come back as zeros: the LS-DEM branch appends contact records
    without ``E_star``/``W``, so a caller cannot assume every key is present.
    """
    n = len(contacts)
    if n == 0:
        return {k: np.zeros(0) for k in names}
    col = getattr(contacts, 'column', None)
    out = {}
    for k in names:
        if col is not None:
            try:
                out[k] = np.asarray(col(k), dtype=float)
                continue
            except (KeyError, TypeError):
                pass
            out[k] = np.zeros(n)
        else:
            out[k] = np.array([c.get(k, 0.0) for c in contacts], dtype=float)
    return out


def _min_image(d, L):
    """Vectorised minimum-image displacement along one axis."""
    if L <= 0:
        return d
    return (d + 0.5 * L) % L - 0.5 * L


def _positions(gs):
    N = gs.N
    if gs.is_3d:
        return np.column_stack((gs.x[:N], gs.y[:N], gs.z[:N]))
    return np.column_stack((gs.x[:N], gs.y[:N], np.zeros(N)))


def _branch(pos, ia, ib, p, periodic, is_3d):
    """``x_A - x_B`` for each interaction, minimum-imaged when periodic.

    Without the minimum image a contact across a periodic face reports a branch
    vector nearly a box long and of the wrong sign, which would dominate the
    sum outright.
    """
    l = pos[ia] - pos[ib]
    if periodic:
        l[:, 0] = _min_image(l[:, 0], float(p.Lx))
        l[:, 1] = _min_image(l[:, 1], float(p.Ly))
        if is_3d:
            l[:, 2] = _min_image(l[:, 2], float(getattr(p, 'Lz', 0.0)))
    return l


def stress_volume(gs, p):
    """The volume the stress is divided by, um^3 (um^2 in 2D).

    A container with a free top is mostly headroom -- ``pmma_well`` is about
    1.4x the bed -- and dividing by the container would dilute the pressure by
    that ratio. So the denominator is the BED envelope whenever the top is
    free, and the container volume otherwise. Reported as ``stress_volume`` so
    the denominator is never hidden, the same discipline
    ``laguerre_valid_frac`` follows.
    """
    from gels.engine import boundary_geometry, domain_volume, bed_surface
    geom = boundary_geometry(p, gs.mode if gs.mode in ('2D', '3D') else None)
    if geom.top_free and gs.N > 0:
        env = float(bed_surface(gs, p).get('bed_envelope_volume', 0.0))
        if env > 0:
            return env
    return float(domain_volume(p))


def virial_stress(gs, p, contacts, volume=None):
    """``(sigma, parts)`` -- the 3x3 Cauchy stress in kPa, compression positive.

    ``contacts`` is the list from the force evaluation at the CURRENT
    configuration: pass the one already in hand, never recompute it. Positions
    and forces must be from the same instant, which is why this is called at
    force time rather than at metrics time -- the step moves the granules
    between the two, and a branch vector taken afterwards would not match the
    force that goes with it.

    Units: ``F_normal`` is nN and the branch vector is um, so the quotient is
    nN/um^2, and 1 nN/um^2 IS 1 kPa. No conversion factor appears here, which
    is worth stating because the engine's ``E_star`` / ``W`` / ``tau_0`` are in
    Pa and every force expression that uses them carries an explicit ``1e-3``.
    Mixing those two conventions is what made viz2's contact energy 1000x too
    large (V3.7).
    """
    N = int(gs.N)
    sig_c = np.zeros((3, 3))
    sig_a = np.zeros((3, 3))
    n_c = 0
    n_b = 0
    V = float(stress_volume(gs, p) if volume is None else volume)
    if N == 0 or V <= 0:
        return np.zeros((3, 3)), {'contact': sig_c, 'active': sig_a,
                                  'volume': V, 'n_contacts': 0, 'n_bridges': 0}

    is_3d = bool(gs.is_3d)
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'
    pos = _positions(gs)

    # ── contact term: f = F_normal * n on j, l = x_j - x_i ──
    if contacts is not None and len(contacts) > 0:
        col = getattr(contacts, 'column', None)
        if col is not None:
            ci = np.asarray(col('i'), dtype=np.int64)
            cj = np.asarray(col('j'), dtype=np.int64)
        else:
            ci = np.array([c['i'] for c in contacts], dtype=np.int64)
            cj = np.array([c['j'] for c in contacts], dtype=np.int64)
        cc = _columns(contacts, ('nx', 'ny', 'nz', 'F_normal'))
        keep = (ci >= 0) & (ci < N) & (cj >= 0) & (cj < N)
        if np.any(keep):
            ci, cj = ci[keep], cj[keep]
            nvec = np.column_stack((cc['nx'][keep], cc['ny'][keep],
                                    cc['nz'][keep] if is_3d else np.zeros(keep.sum())))
            f = cc['F_normal'][keep][:, None] * nvec          # force on j
            l = _branch(pos, cj, ci, p, periodic, is_3d)      # x_j - x_i
            sig_c = f.T @ l
            n_c = int(keep.sum())

    # ── active term: f = the cell's force on its HOST, l = x_host - x_target ──
    off = getattr(gs, 'cell_offset', None)
    if off is not None and int(off[N]) > 0:
        n_cells = int(off[N])
        bt = np.asarray(gs.cell_bridge_target[:n_cells], dtype=np.int64)
        host = np.asarray(gs.cell_granule_id[:n_cells], dtype=np.int64)
        fx = np.asarray(gs.cell_fx[:n_cells], dtype=float)
        fy = np.asarray(gs.cell_fy[:n_cells], dtype=float)
        fz = (np.asarray(gs.cell_fz[:n_cells], dtype=float) if is_3d
              else np.zeros(n_cells))
        sel = ((bt >= 0) & (bt < N) & (host >= 0) & (host < N)
               & ((fx != 0.0) | (fy != 0.0) | (fz != 0.0)))
        if np.any(sel):
            f = np.column_stack((fx[sel], fy[sel], fz[sel]))
            l = _branch(pos, host[sel], bt[sel], p, periodic, is_3d)
            sig_a = f.T @ l
            n_b = int(sel.sum())

    sig_c /= V
    sig_a /= V
    # Symmetrise: the Cauchy stress is symmetric, and the two off-diagonal
    # halves differ only by the contact-point offsets, which are second order.
    sig_c = 0.5 * (sig_c + sig_c.T)
    sig_a = 0.5 * (sig_a + sig_a.T)
    return sig_c + sig_a, {'contact': sig_c, 'active': sig_a, 'volume': V,
                           'n_contacts': n_c, 'n_bridges': n_b}


def _pressure(sig, dim):
    """tr(sigma)/D -- D is 2 in 2D, which is the whole point of passing it."""
    if dim == 2:
        return float((sig[0, 0] + sig[1, 1]) / 2.0)
    return float(np.trace(sig) / 3.0)


def _deviator(sig, dim):
    """The deviator taken in ``dim`` dimensions, not in the 3x3 embedding.

    This has to be dimension-aware for the same reason `_pressure` does, and
    the failure is louder. A 2D stress is stored zero-padded as
    ``diag(P, P, 0)``; subtracting ``tr/3`` from THAT leaves
    ``diag(P/3, P/3, -2P/3)``, whose magnitude is ``P/sqrt(3)``. So a perfectly
    isotropic 2D state reports a deviatoric stress of ``0.577 P`` -- a shear
    that is not there, invented entirely by the padding. Measured on three
    different 2D beds, ``q/p`` came out 0.579, 0.581 and 0.652 before this was
    fixed, the first two being the artefact almost neat.
    """
    d = int(dim)
    blk = np.asarray(sig)[:d, :d]
    return blk - (np.trace(blk) / d) * np.eye(d)


def _J2(sig, dim):
    s = _deviator(sig, dim)
    return 0.5 * float(np.sum(s * s))


def _von_mises(sig, dim):
    """sqrt(3 J2) -- the standard von Mises in 3D, and the same definition
    applied to the in-plane deviator in 2D."""
    return float(np.sqrt(3.0 * _J2(sig, dim)))


def _dev_magnitude(sig, dim):
    """sqrt(J2) -- the equivalent shear stress ``q``."""
    return float(np.sqrt(_J2(sig, dim)))


def record_stress(gs, p, contacts):
    """Compute the stress and stash it on ``gs`` for the metrics twins.

    The same arrangement V3.5 used for the energy audit: computed once, where
    the forces and the positions are consistent, and read back later by
    whichever twin builds the metrics dict.
    """
    sig, parts = virial_stress(gs, p, contacts)
    gs.stress_total = sig
    gs.stress_contact = parts['contact']
    gs.stress_active = parts['active']
    gs.stress_volume = parts['volume']
    gs.stress_n_contacts = parts['n_contacts']
    gs.stress_n_bridges = parts['n_bridges']
    gs.stress_fabric = fabric_stress(gs, p, contacts, sig=sig,
                                     sig_contact=parts['contact'])   # V3.7 SFF
    return sig, parts


def stress_metrics(gs):
    """The ``P_vir`` metric block, called by BOTH metrics twins.

    Every key is always present and always a plain float or int -- they pass
    through ``csv.DictWriter``, whose field names come from the first row, so a
    key set that varied by frame would truncate the history file.
    """
    dim = 3 if getattr(gs, 'is_3d', False) else 2
    # A gs that has never been stepped carries None, not a zero tensor, so this
    # cannot be a getattr default -- the attribute EXISTS and is None.
    Z = np.zeros((3, 3))
    sig = getattr(gs, 'stress_total', None)
    sig = Z if sig is None else sig
    sig_c = getattr(gs, 'stress_contact', None)
    sig_c = Z if sig_c is None else sig_c
    sig_a = getattr(gs, 'stress_active', None)
    sig_a = Z if sig_a is None else sig_a
    P = _pressure(sig, dim)
    Pc = _pressure(sig_c, dim)
    Pa = _pressure(sig_a, dim)
    denom = abs(Pc) + abs(Pa)
    out = {
        'P_vir': P,
        'P_contact': Pc,
        'P_active': Pa,
        'stress_von_mises': _von_mises(sig, dim),
        'stress_dev': _dev_magnitude(sig, dim),
        # bounded in [0, 1] on purpose: a ratio to P_vir would blow up wherever
        # the contact and active parts cancel, which is exactly the interesting
        # regime.
        'stress_active_frac': float(abs(Pa) / denom) if denom > 0 else 0.0,
        'stress_volume': float(getattr(gs, 'stress_volume', 0.0)),
        'stress_n_contacts': int(getattr(gs, 'stress_n_contacts', 0)),
        'stress_n_bridges': int(getattr(gs, 'stress_n_bridges', 0)),
    }
    fab = getattr(gs, 'stress_fabric', None)
    out.update(fab if fab else {k: 0.0 for k in (
        'stress_q_over_p', 'fabric_a_c', 'fabric_a_fn', 'fabric_a_n',
        'sff_q_over_p', 'sff_closure', 'stress_patch_p95', 'stress_patch_max')})
    return out


# ======================================================================
# Reading contact physics back off a snapshot  (V3.7)
# ======================================================================
#
# Three modules -- viz2/energy_stress.py, viz/stress.py and
# analysis/coarse_grain.py -- each rebuilt the contact law from `p.E_modulus`
# because the snapshot did not carry the per-pair values. They now share these
# two functions instead, so there is one contact law downstream of the engine
# rather than three approximations of it.

#: Columns a snapshot must carry for the energy to be the engine's own.
ENERGY_COLUMNS = ('contact_a_contact', 'contact_E_star', 'contact_W',
                  'contact_R_eff', 'contact_overlap')


def has_exact_contact_physics(snap):
    """True when this snapshot was written by V3.7 or later.

    A run made before V3.7 has ten contact columns and no ``E_star``; it must
    still plot, so every consumer falls back -- but says so, because a silent
    fallback to a 750x-wrong number is the thing V3.7 exists to stop.
    """
    return all(k in snap and len(np.asarray(snap[k])) > 0
               for k in ENERGY_COLUMNS)


def snapshot_contact_energy(snap, p):
    """``(energy_per_contact, exact)`` -- JKR contact energy in nN um.

    ``exact`` is False for a pre-V3.7 snapshot, where ``E_star`` is estimated
    from the global ``p.E_modulus`` and the adhesive term is unavailable.

    **Units.** ``E_star`` and ``W`` are in the engine's Pa-based convention and
    every force expression that consumes them carries an explicit ``1e-3`` to
    reach nN/um^2. The fallback below carries it too. Omitting it is exactly
    the factor of 1000 that made the old energy field 750x too large; the
    other 0.75 was a Hertz prefactor of ``(2/5)`` where the integral of
    ``F = (4/3) E* sqrt(R) d^{3/2}`` is ``(8/15)``.
    """
    n = len(np.asarray(snap.get('contact_i', [])))
    if n == 0:
        return np.zeros(0), True

    if has_exact_contact_physics(snap):
        from gels.engine import jkr_contact_energy
        return (np.asarray(jkr_contact_energy(
            np.asarray(snap['contact_a_contact'], dtype=float),
            np.asarray(snap['contact_R_eff'], dtype=float),
            np.asarray(snap['contact_E_star'], dtype=float),
            np.asarray(snap['contact_W'], dtype=float)), dtype=float), True)

    # ── pre-V3.7 fallback: Hertz, global modulus, no adhesion ──
    R = np.asarray(snap.get('contact_R_eff', np.zeros(n)), dtype=float)
    d = np.asarray(snap.get('contact_overlap', np.zeros(n)), dtype=float)
    nu = float(getattr(p, 'poisson_ratio', 0.45))
    E_star = (float(getattr(p, 'E_modulus', 10.0)) * 1e3) / (2.0 * (1.0 - nu * nu))
    d = np.maximum(d, 0.0)
    return (8.0 / 15.0) * E_star * 1e-3 * np.sqrt(np.maximum(R, 0.0)) * d ** 2.5, False


def snapshot_cell_strain_energy(snap, p):
    """Per-cell elastic energy from a snapshot, nN um.  V3.7

    The array form of the engine's `cell_strain_energy` (V3.6): a loaded cell
    is a spring in series with what it grips, ``1/k = 1/k_cell + 1/k_sub``, and
    a BRIDGING cell grips two, so ``U = F^2 / 2k`` with both substrates in the
    series. Returned per cell rather than summed, because the visualization
    stamps it at each cell's own position.

    This replaces ``0.5 * |F| * bridge_length``, which viz2 used before V3.7.
    That expression is the work done dragging a granule the whole length of a
    bridge, not the energy stored in a loaded cell, and it grows with the
    spacing of the granules rather than with how hard the cell is pulling.

    ``k_cell <= 0`` (a Params from before V3.6) falls back to the rigid-cell
    limit, where the substrate holds all of the compliance -- the correct
    ``k_cell -> inf`` limit of the same expression, not a different model.
    """
    fx = np.asarray(snap.get('cell_fx', np.zeros(0)), dtype=float)
    C = fx.size
    if C == 0:
        return np.zeros(0)
    fy = np.asarray(snap.get('cell_fy', np.zeros(C)), dtype=float)
    fz = np.asarray(snap.get('cell_fz', np.zeros(C)), dtype=float)
    F = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)

    E_gran = np.asarray(snap.get('E_gran', np.zeros(0)), dtype=float)
    nu_gran = np.asarray(snap.get('nu_gran', np.zeros(0)), dtype=float)
    if E_gran.size == 0:
        return np.zeros(C)
    a_cell = float(getattr(p, 'cell_diameter', 20.0)) / 2.0
    k_sub = np.pi * E_gran * a_cell / np.maximum(1.0 - nu_gran ** 2, 1e-12)
    k_sub = np.maximum(k_sub, 1e-12)

    gid = np.asarray(snap.get('cell_granule_id', np.zeros(C)), dtype=np.int64)
    gid = np.clip(gid, 0, k_sub.size - 1)
    tgt = np.asarray(snap.get('cell_bridge_target', -np.ones(C)), dtype=np.int64)

    k_cell = float(getattr(p, 'cell_series_stiffness', 0.0) or 0.0)
    inv = (1.0 / k_cell if k_cell > 0 else 0.0) + 1.0 / k_sub[gid]
    bridged = (tgt >= 0) & (tgt < k_sub.size)
    if np.any(bridged):
        inv = inv + np.where(bridged, 1.0 / k_sub[np.where(bridged, tgt, 0)], 0.0)
    return 0.5 * F ** 2 * inv


# ======================================================================
# Stress-force-fabric, and whether the small-strain picture still holds
# ======================================================================

def _anisotropy(T, dim):
    """Rothenburg-Bathurst anisotropy coefficient of a normalised tensor."""
    d = _deviator(T, dim)
    return float(_ANIS_K[dim] * np.sqrt(np.sum(d * d)))


def fabric_stress(gs, p, contacts, sig=None, sig_contact=None):
    """The SFF decomposition, plus the diagnostic that bounds its validity.

    **Why this is here.** Love-Weber gives one number, ``sigma``. The
    stress-force-fabric relation (Rothenburg & Bathurst 1989) says where that
    number comes from: the deviatoric stress ratio of a granular assembly is
    carried by the ANISOTROPY of the contact network and of the forces on it,

        q/p  ~  (a_c + a_n + a_t) / 2      (2D)
        q/p  ~  2 (a_c + a_n + a_t) / 5    (3D)

    with ``a_c`` the fabric (contact-normal) anisotropy and ``a_n`` / ``a_t``
    the normal / tangential force anisotropies. That is the upscaling: two
    scalars that a continuum model can consume, instead of a full tensor.

    ``a_t`` is not available -- the tangential force is applied but not stored
    on the contact record -- so ``sff_closure`` (predicted / measured q/p) is
    reported instead of being quietly absorbed. Measured on four reference
    beds it lands at **0.82-1.11**, and the distance from 1 IS the tangential
    plus higher-order share. Treating it as exact would be the mistake.

    **The validity diagnostic.** Love-Weber itself is exact for this model:
    the particle-centred form ``sum_p sum_c f (x_c - x_p)`` telescopes to the
    branch-vector form whatever the contact point, and the CMN terms beyond it
    -- an unbalanced-moment term and a centripetal one -- are inertial, so they
    are identically zero in an overdamped run. What is NOT exact at large
    deformation is the layer underneath: Hertz and JKR are small-strain contact
    theories, good while the contact patch is small against the particle. So
    the honest flag is ``a_contact / min(r_i, r_j)``, which is measured here
    rather than assumed. On the 2D beds it sits at 0.05-0.12 (comfortable); on
    a gravity-loaded 3D bed it reaches **0.22 at p95**, which is the edge of
    where the contact law itself should be trusted -- not a problem with the
    stress formula, but with the forces going into it.
    """
    dim = 3 if gs.is_3d else 2
    out = {'stress_q_over_p': 0.0, 'fabric_a_c': 0.0, 'fabric_a_fn': 0.0,
           'fabric_a_n': 0.0, 'sff_q_over_p': 0.0, 'sff_closure': 0.0,
           'stress_patch_p95': 0.0, 'stress_patch_max': 0.0}
    if contacts is None or len(contacts) == 0:
        return out
    col = getattr(contacts, 'column', None)
    if col is not None:
        n = np.column_stack((col('nx'), col('ny'), col('nz'))).astype(float)
        fn = np.asarray(col('F_normal'), dtype=float)
        ii = np.asarray(col('i'), dtype=np.int64)
        jj = np.asarray(col('j'), dtype=np.int64)
        a_c_arr = np.asarray(col('a_contact'), dtype=float)
    else:
        n = np.array([[c['nx'], c['ny'], c.get('nz', 0.0)] for c in contacts], dtype=float)
        fn = np.array([c['F_normal'] for c in contacts], dtype=float)
        ii = np.array([c['i'] for c in contacts], dtype=np.int64)
        jj = np.array([c['j'] for c in contacts], dtype=np.int64)
        a_c_arr = np.array([c.get('a_contact', 0.0) for c in contacts], dtype=float)

    Nc = fn.size
    # ── the small-strain flag ──
    N = int(gs.N)
    ok = (ii >= 0) & (ii < N) & (jj >= 0) & (jj < N)
    if np.any(ok):
        r_min = np.minimum(np.asarray(gs.r[:N])[ii[ok]], np.asarray(gs.r[:N])[jj[ok]])
        ratio = a_c_arr[ok] / np.maximum(r_min, 1e-12)
        out['stress_patch_p95'] = float(np.percentile(ratio, 95))
        out['stress_patch_max'] = float(np.max(ratio))

    # ── fabric and force-weighted fabric ──
    fabric = (n.T @ n) / Nc
    out['fabric_a_c'] = _anisotropy(fabric, dim)
    f_bar = float(np.mean(fn))
    if abs(f_bar) > 1e-30:
        chi = (n.T @ (n * fn[:, None])) / Nc / f_bar
        a_fn = _anisotropy(chi, dim)
        out['fabric_a_fn'] = a_fn
        # RB's expansion is additive to second order, so the force-only part is
        # what the force weighting ADDED to the geometry.
        out['fabric_a_n'] = a_fn - out['fabric_a_c']
        out['sff_q_over_p'] = float((0.5 if dim == 2 else 0.4) * a_fn)

    if sig is None:
        sig, _ = virial_stress(gs, p, contacts)
    P = _pressure(sig, dim)
    if abs(P) > 1e-30:
        out['stress_q_over_p'] = float(_dev_magnitude(sig, dim) / abs(P))

    # `sff_closure` is checked against the CONTACT stress, not the total.
    # a_c and a_n are built from contact normals and contact forces, so the
    # only thing they can explain is the contact network's own shear. Once
    # cells are pulling, a large part of the total q/p has no contact-fabric
    # origin at all -- measured, comparing against the total dropped the
    # closure from 0.82-1.11 (no cells) to 0.15-0.64 (cells seeded), which is
    # a statement about the pairing and not about the packing.
    sc = sig_contact if sig_contact is not None else sig
    Pc = _pressure(sc, dim)
    if abs(Pc) > 1e-30:
        qp_c = float(_dev_magnitude(sc, dim) / abs(Pc))
        if qp_c > 0:
            out['sff_closure'] = float(out['sff_q_over_p'] / qp_c)
    return out
