import pickle as pkl
from pathlib import Path
import re
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import HOMmix_Master_Module as hmm
from mode_identification import bright_voxel_CI as bv
import region_detection_data_filtering as rd
import sys
from scipy.special import jn_zeros, jnp_zeros
from scipy import special
from functools import lru_cache
from typing import Tuple, Dict, Union, List
import os
from scipy.optimize import brentq
from pptx.util import Inches
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

hmm_path = r"C:\Users\zup98752\PycharmProjects\HOMmix"
sys.path.insert(0, hmm_path)

C0 = 299_792_458.0

def get_analytical_freqs(m_max, n_max, p_max, mode):
    if mode == "TM" or mode == "TE":
        pass
    else:
        exit(f"mode in get_analytical_freqs() needs to be TM or TE, not {mode}.")
    csol = 299792458.0
    frequency_010 = 1.3e9
    lambda_010 = csol / frequency_010
    R = hmm.pillbox_radius_from_freq(frequency_010)

    analytical_freq_dict = {}
    for m in range(m_max):
        for n in range(1, n_max):
            for p in range(p_max):
                freq_GHz = cavity_frequency(m, n, p, l=1.0, mode=mode, a=R, normalised=False) * 1.e-9
                analytical_freq_dict[f"{m}{n}{p}"] = freq_GHz

    return analytical_freq_dict


# --- Root helpers (1-based n) ---
def tm_root_v_mn(m: int, n: int) -> float:
    """v_mn = nth zero of J_m (TM uses J_m(v)=0)."""
    if n < 1:
        raise ValueError("n must be >= 1 (1-based).")
    return float(jn_zeros(m, n)[-1])


def te_root_vprime_mn(m: int, n: int) -> float:
    """v'_mn = nth zero of J'_m (TE uses J'_m(v')=0)."""
    if n < 1:
        raise ValueError("n must be >= 1 (1-based).")
    return float(jnp_zeros(m, n)[-1])


# --- Frequencies (absolute, Hz) ---
def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    """
    TM_mnp frequency (Hz):
      f = (c / 2π) * sqrt( (v_mn / R)^2 + (pπ / L)^2 )
    """
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be > 0.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    v = tm_root_v_mn(m, n)
    return (c / (2 * np.pi)) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2)


def f_te(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    """
    TE_mnp frequency (Hz):
      f = (c / 2π) * sqrt( (v'_mn / R)^2 + (pπ / L)^2 )
    """
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be > 0.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    vprime = te_root_vprime_mn(m, n)
    return (c / (2 * np.pi)) * np.sqrt((vprime / R) ** 2 + (p * np.pi / L) ** 2)


# --- Unified wrapper (normalised or absolute) ---
def cavity_frequency(m, n, p, l,
                     *,
                     mode="TM",
                     a=1.0,
                     c=C0,
                     normalised=True):
    """
    Compute cylindrical cavity frequency for TM/TE modes.

    Parameters
    ----------
    m : int
        Azimuthal index
    n : int
        Radial root index (1-based)
    p : int
        Axial index (>= 0)
    l : float
        length_factor used in the normalised model (axial term p/l),
        and in the absolute model via L = l * a (keeps your original scaling).
    mode : {"TM","TE"}, optional
        Which family to use.
    a : float, optional
        Cavity radius (meters). Only used if normalised=False and for L = l*a.
    c : float, optional
        Speed of light (m/s)
    normalised : bool, optional
        If True, return f / f_TM010 (same normalisation as your original).
        If False, return absolute frequency (Hz), calling f_tm/f_te.

    Returns
    -------
    float
        Frequency (normalised or absolute)
    """
    mode_u = str(mode).strip().upper()
    if mode_u not in {"TM", "TE"}:
        raise ValueError('mode must be "TM" or "TE".')
    if p < 0:
        raise ValueError("p must be >= 0.")
    if n < 1:
        raise ValueError("n must be >= 1 (1-based).")
    if l <= 0:
        raise ValueError("l must be > 0.")

    # --- Pick the correct radial root ν_mn (TM) or ν'_mn (TE) ---
    if mode_u == "TM":
        nu_mn = tm_root_v_mn(m, n)
    else:
        nu_mn = te_root_vprime_mn(m, n)

    # --- Normalisation reference: TM010 (same as original) ---
    nu_01_tm = tm_root_v_mn(0, 1)  # 2.404825...

    if normalised:
        B_mn = nu_mn / nu_01_tm
        return np.sqrt(B_mn ** 2 + (p / l) ** 2)

    # --- Absolute frequency (Hz) ---
    # Keep your original geometry scaling:
    #   R = a, L = l * a
    R = a
    L = l * a
    if mode_u == "TM":
        return f_tm(m, n, p, R=R, L=L, c=c)
    else:
        return f_te(m, n, p, R=R, L=L, c=c)



def pillbox_freq_from_radius(radius, GHz=False):
    """

    :param radius:
    :return:
    """
    radius = float(radius)
    csol = 299792458.0
    f_Hz = (2.4048 * csol) / (2.0 * np.pi * radius)
    f_GHz = f_Hz / 1e9
    if GHz:
        return f_GHz
    else:
        return f_Hz


def pillbox_radius_from_freq(f_Hz):
    f_Hz = float(f_Hz)
    csol = 299792458.0
    radius_m = (2.4048 * csol) / (2.0 * np.pi * f_Hz)

    return radius_m

def _k_mn_TE(m: int, n: int, R: float):
    """
    k_mn = x'_mn / R
    where x'_mn are zeros of J'_m.
    """
    xprime = special.jnp_zeros(m, n)[-1]
    return xprime / R



def pillbox_field_voxel_grid_xyz(
    R: float, L: float,
    m: int, n: int, p: int,
    x_res: int, y_res: int, z_res: int,
    mode: str = "TM",
    E0: float = 1.0,
    z_range: Tuple[float, float] = (0.0, 1.0),
    dtype=np.float32,
):
    """
    Returns a Cartesian voxel grid whose indexing matches: out[x, y, z].

    mode: "TM" or "TE"
    """

    mode = mode.upper()
    if mode not in ("TM", "TE"):
        raise ValueError("mode must be 'TM' or 'TE'")

    if R <= 0 or L <= 0:
        raise ValueError("R and L must be > 0.")
    if any(k < 2 for k in (x_res, y_res, z_res)):
        raise ValueError("x_res, y_res, z_res must be >= 2.")

    z0 = float(z_range[0]) * L
    z1 = float(z_range[1]) * L
    if not (0.0 <= z_range[0] <= 1.0 and
            0.0 <= z_range[1] <= 1.0 and
            z0 < z1):
        raise ValueError("z_range must be within [0,1] with z_range[0] < z_range[1].")

    x_coords = np.linspace(-R, R, x_res, dtype=float)
    y_coords = np.linspace(-R, R, y_res, dtype=float)
    z_coords = np.linspace(z0, z1, z_res, dtype=float)

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = r <= R

    # ------------------------
    # Select field generator
    # ------------------------
    if mode == "TM":
        Er, Eth, Ez = _E_field_cyl_TM(r, theta, Z, m, n, p, R, L, E0=E0)
    else:
        Er, Eth, Ez = _E_field_cyl_TE(r, theta, Z, m, n, p, R, L, E0=E0)

    # Cyl → Cart
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    Ex = Er * cos_t - Eth * sin_t
    Ey = Er * sin_t + Eth * cos_t

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    nan = np.nan
    Exm = np.where(mask, Ex, nan).astype(dtype, copy=False)
    Eym = np.where(mask, Ey, nan).astype(dtype, copy=False)
    Ezm = np.where(mask, Ez, nan).astype(dtype, copy=False)
    Eperpm = np.where(mask, Eperp, nan).astype(dtype, copy=False)
    Emagm = np.where(mask, Emag, nan).astype(dtype, copy=False)

    return {
        "Ex": Exm,
        "Ey": Eym,
        "Ez": Ezm,
        "Eperp": Eperpm,
        "|E|": Emagm,
    }




@lru_cache(maxsize=None)
def _besselj_zero(m: int, n: int) -> float:
    """n-th positive zero of J_m(x), with n starting at 1."""
    return special.jn_zeros(m, n)[-1]

def _k_mn(m: int, n: int, R: float) -> float:
    return _besselj_zero(m, n) / R

def _k_z(p: int, L: float) -> float:
    return np.pi * p / L

def _E_field_cyl_TM(r, theta, z, m: int, n: int, p: int, R: float, L: float, E0: float = 1.0):
    """
    Mode-shape fields (arb. units) matching the user's equations:
      Ez ∝ J_m(k_mn r) cos(mθ) cos(kz z)
      Er ∝ p J'_m(k_mn r) cos(mθ) sin(kz z)
      Eθ ∝ (p/r) J_m(k_mn r) sin(mθ) sin(kz z)
    """
    kmn = _k_mn(m, n, R)
    kz = _k_z(p, L)

    x = kmn * r
    Jm = special.jv(m, x)
    Jm_prime = special.jvp(m, x, 1)  # d/dx J_m(x)

    cos_mth = np.cos(m * theta)
    sin_mth = np.sin(m * theta)
    cos_kzz = np.cos(kz * z)
    sin_kzz = np.sin(kz * z)

    Ez = E0 * (Jm * cos_mth * cos_kzz)
    Er = E0 * (p * Jm_prime * cos_mth * sin_kzz)

    with np.errstate(divide="ignore", invalid="ignore"):
        Eth = E0 * ((p / r) * Jm * sin_mth * sin_kzz)
        Eth = np.where(r == 0.0, 0.0, Eth)

    return Er, Eth, Ez


def _E_field_cyl_TE(r, theta, z, m: int, n: int, p: int,
                    R: float, L: float, E0: float = 1.0):
    """
    TE mode-shape fields (arb. units) matching the provided equations:

      Ez = 0

      Er  ∝ J_m(k_mn r) sin(mθ) sin(kz z)
      Eθ  ∝ J'_m(k_mn r) cos(mθ) sin(kz z)

    Uses zeros of J'_m for radial eigenvalue.
    """

    # Radial eigenvalue uses derivative root
    kmn = _k_mn_TE(m, n, R)   # <-- must use J'_m zero
    kz = _k_z(p, L)

    x = kmn * r
    Jm = special.jv(m, x)
    Jm_prime = special.jvp(m, x, 1)

    sin_mth = np.sin(m * theta)
    cos_mth = np.cos(m * theta)
    sin_kzz = np.sin(kz * z)

    # From your equations (ignoring constant prefactors)
    Er = E0 * (Jm * sin_mth * sin_kzz)
    Eth = E0 * (Jm_prime * cos_mth * sin_kzz)

    Ez = np.zeros_like(Er)

    return Er, Eth, Ez


def get_analytical_freqs(m_max, n_max, p_max, mode):
    if mode == "TM" or mode == "TE":
        pass
    else:
        exit(f"mode in get_analytical_freqs() needs to be TM or TE, not {mode}.")
    csol = 299792458.0
    frequency_010 = 1.3e9
    lambda_010 = csol / frequency_010
    R = pillbox_radius_from_freq(frequency_010)

    analytical_freq_dict = {}
    for m in range(m_max+1):
        for n in range(1, n_max+1):
            for p in range(p_max+1):
                freq_GHz = cavity_frequency(m, n, p, l=1.0, mode=mode, a=R, normalised=False) * 1.e-9
                analytical_freq_dict[f"{m}{n}{p}"] = freq_GHz

    print(f"{analytical_freq_dict['222'] = }")
    return analytical_freq_dict

def assemble_all_data_dict(m_max, n_max, p_max,
                           frequency_010 = 1.3e9,
                           LF_start = 0.9,
                           LF_stop = 1.1,
                           param_sweep_resolution = 1000,
                           voxel_res = 21,
):

    csol = 299_792_458.0  # speed of light (m/s)
    lambda_010 = csol / frequency_010
    R = pillbox_radius_from_freq(frequency_010)

    m_list = np.linspace(0, m_max, m_max+1, endpoint=True)
    mint = [int(m) for m in m_list]
    n_list = np.linspace(1, n_max, n_max+1, endpoint=True)
    nint = [int(n) for n in n_list]
    p_list = np.linspace(0, p_max, p_max + 1, endpoint=True)
    pint = [int(p) for p in p_list]
    length_factor_vector = np.linspace(LF_start, LF_stop, param_sweep_resolution, endpoint=True)
    length_factor_vector_floats = [float(i) for i in length_factor_vector]

    all_data = {}
    all_data['TM'] = {}
    all_data['TE'] = {}
    all_data['length_factor_vector'] = length_factor_vector_floats

    for pidx, p in enumerate(pint):
        for nidx, n in enumerate(nint):
            for midx, m in enumerate(mint):

                """ TM and TE fields """

                """
                data = {"Ex": Exm, "Ey": Eym, "Ez": Ezm, "Eperp": Eperpm, "|E|": Emagm}
                """

                print(f"{m}{n}{p}")
                all_data['TM'][f"{m}{n}{p}"] = {}
                all_data['TE'][f"{m}{n}{p}"] = {}

                all_data['TM'][f"{m}{n}{p}"]['3D_Efield'] = pillbox_field_voxel_grid_xyz(
                    R=R,
                    L=lambda_010 / 2.,
                    m=m,
                    n=n,
                    p=p,
                    x_res=voxel_res,
                    y_res=voxel_res,
                    z_res=voxel_res,
                    E0=1.0,
                    mode="TM",
                    z_range=(0.0, 1.0),
                    dtype=np.float32,
                )

                all_data['TE'][f"{m}{n}{p}"]['3D_Efield'] = pillbox_field_voxel_grid_xyz(
                    R=R,
                    L=lambda_010 / 2.,
                    m=m,
                    n=n,
                    p=p,
                    x_res=voxel_res,
                    y_res=voxel_res,
                    z_res=voxel_res,
                    E0=1.0,
                    mode="TE",
                    z_range=(0.0, 1.0),
                    dtype=np.float32,
                )


                TM_mode_data = []
                TE_mode_data = []
                TM_normalised_mode_data = []
                TE_normalised_mode_data = []
                for lidx, length_factor in enumerate(length_factor_vector_floats):
                    L = length_factor*lambda_010/2.

                    """ TM and TE frequencies """

                    f_tm_val = f_tm(m, n, p, R, L)
                    f_te_val = f_te(m, n, p, R, L)
                    # input(f"TM{m}{n}{p} at {length_factor} = {f_tm_val}")
                    TM_mode_data.append(f_tm_val)
                    TE_mode_data.append(f_te_val)
                    TM_normalised_mode_data.append(f_tm_val/frequency_010)
                    TE_normalised_mode_data.append(f_te_val/frequency_010)

                    all_data['TM'][f"{m}{n}{p}"]['frequency_Hz'] = TM_mode_data
                    all_data['TE'][f"{m}{n}{p}"]['frequency_Hz'] = TE_mode_data
                    all_data['TM'][f"{m}{n}{p}"]['frequency_normalised'] = TM_normalised_mode_data
                    all_data['TE'][f"{m}{n}{p}"]['frequency_normalised'] = TE_normalised_mode_data

    return all_data

def pickle_save(data_dict, dir_fname):
    with open(dir_fname, "wb") as handle:
        pkl.dump(data_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

def pickle_load(dir_fname):
    with open(dir_fname, "rb") as handle:
        data_dict = pkl.load(handle)

    return data_dict

def find_mode_crossings_from_all_data(all_data, mode_type="TM"):



    mode_type = mode_type.upper()
    if mode_type not in ("TM", "TE", "BOTH"):
        raise ValueError("mode_type must be 'TM', 'TE', or 'BOTH'")

    results = {}
    L = np.array(all_data["length_factor_vector"])

    # ===============================================================
    # Helper: reorder modes by frequency at x=1.0
    # ===============================================================
    def reorder_modes(mode_dict):

        target_x = 1.0
        rename_list = []

        for mnp, data in mode_dict.items():
            F = np.array(data["frequency_Hz"])

            if not (L.min() <= target_x <= L.max()):
                raise ValueError(f"{mnp} does not span x=1.0")

            f_interp = np.interp(target_x, L, F)
            rename_list.append((mnp, f_interp))

        rename_list.sort(key=lambda t: t[1])
        return [t[0] for t in rename_list]

    # ===============================================================
    # Helper: crossing detector between two frequency arrays
    # ===============================================================
    def detect_crossings(name_i, name_j, fi, fj):

        crossings = {}
        crossed_modes = []

        g = fi - fj
        idxs = np.where(np.diff(np.sign(g)) != 0)[0]

        for idx in idxs:

            L1, L2 = L[idx], L[idx + 1]

            def gfun(x):
                fi_x = np.interp(x, L, fi)
                fj_x = np.interp(x, L, fj)
                return fi_x - fj_x

            try:
                Lc = brentq(gfun, L1, L2)
            except ValueError:
                continue

            Fc = np.interp(Lc, L, fi)

            key = f"{name_i}–{name_j}"

            crossings[key] = {
                "mode_i": name_i,
                "mode_j": name_j,
                "length_factor": float(f"{Lc:.12g}"),
                "frequency_Hz": float(f"{Fc:.12g}"),
            }

            crossed_modes.extend([name_i, name_j])

        return crossings, crossed_modes

    # ===============================================================
    # Process TM
    # ===============================================================
    if mode_type in ("TM", "BOTH"):

        reordered_TM = reorder_modes(all_data["TM"])
        crossings_TM = {}
        cross_modes_TM = []

        for i in range(len(reordered_TM)):
            for j in range(i + 1, len(reordered_TM)):

                Mi = reordered_TM[i]
                Mj = reordered_TM[j]

                fi = np.array(all_data["TM"][Mi]["frequency_Hz"])
                fj = np.array(all_data["TM"][Mj]["frequency_Hz"])

                c, cm = detect_crossings(f"TM_{Mi}", f"TM_{Mj}", fi, fj)

                crossings_TM.update(c)
                cross_modes_TM.extend(cm)

        results["TM"] = {
            "crossings": crossings_TM,
            "modes_that_cross": list(set(cross_modes_TM)),
        }

    # ===============================================================
    # Process TE
    # ===============================================================
    if mode_type in ("TE", "BOTH"):

        reordered_TE = reorder_modes(all_data["TE"])
        crossings_TE = {}
        cross_modes_TE = []

        for i in range(len(reordered_TE)):
            for j in range(i + 1, len(reordered_TE)):

                Mi = reordered_TE[i]
                Mj = reordered_TE[j]

                fi = np.array(all_data["TE"][Mi]["frequency_Hz"])
                fj = np.array(all_data["TE"][Mj]["frequency_Hz"])

                c, cm = detect_crossings(f"TE_{Mi}", f"TE_{Mj}", fi, fj)

                crossings_TE.update(c)
                cross_modes_TE.extend(cm)

        results["TE"] = {
            "crossings": crossings_TE,
            "modes_that_cross": list(set(cross_modes_TE)),
        }

    # ===============================================================
    # HYBRID TM–TE crossings
    # ===============================================================
    if mode_type == "BOTH":

        reordered_TM = reorder_modes(all_data["TM"])
        reordered_TE = reorder_modes(all_data["TE"])

        crossings_hybrid = {}
        cross_modes_hybrid = []

        for Mi in reordered_TM:
            for Mj in reordered_TE:

                fi = np.array(all_data["TM"][Mi]["frequency_Hz"])
                fj = np.array(all_data["TE"][Mj]["frequency_Hz"])

                c, cm = detect_crossings(
                    f"TM_{Mi}",
                    f"TE_{Mj}",
                    fi,
                    fj
                )

                crossings_hybrid.update(c)
                cross_modes_hybrid.extend(cm)

        results["HYBRID"] = {
            "crossings": crossings_hybrid,
            "modes_that_cross": list(set(cross_modes_hybrid)),
        }

    return results

def plot_crossing_population_heatmap(
    crossing_results: dict,
    savepath: str,
    savename: str,
    *,
    m_values=(0, 1, 2),
    include_families=("TM", "TE"),
    inspect: bool = False,
    dpi: int = 300,
):
    """
    Category heatmap of crossing populations from find_mode_crossings_from_all_data() output.

    Categories (rows/cols):
        TM0np, TM1np, TM2np, TE0np, TE1np, TE2np   (default m_values=(0,1,2))

    Changes vs previous version:
      - Diagonal IS included.
      - Data is displayed in the LOWER-LEFT half of the grid (i >= j).
      - Each populated cell is annotated with count in black text on a translucent white box.

    Notes:
      - For like-like crossings, diagonal entries are meaningful.
      - HYBRID crossings (TM vs TE) will naturally populate off-diagonal blocks.
      - We count each crossing once into the lower triangle by enforcing i >= j.
    """


    include_families = tuple(f.upper() for f in include_families)
    valid_fams = {"TM", "TE"}
    if any(f not in valid_fams for f in include_families):
        raise ValueError("include_families must be a subset of ('TM','TE').")

    # ------------------------------------------------------------------
    # Build category labels + index mapping
    # ------------------------------------------------------------------
    categories = [f"{fam}{m}np" for fam in include_families for m in m_values]
    cat_index = {c: i for i, c in enumerate(categories)}
    N = len(categories)

    counts = np.zeros((N, N), dtype=int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def parse_mode_tag(tag: str):
        """
        tag like: "TM_012" or "TE_1012" etc.
        Returns (fam, m, n, p, mnp_str)
        """
        fam, mnp = tag.split("_", 1)
        fam = fam.upper()
        s = str(mnp)

        if len(s) < 2:
            raise ValueError(f"Invalid mnp key in tag: {tag!r}")

        m = int(s[0])
        n = int(s[1])
        p = int(s[2:]) if len(s) > 2 else 0
        return fam, m, n, p, s

    def to_category(fam: str, m: int):
        return f"{fam}{m}np"

    # ------------------------------------------------------------------
    # Collect all crossings present (TM, TE, HYBRID)
    # ------------------------------------------------------------------
    all_crossings = []

    for fam in ("TM", "TE"):
        if fam in crossing_results and "crossings" in crossing_results[fam]:
            all_crossings.extend(crossing_results[fam]["crossings"].values())

    if "HYBRID" in crossing_results and "crossings" in crossing_results["HYBRID"]:
        all_crossings.extend(crossing_results["HYBRID"]["crossings"].values())

    # ------------------------------------------------------------------
    # Populate counts into LOWER-LEFT triangle (i >= j), INCLUDING diagonal
    # ------------------------------------------------------------------
    for c in all_crossings:
        mi = c.get("mode_i")
        mj = c.get("mode_j")
        if not mi or not mj:
            continue

        fam_i, m_i, *_ = parse_mode_tag(mi)
        fam_j, m_j, *_ = parse_mode_tag(mj)

        # Filter by included families and m_values
        if fam_i not in include_families or fam_j not in include_families:
            continue
        if m_i not in m_values or m_j not in m_values:
            continue

        ci = to_category(fam_i, m_i)
        cj = to_category(fam_j, m_j)

        if ci not in cat_index or cj not in cat_index:
            continue

        i = cat_index[ci]
        j = cat_index[cj]

        # force LOWER-LEFT half: i >= j
        if i < j:
            i, j = j, i

        counts[i, j] += 1

    # ------------------------------------------------------------------
    # Mask UPPER-RIGHT half (i < j) only; keep diagonal
    # ------------------------------------------------------------------
    mask = np.triu(np.ones_like(counts, dtype=bool), k=1)  # True strictly above diagonal
    heat = counts.astype(float)
    heat[mask] = np.nan

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heat, aspect="equal")

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)

    ax.set_title("Crossing population by (family, m) category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Category")

    # Annotate counts (lower triangle + diagonal)
    bbox_style = dict(facecolor="white", edgecolor="none", alpha=0.65, boxstyle="square,pad=0.25")
    for i in range(N):
        for j in range(N):
            if np.isnan(heat[i, j]):
                continue
            ax.text(
                j, i, str(counts[i, j]),
                ha="center", va="center",
                fontsize=10,
                color="black",
                bbox=bbox_style,
            )

    # Cell boundaries for readability
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Crossing count")
    fig.tight_layout()

    if inspect:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=dpi)
        plt.close("all")

    return counts, categories, all_crossings



def plot_modes_from_all_data(
    all_data,
    crossing_results,
    savepath,
    savename,
    *,
    mode_type="TM",          # "TM", "TE", "both"
    m_filter=None,           # None, int, iterable[int]
    n_filter=None,           # None, int, iterable[int]
    p_filter=None,           # None, int, iterable[int]
    normalised=False,
    inspect=False,
    loglog=False,
    acceptance_fraction=None,
):
    """
    Plot modes from shared dictionary with optional (m,n,p) index filtering.

    mnp keys are strings like "012" meaning m=0,n=1,p=2.
    Filters may be None, an int, or an iterable of ints.

    If mode_type="both", TM and TE modes that pass the same filters are plotted,
    and crossings are filtered to include only visible modes (TM-TM, TE-TE, and HYBRID if provided).
    """


    mode_type = mode_type.upper()
    if mode_type not in ("TM", "TE", "BOTH"):
        raise ValueError("mode_type must be 'TM', 'TE', or 'both'")

    def _to_set(v, name):
        if v is None:
            return None
        if isinstance(v, int):
            return {v}
        try:
            s = set(int(x) for x in v)
        except Exception as e:
            raise ValueError(f"{name} must be None, int, or iterable of ints") from e
        return s

    m_filter = _to_set(m_filter, "m_filter")
    n_filter = _to_set(n_filter, "n_filter")
    p_filter = _to_set(p_filter, "p_filter")

    L = np.array(all_data["length_factor_vector"], dtype=float)

    fig, ax = plt.subplots(figsize=(18, 8))

    # --------------------------------------------------------------
    # Determine families
    # --------------------------------------------------------------
    families = []
    if mode_type in ("TM", "BOTH"):
        families.append("TM")
    if mode_type in ("TE", "BOTH"):
        families.append("TE")

    # --------------------------------------------------------------
    # Parse mnp safely (supports multi-digit n/p if ever needed)
    # Default assumption remains "012" (single digits). If longer:
    # - m = first char
    # - n = second char
    # - p = remaining (or 0 if empty)
    # --------------------------------------------------------------
    def parse_mnp(mnp: str):
        s = str(mnp)
        if len(s) < 2:
            raise ValueError(f"Invalid mnp key: {mnp!r}")
        m = int(s[0])
        n = int(s[1])
        p = int(s[2:]) if len(s) > 2 else 0
        return m, n, p

    # --------------------------------------------------------------
    # Collect filtered modes
    # --------------------------------------------------------------
    visible_modes = []  # list of (fam, mnp)

    for fam in families:
        for mnp in all_data[fam].keys():
            m_val, n_val, p_val = parse_mnp(mnp)

            if m_filter is not None and m_val not in m_filter:
                continue
            if n_filter is not None and n_val not in n_filter:
                continue
            if p_filter is not None and p_val not in p_filter:
                continue

            visible_modes.append((fam, mnp))

    if not visible_modes:
        raise ValueError("No modes remain after applying m/n/p filters.")

    visible_set = set(visible_modes)

    # --------------------------------------------------------------
    # Colors
    # --------------------------------------------------------------
    cmap = plt.cm.get_cmap("nipy_spectral", len(visible_modes))
    mode_colors = {
        f"{fam}_{mnp}": cmap(i)
        for i, (fam, mnp) in enumerate(visible_modes)
    }

    # --------------------------------------------------------------
    # 1. Plot mode curves
    # --------------------------------------------------------------
    for fam, mnp in visible_modes:
        data = all_data[fam][mnp]

        if normalised:
            ys = np.asarray(data["frequency_normalised"], dtype=float)
        else:
            ys = np.asarray(data["frequency_Hz"], dtype=float) / 1e9

        xs = L.copy()

        if loglog:
            mask = (xs > 0) & (ys > 0)
            xs, ys = xs[mask], ys[mask]

        label = f"{fam}_{mnp}"
        ax.plot(xs, ys, lw=1.0, alpha=0.85, color=mode_colors[label])

    # --------------------------------------------------------------
    # 2. Plot crossings (filtered to visible modes)
    # --------------------------------------------------------------
    cross_entries = []

    def crossing_visible(c):
        fam_i, mnp_i = c["mode_i"].split("_", 1)
        fam_j, mnp_j = c["mode_j"].split("_", 1)
        return (fam_i, mnp_i) in visible_set and (fam_j, mnp_j) in visible_set

    for fam in families:
        if fam in crossing_results:
            for c in crossing_results[fam]["crossings"].values():
                if crossing_visible(c):
                    cross_entries.append(c)

    if mode_type == "BOTH" and "HYBRID" in crossing_results:
        for c in crossing_results["HYBRID"]["crossings"].values():
            if crossing_visible(c):
                cross_entries.append(c)

    cross_cmap = plt.cm.get_cmap("tab20", max(4, len(cross_entries)))
    cross_colors = [cross_cmap(i) for i in range(len(cross_entries))]

    for idx, c in enumerate(cross_entries):
        x = float(c["length_factor"])

        if normalised:
            # re-evaluate y from the visible curve for mode_i at x
            fam_i, mnp_i = c["mode_i"].split("_", 1)
            y_arr = np.asarray(all_data[fam_i][mnp_i]["frequency_normalised"], dtype=float)
            y = float(np.interp(x, L, y_arr))
        else:
            y = float(c["frequency_Hz"]) / 1e9

        if loglog and (x <= 0 or y <= 0):
            continue

        ax.scatter(
            x, y,
            s=120,
            facecolors="none",
            edgecolors=cross_colors[idx],
            linewidths=2.0,
        )

    # --------------------------------------------------------------
    # 3. Axis setup
    # --------------------------------------------------------------
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # ax.set_xlim(0.68, 1.32)
    ax.set_xlim(0.95, 1.05)
    fig.subplots_adjust(right=0.80)

    # --------------------------------------------------------------
    # 4. Distributed labels at x = 1.5
    # --------------------------------------------------------------
    label_x = 1.52
    min_sep = 0.004
    repulsion = 0.0015

    mode_positions = []
    for fam, mnp in visible_modes:
        data = all_data[fam][mnp]
        if normalised:
            F = np.asarray(data["frequency_normalised"], dtype=float)
        else:
            F = np.asarray(data["frequency_Hz"], dtype=float) / 1e9

        if L.min() <= 1.5 <= L.max():
            f_interp = float(np.interp(1.5, L, F))
            mode_positions.append((f"{fam}_{mnp}", f_interp))

    mode_positions.sort(key=lambda t: t[1])

    adjusted = []
    for i, (_, f) in enumerate(mode_positions):
        if i == 0:
            adjusted.append(f)
        else:
            prev = adjusted[-1]
            new_y = f
            if not loglog and new_y - prev < min_sep:
                new_y = prev + min_sep + repulsion * (i % 5)
            adjusted.append(new_y)

    for (mode, _), f_adj in zip(mode_positions, adjusted):
        ax.text(
            label_x,
            f_adj,
            mode,
            fontsize=8,
            weight="bold",
            color=mode_colors[mode],
            ha="left",
            va="center",
            alpha=0.85,
            clip_on=False,
        )

    ax.axvline(1.0, ls="--", color="k", alpha=0.65, lw=1.0)
    if acceptance_fraction:
        acceptance_minus = 1. - acceptance_fraction
        acceptance_plus = 1. + acceptance_fraction
        ax.axvline(acceptance_minus,ls="--", color="r", alpha=0.65, lw=1.0)
        ax.axvline(acceptance_plus,ls="--", color="r", alpha=0.65, lw=1.0)

    # --------------------------------------------------------------
    # 5. Labels + grid
    # --------------------------------------------------------------

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=13)
    ax.set_ylabel("$f_{mnp} / f_{010}$" if normalised else "f [GHz]", fontsize=13)


    if loglog:
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.35, lw=0.6)
        ax.grid(which="minor", alpha=0.15, lw=0.4)
    else:
        ax.grid(alpha=0.25, lw=0.5)

    if inspect:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")




# ---------------------------
# Helpers
# ---------------------------
def parse_mode_tag(tag: str):
    """
    tag like "TM_012" or "TE_211"
    Returns (fam, mnp, m, n, p)
    """
    fam, mnp = tag.split("_", 1)
    fam = fam.upper()
    s = str(mnp)
    if len(s) < 2:
        raise ValueError(f"Invalid mnp in tag: {tag!r}")
    m = int(s[0])
    n = int(s[1])
    p = int(s[2:]) if len(s) > 2 else 0
    return fam, s, m, n, p


def mode_category(fam: str, m: int) -> str:
    """Category label used by your heatmap: TM0np, TE2np, etc."""
    return f"{fam}{m}np"


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def get_fieldmap_at_length(
    all_data: dict,
    fam: str,
    mnp: str,
    Lc: float,
    *,
    field_container_key: str = "E_field_maps",
    component_keys=("Ex", "Ey", "Ez"),
):
    """
    Fetch E-field components for (fam, mnp) at length factor closest to Lc.

    Supported storage patterns:

    A) all_data[fam][mnp][field_container_key] is a list/array aligned with length_factor_vector:
         - list of dicts: [{"Ex":..., "Ey":..., "Ez":...}, ...]
         - OR dict of arrays with a leading "length" axis:
              {"Ex": arr[t,x,y,z], ...}

    B) all_data[fam][mnp][field_container_key] is a dict mapping length -> field dict

    Returns
    -------
    (Ex, Ey, Ez) as numpy arrays with shape (x,y,z)
    """
    fam = fam.upper()
    Lvec = _as_float_array(all_data["length_factor_vector"])
    idx = int(np.argmin(np.abs(Lvec - float(Lc))))

    mode_entry = all_data[fam][mnp]

    if field_container_key not in mode_entry:
        raise KeyError(
            f"Could not find '{field_container_key}' in all_data['{fam}']['{mnp}']."
        )

    container = mode_entry[field_container_key]

    # Pattern B: dict keyed by length factor
    if isinstance(container, dict) and all(isinstance(k, (int, float, np.floating)) for k in container.keys()):
        # pick closest key
        keys = np.array(list(container.keys()), dtype=float)
        kidx = int(np.argmin(np.abs(keys - float(Lc))))
        fmap = container[float(keys[kidx])]
        Ex = np.asarray(fmap[component_keys[0]])
        Ey = np.asarray(fmap[component_keys[1]])
        Ez = np.asarray(fmap[component_keys[2]])
        return Ex, Ey, Ez

    # Pattern A1: list of dicts aligned with Lvec
    if isinstance(container, (list, tuple)):
        fmap = container[idx]
        Ex = np.asarray(fmap[component_keys[0]])
        Ey = np.asarray(fmap[component_keys[1]])
        Ez = np.asarray(fmap[component_keys[2]])
        return Ex, Ey, Ez

    # Pattern A2: dict of arrays that might be:
    #   - 4D with a length axis (e.g. [len(L), x, y, z] or [x, y, z, len(L)] etc.)
    #   - 3D single voxel grid (e.g. [x, y, z]) with NO length axis
    if isinstance(container, dict) and all(k in container for k in component_keys):
        Ex_arr = np.asarray(container[component_keys[0]])
        Ey_arr = np.asarray(container[component_keys[1]])
        Ez_arr = np.asarray(container[component_keys[2]])

        # If already single maps (no length axis)
        if Ex_arr.ndim == 3 and Ey_arr.ndim == 3 and Ez_arr.ndim == 3:
            return Ex_arr, Ey_arr, Ez_arr

        # If there is a length axis somewhere (common cases: axis 0 or last axis)
        Llen = len(Lvec)

        def take_length_slice(arr):
            if arr.ndim != 4:
                raise TypeError(f"Expected 3D or 4D field array, got shape {arr.shape}")
            # Find which axis matches the length vector
            axes = [ax for ax, s in enumerate(arr.shape) if s == Llen]
            if not axes:
                raise ValueError(
                    f"Cannot find length axis of size {Llen} in field array shape {arr.shape}"
                )
            if len(axes) > 1:
                # Ambiguous; prefer axis 0 if it matches, else the last matching
                ax = 0 if 0 in axes else axes[-1]
            else:
                ax = axes[0]

            return np.take(arr, idx, axis=ax)

        Ex = take_length_slice(Ex_arr)
        Ey = take_length_slice(Ey_arr)
        Ez = take_length_slice(Ez_arr)
        return Ex, Ey, Ez

    raise TypeError(
        f"Unrecognized '{field_container_key}' structure for {fam}_{mnp}."
    )







def make_crossing_plots_trans_and_longit(
    Ex_i, Ey_i, Ez_i,
    Ex_j, Ey_j, Ez_j,
    *,
    title_base: str,             # e.g. "TM012_TE211"
    out_dir: str,
    inspect: bool = False,
):
    """
    Creates two PNGs in out_dir:
      - {title_base}_trans.png   (xy plane @ zmid)
      - {title_base}_longit.png  (yz plane @ xmid)
    """

    Ex_i = np.asarray(Ex_i); Ey_i = np.asarray(Ey_i); Ez_i = np.asarray(Ez_i)
    Ex_j = np.asarray(Ex_j); Ey_j = np.asarray(Ey_j); Ez_j = np.asarray(Ez_j)

    if Ex_i.shape != Ex_j.shape:
        raise ValueError(f"Shape mismatch Ei {Ex_i.shape} vs Ej {Ex_j.shape}")

    x_res, y_res, z_res = Ex_i.shape
    xmid = x_res // 2
    zmid = z_res // 2

    # Magnitudes (3D)
    Ei_mag3 = np.sqrt(Ex_i**2 + Ey_i**2 + Ez_i**2)
    Ej_mag3 = np.sqrt(Ex_j**2 + Ey_j**2 + Ez_j**2)

    # Component-wise add/sub (3D)
    Eaddx3, Eaddy3, Eaddz3 = Ex_i + Ex_j, Ey_i + Ey_j, Ez_i + Ez_j
    Esubx3, Esuby3, Esubz3 = Ex_i - Ex_j, Ey_i - Ey_j, Ez_i - Ez_j

    # Magnitudes add/sub (3D)
    Eadd_mag3 = np.sqrt(Eaddx3**2 + Eaddy3**2 + Eaddz3**2)
    Esub_mag3 = np.sqrt(Esubx3**2 + Esuby3**2 + Esubz3**2)

    # ----------------------------
    # Transverse (xy @ zmid)
    # Use transpose so x is horizontal, y is vertical, with origin="lower"
    # ----------------------------
    trans = dict(
        Eix=Ex_i[:, :, zmid].T, Ejx=Ex_j[:, :, zmid].T, Eaddx=Eaddx3[:, :, zmid].T, Esubx=Esubx3[:, :, zmid].T,
        Eiy=Ey_i[:, :, zmid].T, Ejy=Ey_j[:, :, zmid].T, Eaddy=Eaddy3[:, :, zmid].T, Esuby=Esuby3[:, :, zmid].T,
        Eiz=Ez_i[:, :, zmid].T, Ejz=Ez_j[:, :, zmid].T, Eaddz=Eaddz3[:, :, zmid].T, Esubz=Esubz3[:, :, zmid].T,
        Ei_mag=Ei_mag3[:, :, zmid].T, Ej_mag=Ej_mag3[:, :, zmid].T, Eadd_mag=Eadd_mag3[:, :, zmid].T, Esub_mag=Esub_mag3[:, :, zmid].T,
    )

    out_trans = os.path.join(out_dir, f"{title_base}_trans.png")
    _plot_4x4_grid(
        **trans,
        title=f"{title_base}  (transverse: xy @ zmid)",
        out_png_path=out_trans,
        inspect=inspect,
    )

    # ----------------------------
    # Longitudinal (yz @ xmid)
    # yz slice is (y,z) already; no transpose. origin="lower" makes y increase upward.
    # ----------------------------
    longit = dict(
        Eix=Ex_i[xmid, :, :], Ejx=Ex_j[xmid, :, :], Eaddx=Eaddx3[xmid, :, :], Esubx=Esubx3[xmid, :, :],
        Eiy=Ey_i[xmid, :, :], Ejy=Ey_j[xmid, :, :], Eaddy=Eaddy3[xmid, :, :], Esuby=Esuby3[xmid, :, :],
        Eiz=Ez_i[xmid, :, :], Ejz=Ez_j[xmid, :, :], Eaddz=Eaddz3[xmid, :, :], Esubz=Esubz3[xmid, :, :],
        Ei_mag=Ei_mag3[xmid, :, :], Ej_mag=Ej_mag3[xmid, :, :], Eadd_mag=Eadd_mag3[xmid, :, :], Esub_mag=Esub_mag3[xmid, :, :],
    )

    out_longit = os.path.join(out_dir, f"{title_base}_longit.png")
    _plot_4x4_grid(
        **longit,
        title=f"{title_base}  (longitudinal: yz @ xmid)",
        out_png_path=out_longit,
        inspect=inspect,
    )

    return out_trans, out_longit

def _plot_4x4_grid(
    *,
    Eix, Ejx, Eaddx, Esubx,
    Eiy, Ejy, Eaddy, Esuby,
    Eiz, Ejz, Eaddz, Esubz,
    Ei_mag, Ej_mag, Eadd_mag, Esub_mag,
    title: str,
    out_png_path: str,
    inspect: bool = False,
):
    """
    4×4 layout, tuned to fit as HALF of a 16:9 slide when placed side-by-side.
    - More square aspect
    - Smaller title footprint
    - Smaller colorbars
    - Max-abs annotations per subplot
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # More square so two images can sit side-by-side in PPT
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 10.0), constrained_layout=True)

    # Smaller title; keep it inside top margin
    fig.suptitle(title, fontsize=11, y=0.99)

    row_names = ["Ex", "Ey", "Ez", "|E|"]
    col_names = ["Ei", "Ej", "Eadd", "Esub"]

    grid = [
        [Eix,    Ejx,    Eaddx,    Esubx],
        [Eiy,    Ejy,    Eaddy,    Esuby],
        [Eiz,    Ejz,    Eaddz,    Esubz],
        [Ei_mag, Ej_mag, Eadd_mag, Esub_mag],
    ]

    bbox_style = dict(facecolor="white", edgecolor="none", alpha=0.65,
                      boxstyle="round,pad=0.2")

    for r in range(4):
        row_slices = grid[r]

        if r < 3:
            vmax = np.nanmax(np.abs(row_slices))
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmin = 0.0
            vmax = np.nanmax(row_slices)
            cmap = "viridis"

        for c in range(4):
            ax = axes[r, c]
            data = row_slices[c]

            ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(col_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_names[r], fontsize=10)

            max_abs = np.nanmax(np.abs(data))
            ax.text(
                0.02, 0.98,
                f"max| |={max_abs:.2e}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                color="black",
                bbox=bbox_style,
            )

        # Smaller per-row colorbar so it doesn't dominate
        fig.colorbar(
            axes[r, 0].images[0],
            ax=axes[r, :],
            fraction=0.018,
            pad=0.01,
        )

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    if inspect:
        plt.show()
    else:
        fig.savefig(out_png_path, dpi=300)
        plt.close(fig)



def generate_crossing_fieldmap_figures(
    all_data: dict,
    crossing_results: dict,
    counts,
    categories: list,
    out_root_dir: str,
    *,
    mode_type: str = "BOTH",
    field_container_key: str = "E_field_maps",
    inspect: bool = False,
):
    """
    Creates TWO PNGs per crossing:
      - ..._trans.png
      - ..._longit.png

    Saved in:
      out_root_dir / "{CAT_A}_{CAT_B}" / "{MODEI}_{MODEJ}_trans.png"
      out_root_dir / "{CAT_A}_{CAT_B}" / "{MODEI}_{MODEJ}_longit.png"
    """


    mode_type = mode_type.upper()
    if mode_type not in ("TM", "TE", "BOTH"):
        raise ValueError("mode_type must be 'TM', 'TE', or 'BOTH'")

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    def parse_mode_tag(tag: str):
        fam, mnp = tag.split("_", 1)
        fam = fam.upper()
        s = str(mnp)
        m = int(s[0]); n = int(s[1]); p = int(s[2:]) if len(s) > 2 else 0
        return fam, s, m, n, p

    def mode_category(fam: str, m: int) -> str:
        return f"{fam}{m}np"

    # gather crossings
    crossings_to_process = []
    if mode_type in ("TM", "BOTH") and "TM" in crossing_results:
        crossings_to_process.extend(crossing_results["TM"]["crossings"].values())
    if mode_type in ("TE", "BOTH") and "TE" in crossing_results:
        crossings_to_process.extend(crossing_results["TE"]["crossings"].values())
    if mode_type == "BOTH" and "HYBRID" in crossing_results:
        crossings_to_process.extend(crossing_results["HYBRID"]["crossings"].values())

    made = 0

    for cidx, c in enumerate(crossings_to_process):
        mode_i = c["mode_i"]           # "TM_012"
        mode_j = c["mode_j"]           # "TE_211"
        Lc = float(c["length_factor"])

        print(f"\n{c['mode_i']} with {c['mode_j']}: {(cidx+1)/len(crossings_to_process)}")

        fam_i, mnp_i, m_i, n_i, p_i = parse_mode_tag(mode_i)
        fam_j, mnp_j, m_j, n_j, p_j = parse_mode_tag(mode_j)

        cat_i = mode_category(fam_i, m_i)
        cat_j = mode_category(fam_j, m_j)

        # folder ordering by categories list to avoid duplicates
        if cat_to_idx.get(cat_i, 10**9) <= cat_to_idx.get(cat_j, 10**9):
            folder = f"{cat_i}_{cat_j}"
            title_base = f"{fam_i}{mnp_i}_{fam_j}{mnp_j}"   # e.g. TM012_TE211
        else:
            folder = f"{cat_j}_{cat_i}"
            title_base = f"{fam_j}{mnp_j}_{fam_i}{mnp_i}"

        out_dir = os.path.join(out_root_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        # --- fetch fields at nearest length index ---
        Ex_i, Ey_i, Ez_i = get_fieldmap_at_length(
            all_data, fam_i, mnp_i, Lc, field_container_key=field_container_key
        )
        Ex_j, Ey_j, Ez_j = get_fieldmap_at_length(
            all_data, fam_j, mnp_j, Lc, field_container_key=field_container_key
        )

        # --- make 2 figures ---
        make_crossing_plots_trans_and_longit(
            Ex_i, Ey_i, Ez_i,
            Ex_j, Ey_j, Ez_j,
            title_base=title_base,
            out_dir=out_dir,
            inspect=inspect,
        )

        made += 1

    return made



def plot_transverse_plane_field(
    m: int, n: int, p: int,
    R: float, L: float,
    z: float | None = None,
    N: int = 301,
    E0: float = 1.0,
    what: str = "|E|",
    vectors: bool = False,
    vector_stride: int = 12,
    quiver_scale: float | None = None,
    vector_what: str = "Eperp",
):
    """
    Plot the transverse (x,y) plane at z (default z=L/2).

    Plottable scalars via `what`:
      "|E|" (default), "Ex", "Ey", "Ez", "Eperp"

    Optional vector overlay (`vectors=True`) in the transverse plane:
      vector_what="Eperp" (default) -> arrows are (Ex, Ey)
      vector_what="ErEth"           -> arrows are (Er, Eθ) projected into (x,y) (same result as Ex,Ey),
                                       kept for explicitness.

    Returns (fig, ax).
    """
    if z is None:
        z = L / 2.0

    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    mask = r <= R

    Er, Eth, Ez = _E_field_cyl_TM(r, theta, z, m, n, p, R, L, E0=E0)

    # Cyl -> Cart
    Ex = Er * np.cos(theta) - Eth * np.sin(theta)
    Ey = Er * np.sin(theta) + Eth * np.cos(theta)

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    key = what.strip().lower()
    if key in ["|e|", "e", "emag", "mag", "magnitude"]:
        F = Emag
        label = r"$|E|$ (arb. units)"
        title = r"$|E(x,y)|$"
    elif key == "ex":
        F = Ex
        label = r"$E_x$ (arb. units)"
        title = r"$E_x(x,y)$"
    elif key == "ey":
        F = Ey
        label = r"$E_y$ (arb. units)"
        title = r"$E_y(x,y)$"
    elif key == "ez":
        F = Ez
        label = r"$E_z$ (arb. units)"
        title = r"$E_z(x,y)$"
    elif key in ["eperp", "e_perp", "transverse"]:
        F = Eperp
        label = r"$|E_\perp|$ (arb. units)"
        title = r"$|E_\perp(x,y)|$"
    else:
        raise ValueError('what must be one of: "|E|", "Ex", "Ey", "Ez", "Eperp"')

    Fm = np.where(mask, F, np.nan)

    fig, ax = plt.subplots()
    im = ax.imshow(
        Fm,
        extent=[-R, R, -R, R],
        origin="lower",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label=label)

    # Optional vector overlay
    if vectors:
        s = vector_stride

        vkey = vector_what.strip().lower()
        if vkey in ["eperp", "exey", "xy", "cart"]:
            U = Ex
            V = Ey
        elif vkey in ["ereth", "cyl"]:
            # Convert (Er,Eθ) to (Ex,Ey) for plotting arrows in the xy plane
            U = Ex
            V = Ey
        else:
            raise ValueError('vector_what must be "Eperp" (Ex,Ey arrows) or "ErEth"')

        Um = np.where(mask, U, np.nan)
        Vm = np.where(mask, V, np.nan)

        ax.quiver(
            X[::s, ::s], Y[::s, ::s],
            Um[::s, ::s], Vm[::s, ::s],
            scale=quiver_scale
        )

    # Draw cavity boundary
    t = np.linspace(0, 2*np.pi, 600)
    ax.plot(R*np.cos(t), R*np.sin(t), linewidth=1)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title} at z={z:.4g} m   (m={m}, n={n}, p={p})")

    plt.tight_layout()
    plt.show()
    return fig, ax, Fm


def Vacc_from_real_Ez(z_mm, Ez_Vperm, omega_rad_s, beta=1.0):
    """
    Compute accelerating voltage from real on-axis Ez for a standing-wave mode.

    Parameters
    ----------
    z_mm : array-like
        z coordinates in mm
    Ez_Vperm : array-like
        Real Ez field values in V/m
    omega_rad_s : float
        Angular frequency omega = 2*pi*f in rad/s
    beta : float, optional
        Particle velocity / c

    Returns
    -------
    float
        Accelerating voltage in V
    """
    c = 299792458.0

    z_m = np.asarray(z_mm, dtype=float) / 1000.0
    Ez = np.asarray(Ez_Vperm, dtype=float)

    # Optional: center z so that cavity center is at 0
    z_m = z_m - z_m[len(z_m) // 2]

    phase = omega_rad_s * z_m / (beta * c)
    integrand = Ez * np.cos(phase)

    Vacc = np.abs(np.trapezoid(integrand, z_m))
    return Vacc


def loss_from_Vz(Vz):
    loss = Vz ** 2. / 4.

    return loss



def Vz_loss_from_field(datapath, field_saved_fname, f_010, f_mnp, l_factor, Req_m, Rir_div_Req, plot=False):
    """
    for 150x150x150 cavity data
    N.B mid-pixel is 75
    """

    # load field
    c_sol = 299792458.0

    Ez = np.load(f"{datapath}\\{field_saved_fname}")

    # manipulate variables
    w_mnp = f_mnp * 2. * np.pi  # angular_freq
    lambda_010 = c_sol / f_010  # wavelength
    l_1_m = lambda_010 / 2.  # design cell length

    d_m = l_factor * l_1_m  # degenerate cell length

    # calculate longitudinal pixel length in metres
    n_pixels_longit = len(Ez[75, 75, :])
    len_pixel_longit = d_m / float((n_pixels_longit) - 1.)

    # generate z-vector in mm
    z_mm = [i * len_pixel_longit * 1.e3 for i in range(n_pixels_longit)]

    # calculate transverse pixel length in metres
    cav_diameter_m = 2. * Req_m
    n_pixel_trans = len(Ez[75, :, 75])
    len_pixel_trans = cav_diameter_m / float((n_pixel_trans) - 1.)

    # we want to analyse transversely up to quarter of the iris radius
    iris_radius_m = Req_m * Rir_div_Req
    iris_radius_pix = iris_radius_m / len_pixel_trans
    quarter_iris_radius_pix = int(np.floor(iris_radius_pix / 4.))
    print(f"{d_m = }")
    print(f"{n_pixel_trans = }")
    print(f"{len_pixel_trans = }")
    print(f"{iris_radius_m = }")
    print(f"{iris_radius_pix = }")
    print(f"{quarter_iris_radius_pix = }")

    # construct the radial coordinate vector in metres
    r_m = [i * len_pixel_trans for i in range(1, quarter_iris_radius_pix)]
    print(f"{r_m = }")

    # calculate the accelerating voltage for each radius up to Rir/2
    # ASSUME - high field is vertical - should be so from previous rotations
    Ez_vperm = Ez[75, 75, :]
    Vz_axis = Vacc_from_real_Ez(z_mm, Ez_vperm, w_mnp)

    # calculate the loss factors for each radius up to Rir/2
    loss_axis = loss_from_Vz(Vz_axis)

    return Vz_axis, loss_axis


def get_3D_data_monopole(
    E1,
    E2,
    array_path: str,
    plot: bool = False,
    create_fields: bool = True,
    coord_unit: str = "mm",
):
    """
    Uses read_3D_CST_field_data() (must be defined/imported) to read two CST 3D E-field maps,
    then saves/loads ALL arrays as 3D arrays.

    Returned dict matches your original keys, but with the NECESSARY change:
      - it now ALSO returns the x coordinate vectors as xs1 and xs2
        (you already return y and z as ys*/zs*).
    """



    os.makedirs(array_path, exist_ok=True)

    if create_fields:
        # # --- Read full 3D field dictionaries ---
        # E1 = pd.read_csv(field_map_filename_E1, sep=r"\s+", skiprows=2, names=cols, engine="python")
        # E2 = pd.read_csv(field_map_filename_E2, sep=r"\s+", skiprows=2, names=cols, engine="python")

        # --- Extract 3D components ---
        E1_Ex, E1_Ey, E1_Ez = E1["Ex"], E1["Ey"], E1["Ez"]
        E2_Ex, E2_Ey, E2_Ez = E2["Ex"], E2["Ey"], E2["Ez"]


        # --- Plus/minus combos (3D) ---
        Ex_plus = E1_Ex + E2_Ex
        Ey_plus = E1_Ey + E2_Ey
        Ez_plus = E1_Ez + E2_Ez
        Ex_minus = E1_Ex - E2_Ex
        Ey_minus = E1_Ey - E2_Ey
        Ez_minus = E1_Ez - E2_Ez

        # --- Magnitudes (3D) ---
        abs_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2 + np.abs(E1_Ez) ** 2)
        abs_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2 + np.abs(E2_Ez) ** 2)
        abs_add = np.sqrt(
            np.abs(Ex_plus) ** 2 + np.abs(Ey_plus) ** 2 + np.abs(Ez_plus) ** 2
        )
        abs_sub = np.sqrt(
            np.abs(Ex_minus) ** 2 + np.abs(Ey_minus) ** 2 + np.abs(Ez_minus) ** 2
        )

        # --- Transverse Fields (3D) ---
        trans_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2)
        trans_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2)
        trans_plus = np.sqrt(np.abs(Ex_plus) ** 2 + np.abs(Ey_plus) ** 2)
        trans_minus = np.sqrt(np.abs(Ex_minus) ** 2 + np.abs(Ey_minus) ** 2)

        # --- Save everything (3D arrays + coordinate vectors) ---
        np.save(os.path.join(array_path, "abs_E1.npy"), abs_E1)
        np.save(os.path.join(array_path, "E1_Ex.npy"), E1_Ex)
        np.save(os.path.join(array_path, "E1_Ey.npy"), E1_Ey)
        np.save(os.path.join(array_path, "E1_Ez.npy"), E1_Ez)

        np.save(os.path.join(array_path, "abs_E2.npy"), abs_E2)
        np.save(os.path.join(array_path, "E2_Ex.npy"), E2_Ex)
        np.save(os.path.join(array_path, "E2_Ey.npy"), E2_Ey)
        np.save(os.path.join(array_path, "E2_Ez.npy"), E2_Ez)

        np.save(os.path.join(array_path, "trans_E1.npy"), trans_E1)
        np.save(os.path.join(array_path, "trans_E2.npy"), trans_E2)

        np.save(os.path.join(array_path, "abs_add.npy"), abs_add)
        np.save(os.path.join(array_path, "Ex_plus.npy"), Ex_plus)
        np.save(os.path.join(array_path, "Ey_plus.npy"), Ey_plus)
        np.save(os.path.join(array_path, "Ez_plus.npy"), Ez_plus)
        np.save(os.path.join(array_path, "trans_plus.npy"), trans_plus)

        np.save(os.path.join(array_path, "abs_sub.npy"), abs_sub)
        np.save(os.path.join(array_path, "Ex_minus.npy"), Ex_minus)
        np.save(os.path.join(array_path, "Ey_minus.npy"), Ey_minus)
        np.save(os.path.join(array_path, "Ez_minus.npy"), Ez_minus)
        np.save(os.path.join(array_path, "trans_minus.npy"), trans_minus)

    else:
        # --- Load everything (3D arrays + coordinate vectors) ---
        abs_E1 = np.load(os.path.join(array_path, "abs_E1.npy"))
        E1_Ex = np.load(os.path.join(array_path, "E1_Ex.npy"))
        E1_Ey = np.load(os.path.join(array_path, "E1_Ey.npy"))
        E1_Ez = np.load(os.path.join(array_path, "E1_Ez.npy"))


        abs_E2 = np.load(os.path.join(array_path, "abs_E2.npy"))
        E2_Ex = np.load(os.path.join(array_path, "E2_Ex.npy"))
        E2_Ey = np.load(os.path.join(array_path, "E2_Ey.npy"))
        E2_Ez = np.load(os.path.join(array_path, "E2_Ez.npy"))


        trans_E1 = np.load(os.path.join(array_path, "trans_E1.npy"))
        trans_E2 = np.load(os.path.join(array_path, "trans_E2.npy"))
        trans_plus = np.load(os.path.join(array_path, "trans_plus.npy"))
        trans_minus = np.load(os.path.join(array_path, "trans_minus.npy"))

        abs_add = np.load(os.path.join(array_path, "abs_add.npy"))
        Ex_plus = np.load(os.path.join(array_path, "Ex_plus.npy"))
        Ey_plus = np.load(os.path.join(array_path, "Ey_plus.npy"))
        Ez_plus = np.load(os.path.join(array_path, "Ez_plus.npy"))

        abs_sub = np.load(os.path.join(array_path, "abs_sub.npy"))
        Ex_minus = np.load(os.path.join(array_path, "Ex_minus.npy"))
        Ey_minus = np.load(os.path.join(array_path, "Ey_minus.npy"))
        Ez_minus = np.load(os.path.join(array_path, "Ez_minus.npy"))

    print(f"{E1_Ex.shape = }")
    print(f"{E1_Ey.shape = }")
    print(f"{E1_Ez.shape = }")

    # Keep original naming scheme; add xs1/xs2 as the necessary change.
    return {
        "abs_E1": abs_E1,
        "E1_Ex": E1_Ex,
        "E1_Ey": E1_Ey,
        "E1_Ez": E1_Ez,
        "trans_E1": trans_E1,

        "abs_E2": abs_E2,
        "E2_Ex": E2_Ex,
        "E2_Ey": E2_Ey,
        "E2_Ez": E2_Ez,
        "trans_E2": trans_E2,

        "abs_add": abs_add,
        "Ex_plus": Ex_plus,
        "Ey_plus": Ey_plus,
        "Ez_plus": Ez_plus,
        "trans_plus": trans_plus,

        "abs_sub": abs_sub,
        "Ex_minus": Ex_minus,
        "Ey_minus": Ey_minus,
        "Ez_minus": Ez_minus,
        "trans_minus": trans_minus,
    }



def extract_3d_field_slices(
    data: Dict[str, np.ndarray],
    field_keys: List[str],
) -> Dict[str, np.ndarray]:
    """
    From a data dict containing multiple 3D fields, extract three slices
    from each field:

      iris             = field[:, :, 0]
      transverse_mid   = field[:, :, mid_pixel]
      longitudinal_mid = field[mid_pixel, :, :]

    Parameters
    ----------
    data : dict
        Dictionary containing 3D numpy arrays.
    field_keys : list of str
        Keys in `data` corresponding to 3D fields to slice.

    Returns
    -------
    slices : dict
        Flat dictionary with keys like:
          '<field>_iris'
          '<field>_transverse_mid'
          '<field>_longitudinal_mid'
    """

    slices: Dict[str, np.ndarray] = {}

    for key in field_keys:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in data dict")

        field = data[key]
        if not isinstance(field, np.ndarray) or field.ndim != 3:
            raise ValueError(
                f"'{key}' must be a 3D numpy array, got shape {getattr(field, 'shape', None)}"
            )

        Nx, Ny, Nz = field.shape
        mid_trans_pixel = Nx // 2  # consistent with earlier convention
        mid_longit_pixel = Nz // 2  # consistent with earlier convention

        slices[f"{key}_iris_1"] = field[:, :, 0]
        slices[f"{key}_iris_2"] = field[:, :, Nz-1]
        slices[f"{key}_transverse_mid"] = field[:, :, mid_longit_pixel]
        slices[f"{key}_longitudinal_mid"] = field[mid_trans_pixel, :, :]

    return slices


# Convenience wrappers
def plot_all_plus(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids_with_txt(
        slice_dict, save_directory_fname, ops=("plus",), **kwargs
    )

def plot_all_minus(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids_with_txt(
        slice_dict, save_directory_fname, ops=("minus",), **kwargs
    )


def plot_slice_field_grids_with_txt(
    slice_dict,
    save_directory_fname,
    slice_types=("iris_1", "iris_2", "transverse_mid", "longitudinal_mid"),
    ops=("plus", "minus"),
    *,
    extent_by_type=None,
    cmap_div="RdBu_r",
    cmap_abs="viridis",
    vlim_xyz=None,
    vlim_abs=None,
    complex_component="real",
    abs_for_absE=True,
    figsize=(11, 10),
    tight=True,
    save_fig=True,
):
    """
    Plots 4x3 grids for each slice type and op, annotates each subplot with the
    maximum magnitude value (max(|pixel|)) of that subplot, and returns those
    maxima in a dict with keys like: 'minus_iris_1_E1_Ex'.
    """
    rows = ["Ex", "Ey", "Ez", "absE"]

    def _need(k):
        if k not in slice_dict:
            raise KeyError(f"Missing key in slice_dict: {k}")

    def _get_vlim_xyz_for_row(rowname, keys_for_row):
        # explicit limits
        if isinstance(vlim_xyz, (int, float)):
            v = float(vlim_xyz)
            return (-v, v)
        if isinstance(vlim_xyz, dict) and rowname in vlim_xyz:
            v = float(vlim_xyz[rowname])
            return (-v, v)

        # auto symmetric limits per-row across 3 panels
        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=abs_for_absE,
                rowname=rowname,
            )
            for k in keys_for_row
        ]
        m = max(np.nanmax(np.abs(im)) for im in imgs)
        return (-m, m) if np.isfinite(m) and m > 0 else (-1.0, 1.0)

    def _get_vlim_abs(stype, abs_keys):
        if isinstance(vlim_abs, tuple) and len(vlim_abs) == 2:
            return (float(vlim_abs[0]), float(vlim_abs[1]))
        if isinstance(vlim_abs, dict) and stype in vlim_abs:
            t = vlim_abs[stype]
            return (float(t[0]), float(t[1]))

        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=True,
                rowname="absE",
            )
            for k in abs_keys
        ]
        vmin = min(np.nanmin(im) for im in imgs)
        vmax = max(np.nanmax(im) for im in imgs)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            return (0.0, 1.0)
        return (float(vmin), float(vmax))

    def _max_magnitude(img: np.ndarray) -> float:
        """Max magnitude for any real/complex image; NaN-safe."""
        m = np.nanmax(np.abs(img))
        return float(m) if np.isfinite(m) else float("nan")

    figs = {}
    axes_out = {}
    max_dict = {}  # <- NEW: subplot_id -> max(|pixel|)

    for stype in slice_types:
        extent = extent_by_type.get(stype) if extent_by_type else None

        for op in ops:
            op = op.lower().strip()
            if op not in {"plus", "minus"}:
                raise ValueError("ops must be drawn from {'plus','minus'}")

            abs_op_key = "abs_add" if op == "plus" else "abs_sub"
            comp_op_suffix = "plus" if op == "plus" else "minus"

            key_map = {
                ("Ex", "E1"): f"E1_Ex_{stype}",
                ("Ex", "E2"): f"E2_Ex_{stype}",
                ("Ex", op):   f"Ex_{comp_op_suffix}_{stype}",

                ("Ey", "E1"): f"E1_Ey_{stype}",
                ("Ey", "E2"): f"E2_Ey_{stype}",
                ("Ey", op):   f"Ey_{comp_op_suffix}_{stype}",

                ("Ez", "E1"): f"E1_Ez_{stype}",
                ("Ez", "E2"): f"E2_Ez_{stype}",
                ("Ez", op):   f"Ez_{comp_op_suffix}_{stype}",

                ("absE", "E1"): f"abs_E1_{stype}",
                ("absE", "E2"): f"abs_E2_{stype}",
                ("absE", op):   f"{abs_op_key}_{stype}",
            }

            # validate keys exist
            for r in rows:
                for c in ("E1", "E2", op):
                    _need(key_map[(r, c)])

            # limits
            vlims_xyz = {}
            for r in ["Ex", "Ey", "Ez"]:
                keys_for_row = [key_map[(r, "E1")], key_map[(r, "E2")], key_map[(r, op)]]
                vlims_xyz[r] = _get_vlim_xyz_for_row(r, keys_for_row)

            abs_keys = [key_map[("absE", "E1")], key_map[("absE", "E2")], key_map[("absE", op)]]
            vmin_abs, vmax_abs = _get_vlim_abs(stype, abs_keys)

            # plot
            fig, ax = plt.subplots(4, 3, figsize=figsize, sharex=True, sharey=True)
            fig.suptitle(f"{stype} — E1 {'+' if op=='plus' else '-'} E2", y=0.98)

            col_labels = ["E1", "E2", op]
            for ci, clab in enumerate(col_labels):
                ax[0, ci].set_title(clab)

            for ri, rname in enumerate(rows):
                ax[ri, 0].set_ylabel(rname)

                for ci, cname in enumerate(["E1", "E2", op]):
                    a = ax[ri, ci]
                    raw = slice_dict[key_map[(rname, cname)]]

                    img = _to_real_image(
                        raw,
                        component=complex_component,
                        abs_for_absE=abs_for_absE,
                        rowname=rname,
                    )


                    #TODO remove if deemed incorrect
                    # img = img.T

                    # max magnitude + annotate + store
                    mval = _max_magnitude(img)
                    subplot_id = f"{op}_{stype}_{cname}_{rname}"  # e.g. minus_iris_1_E1_Ex
                    max_dict[subplot_id] = mval

                    a.text(
                        0.02, 0.98, f"max|·|={mval:.3g}",
                        transform=a.transAxes,
                        ha="left", va="top",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0),
                    )

                    # draw
                    if rname in {"Ex", "Ey", "Ez"}:
                        vmin, vmax = vlims_xyz[rname]
                        im = a.imshow(
                            img, origin="lower", extent=extent,
                            cmap=cmap_div, vmin=vmin, vmax=vmax, aspect="auto"
                        )
                    else:
                        im = a.imshow(
                            img, origin="lower", extent=extent,
                            cmap=cmap_abs, vmin=vmin_abs, vmax=vmax_abs, aspect="auto"
                        )

                    # colorbar on rightmost col
                    if ci == 2:
                        divider = make_axes_locatable(a)
                        cax = divider.append_axes("right", size="4%", pad=0.05)
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.ax.tick_params(labelsize=8)

                mid = ax[ri, 1]
                mid.text(
                    1.02, 1.02, f"{'+' if op=='plus' else '-'}  =",
                    transform=mid.transAxes, ha="left", va="bottom", fontsize=10
                )

            if tight:
                plt.tight_layout()

            figs[(stype, op)] = fig
            axes_out[(stype, op)] = ax

            if save_fig:
                plt.savefig(f"{save_directory_fname}_{op}_{stype}.png")
                plt.close("all")
            else:
                plt.show()

    return figs, axes_out, max_dict







def _to_real_image(arr, *, component="real", abs_for_absE=True, rowname=None):
    """
    Convert possibly-complex arrays to real-valued 2D arrays for plotting.

    component: "real" | "imag" | "abs" | "phase"
      - for Ex/Ey/Ez rows, default uses 'component' (real/imag/abs/phase)
      - for absE row, default uses abs if abs_for_absE=True
    """
    a = np.asarray(arr)

    # Special-case absE row: typically magnitude
    if rowname == "absE" and abs_for_absE:
        return np.abs(a).astype(float)

    if np.iscomplexobj(a):
        if component == "real":
            return np.real(a).astype(float)
        if component == "imag":
            return np.imag(a).astype(float)
        if component == "abs":
            return np.abs(a).astype(float)
        if component == "phase":
            return np.angle(a).astype(float)
        raise ValueError("component must be one of: 'real', 'imag', 'abs', 'phase'")

    # Already real
    return a.astype(float)

