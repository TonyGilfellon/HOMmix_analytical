"""
Monopole pillbox crossing and on-axis loss analysis, Method-2 aligned.

This file is intentionally standalone: the functions previously needed from
HOMmix_analytical_master_module.py have been moved here so the runnable analysis
is easier to audit and reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.optimize import brentq
from scipy.special import jn_zeros, jnp_zeros

from matplotlib.backends.backend_pdf import PdfPages

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1.0e-12


# -----------------------------------------------------------------------------
# Save/load helpers
# -----------------------------------------------------------------------------

def pickle_save(obj: Any, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename: str | Path) -> Any:
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def save_npz_dict(filename: str | Path, data: dict[str, np.ndarray]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, **data)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    with np.load(filename) as loaded:
        return {k: loaded[k] for k in loaded.files}


# -----------------------------------------------------------------------------
# Pillbox frequencies and fields
# -----------------------------------------------------------------------------

def tm_root_v_mn(m: int, n: int) -> float:
    """nth positive zero of J_m(x); n is 1-based."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    return float(jn_zeros(int(m), int(n))[-1])


def te_root_vprime_mn(m: int, n: int) -> float:
    """nth positive zero of J'_m(x); n is 1-based."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    return float(jnp_zeros(int(m), int(n))[-1])


def pillbox_radius_from_freq(f_Hz: float, c: float = C0) -> float:
    """Radius for TM010 at f_Hz."""
    return tm_root_v_mn(0, 1) * c / (2.0 * np.pi * float(f_Hz))


def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    v = tm_root_v_mn(m, n)
    return float(c / (2.0 * np.pi) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2))


def f_te(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    vprime = te_root_vprime_mn(m, n)
    return float(c / (2.0 * np.pi) * np.sqrt((vprime / R) ** 2 + (p * np.pi / L) ** 2))


def _cylindrical_tm_e_field(
    r: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    m: int,
    n: int,
    p: int,
    R: float,
    L: float,
    E0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical TM_mnp electric-field shape in a closed pillbox.

    Ez = E0 J_m(kc r) cos(m theta) cos(kz z)
    E_perp is derived from -grad_perp(Ez) * kz / kc^2, up to an overall sign.
    The sign does not affect the plus/minus-loss comparison as long as it is
    used consistently. For p=0, Er=Etheta=0, so TM010 is not repeated elsewhere.
    """
    kc = tm_root_v_mn(m, n) / R
    kz = p * np.pi / L
    x = kc * r

    Jm = special.jv(m, x)
    Jmp = special.jvp(m, x, 1)
    cos_mth = np.cos(m * theta)
    sin_mth = np.sin(m * theta)
    cos_kzz = np.cos(kz * z)
    sin_kzz = np.sin(kz * z)

    Ez = E0 * Jm * cos_mth * cos_kzz

    if p == 0:
        return np.zeros_like(Ez), np.zeros_like(Ez), Ez

    # Dimensionally consistent transverse terms. Overall sign is conventional.
    Er = -E0 * (kz / kc) * Jmp * cos_mth * sin_kzz
    with np.errstate(divide="ignore", invalid="ignore"):
        Eth = E0 * (kz / kc**2) * (m / r) * Jm * sin_mth * sin_kzz
        Eth = np.where(r == 0.0, 0.0, Eth)
    return Er, Eth, Ez


def _cylindrical_te_e_field(
    r: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    m: int,
    n: int,
    p: int,
    R: float,
    L: float,
    E0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple TE field-shape helper. Ez is exactly zero for TE modes."""
    kc = te_root_vprime_mn(m, n) / R
    kz = p * np.pi / L
    x = kc * r
    Jm = special.jv(m, x)
    Jmp = special.jvp(m, x, 1)
    sin_mth = np.sin(m * theta)
    cos_mth = np.cos(m * theta)
    sin_kzz = np.sin(kz * z)

    Er = E0 * Jm * sin_mth * sin_kzz
    Eth = E0 * Jmp * cos_mth * sin_kzz
    Ez = np.zeros_like(Er)
    return Er, Eth, Ez


def pillbox_field_voxel_grid_xyz(
    R: float,
    L: float,
    m: int,
    n: int,
    p: int,
    x_res: int,
    y_res: int,
    z_res: int,
    *,
    mode: str = "TM",
    E0: float = 1.0,
    z_range: tuple[float, float] = (0.0, 1.0),
    dtype=np.float32,
) -> dict[str, np.ndarray]:
    """Return Cartesian field arrays indexed as [x, y, z]."""
    mode = mode.upper()
    if mode not in {"TM", "TE"}:
        raise ValueError("mode must be 'TM' or 'TE'.")
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive.")
    if min(x_res, y_res, z_res) < 2:
        raise ValueError("x_res, y_res and z_res must all be >= 2.")
    if not (0.0 <= z_range[0] < z_range[1] <= 1.0):
        raise ValueError("z_range must be within [0, 1] and increasing.")

    x = np.linspace(-R, R, x_res, dtype=float)
    y = np.linspace(-R, R, y_res, dtype=float)
    z = np.linspace(z_range[0] * L, z_range[1] * L, z_res, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    mask = r <= R

    if mode == "TM":
        Er, Eth, Ez = _cylindrical_tm_e_field(r, theta, Z, m, n, p, R, L, E0)
    else:
        Er, Eth, Ez = _cylindrical_te_e_field(r, theta, Z, m, n, p, R, L, E0)

    Ex = Er * np.cos(theta) - Eth * np.sin(theta)
    Ey = Er * np.sin(theta) + Eth * np.cos(theta)
    Eperp = np.hypot(Ex, Ey)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    def inside(a: np.ndarray) -> np.ndarray:
        return np.where(mask, a, np.nan).astype(dtype, copy=False)

    return {"Ex": inside(Ex), "Ey": inside(Ey), "Ez": inside(Ez), "Eperp": inside(Eperp), "|E|": inside(Emag)}


# -----------------------------------------------------------------------------
# Mode data and crossing search
# -----------------------------------------------------------------------------

def mode_key(m: int, n: int, p: int) -> str:
    """Keep your existing mnp string convention, e.g. 010, 032."""
    return f"{int(m)}{int(n)}{int(p)}"


def parse_mode_name(name: str) -> tuple[str, str]:
    """Return (family, mnp) from 'TM_010', 'TM010' or '010'."""
    s = str(name)
    if "_" in s:
        family, mnp = s.split("_", 1)
        return family.upper(), mnp
    if s[:2].upper() in {"TM", "TE"}:
        return s[:2].upper(), s[2:]
    return "", s


def assemble_all_data_dict(
    m_max: int,
    n_max: int,
    p_max: int,
    *,
    frequency_010: float = 1.3e9,
    LF_start: float = 0.9,
    LF_stop: float = 1.1,
    param_sweep_resolution: int = 1000,
    voxel_res: int = 21,
    families: Iterable[str] = ("TM",),
    create_field_maps: bool = True,
) -> dict[str, Any]:
    """Assemble frequency sweeps and optional design-length field maps."""
    families = tuple(f.upper() for f in families)
    if any(f not in {"TM", "TE"} for f in families):
        raise ValueError("families must contain only 'TM' and/or 'TE'.")

    lambda_010 = C0 / frequency_010
    design_length_m = lambda_010 / 2.0
    R = pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, param_sweep_resolution, dtype=float)

    all_data: dict[str, Any] = {
        "meta": {
            "frequency_010_Hz": float(frequency_010),
            "lambda_010_m": float(lambda_010),
            "design_length_m": float(design_length_m),
            "pillbox_radius_m": float(R),
            "voxel_res": int(voxel_res),
        },
        "length_factor_vector": length_factors.tolist(),
    }
    for family in families:
        all_data[family] = {}

    for p in range(p_max + 1):
        for n in range(1, n_max + 1):
            for m in range(m_max + 1):
                mnp = mode_key(m, n, p)
                print(f"Building {mnp}")
                for family in families:
                    freq_fun = f_tm if family == "TM" else f_te
                    freqs = np.array([freq_fun(m, n, p, R, lf * design_length_m) for lf in length_factors], dtype=float)
                    entry: dict[str, Any] = {
                        "m": int(m),
                        "n": int(n),
                        "p": int(p),
                        "frequency_Hz": freqs.tolist(),
                        "frequency_normalised": (freqs / frequency_010).tolist(),
                        "design_frequency_Hz": float(freq_fun(m, n, p, R, design_length_m)),
                        "design_frequency_normalised": float(freq_fun(m, n, p, R, design_length_m) / frequency_010),
                    }
                    if create_field_maps:
                        entry["3D_Efield"] = pillbox_field_voxel_grid_xyz(
                            R=R,
                            L=design_length_m,
                            m=m,
                            n=n,
                            p=p,
                            x_res=voxel_res,
                            y_res=voxel_res,
                            z_res=voxel_res,
                            mode=family,
                            E0=1.0,
                            dtype=np.float32,
                        )
                    all_data[family][mnp] = entry
    return all_data


def find_mode_crossings_from_all_data(
    all_data: dict[str, Any],
    *,
    mode_type: str = "TM",
    x_target_for_ordering: float = 1.0,
    tolerance_Hz: float = 1e-6,
) -> dict[str, Any]:
    """
    Find mode crossings in frequency sweeps.

    Robust against exact zero samples, avoids duplicate crossings from adjacent
    zero-valued intervals, and uses interpolation + brentq on each sign change.
    """
    mode_type = mode_type.upper()
    if mode_type not in {"TM", "TE", "BOTH"}:
        raise ValueError("mode_type must be 'TM', 'TE' or 'BOTH'.")

    L = np.asarray(all_data["length_factor_vector"], dtype=float)
    if np.any(np.diff(L) <= 0):
        raise ValueError("length_factor_vector must be strictly increasing.")
    if not (L[0] <= x_target_for_ordering <= L[-1]):
        raise ValueError("The sweep must include x=1.0 for ordering.")

    def ordered_modes(family: str) -> list[str]:
        return sorted(
            all_data[family].keys(),
            key=lambda mnp: np.interp(x_target_for_ordering, L, np.asarray(all_data[family][mnp]["frequency_Hz"], dtype=float)),
        )

    def pair_crossings(tag_i: str, tag_j: str, fi: np.ndarray, fj: np.ndarray) -> list[dict[str, Any]]:
        g = fi - fj
        found: list[dict[str, Any]] = []
        last_lc: float | None = None

        for idx in range(len(L) - 1):
            g0, g1 = g[idx], g[idx + 1]
            if abs(g0) <= tolerance_Hz:
                Lc = float(L[idx])
            elif g0 * g1 < 0:
                def gfun(x: float) -> float:
                    return float(np.interp(x, L, fi) - np.interp(x, L, fj))
                Lc = float(brentq(gfun, float(L[idx]), float(L[idx + 1])))
            else:
                continue

            if last_lc is not None and abs(Lc - last_lc) < 10 * np.finfo(float).eps:
                continue
            last_lc = Lc
            Fc = float(np.interp(Lc, L, fi))
            found.append({
                "mode_i": tag_i,
                "mode_j": tag_j,
                "length_factor": Lc,
                "frequency_Hz": Fc,
                "frequency_normalised": Fc / float(all_data["meta"]["frequency_010_Hz"]),
            })
        return found

    results: dict[str, Any] = {}

    def process_like_family(family: str) -> None:
        crossings: dict[str, Any] = {}
        modes = ordered_modes(family)
        for i, mi in enumerate(modes):
            for mj in modes[i + 1:]:
                fi = np.asarray(all_data[family][mi]["frequency_Hz"], dtype=float)
                fj = np.asarray(all_data[family][mj]["frequency_Hz"], dtype=float)
                tag_i = f"{family}_{mi}"
                tag_j = f"{family}_{mj}"
                for c in pair_crossings(tag_i, tag_j, fi, fj):
                    key = f"{tag_i}-{tag_j}@{c['length_factor']:.12g}"
                    crossings[key] = c
        results[family] = {
            "crossings": crossings,
            "modes_that_cross": sorted({m for c in crossings.values() for m in (c["mode_i"], c["mode_j"])}),
        }

    if mode_type in {"TM", "BOTH"}:
        process_like_family("TM")
    if mode_type in {"TE", "BOTH"} and "TE" in all_data:
        process_like_family("TE")
    if mode_type == "BOTH" and "TM" in all_data and "TE" in all_data:
        crossings: dict[str, Any] = {}
        for mi in ordered_modes("TM"):
            for mj in ordered_modes("TE"):
                fi = np.asarray(all_data["TM"][mi]["frequency_Hz"], dtype=float)
                fj = np.asarray(all_data["TE"][mj]["frequency_Hz"], dtype=float)
                for c in pair_crossings(f"TM_{mi}", f"TE_{mj}", fi, fj):
                    key = f"{c['mode_i']}-{c['mode_j']}@{c['length_factor']:.12g}"
                    crossings[key] = c
        results["HYBRID"] = {
            "crossings": crossings,
            "modes_that_cross": sorted({m for c in crossings.values() for m in (c["mode_i"], c["mode_j"])}),
        }
    return results


# -----------------------------------------------------------------------------
# Crossing field data, Vz and loss
# -----------------------------------------------------------------------------

def combine_crossing_fields(E1: dict[str, np.ndarray], E2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return one unified field-data dictionary for E1, E2, plus and minus."""
    out = {
        "E1_Ex": np.asarray(E1["Ex"]), "E1_Ey": np.asarray(E1["Ey"]), "E1_Ez": np.asarray(E1["Ez"]),
        "E2_Ex": np.asarray(E2["Ex"]), "E2_Ey": np.asarray(E2["Ey"]), "E2_Ez": np.asarray(E2["Ez"]),
    }
    for comp in ("Ex", "Ey", "Ez"):
        out[f"{comp}_plus"] = out[f"E1_{comp}"] + out[f"E2_{comp}"]
        out[f"{comp}_minus"] = out[f"E1_{comp}"] - out[f"E2_{comp}"]

    out["abs_E1"] = np.sqrt(out["E1_Ex"]**2 + out["E1_Ey"]**2 + out["E1_Ez"]**2)
    out["abs_E2"] = np.sqrt(out["E2_Ex"]**2 + out["E2_Ey"]**2 + out["E2_Ez"]**2)
    out["trans_E1"] = np.hypot(out["E1_Ex"], out["E1_Ey"])
    out["trans_E2"] = np.hypot(out["E2_Ex"], out["E2_Ey"])
    out["abs_plus"] = np.sqrt(out["Ex_plus"]**2 + out["Ey_plus"]**2 + out["Ez_plus"]**2)
    out["abs_minus"] = np.sqrt(out["Ex_minus"]**2 + out["Ey_minus"]**2 + out["Ez_minus"]**2)
    out["trans_plus"] = np.hypot(out["Ex_plus"], out["Ey_plus"])
    out["trans_minus"] = np.hypot(out["Ex_minus"], out["Ey_minus"])
    return out


def get_or_create_crossing_field_data(
    E1: dict[str, np.ndarray],
    E2: dict[str, np.ndarray],
    array_path: str | Path,
    *,
    create_fields: bool = True,
) -> dict[str, np.ndarray]:
    """Save/load one NPZ per crossing instead of many loose .npy files."""
    array_path = Path(array_path)
    npz_path = array_path / "field_data.npz"
    if create_fields or not npz_path.exists():
        data = combine_crossing_fields(E1, E2)
        save_npz_dict(npz_path, data)
        return data
    return load_npz_dict(npz_path)


def accelerating_voltage_from_real_Ez(
    Ez_V_per_m: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    beta: float = 1.0,
    centre_z: bool = False,
) -> complex:
    """
    Complex transit-time-corrected on-axis voltage.

    This deliberately uses only Ez(0,0,z), but follows the same transit-time
    convention as the current Method-2 comparison script:

        Vz = integral_0^L Ez(0,0,z) exp(i omega z / beta c) dz.

    The returned value is complex.  Use abs(Vz) for the accelerating-voltage
    magnitude.  centre_z=False is the default to match
    compare_Ez_loss_kick_focus_parent_degen_freqs_localLS.py.
    """
    Ez = np.asarray(Ez_V_per_m, dtype=float)
    if Ez.ndim != 1:
        raise ValueError("Ez_V_per_m must be a 1D on-axis trace.")
    if Ez.size < 2:
        raise ValueError("Ez_V_per_m must contain at least two pixels.")

    z_m = np.linspace(0.0, float(length_m), Ez.size)
    if centre_z:
        z_m = z_m - 0.5 * float(length_m)

    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = omega * z_m / (float(beta) * C0)
    return complex(np.trapezoid(Ez * np.exp(1j * phase), z_m))


def loss_from_Vz(Vz: complex, U_J: float) -> float:
    """Return stored-energy-normalised k_parallel in V/C."""
    U_J = float(U_J)
    if U_J <= 0.0 or not np.isfinite(U_J):
        raise ValueError(f"Stored energy U_J must be positive and finite, got {U_J!r}.")
    return float(abs(Vz) ** 2 / (4.0 * U_J))



def _field_spacing_and_mask(
    shape: tuple[int, int, int],
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> tuple[float, float, float, np.ndarray, float, float, float]:
    """Return dx, dy, dz and the cylindrical transverse mask.

    For the standard 151x151 maps the physical axis is array[75,75,:]
    and the cylinder radius is 75 pixels from the axis, so dx = Req_m / 75.
    """
    nx, ny, nz = shape
    if min(nx, ny, nz) < 2:
        raise ValueError("Field arrays must have at least two points on every axis.")

    if axis_i is None:
        axis_i = float(nx // 2)
    if axis_j is None:
        axis_j = float(ny // 2)
    if radius_pixels is None:
        radius_pixels = float(min(axis_i, axis_j, nx - 1 - axis_i, ny - 1 - axis_j))
    if radius_pixels <= 0.0:
        raise ValueError(f"radius_pixels must be positive, got {radius_pixels!r}.")

    dx = float(Req_m) / float(radius_pixels)
    dy = dx
    dz = float(length_m) / (nz - 1)

    x = (np.arange(nx, dtype=float) - float(axis_i)) * dx
    y = (np.arange(ny, dtype=float) - float(axis_j)) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask_xy = (X * X + Y * Y) <= float(Req_m) * float(Req_m)
    return dx, dy, dz, mask_xy, float(axis_i), float(axis_j), float(radius_pixels)


def stored_energy_from_Ez_only_peak_equivalent(
    Ez: np.ndarray,
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> float:
    """Old Ez-only peak-electric-energy proxy: 0.5 eps0 integral Ez^2 dV.

    This is retained only as a diagnostic. It is not the CST stored energy.
    """
    Ez = np.nan_to_num(np.asarray(Ez, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Ez.ndim != 3:
        raise ValueError(f"Ez must be a 3D array, got shape {Ez.shape}.")
    dx, dy, dz, mask_xy, *_ = _field_spacing_and_mask(
        Ez.shape,
        Req_m=Req_m,
        length_m=length_m,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )
    return float(0.5 * EPS0 * np.sum(Ez * Ez * mask_xy[:, :, None]) * dx * dy * dz)


def stored_energy_from_Etotal_CST_equivalent(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> dict[str, float]:
    """CST-equivalent stored energy using the electric field components.

    CST eigenmode stored energy is the total time-averaged electromagnetic
    stored energy,

        U_CST = 1/4 integral (eps |E|^2 + mu |H|^2) dV.

    For a lossless resonant eigenmode, the time-averaged electric and magnetic
    energies are equal, so this can be computed from the electric field alone as

        U_CST = 2 U_E,time = 0.5 eps0 integral |E|^2 dV,

    provided Ex,Ey,Ez are the peak/phasor electric-field amplitudes in the same
    normalisation.

    The returned dictionary also includes diagnostic electric-only components.
    """
    Ex = np.nan_to_num(np.asarray(Ex, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ey = np.nan_to_num(np.asarray(Ey, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ez = np.nan_to_num(np.asarray(Ez, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Ex.shape != Ey.shape or Ex.shape != Ez.shape or Ex.ndim != 3:
        raise ValueError(f"Ex, Ey, Ez must be matching 3D arrays; got {Ex.shape}, {Ey.shape}, {Ez.shape}.")

    dx, dy, dz, mask_xy, axis_i, axis_j, radius_pixels = _field_spacing_and_mask(
        Ez.shape,
        Req_m=Req_m,
        length_m=length_m,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )
    dV = dx * dy * dz
    mask = mask_xy[:, :, None]

    int_Ex2 = float(np.sum(Ex * Ex * mask) * dV)
    int_Ey2 = float(np.sum(Ey * Ey * mask) * dV)
    int_Ez2 = float(np.sum(Ez * Ez * mask) * dV)
    int_Etot2 = int_Ex2 + int_Ey2 + int_Ez2

    U_E_time = 0.25 * EPS0 * int_Etot2
    U_CST = 2.0 * U_E_time

    if not np.isfinite(U_CST) or U_CST <= 0.0:
        raise ValueError(f"Calculated non-positive CST-equivalent stored energy U={U_CST!r}.")

    return {
        "U_CST_J": float(U_CST),
        "U_Etotal_time_average_J": float(U_E_time),
        "U_Etotal_peak_J": float(0.5 * EPS0 * int_Etot2),
        "U_Ez_only_time_average_J": float(0.25 * EPS0 * int_Ez2),
        "U_Ez_only_peak_J": float(0.5 * EPS0 * int_Ez2),
        "int_Ex2_dV": int_Ex2,
        "int_Ey2_dV": int_Ey2,
        "int_Ez2_dV": int_Ez2,
        "int_Etotal2_dV": int_Etot2,
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": float(dz),
        "axis_i": float(axis_i),
        "axis_j": float(axis_j),
        "radius_pixels": float(radius_pixels),
    }


def stored_energy_from_components(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> float:
    """Return CST-equivalent total time-averaged stored energy in joules.

    This replaces the previous Ez-only proxy. It uses Ex,Ey,Ez and assumes a
    lossless eigenmode so that U_CST = 2 U_E,time.
    """
    return stored_energy_from_Etotal_CST_equivalent(
        Ex, Ey, Ez,
        Req_m=Req_m,
        length_m=length_m,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )["U_CST_J"]


def stored_energy_from_field_data(
    field_data: dict[str, np.ndarray],
    name: str,
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
    return_diagnostics: bool = False,
) -> float | dict[str, float]:
    """CST-equivalent stored energy for E1, E2, plus, or minus.

    Uses all three electric components and returns the total time-averaged
    electromagnetic stored energy equivalent to CST's eigenmode stored energy:
    U_CST = 0.5 eps0 integral |E|^2 dV for lossless resonant fields.
    """
    if name == "E1":
        keys = ("E1_Ex", "E1_Ey", "E1_Ez")
    elif name == "E2":
        keys = ("E2_Ex", "E2_Ey", "E2_Ez")
    elif name == "plus":
        keys = ("Ex_plus", "Ey_plus", "Ez_plus")
    elif name == "minus":
        keys = ("Ex_minus", "Ey_minus", "Ez_minus")
    else:
        raise ValueError(f"Unknown field name {name!r}; expected E1, E2, plus or minus.")

    diag = stored_energy_from_Etotal_CST_equivalent(
        field_data[keys[0]],
        field_data[keys[1]],
        field_data[keys[2]],
        Req_m=Req_m,
        length_m=length_m,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )
    return diag if return_diagnostics else diag["U_CST_J"]

def Vz_loss_from_field(
    field_or_path: np.ndarray | str | Path,
    field_saved_fname: str | None = None,
    *,
    f_mnp: float,
    length_m: float,
    U_J: float,
    beta: float = 1.0,
    axis: tuple[int | None, int | None, slice] | None = None,
    centre_z: bool = False,
) -> tuple[complex, float]:
    """
    Calculate complex on-axis Vz and U-normalised k_parallel from a 3D Ez field.

    This deliberately uses only Ez(0,0,z), not a transverse polynomial fit.
    The default axis is the central x,y pixel of the supplied grid, e.g.
    array[75, 75, :] for a 151^3 field map.
    """
    if isinstance(field_or_path, np.ndarray):
        Ez = np.asarray(field_or_path)
    else:
        if field_saved_fname is None:
            Ez = np.load(Path(field_or_path))
        else:
            Ez = np.load(Path(field_or_path) / field_saved_fname)

    if Ez.ndim != 3:
        raise ValueError(f"Expected a 3D Ez array, got shape {Ez.shape}.")

    x_mid, y_mid = Ez.shape[0] // 2, Ez.shape[1] // 2
    if axis is None:
        Ez_axis = Ez[x_mid, y_mid, :]
    else:
        ix = x_mid if axis[0] is None else axis[0]
        iy = y_mid if axis[1] is None else axis[1]
        Ez_axis = Ez[ix, iy, axis[2]]

    Vz = accelerating_voltage_from_real_Ez(
        Ez_axis,
        length_m=length_m,
        frequency_Hz=f_mnp,
        beta=beta,
        centre_z=centre_z,
    )
    return Vz, loss_from_Vz(Vz, U_J)

def analyse_crossing(
    all_data: dict[str, Any],
    crossing: dict[str, Any],
    out_dir: str | Path,
    *,
    create_fields: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    """One crossing in, one summary dict out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fam_i, mnp_i = parse_mode_name(crossing["mode_i"])
    fam_j, mnp_j = parse_mode_name(crossing["mode_j"])
    if fam_i != "TM" or fam_j != "TM":
        raise ValueError("This monopole loss workflow expects TM-TM crossings.")

    E1 = all_data[fam_i][mnp_i]["3D_Efield"]
    E2 = all_data[fam_j][mnp_j]["3D_Efield"]
    field_data = get_or_create_crossing_field_data(E1, E2, out_dir, create_fields=create_fields)

    design_length_m = float(all_data["meta"]["design_length_m"])
    mixed_length_m = design_length_m * float(crossing["length_factor"])
    Req_m = float(all_data["meta"]["pillbox_radius_m"])
    # Match Method-2 convention for 151^3 maps: axis is array[75,75,:]
    # and the cylinder radius is 75 pixels from the axis.  The fallbacks also
    # work for other odd grid sizes.
    voxel_res = int(all_data.get("meta", {}).get("voxel_res", field_data["E1_Ez"].shape[0]))
    axis_i = float((voxel_res - 1) // 2)
    axis_j = float((voxel_res - 1) // 2)
    radius_pixels = axis_i

    analyses = {
        "E1": {
            "Ez_key": "E1_Ez",
            "frequency_Hz": float(all_data[fam_i][mnp_i]["design_frequency_Hz"]),
            "length_m": design_length_m,
        },
        "E2": {
            "Ez_key": "E2_Ez",
            "frequency_Hz": float(all_data[fam_j][mnp_j]["design_frequency_Hz"]),
            "length_m": design_length_m,
        },
        "plus": {"Ez_key": "Ez_plus", "frequency_Hz": float(crossing["frequency_Hz"]), "length_m": mixed_length_m},
        "minus": {"Ez_key": "Ez_minus", "frequency_Hz": float(crossing["frequency_Hz"]), "length_m": mixed_length_m},
    }

    for name, item in analyses.items():
        energy_diag = stored_energy_from_field_data(
            field_data,
            name,
            Req_m=Req_m,
            length_m=item["length_m"],
            axis_i=axis_i,
            axis_j=axis_j,
            radius_pixels=radius_pixels,
            return_diagnostics=True,
        )
        U_J = float(energy_diag["U_CST_J"])
        Vz, k_parallel = Vz_loss_from_field(
            field_data[item["Ez_key"]],
            f_mnp=item["frequency_Hz"],
            length_m=item["length_m"],
            U_J=U_J,
        )

        print(f"\n{mnp_i = }")
        print(f"{mnp_j = }")
        print(f'{float(all_data[fam_i][mnp_i]["design_frequency_Hz"]) = }')
        print(f'{float(all_data[fam_j][mnp_j]["design_frequency_Hz"]) = }')
        print(f'{float(crossing["frequency_Hz"]) = }')
        print(f"{design_length_m = }")
        print(f"{name = }")
        print(f"{U_J = }  # U_CST = 0.5 eps0 integral |E|^2 dV")
        print(f"U_Etotal_time_average_J = {energy_diag['U_Etotal_time_average_J']}")
        print(f"U_Ez_only_peak_J = {energy_diag['U_Ez_only_peak_J']}")
        print(f"{Vz = }")
        print(f"{k_parallel = }")
        print(f"{k_parallel * PC = }\n")


        item["U_J"] = U_J
        item["U_CST_J"] = U_J
        item["stored_energy_diagnostics"] = energy_diag
        item["Vz_complex_V"] = Vz
        item["Vz_abs_V"] = float(abs(Vz))
        item["k_parallel_V_per_C"] = k_parallel
        item["k_parallel_V_per_pC"] = k_parallel * PC
        # Backwards-compatible aliases.  Use V/pC here because older tables
        # labelled this quantity as V/pC/m-like rather than V/C.
        item["Vz_V"] = float(abs(Vz))
        item["loss"] = k_parallel * PC
        item["loss_V_per_pC"] = k_parallel * PC
        item["normalisation"] = "CST-equivalent total time-averaged U; k_parallel=|Vz|^2/(4U_CST)"
        item["axis_i"] = axis_i
        item["axis_j"] = axis_j
        item["radius_pixels"] = radius_pixels

    summary_pdf = None

    if make_plots:
        slice_dict = extract_3d_field_slices(field_data)
        pickle_save(slice_dict, out_dir / "slice_dict.pkl")
        plot_all_plus(slice_dict, out_dir / "TM_plus")
        plot_all_minus(slice_dict, out_dir / "TM_minus")
        plot_all_plus_minus_combined(slice_dict, out_dir / "TM_combined")

        summary_pdf = save_four_slice_pdfs_and_merge(
            slice_dict=slice_dict,
            out_dir=out_dir / "slice_summary_pdfs",
            merged_pdf_name=f"TM{mnp_i}_TM{mnp_j}_field_summary.pdf",
        )

    summary = {
        "crossing": crossing,
        "modes": {
            "E1": f"{fam_i}_{mnp_i}",
            "E2": f"{fam_j}_{mnp_j}",
        },
        "lengths_m": {
            "design": design_length_m,
            "mixed": mixed_length_m,
        },
        "pillbox_radius_m": Req_m,
        "stored_energy_convention": "CST-equivalent total time-averaged U = 0.5 eps0 integral |E|^2 dV (electric-field-only equivalent for lossless eigenmodes)",
        "analysis": analyses,
        "files": {
            "slice_dict": str(out_dir / "slice_dict.pkl"),
            "merged_slice_pdf": str(summary_pdf) if summary_pdf is not None else None,
            "slice_summary_dir": str(out_dir / "slice_summary_pdfs"),
        },
    }

    pickle_save(summary, out_dir / "crossing_analysis.pkl")

    return summary


# -----------------------------------------------------------------------------
# Slices and compact plotting
# -----------------------------------------------------------------------------

def extract_3d_field_slices(data: dict[str, np.ndarray], field_keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    if field_keys is None:
        field_keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 3]
    out: dict[str, np.ndarray] = {}
    for key in field_keys:
        field = np.asarray(data[key])
        if field.ndim != 3:
            continue
        nx, ny, nz = field.shape
        out[f"{key}_iris_1"] = field[:, :, 0]
        out[f"{key}_iris_2"] = field[:, :, nz - 1]
        out[f"{key}_transverse_mid"] = field[:, :, nz // 2]
        out[f"{key}_longitudinal_mid"] = field[nx // 2, :, :]
    return out


def _real_image(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return np.real(a) if np.iscomplexobj(a) else a


def plot_all_plus(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path) -> None:
    _plot_slice_group(slice_dict, save_directory_fname, op="plus")


def plot_all_minus(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path) -> None:
    _plot_slice_group(slice_dict, save_directory_fname, op="minus")


def plot_all_plus_minus_combined(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path) -> None:
    _plot_slice_group_combined(slice_dict, save_directory_fname)


def _plot_slice_group(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path, *, op: str) -> None:
    save_directory_fname = Path(save_directory_fname)
    save_directory_fname.mkdir(parents=True, exist_ok=True)

    slice_types = ["iris_1", "iris_2", "transverse_mid", "longitudinal_mid"]
    rows = [
        ("E1_Ex", "E2_Ex", f"Ex_{op}"),
        ("E1_Ey", "E2_Ey", f"Ey_{op}"),
        ("E1_Ez", "E2_Ez", f"Ez_{op}"),
        ("abs_E1", "abs_E2", f"abs_{op}"),
    ]

    for stype in slice_types:
        fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(f"{op}: {stype}")

        for r, row_keys in enumerate(rows):
            row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]

            is_abs_row = (r == 3)
            if is_abs_row:
                vmax = max(float(np.nanmax(x)) for x in row_data) or 1.0
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                vmin = 0.0
                cmap = "viridis"
            else:
                vmax = max(float(np.nanmax(np.abs(x))) for x in row_data) or 1.0
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                vmin = -vmax
                cmap = "RdBu_r"

            for c, (key, arr) in enumerate(zip(row_keys, row_data)):
                ax = axes[r, c]
                plot_arr = arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr
                im = ax.imshow(
                    plot_arr,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )
                ax.set_title(key)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.02,
                    0.98,
                    f"max={np.nanmax(np.abs(arr)):.2e}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
                )

            fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)

        fig.savefig(save_directory_fname / f"{op}_{stype}.png", dpi=300)
        plt.close(fig)

def _plot_slice_group_combined_A4_pdf(
    slice_dict: dict[str, np.ndarray],
    save_directory_fname: str | Path,
    pdf_name: str = "combined_A4_summary.pdf",
) -> None:

    save_directory_fname = Path(save_directory_fname)
    save_directory_fname.mkdir(parents=True, exist_ok=True)

    slice_types = ["iris_1", "iris_2", "transverse_mid", "longitudinal_mid"]

    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]

    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [r"$E_z/E_{z,\mathrm{ref}}$", r"$|E|/|E|_{\mathrm{ref}}$"]

    block_titles = {
        "iris_1": "Iris 1",
        "iris_2": "Iris 2",
        "transverse_mid": "Transverse mid-plane",
        "longitudinal_mid": "Longitudinal mid-plane",
    }

    def _real_image(arr):
        return np.real(np.asarray(arr))

    def _plot_orient(arr, stype):
        return arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr

    def _safe_ref(arrays):
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        return ref if np.isfinite(ref) and ref > 0.0 else 1.0

    def _safe_vmax(arrays, ref):
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        return max(1.0, scaled_max) if np.isfinite(scaled_max) and scaled_max > 0.0 else 1.0

    # Shorter than A4 so LaTeX has room above for a table.
    fig = plt.figure(figsize=(8.27, 7.2), constrained_layout=False)

    outer = fig.add_gridspec(
        4,
        1,
        left=0.08,
        right=0.96,
        bottom=0.03,
        top=0.96,
        hspace=0.30,
    )

    for block_index, stype in enumerate(slice_types):
        inner = outer[block_index, 0].subgridspec(
            2,
            5,
            width_ratios=[1, 1, 1, 1, 0.03],
            height_ratios=[1, 1],
            wspace=0.00,
            hspace=0.1,
        )

        axes = np.empty((2, 4), dtype=object)

        parent_ez_ref = _safe_ref([
            _real_image(slice_dict[f"E1_Ez_{stype}"]),
            _real_image(slice_dict[f"E2_Ez_{stype}"]),
        ])

        parent_abs_ref = _safe_ref([
            _real_image(slice_dict[f"abs_E1_{stype}"]),
            _real_image(slice_dict[f"abs_E2_{stype}"]),
        ])

        for r, row_keys in enumerate(rows):
            raw_row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]
            is_abs_row = r == 1

            if is_abs_row:
                ref = parent_abs_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmin, vmax, cmap = 0.0, _safe_vmax(raw_row_data, ref), "viridis"
            else:
                ref = parent_ez_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmax = _safe_vmax(raw_row_data, ref)
                vmin, cmap = -vmax, "RdBu_r"

            im = None

            for c, (arr_raw, arr_scaled) in enumerate(zip(raw_row_data, row_data)):
                ax = fig.add_subplot(inner[r, c])
                axes[r, c] = ax

                plot_arr = _plot_orient(arr_scaled, stype)

                im = ax.imshow(
                    plot_arr,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.margins(0)
                ax.set_anchor("C")

                # Make the axes exactly the size of the image.
                ny, nx = plot_arr.shape
                ax.set_box_aspect(ny / nx)
                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_linewidth(0.2)

                if r == 0:
                    ax.set_title(column_titles[c], fontsize=9, fontweight="bold", pad=2)

                if c == 0:
                    ax.set_ylabel(row_titles[r], fontsize=8, rotation=90, labelpad=7)

                ax.text(
                    0.04,
                    0.96,
                    f"{np.nanmax(np.abs(arr_scaled)):.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
                )

            cax = fig.add_subplot(inner[r, 4])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)

        axes[0, 0].text(
            0.0,
            1.22,
            block_titles.get(stype, stype),
            transform=axes[0, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )

    out_file = save_directory_fname / pdf_name
    fig.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote {out_file}")

def _plot_slice_group_combined(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path) -> None:
    """Save 4x4 combined plots with columns [E1, E2, E+, E-].

    Colour meaning:
        - Ex, Ey, Ez rows are divided by max(abs(E1_Ez), abs(E2_Ez)).
          Therefore colourbar value 1 means the parent Ez reference amplitude.
        - |E| row is divided by max(|E1|, |E2|).
          Therefore colourbar value 1 means the parent |E| reference amplitude.
        - The colourbar limits are allowed to extend beyond +/-1 or 1 if E+
          or E- exceed the parent reference.
    """
    save_directory_fname = Path(save_directory_fname)
    save_directory_fname.mkdir(parents=True, exist_ok=True)

    slice_types = ["iris_1", "iris_2", "transverse_mid", "longitudinal_mid"]
    rows = [
        ("E1_Ex", "E2_Ex", "Ex_plus", "Ex_minus"),
        ("E1_Ey", "E2_Ey", "Ey_plus", "Ey_minus"),
        ("E1_Ez", "E2_Ez", "Ez_plus", "Ez_minus"),
        ("abs_E1", "abs_E2", "abs_plus", "abs_minus"),
    ]

    column_titles = ["E₁", "E₂", "E₊", "E₋"]
    row_titles = [r"$E_x/E_{z,\mathrm{ref}}$", r"$E_y/E_{z,\mathrm{ref}}$", r"$E_z/E_{z,\mathrm{ref}}$", r"$|E|/|E|_{\mathrm{ref}}$"]

    def _plot_orient(arr: np.ndarray, stype: str) -> np.ndarray:
        return arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr

    def _safe_ref(arrays: list[np.ndarray]) -> float:
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        if not np.isfinite(ref) or ref <= 0.0:
            ref = 1.0
        return ref

    def _safe_vmax(arrays: list[np.ndarray], ref: float, *, abs_only: bool) -> float:
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        if not np.isfinite(scaled_max) or scaled_max <= 0.0:
            scaled_max = 1.0
        return max(1.0, scaled_max)

    for stype in slice_types:
        fig, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"plus/minus comparison: {stype}")

        parent_ez_ref = _safe_ref([
            _real_image(slice_dict[f"E1_Ez_{stype}"]),
            _real_image(slice_dict[f"E2_Ez_{stype}"]),
        ])

        parent_abs_ref = _safe_ref([
            _real_image(slice_dict[f"abs_E1_{stype}"]),
            _real_image(slice_dict[f"abs_E2_{stype}"]),
        ])

        for r, row_keys in enumerate(rows):
            raw_row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]

            is_abs_row = r == 3
            if is_abs_row:
                ref = parent_abs_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmin = 0.0
                vmax = _safe_vmax(raw_row_data, ref, abs_only=True)
                cmap = "viridis"
            else:
                ref = parent_ez_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmax = _safe_vmax(raw_row_data, ref, abs_only=False)
                vmin = -vmax
                cmap = "RdBu_r"

            for c, (key, arr_raw, arr_scaled) in enumerate(zip(row_keys, raw_row_data, row_data)):
                ax = axes[r, c]
                plot_arr = _plot_orient(arr_scaled, stype)

                im = ax.imshow(
                    plot_arr,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )

                if r == 0:
                    ax.text(
                        0.5,
                        1.02,
                        column_titles[c],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontstyle="normal",
                        fontweight="bold",
                        zorder=100,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.90,
                            pad=2.0,
                        ),
                        clip_on=False,
                    )

                if c == 0:
                    ax.text(
                        -0.12,
                        0.5,
                        row_titles[r],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=12,
                        zorder=100,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.90,
                            pad=2.0,
                        ),
                        clip_on=False,
                    )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.02,
                    0.98,
                    f"max={np.nanmax(np.abs(arr_raw)):.2e}\n"
                    f"norm={np.nanmax(np.abs(arr_scaled)):.2g}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
                )

            fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)

        fig.savefig(save_directory_fname / f"{stype}.png", dpi=300)
        plt.close(fig)




def _save_one_2x4_slice_pdf(
    slice_dict: dict[str, np.ndarray],
    stype: str,
    out_dir: str | Path,
    pdf_name: str,
) -> Path:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]

    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]

    title_map = {
        "iris_1": "Transverse iris 1",
        "iris_2": "Transverse iris 2",
        "longitudinal_mid": "Longitudinal vertical mid-plane",
        "transverse_mid": "Transverse mid-plane",
    }

    def _real_image(arr):
        return np.real(np.asarray(arr))

    def _plot_orient(arr, stype):
        return arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr

    def _safe_ref(arrays):
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        return ref if np.isfinite(ref) and ref > 0 else 1.0

    def _safe_vmax(arrays, ref):
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        return max(1.0, scaled_max) if np.isfinite(scaled_max) and scaled_max > 0 else 1.0

    parent_ez_ref = _safe_ref([
        _real_image(slice_dict[f"E1_Ez_{stype}"]),
        _real_image(slice_dict[f"E2_Ez_{stype}"]),
    ])

    parent_abs_ref = _safe_ref([
        _real_image(slice_dict[f"abs_E1_{stype}"]),
        _real_image(slice_dict[f"abs_E2_{stype}"]),
    ])

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=False)

    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 0.045],
        left=0.075,
        right=0.965,
        bottom=0.08,
        top=0.86,
        wspace=0.0,
        hspace=0.1,
    )

    fig.suptitle(title_map.get(stype, stype), fontsize=12, fontweight="bold", y=0.965)

    for r, row_keys in enumerate(rows):
        raw_row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]

        if r == 0:
            ref = parent_ez_ref
            row_data = [arr / ref for arr in raw_row_data]
            vmax = _safe_vmax(raw_row_data, ref)
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            ref = parent_abs_ref
            row_data = [arr / ref for arr in raw_row_data]
            vmin = 0.0
            vmax = _safe_vmax(raw_row_data, ref)
            cmap = "viridis"

        im = None

        for c, (arr_raw, arr_scaled) in enumerate(zip(raw_row_data, row_data)):
            ax = fig.add_subplot(gs[r, c])
            ax.set_anchor("C")

            plot_arr = _plot_orient(arr_scaled, stype)

            im = ax.imshow(
                plot_arr,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

            ny, nx = plot_arr.shape
            ax.set_box_aspect(ny / nx)
            ax.margins(0)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(column_titles[c], fontsize=10, fontweight="bold", pad=2)

            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9, rotation=90, labelpad=7)

            ax.text(
                0.04,
                0.96,
                f"{np.nanmax(np.abs(arr_scaled)):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
            )

        cax = fig.add_subplot(gs[r, 4])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7, length=2, pad=1)

    out_file = out_dir / pdf_name
    fig.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote {out_file}")
    return out_file

def save_four_slice_pdfs_and_merge(
    slice_dict: dict[str, np.ndarray],
    out_dir: str | Path,
    merged_pdf_name: str = "combined_four_slice_summary.pdf",
) -> Path:
    """
    Save one tall single-page PDF containing all four 2x4 slice summaries.

    Output location and final filename are unchanged, e.g.
        TM012_TM020/slice_summary_pdfs/TM012_TM020_field_summary.pdf

    The old individual page PDFs are no longer needed.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("iris_1", "Transverse iris 1"),
        ("iris_2", "Transverse iris 2"),
        ("longitudinal_mid", "Longitudinal vertical mid-plane"),
        ("transverse_mid", "Transverse mid-plane"),
    ]

    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]

    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]

    def _real_image(arr):
        return np.real(np.asarray(arr))

    def _plot_orient(arr, stype):
        return arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr

    def _safe_ref(arrays):
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        return ref if np.isfinite(ref) and ref > 0 else 1.0

    def _safe_vmax(arrays, ref):
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        return max(1.0, scaled_max) if np.isfinite(scaled_max) and scaled_max > 0 else 1.0

    # Tall but still comfortable when included at ~0.75 textwidth in LaTeX.
    fig = plt.figure(figsize=(7.2, 12.4), constrained_layout=False)

    outer = fig.add_gridspec(
        4,
        1,
        left=0.075,
        right=0.965,
        bottom=0.025,
        top=0.975,
        hspace=0.18,
    )

    for block_idx, (stype, block_title) in enumerate(specs):
        gs = outer[block_idx, 0].subgridspec(
            2,
            5,
            width_ratios=[1, 1, 1, 1, 0.045],
            height_ratios=[1, 1],
            wspace=0.0,
            hspace=0.08,
        )

        parent_ez_ref = _safe_ref([
            _real_image(slice_dict[f"E1_Ez_{stype}"]),
            _real_image(slice_dict[f"E2_Ez_{stype}"]),
        ])

        parent_abs_ref = _safe_ref([
            _real_image(slice_dict[f"abs_E1_{stype}"]),
            _real_image(slice_dict[f"abs_E2_{stype}"]),
        ])

        first_ax = None

        for r, row_keys in enumerate(rows):
            raw_row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]

            if r == 0:
                ref = parent_ez_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmax = _safe_vmax(raw_row_data, ref)
                vmin = -vmax
                cmap = "RdBu_r"
            else:
                ref = parent_abs_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmin = 0.0
                vmax = _safe_vmax(raw_row_data, ref)
                cmap = "viridis"

            im = None

            for c, (arr_raw, arr_scaled) in enumerate(zip(raw_row_data, row_data)):
                ax = fig.add_subplot(gs[r, c])

                if first_ax is None:
                    first_ax = ax

                ax.set_anchor("C")

                plot_arr = _plot_orient(arr_scaled, stype)

                im = ax.imshow(
                    plot_arr,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )

                ny, nx = plot_arr.shape
                ax.set_box_aspect(ny / nx)
                ax.margins(0)

                ax.set_xticks([])
                ax.set_yticks([])

                if r == 0:
                    ax.set_title(
                        column_titles[c],
                        fontsize=10,
                        fontweight="bold",
                        pad=2,
                    )

                if c == 0:
                    ax.set_ylabel(
                        row_titles[r],
                        fontsize=9,
                        rotation=90,
                        labelpad=7,
                    )

                ax.text(
                    0.04,
                    0.96,
                    f"{np.nanmax(np.abs(arr_scaled)):.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                        pad=1.2,
                    ),
                )

            cax = fig.add_subplot(gs[r, 4])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=7, length=2, pad=1)

        if first_ax is not None:
            first_ax.text(
                0.0,
                1.12,
                block_title,
                transform=first_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                clip_on=False,
            )

    out_file = out_dir / merged_pdf_name
    fig.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote single-page summary PDF {out_file}")
    return out_file
# -----------------------------------------------------------------------------
# Main script configuration
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    datapath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical")
    savepath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_monopoles")


    m_max: int = 0
    n_max: int = 3
    p_max: int = 3
    voxel_res: int = 151
    f_010: float = 1.3e9
    LF_start: float = 0.7
    LF_stop: float = 1.3
    param_sweep_resolution: int = 1000
    create_data: bool = False
    create_fields: bool = False
    make_plots: bool = True

    @property
    def data_file(self) -> Path:
        return self.datapath / f"TMm0_TMm0_data_dict_{self.voxel_res}x{self.voxel_res}x{self.voxel_res}.pkl"


def main(cfg: RunConfig = RunConfig()) -> list[dict[str, Any]]:
    cfg.datapath.mkdir(parents=True, exist_ok=True)
    cfg.savepath.mkdir(parents=True, exist_ok=True)

    if cfg.create_data:
        data_dict = assemble_all_data_dict(
            cfg.m_max,
            cfg.n_max,
            cfg.p_max,
            frequency_010=cfg.f_010,
            LF_start=cfg.LF_start,
            LF_stop=cfg.LF_stop,
            param_sweep_resolution=cfg.param_sweep_resolution,
            voxel_res=cfg.voxel_res,
            families=("TM",),
            create_field_maps=True,
        )
        pickle_save(data_dict, cfg.data_file)
    else:
        data_dict = pickle_load(cfg.data_file)

    crossing_results = find_mode_crossings_from_all_data(data_dict, mode_type="TM")
    pickle_save(crossing_results, cfg.savepath / "crossing_results.pkl")

    summaries: list[dict[str, Any]] = []
    for key, crossing in crossing_results["TM"]["crossings"].items():
        _, mode_i = parse_mode_name(crossing["mode_i"])
        _, mode_j = parse_mode_name(crossing["mode_j"])
        print(f"Analysing {key}: LF={crossing['length_factor']:.8g}, f={crossing['frequency_Hz'] / 1e9:.6g} GHz")
        out_dir = cfg.savepath / f"TM{mode_i}_TM{mode_j}"
        summaries.append(analyse_crossing(data_dict, crossing, out_dir, create_fields=cfg.create_fields, make_plots=cfg.make_plots))

    pickle_save(summaries, cfg.savepath / "all_crossing_analyses.pkl")
    return summaries


if __name__ == "__main__":
    main()
