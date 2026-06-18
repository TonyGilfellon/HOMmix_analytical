"""
Monopole pillbox crossing and on-axis loss analysis.

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

C0 = 299_792_458.0


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
) -> float:
    """
    Integrate Ez(z) cos(omega z / beta c) dz along the supplied field axis.

    The pixel-to-metre conversion is controlled only by length_m and Nz:
      dz = length_m / (Nz - 1).
    For E1/E2 use length_m = design_length_m. For mixed fields use
    length_m = design_length_m * crossing length factor.
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
    phase = omega * z_m / (beta * C0)
    return float(abs(np.trapezoid(Ez * np.cos(phase), z_m)))


def loss_from_Vz(Vz: float) -> float:
    """Original convention retained: k_loss = Vz^2 / 4."""
    return float(Vz**2 / 4.0)


def Vz_loss_from_field(
    field_or_path: np.ndarray | str | Path,
    field_saved_fname: str | None = None,
    *,
    f_mnp: float,
    length_m: float,
    beta: float = 1.0,
    axis: tuple[int | None, int | None, slice] | None = None,
    centre_z: bool = False,
) -> tuple[float, float]:
    """
    Calculate on-axis Vz and loss from a 3D Ez field or a saved .npy file.

    This replaces the old hard-coded [75, 75, :] indexing. The default uses the
    central x,y pixels of whatever grid is supplied.
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

    Vz = accelerating_voltage_from_real_Ez(Ez_axis, length_m=length_m, frequency_Hz=f_mnp, beta=beta, centre_z=centre_z)
    return Vz, loss_from_Vz(Vz)


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
        Vz, loss = Vz_loss_from_field(field_data[item["Ez_key"]], f_mnp=item["frequency_Hz"], length_m=item["length_m"])
        item["Vz_V"] = Vz
        item["loss"] = loss

    summary = {
        "crossing": crossing,
        "modes": {"E1": f"{fam_i}_{mnp_i}", "E2": f"{fam_j}_{mnp_j}"},
        "lengths_m": {"design": design_length_m, "mixed": mixed_length_m},
        "analysis": analyses,
    }
    pickle_save(summary, out_dir / "crossing_analysis.pkl")

    if make_plots:
        slice_dict = extract_3d_field_slices(field_data)
        pickle_save(slice_dict, out_dir / "slice_dict.pkl")
        plot_all_plus(slice_dict, out_dir / "TM_plus")
        plot_all_minus(slice_dict, out_dir / "TM_minus")

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


def _plot_slice_group(slice_dict: dict[str, np.ndarray], save_directory_fname: str | Path, *, op: str) -> None:
    save_directory_fname = Path(save_directory_fname)
    save_directory_fname.mkdir(parents=True, exist_ok=True)
    slice_types = ["iris_1", "iris_2", "transverse_mid", "longitudinal_mid"]
    rows = [("E1_Ex", "E2_Ex", f"Ex_{op}"), ("E1_Ey", "E2_Ey", f"Ey_{op}"), ("E1_Ez", "E2_Ez", f"Ez_{op}"), ("abs_E1", "abs_E2", f"abs_{op}")]

    for stype in slice_types:
        fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(f"{op}: {stype}")
        for r, row_keys in enumerate(rows):
            row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]
            vmax = max(float(np.nanmax(np.abs(x))) for x in row_data) or 1.0
            for c, (key, arr) in enumerate(zip(row_keys, row_data)):
                ax = axes[r, c]
                im = ax.imshow(arr.T if stype.startswith("iris") or stype == "transverse_mid" else arr, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_title(key)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.02, 0.98, f"max={np.nanmax(np.abs(arr)):.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
            fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)
        fig.savefig(save_directory_fname / f"{op}_{stype}.png", dpi=300)
        plt.close(fig)


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
    create_data: bool = True
    create_fields: bool = True
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
