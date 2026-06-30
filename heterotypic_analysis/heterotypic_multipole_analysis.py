"""Analyse heterotypic mixed fields saved by heterotypic_crossing_field_mixing.py.

The mixed fields are not pure monopole/dipole/quadrupole modes.  This module
therefore characterises each parent and mixed field by a near-axis Taylor / multipole
expansion of the complex longitudinal voltage

    Vz(x,y) = integral Ez(x,y,z) exp(i omega z / beta c) dz.

The fitted coefficients are then converted, using Panofsky-Wenzel, into:
    - monopole voltage / loss-like term,
    - dipole kick vector,
    - quadrupole focusing matrix.

Expected crossing folder contents from heterotypic_crossing_field_mixing.py:
    field_data.npz
    heterotypic_crossing_analysis.pkl

Optional family data dictionaries from the earlier mono/di/quad runs are used to
obtain parent design frequencies.  If they are not supplied, parent fields are
analysed at the crossing frequency as a fallback, which is less strict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np

C0 = 299_792_458.0


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_save(obj: Any, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# Voltage map and near-axis expansion
# -----------------------------------------------------------------------------

def centred_transverse_coords(nx: int, ny: int, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinate vectors for field[x,y,z] over [-R,R]."""
    x = np.linspace(-float(radius_m), float(radius_m), int(nx))
    y = np.linspace(-float(radius_m), float(radius_m), int(ny))
    return x, y


def longitudinal_coords(nz: int, length_m: float, *, centred: bool = True) -> np.ndarray:
    """Return z positions for integration.

    centred=True uses z in [-L/2,L/2], which is convenient for a standing-wave
    phase convention and matches the dipole/quadrupole helper scripts.
    """
    z = np.linspace(0.0, float(length_m), int(nz))
    if centred:
        z = z - 0.5 * float(length_m)
    return z


def complex_voltage_map_from_Ez(
    Ez_xyz: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    beta: float = 1.0,
    centred_z: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate Ez along z and return Vz(x,y) and z_m.

    Parameters
    ----------
    Ez_xyz:
        3D longitudinal field map with convention Ez[x_index, y_index, z_index].
    length_m:
        Physical length represented by the z-axis.  Use design length for E1/E2
        and design length * length_factor for plus/minus mixed fields.
    frequency_Hz:
        Mode frequency used in the transit-time phase factor.

    Returns
    -------
    Vz_xy:
        Complex longitudinal voltage map in V for a field map in V/m.
    z_m:
        Longitudinal coordinate vector used in the integration.
    """
    Ez = np.nan_to_num(np.asarray(Ez_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Ez.ndim != 3:
        raise ValueError(f"Ez_xyz must be 3D, got shape {Ez.shape}")
    z_m = longitudinal_coords(Ez.shape[2], length_m, centred=centred_z)
    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = np.exp(1j * omega * z_m / (float(beta) * C0))
    Vz_xy = np.trapezoid(Ez * phase[None, None, :], z_m, axis=2)
    return Vz_xy, z_m


def fit_near_axis_voltage_taylor(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    fit_pixels: int = 8,
    include_quadratic: bool = True,
) -> dict[str, Any]:
    """Fit near-axis Vz(x,y) to a 2D Taylor expansion.

    Fit form:
        Vz(x,y) = V0 + ax x + ay y + bxx x^2 + bxy x y + byy y^2

    The fit is complex: real and imaginary parts are represented naturally by
    a complex least-squares solve.
    """
    V = np.asarray(Vz_xy, dtype=complex)
    nx, ny = V.shape
    ix0, iy0 = nx // 2, ny // 2
    fp = int(fit_pixels)
    if fp < 1:
        raise ValueError("fit_pixels must be >= 1")
    ix1 = max(0, ix0 - fp); ix2 = min(nx, ix0 + fp + 1)
    iy1 = max(0, iy0 - fp); iy2 = min(ny, iy0 + fp + 1)

    X, Y = np.meshgrid(x_m[ix1:ix2], y_m[iy1:iy2], indexing="ij")
    Z = V[ix1:ix2, iy1:iy2]
    mask = np.isfinite(Z.real) & np.isfinite(Z.imag)

    cols = [np.ones(mask.sum()), X[mask], Y[mask]]
    names = ["V0", "ax", "ay"]
    if include_quadratic:
        cols += [X[mask] ** 2, X[mask] * Y[mask], Y[mask] ** 2]
        names += ["bxx", "bxy", "byy"]

    A = np.column_stack(cols)
    coeff, residuals, rank, svals = np.linalg.lstsq(A, Z[mask].ravel(), rcond=None)
    coeffs = {name: coeff[i] for i, name in enumerate(names)}
    for missing in ("bxx", "bxy", "byy"):
        coeffs.setdefault(missing, 0.0 + 0.0j)

    fitted = (A @ coeff).reshape(-1)
    data = Z[mask].ravel()
    rms = float(np.sqrt(np.mean(np.abs(data - fitted) ** 2))) if data.size else float("nan")
    scale = float(np.sqrt(np.mean(np.abs(data) ** 2))) if data.size else float("nan")

    return {
        "coefficients": coeffs,
        "fit_pixels": fp,
        "n_points": int(mask.sum()),
        "rank": int(rank),
        "rms_residual_V": rms,
        "relative_rms_residual": rms / scale if scale > 0 else float("nan"),
        "x_fit_range_m": (float(x_m[ix1]), float(x_m[ix2 - 1])),
        "y_fit_range_m": (float(y_m[iy1]), float(y_m[iy2 - 1])),
    }


def figures_of_merit_from_taylor(
    fit: dict[str, Any],
    *,
    frequency_Hz: float,
    charge_C: float = 1.0,
) -> dict[str, Any]:
    """Convert Taylor coefficients into loss, kick and focusing figures.

    Definitions:
        Vz = V0 + ax x + ay y + bxx x^2 + bxy xy + byy y^2

        Vperp = (i c / omega) grad_perp Vz

    Reported units assume the field map is normalised to charge_C.  For the
    present analytical maps charge_C is normally 1 C, so the reported values are
    V/C, V/C/m and V/C/m/m.
    """
    c = fit["coefficients"]
    omega = 2.0 * np.pi * float(frequency_Hz)
    pw = 1j * C0 / omega

    V0 = c["V0"] / charge_C
    ax = c["ax"] / charge_C
    ay = c["ay"] / charge_C
    bxx = c["bxx"] / charge_C
    bxy = c["bxy"] / charge_C
    byy = c["byy"] / charge_C

    # Monopole-like term.  This follows the convention used in the earlier code.
    loss_like = abs(V0) ** 2 / 4.0

    # Dipole-like steering kick components.
    kick_x = pw * ax
    kick_y = pw * ay
    kick_mag = float(np.sqrt(abs(kick_x) ** 2 + abs(kick_y) ** 2))
    kick_angle_deg = float(np.degrees(np.arctan2(abs(kick_y), abs(kick_x)))) if kick_mag > 0 else float("nan")

    # Quadrupole-like linear focusing matrix from gradients of Vperp.
    Kxx = pw * 2.0 * bxx
    Kxy = pw * bxy
    Kyx = pw * bxy
    Kyy = pw * 2.0 * byy
    Kmat = np.array([[Kxx, Kxy], [Kyx, Kyy]], dtype=complex)
    quad_norm = float(np.linalg.norm(Kmat))

    # Normal/skew quadrupole decomposition based on the longitudinal-voltage Hessian.
    normal_longitudinal = bxx - byy
    skew_longitudinal = bxy
    quad_angle_deg = 0.5 * float(np.degrees(np.arctan2(skew_longitudinal.real, normal_longitudinal.real))) if abs(normal_longitudinal) + abs(skew_longitudinal) > 0 else float("nan")

    return {
        "frequency_Hz": float(frequency_Hz),
        "omega_rad_s": float(omega),
        "V0_V_per_C": V0,
        "loss_like_V2_per_C2": float(loss_like),
        "dVz_dx_V_per_C_per_m": ax,
        "dVz_dy_V_per_C_per_m": ay,
        "kick_x_V_per_C_per_m": kick_x,
        "kick_y_V_per_C_per_m": kick_y,
        "kick_magnitude_V_per_C_per_m": kick_mag,
        "kick_angle_deg_abs_components": kick_angle_deg,
        "bxx_V_per_C_per_m2": bxx,
        "bxy_V_per_C_per_m2": bxy,
        "byy_V_per_C_per_m2": byy,
        "Kxx_V_per_C_per_m_per_m": Kxx,
        "Kxy_V_per_C_per_m_per_m": Kxy,
        "Kyx_V_per_C_per_m_per_m": Kyx,
        "Kyy_V_per_C_per_m_per_m": Kyy,
        "quadrupole_matrix_norm_V_per_C_per_m_per_m": quad_norm,
        "normal_quadrupole_longitudinal_V_per_C_per_m2": normal_longitudinal,
        "skew_quadrupole_longitudinal_V_per_C_per_m2": skew_longitudinal,
        "quadrupole_orientation_deg_from_real_coeffs": quad_angle_deg,
    }


# -----------------------------------------------------------------------------
# Parent/mixed field analysis
# -----------------------------------------------------------------------------

@dataclass
class FieldAnalysisConfig:
    f_010: float = 1.3e9
    radius_m: float | None = None
    design_length_m: float | None = None
    beta: float = 1.0
    fit_pixels: int = 8
    charge_C: float = 1.0
    centred_z: bool = True


def pillbox_radius_from_f010(f_010: float) -> float:
    # TM010 Bessel root.
    v01 = 2.404825557695773
    return float(v01 * C0 / (2.0 * np.pi * float(f_010)))


def design_length_from_f010(f_010: float) -> float:
    return float((C0 / float(f_010)) / 2.0)


def lookup_parent_frequency(
    family_data_by_m: dict[int, dict] | None,
    *,
    mode_name: str,
    fallback_Hz: float,
) -> float:
    """Look up design_frequency_Hz for mode like 'TM_123'."""
    if family_data_by_m is None:
        return float(fallback_Hz)
    _, mnp = mode_name.split("_", 1)
    m = int(mnp[0])
    try:
        return float(family_data_by_m[m]["TM"][mnp]["design_frequency_Hz"])
    except Exception:
        return float(fallback_Hz)


def analyse_field_Ez(
    Ez_xyz: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    radius_m: float,
    config: FieldAnalysisConfig,
) -> dict[str, Any]:
    nx, ny, nz = np.asarray(Ez_xyz).shape
    x_m, y_m = centred_transverse_coords(nx, ny, radius_m)
    Vz_xy, z_m = complex_voltage_map_from_Ez(
        Ez_xyz,
        length_m=length_m,
        frequency_Hz=frequency_Hz,
        beta=config.beta,
        centred_z=config.centred_z,
    )
    fit = fit_near_axis_voltage_taylor(Vz_xy, x_m, y_m, fit_pixels=config.fit_pixels)
    fom = figures_of_merit_from_taylor(fit, frequency_Hz=frequency_Hz, charge_C=config.charge_C)
    return {
        "length_m": float(length_m),
        "frequency_Hz": float(frequency_Hz),
        "transverse_pixel_x_m": float(x_m[1] - x_m[0]),
        "transverse_pixel_y_m": float(y_m[1] - y_m[0]),
        "longitudinal_pixel_m": float(z_m[1] - z_m[0]) if len(z_m) > 1 else float("nan"),
        "fit": fit,
        "figures_of_merit": fom,
    }


def analyse_heterotypic_crossing_folder(
    crossing_folder: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None = None,
    config: FieldAnalysisConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Analyse E1, E2, plus and minus fields in one saved crossing folder."""
    if config is None:
        config = FieldAnalysisConfig()

    folder = Path(crossing_folder)
    field_data = load_npz_dict(folder / "field_data.npz")
    meta = pickle_load(folder / "heterotypic_crossing_analysis.pkl")
    crossing = meta["crossing"]

    radius_m = config.radius_m or pillbox_radius_from_f010(config.f_010)
    design_L = config.design_length_m or design_length_from_f010(config.f_010)
    mixed_L = design_L * float(crossing["length_factor"])
    f_cross = float(crossing["frequency_Hz"])

    f_E1 = lookup_parent_frequency(family_data_by_m, mode_name=meta["mode_i"], fallback_Hz=f_cross)
    f_E2 = lookup_parent_frequency(family_data_by_m, mode_name=meta["mode_j"], fallback_Hz=f_cross)

    jobs = {
        "E1": {"Ez_key": "E1_Ez", "length_m": design_L, "frequency_Hz": f_E1, "mode": meta["mode_i"]},
        "E2": {"Ez_key": "E2_Ez", "length_m": design_L, "frequency_Hz": f_E2, "mode": meta["mode_j"]},
        "plus": {"Ez_key": "Ez_plus", "length_m": mixed_L, "frequency_Hz": f_cross, "mode": "plus"},
        "minus": {"Ez_key": "Ez_minus", "length_m": mixed_L, "frequency_Hz": f_cross, "mode": "minus"},
    }

    fields = {}
    for name, job in jobs.items():
        fields[name] = analyse_field_Ez(
            field_data[job["Ez_key"]],
            length_m=job["length_m"],
            frequency_Hz=job["frequency_Hz"],
            radius_m=radius_m,
            config=config,
        )
        fields[name]["mode"] = job["mode"]
        fields[name]["Ez_key"] = job["Ez_key"]

    comparison = compare_parent_and_mixed_figures(fields)

    out = {
        "crossing_folder": str(folder),
        "crossing": crossing,
        "mode_i": meta["mode_i"],
        "mode_j": meta["mode_j"],
        "units": {
            "V0": "V/C",
            "loss_like": "V^2/C^2 using |V0|^2/4 convention",
            "kick": "V/C/m",
            "focusing_matrix": "V/C/m/m",
        },
        "analysis_config": config.__dict__,
        "fields": fields,
        "comparison": comparison,
    }

    if save:
        pickle_save(out, folder / "heterotypic_multipole_analysis.pkl")
        write_summary_txt(out, folder / "heterotypic_multipole_summary.txt")

    return out


# -----------------------------------------------------------------------------
# Comparison / enhancement metrics
# -----------------------------------------------------------------------------

def metric_magnitudes(field_analysis: dict[str, Any]) -> dict[str, float]:
    f = field_analysis["figures_of_merit"]
    return {
        "loss_like": float(abs(f["loss_like_V2_per_C2"])),
        "V0": float(abs(f["V0_V_per_C"])),
        "kick_mag": float(abs(f["kick_magnitude_V_per_C_per_m"])),
        "kick_x": float(abs(f["kick_x_V_per_C_per_m"])),
        "kick_y": float(abs(f["kick_y_V_per_C_per_m"])),
        "Kxx": float(abs(f["Kxx_V_per_C_per_m_per_m"])),
        "Kxy": float(abs(f["Kxy_V_per_C_per_m_per_m"])),
        "Kyy": float(abs(f["Kyy_V_per_C_per_m_per_m"])),
        "quad_norm": float(abs(f["quadrupole_matrix_norm_V_per_C_per_m_per_m"])),
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den and np.isfinite(den) and den > 0 else float("nan")


def compare_parent_and_mixed_figures(fields: dict[str, dict]) -> dict[str, Any]:
    """Compare plus/minus figures with E1/E2 parents.

    Enhancement is reported relative to:
        - the larger parent value,
        - the coherent parent sum of magnitudes.
    """
    mags = {name: metric_magnitudes(d) for name, d in fields.items()}
    metrics = list(mags["E1"].keys())
    out: dict[str, Any] = {"metric_magnitudes": mags, "enhancement": {}}

    for metric in metrics:
        p1 = mags["E1"][metric]
        p2 = mags["E2"][metric]
        parent_max = max(p1, p2)
        parent_sum = p1 + p2
        out["enhancement"][metric] = {
            "parent_E1": p1,
            "parent_E2": p2,
            "parent_max": parent_max,
            "parent_sum": parent_sum,
            "plus": mags["plus"][metric],
            "minus": mags["minus"][metric],
            "plus_over_parent_max": safe_ratio(mags["plus"][metric], parent_max),
            "minus_over_parent_max": safe_ratio(mags["minus"][metric], parent_max),
            "plus_over_parent_sum": safe_ratio(mags["plus"][metric], parent_sum),
            "minus_over_parent_sum": safe_ratio(mags["minus"][metric], parent_sum),
        }

    return out


def write_summary_txt(result: dict[str, Any], filename: str | Path) -> None:
    lines = []
    lines.append(f"{result['crossing'].get('pair_type', 'crossing')}: {result['mode_i']} -- {result['mode_j']}")
    lines.append(f"ell = {result['crossing']['length_factor']:.8g}")
    lines.append(f"f = {result['crossing']['frequency_Hz']:.8e} Hz")
    lines.append("")

    for name in ("E1", "E2", "plus", "minus"):
        fa = result["fields"][name]["figures_of_merit"]
        lines.append(f"{name} ({result['fields'][name]['mode']}):")
        lines.append(f"  |V0|      = {abs(fa['V0_V_per_C']):.6e} V/C")
        lines.append(f"  loss_like = {fa['loss_like_V2_per_C2']:.6e} V^2/C^2")
        lines.append(f"  |kick|    = {fa['kick_magnitude_V_per_C_per_m']:.6e} V/C/m")
        lines.append(f"  |K|_F     = {fa['quadrupole_matrix_norm_V_per_C_per_m_per_m']:.6e} V/C/m/m")
        lines.append(f"  Kxx,Kxy,Kyy magnitudes = {abs(fa['Kxx_V_per_C_per_m_per_m']):.6e}, {abs(fa['Kxy_V_per_C_per_m_per_m']):.6e}, {abs(fa['Kyy_V_per_C_per_m_per_m']):.6e}")
        lines.append(f"  fit relative RMS = {result['fields'][name]['fit']['relative_rms_residual']:.3e}")
        lines.append("")

    lines.append("Enhancement relative to max(parent):")
    for metric, row in result["comparison"]["enhancement"].items():
        lines.append(f"  {metric:10s}: plus={row['plus_over_parent_max']:.4g}, minus={row['minus_over_parent_max']:.4g}")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Batch helpers
# -----------------------------------------------------------------------------

def find_crossing_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p.parent for p in root.rglob("heterotypic_crossing_analysis.pkl") if (p.parent / "field_data.npz").exists())


def load_family_data_files(*files: str | Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in files:
        data = pickle_load(f)
        # Prefer metadata; otherwise infer m from first mnp key.
        if "metadata" in data and "family_m" in data["metadata"]:
            m = int(data["metadata"]["family_m"])
        else:
            first = next(iter(data["TM"].keys()))
            m = int(first[0])
        out[m] = data
    return out


def analyse_all_heterotypic_crossings(
    root: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None = None,
    config: FieldAnalysisConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    folders = find_crossing_folders(root)
    results = {}
    for folder in folders:
        print(f"Analysing {folder}")
        results[str(folder)] = analyse_heterotypic_crossing_folder(
            folder,
            family_data_by_m=family_data_by_m,
            config=config,
            save=save,
        )
    if save:
        pickle_save(results, Path(root) / "all_heterotypic_multipole_analyses.pkl")
    return results


# -----------------------------------------------------------------------------
# Example command-line use
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit these paths for your machine.
    heterotypic_root = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings")
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical")
    voxel_res = 151

    family_files = [
        datapath / f"TMm0_TMm0_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
    ]
    existing = [f for f in family_files if f.exists()]
    family_data = load_family_data_files(*existing) if existing else None

    cfg = FieldAnalysisConfig(
        f_010=1.3e9,
        fit_pixels=8,
        charge_C=1.0,
    )
    analyse_all_heterotypic_crossings(
        heterotypic_root,
        family_data_by_m=family_data,
        config=cfg,
        save=True,
    )
