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
import zipfile
from typing import Any, Iterable

import numpy as np

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1.0e-12



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
    """Load an .npz field dictionary with an explicit error for corrupt files."""
    filename = Path(filename)
    try:
        with np.load(filename, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(
            f"{filename} is not a valid .npz zip archive. "
            "This usually means field generation was interrupted, the file is "
            "zero-length/corrupt, or a non-npz file was accidentally named "
            "field_data.npz. Delete this crossing folder's field_data.npz "
            "and regenerate it with the heterotypic crossing field-mixing script "
            "using create_fields=True."
        ) from exc


def is_valid_npz_file(filename: str | Path) -> bool:
    """Return True only for existing, non-empty, valid .npz zip files."""
    filename = Path(filename)
    return filename.exists() and filename.stat().st_size > 0 and zipfile.is_zipfile(filename)


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



def _trapezoid3(values: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray) -> float:
    """3D trapezoidal integral for arrays indexed as [x,y,z]."""
    a = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    integ_z = np.trapezoid(a, z_m, axis=2)
    integ_y = np.trapezoid(integ_z, y_m, axis=1)
    integ_x = np.trapezoid(integ_y, x_m, axis=0)
    return float(integ_x)


def electric_energy_diagnostics_from_components(
    *,
    Ex_xyz: np.ndarray | None,
    Ey_xyz: np.ndarray | None,
    Ez_xyz: np.ndarray,
    radius_m: float,
    length_m: float,
) -> dict[str, float]:
    """Return electric-field stored-energy diagnostics.

    The saved heterotypic field_data.npz files contain E-fields but not H-fields.
    For a lossless resonant eigenmode the CST reported stored energy is the total
    time-averaged electromagnetic energy,

        U_CST = 1/4 int (eps |E|^2 + mu |H|^2) dV.

    Since U_E = U_H at resonance, the E-field-only equivalent is

        U_CST = 1/2 eps0 int |E|^2 dV,

    using peak-amplitude real E-field maps. This function reports old
    Ez-only and E-only time-average diagnostics as checks, but all primary
    normalised figures use U_CST_equiv_J. No runtime option is provided to
    use the legacy time-average E-only value, avoiding the previous factor-of-two
    ambiguity.
    """
    Ez = np.nan_to_num(np.asarray(Ez_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nx, ny, nz = Ez.shape
    x_m, y_m = centred_transverse_coords(nx, ny, radius_m)
    z_m = np.linspace(0.0, float(length_m), nz)

    Ex = np.zeros_like(Ez) if Ex_xyz is None else np.nan_to_num(np.asarray(Ex_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ey = np.zeros_like(Ez) if Ey_xyz is None else np.nan_to_num(np.asarray(Ey_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    int_Ez2 = _trapezoid3(Ez * Ez, x_m, y_m, z_m)
    int_Etot2 = _trapezoid3(Ex * Ex + Ey * Ey + Ez * Ez, x_m, y_m, z_m)

    factor_peak = 0.5 * EPS0
    factor_avg = 0.25 * EPS0

    U_Ez_only_time = factor_avg * int_Ez2
    U_Etotal_time = factor_avg * int_Etot2
    U_Ez_only_peak = factor_peak * int_Ez2
    U_Etotal_peak = factor_peak * int_Etot2

    # CST-equivalent total stored energy for a lossless eigenmode when only
    # peak-amplitude E-field maps are available: total average EM energy equals
    # maximum electric energy.
    U_CST_equiv = U_Etotal_peak

    return {
        "int_Ez2_dV": int_Ez2,
        "int_Etot2_dV": int_Etot2,
        "U_Ez_only_time_average_J": U_Ez_only_time,
        "U_Etotal_time_average_J": U_Etotal_time,
        "U_Ez_only_peak_J": U_Ez_only_peak,
        "U_Etotal_peak_J": U_Etotal_peak,
        "U_CST_equiv_J": U_CST_equiv,
        "U_used_J": U_CST_equiv,
        "U_used_label": "U_CST_equiv = 0.5*eps0*int(|E|^2)dV",
        # Legacy keys retained only as aliases for side-by-side diagnostics.
        # They are deliberately set to the same CST-equivalent convention used
        # in the monopole on-axis analysis to avoid the old factor-of-two split.
        "U_Ez_only_used_J": U_Ez_only_peak,
        "U_Etotal_used_J": U_CST_equiv,
        "energy_time_average_used_for_legacy_keys": False,
        "energy_note": "Primary U is CST-equivalent from E only; H not present in field_data.npz. Legacy used keys now follow the same peak/CST-equivalent convention.",
    }


def kparallel_from_voltage_and_U(V: complex, U_J: float, *, length_m: float | None = None) -> dict[str, float]:
    """Return k_parallel with explicit V/C, V/pC, and optional per-metre units."""
    if U_J is None or not np.isfinite(U_J) or U_J <= 0.0:
        return {
            "V_abs": float(abs(V)),
            "k_V_per_C": float("nan"),
            "k_V_per_pC": float("nan"),
            "k_V_per_C_per_m": float("nan"),
            "k_V_per_pC_per_m": float("nan"),
        }
    k_v_per_c = float(abs(V) ** 2 / (4.0 * U_J))
    out = {
        "V_abs": float(abs(V)),
        "k_V_per_C": k_v_per_c,
        "k_V_per_pC": k_v_per_c * PC,
    }
    if length_m is not None and length_m > 0:
        out["k_V_per_C_per_m"] = k_v_per_c / float(length_m)
        out["k_V_per_pC_per_m"] = k_v_per_c * PC / float(length_m)
    else:
        out["k_V_per_C_per_m"] = float("nan")
        out["k_V_per_pC_per_m"] = float("nan")
    return out


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


def add_u_cst_normalised_figures(
    fom: dict[str, Any],
    energy: dict[str, float],
    *,
    length_m: float,
) -> dict[str, Any]:
    """Add integrated ``k`` and length-normalised ``K`` metrics.

    The explicit public keys and conventions match the companion RF/Fourier
    multipole analysis:

        k_parallel = |V0|^2/(4U)
        K_parallel = k_parallel/length_m

        k_perp = (c/omega)(|ax|^2 + |ay|^2)/(4U)
        K_perp = k_perp/length_m

        k_Q = sqrt((kxx-kyy)^2 + 4 kxy^2),

    where the phase-aligned quadrupole matrix components are normalised by
    ``sqrt(4U)`` but not by the structure length.  The reported quadrupole
    metric is then ``K_Q = k_Q/length_m``.

    Legacy output aliases are retained so existing appendix/report scripts do
    not break, but the explicit k_/K_ keys should be used by new code.
    """
    U = float(energy.get("U_CST_equiv_J", energy.get("U_used_J", float("nan"))))
    if not np.isfinite(U) or U <= 0.0:
        return fom

    if not np.isfinite(length_m) or length_m <= 0.0:
        raise ValueError(f"length_m must be positive and finite, got {length_m!r}")

    denom = 4.0 * U
    omega = float(fom.get("omega_rad_s", float("nan")))
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError(f"omega_rad_s must be positive and finite, got {omega!r}")

    # ------------------------------------------------------------------
    # Monopole: integrated k_parallel and length-normalised K_parallel.
    # ------------------------------------------------------------------
    V0 = complex(fom.get("V0_V_per_C", 0.0 + 0.0j))
    kpar_C = float(abs(V0) ** 2 / denom)
    Kpar_C = float(kpar_C / length_m)

    fom["k_parallel_V_per_C"] = kpar_C
    fom["k_parallel_V_per_pC"] = kpar_C * PC
    fom["K_parallel_V_per_C_per_m"] = Kpar_C
    fom["K_parallel_V_per_pC_per_m"] = Kpar_C * PC

    # Backwards-compatible longitudinal aliases.
    fom["loss_like_V_per_C"] = kpar_C
    fom["loss_like_V_per_pC"] = kpar_C * PC
    fom["loss_like_V_per_C_per_m"] = Kpar_C
    fom["loss_like_V_per_pC_per_m"] = Kpar_C * PC

    # ------------------------------------------------------------------
    # Dipole: use the conventional PW kick-factor normalisation, with one
    # factor c/omega, directly from the fitted longitudinal gradients ax, ay.
    # ------------------------------------------------------------------
    ax = complex(fom.get("dVz_dx_V_per_C_per_m", 0.0 + 0.0j))
    ay = complex(fom.get("dVz_dy_V_per_C_per_m", 0.0 + 0.0j))
    dipole_coeff_sq = abs(ax) ** 2 + abs(ay) ** 2

    kperp_C = float((C0 / omega) * dipole_coeff_sq / denom)
    Kperp_C = float(kperp_C / length_m)

    # Direction-resolved contributions add to the total magnitude metric.
    kperp_x_C = float((C0 / omega) * abs(ax) ** 2 / denom)
    kperp_y_C = float((C0 / omega) * abs(ay) ** 2 / denom)
    Kperp_x_C = float(kperp_x_C / length_m)
    Kperp_y_C = float(kperp_y_C / length_m)

    fom["k_perp_V_per_C_per_m"] = kperp_C
    fom["k_perp_V_per_pC_per_m"] = kperp_C * PC
    fom["K_perp_V_per_C_per_m2"] = Kperp_C
    fom["K_perp_V_per_pC_per_m2"] = Kperp_C * PC

    fom["k_perp_x_V_per_C_per_m"] = kperp_x_C
    fom["k_perp_y_V_per_C_per_m"] = kperp_y_C
    fom["k_perp_x_V_per_pC_per_m"] = kperp_x_C * PC
    fom["k_perp_y_V_per_pC_per_m"] = kperp_y_C * PC
    fom["K_perp_x_V_per_C_per_m2"] = Kperp_x_C
    fom["K_perp_y_V_per_C_per_m2"] = Kperp_y_C
    fom["K_perp_x_V_per_pC_per_m2"] = Kperp_x_C * PC
    fom["K_perp_y_V_per_pC_per_m2"] = Kperp_y_C * PC

    # Backwards-compatible dipole aliases now point to the agreed reported
    # length-normalised K_perp, matching the RF/Fourier companion workflow.
    fom["kick_x_V_per_C_per_m2"] = Kperp_x_C
    fom["kick_y_V_per_C_per_m2"] = Kperp_y_C
    fom["kick_magnitude_V_per_C_per_m2"] = Kperp_C
    fom["kick_x_V_per_pC_per_m2"] = Kperp_x_C * PC
    fom["kick_y_V_per_pC_per_m2"] = Kperp_y_C * PC
    fom["kick_magnitude_V_per_pC_per_m2"] = Kperp_C * PC

    # ------------------------------------------------------------------
    # Quadrupole: phase-align the complex transverse-voltage-gradient matrix.
    # First obtain integrated, U-normalised lower-case components (no /length),
    # then divide by length for the reported upper-case components.
    # ------------------------------------------------------------------
    K_complex = np.array(
        [
            [
                complex(fom.get("Kxx_V_per_C_per_m_per_m", 0.0 + 0.0j)),
                complex(fom.get("Kxy_V_per_C_per_m_per_m", 0.0 + 0.0j)),
            ],
            [
                complex(fom.get("Kxy_V_per_C_per_m_per_m", 0.0 + 0.0j)),
                complex(fom.get("Kyy_V_per_C_per_m_per_m", 0.0 + 0.0j)),
            ],
        ],
        dtype=complex,
    )

    idx = np.unravel_index(int(np.nanargmax(np.abs(K_complex))), K_complex.shape)
    ref = K_complex[idx]
    phase = float(np.angle(ref)) if abs(ref) > 0.0 else 0.0
    K_signed_raw = (K_complex * np.exp(-1j * phase)).real

    scale_k_C = 1.0 / np.sqrt(4.0 * U)
    scale_K_C = scale_k_C / float(length_m)

    k_signed_C = K_signed_raw * scale_k_C
    K_signed_C = K_signed_raw * scale_K_C

    kxx_C = float(k_signed_C[0, 0])
    kxy_C = float(k_signed_C[0, 1])
    kyy_C = float(k_signed_C[1, 1])
    kiso_C = 0.5 * (kxx_C + kyy_C)
    kQ_C = float(np.sqrt((kxx_C - kyy_C) ** 2 + 4.0 * kxy_C ** 2))

    Kxx_C = float(K_signed_C[0, 0])
    Kxy_C = float(K_signed_C[0, 1])
    Kyy_C = float(K_signed_C[1, 1])
    Kiso_C = 0.5 * (Kxx_C + Kyy_C)
    KQ_C = float(np.sqrt((Kxx_C - Kyy_C) ** 2 + 4.0 * Kxy_C ** 2))

    # Explicit integrated lower-case quadrupole outputs.
    fom["k_xx_V_per_C_per_m2"] = kxx_C
    fom["k_xy_V_per_C_per_m2"] = kxy_C
    fom["k_yy_V_per_C_per_m2"] = kyy_C
    fom["k_iso_V_per_C_per_m2"] = kiso_C
    fom["k_Q_V_per_C_per_m2"] = kQ_C

    fom["k_xx_V_per_pC_per_m2"] = kxx_C * PC
    fom["k_xy_V_per_pC_per_m2"] = kxy_C * PC
    fom["k_yy_V_per_pC_per_m2"] = kyy_C * PC
    fom["k_iso_V_per_pC_per_m2"] = kiso_C * PC
    fom["k_Q_V_per_pC_per_m2"] = kQ_C * PC

    # Explicit reported upper-case quadrupole outputs.
    fom["K_xx_V_per_C_per_m3"] = Kxx_C
    fom["K_xy_V_per_C_per_m3"] = Kxy_C
    fom["K_yy_V_per_C_per_m3"] = Kyy_C
    fom["K_iso_V_per_C_per_m3"] = Kiso_C
    fom["K_Q_V_per_C_per_m3"] = KQ_C

    fom["K_xx_V_per_pC_per_m3"] = Kxx_C * PC
    fom["K_xy_V_per_pC_per_m3"] = Kxy_C * PC
    fom["K_yy_V_per_pC_per_m3"] = Kyy_C * PC
    fom["K_iso_V_per_pC_per_m3"] = Kiso_C * PC
    fom["K_Q_V_per_pC_per_m3"] = KQ_C * PC

    # Legacy Hessian aliases retained for existing appendix/report code.
    fom["Kxx_V_per_C_per_m3"] = Kxx_C
    fom["Kxy_V_per_C_per_m3"] = Kxy_C
    fom["Kyy_V_per_C_per_m3"] = Kyy_C
    fom["Kiso_V_per_C_per_m3"] = Kiso_C
    fom["KQ_V_per_C_per_m3"] = KQ_C

    fom["Kxx_V_per_pC_per_m3"] = Kxx_C * PC
    fom["Kxy_V_per_pC_per_m3"] = Kxy_C * PC
    fom["Kyy_V_per_pC_per_m3"] = Kyy_C * PC
    fom["Kiso_V_per_pC_per_m3"] = Kiso_C * PC
    fom["KQ_V_per_pC_per_m3"] = KQ_C * PC

    fom["K_phase_reference_rad"] = phase
    fom["K_signed_phase_aligned_raw_V_per_C_per_m_per_m"] = K_signed_raw
    fom["U_CST_normalisation_J"] = U
    fom["U_CST_normalisation_note"] = (
        "Explicit agreed metrics: k_parallel=|V0|^2/(4U), "
        "K_parallel=k_parallel/length; "
        "k_perp=(c/omega)(|ax|^2+|ay|^2)/(4U), "
        "K_perp=k_perp/length; phase-aligned k_xx,k_xy,k_yy are raw "
        "Hessian PW components divided by sqrt(4U), with "
        "k_Q=sqrt((k_xx-k_yy)^2+4k_xy^2), and K_Q=k_Q/length."
    )

    return fom


# -----------------------------------------------------------------------------
# Parent/mixed field analysis
# -----------------------------------------------------------------------------

@dataclass
class FieldAnalysisConfig:
    f_010: float = 1.3e9
    radius_m: float | None = None
    design_length_m: float | None = None
    beta: float = 1.0
    # Backwards-compatible default fit radius.  The analysis now uses separate
    # Taylor windows for dipole-like and quadrupole-like quantities because the
    # validated finite-radius convergence differs by multipole order.
    fit_pixels: int = 4
    fit_pixels_dipole: int = 4
    fit_pixels_quadrupole: int = 8
    charge_C: float = 1.0
    centred_z: bool = False  # Method-2/monopole-aligned convention: z in [0,L], not [-L/2,L/2].


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
    """Look up the parent design frequency for a mode like 'TM_123'.

    If family data are supplied, this is deliberately strict: a missing family
    or mode now raises an error instead of silently falling back to the crossing
    frequency. Silent fallback was too easy to miss and can make parent fields
    appear to have the wrong frequency.
    """
    if family_data_by_m is None:
        return float(fallback_Hz)

    try:
        _, mnp = mode_name.split("_", 1)
        m = int(mnp[0])
    except Exception as exc:
        raise ValueError(f"Could not parse mode_name={mode_name!r}; expected form 'TM_123'.") from exc

    if m not in family_data_by_m:
        raise KeyError(
            f"No family data supplied for m={m} while looking up {mode_name}. "
            f"Available families: {sorted(family_data_by_m)}"
        )

    tm_table = family_data_by_m[m].get("TM", {})
    if mnp not in tm_table:
        raise KeyError(
            f"Mode {mnp!r} not found in family m={m} while looking up {mode_name}. "
            f"Available modes: {sorted(tm_table)}"
        )

    return float(tm_table[mnp]["design_frequency_Hz"])


def analyse_field_Ez(
    Ez_xyz: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    radius_m: float,
    config: FieldAnalysisConfig,
    Ex_xyz: np.ndarray | None = None,
    Ey_xyz: np.ndarray | None = None,
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
    # Use separate, validated finite windows for different Taylor orders:
    #   * dipole-like kick terms from the linear coefficients: fit_pixels_dipole=4
    #   * quadrupole-like focusing terms from the quadratic coefficients: fit_pixels_quadrupole=8
    # The returned figures_of_merit dictionary keeps the same public keys as before,
    # but its kick entries come from the dipole fit and its K entries from the
    # quadrupole fit.
    fit_dipole = fit_near_axis_voltage_taylor(
        Vz_xy, x_m, y_m, fit_pixels=config.fit_pixels_dipole
    )
    fit_quadrupole = fit_near_axis_voltage_taylor(
        Vz_xy, x_m, y_m, fit_pixels=config.fit_pixels_quadrupole
    )

    fom_dipole = figures_of_merit_from_taylor(
        fit_dipole, frequency_Hz=frequency_Hz, charge_C=config.charge_C
    )
    fom_quadrupole = figures_of_merit_from_taylor(
        fit_quadrupole, frequency_Hz=frequency_Hz, charge_C=config.charge_C
    )

    fom = dict(fom_dipole)
    for key in (
        "bxx_V_per_C_per_m2",
        "bxy_V_per_C_per_m2",
        "byy_V_per_C_per_m2",
        "Kxx_V_per_C_per_m_per_m",
        "Kxy_V_per_C_per_m_per_m",
        "Kyx_V_per_C_per_m_per_m",
        "Kyy_V_per_C_per_m_per_m",
        "quadrupole_matrix_norm_V_per_C_per_m_per_m",
        "normal_quadrupole_longitudinal_V_per_C_per_m2",
        "skew_quadrupole_longitudinal_V_per_C_per_m2",
        "quadrupole_orientation_deg_from_real_coeffs",
    ):
        fom[key] = fom_quadrupole[key]

    ix0, iy0 = nx // 2, ny // 2
    V_axis = Vz_xy[ix0, iy0] / config.charge_C
    V_fit = fom["V0_V_per_C"]

    energy = electric_energy_diagnostics_from_components(
        Ex_xyz=Ex_xyz,
        Ey_xyz=Ey_xyz,
        Ez_xyz=Ez_xyz,
        radius_m=radius_m,
        length_m=length_m,
    )

    fom = add_u_cst_normalised_figures(fom, energy, length_m=length_m)

    k_diagnostics = {
        "axis_U_CST": kparallel_from_voltage_and_U(V_axis, energy["U_CST_equiv_J"], length_m=length_m),
        "fit_V0_U_CST": kparallel_from_voltage_and_U(V_fit, energy["U_CST_equiv_J"], length_m=length_m),
        # Legacy/diagnostic normalisations retained so old discrepancies can be
        # traced explicitly.
        "axis_Ez_only_U": kparallel_from_voltage_and_U(V_axis, energy["U_Ez_only_used_J"], length_m=length_m),
        "axis_Etotal_used_U_CST_alias": kparallel_from_voltage_and_U(V_axis, energy["U_Etotal_used_J"], length_m=length_m),
        "fit_V0_Ez_only_U": kparallel_from_voltage_and_U(V_fit, energy["U_Ez_only_used_J"], length_m=length_m),
        "fit_V0_Etotal_used_U_CST_alias": kparallel_from_voltage_and_U(V_fit, energy["U_Etotal_used_J"], length_m=length_m),
    }

    return {
        "length_m": float(length_m),
        "frequency_Hz": float(frequency_Hz),
        "transverse_pixel_x_m": float(x_m[1] - x_m[0]),
        "transverse_pixel_y_m": float(y_m[1] - y_m[0]),
        "axis_indices_xy": (int(ix0), int(iy0)),
        "longitudinal_pixel_m": float(z_m[1] - z_m[0]) if len(z_m) > 1 else float("nan"),
        "z_start_m": float(z_m[0]),
        "z_stop_m": float(z_m[-1]),
        "centred_z": bool(config.centred_z),
        "Vz_axis_complex_V_per_C": V_axis,
        "Vz_axis_abs_V_per_C": float(abs(V_axis)),
        # Backwards-compatible alias: the main dipole/loss fit.
        "fit": fit_dipole,
        "fit_dipole": fit_dipole,
        "fit_quadrupole": fit_quadrupole,
        "fit_pixels_used": {
            "dipole": int(config.fit_pixels_dipole),
            "quadrupole": int(config.fit_pixels_quadrupole),
        },
        "figures_of_merit": fom,
        "energy_diagnostics": energy,
        "kparallel_diagnostics": k_diagnostics,
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
        "E1": {"Ez_key": "E1_Ez", "Ex_key": "E1_Ex", "Ey_key": "E1_Ey", "length_m": design_L, "frequency_Hz": f_E1, "frequency_source": "parent_design_frequency" if family_data_by_m is not None else "fallback_crossing_frequency", "mode": meta["mode_i"]},
        "E2": {"Ez_key": "E2_Ez", "Ex_key": "E2_Ex", "Ey_key": "E2_Ey", "length_m": design_L, "frequency_Hz": f_E2, "frequency_source": "parent_design_frequency" if family_data_by_m is not None else "fallback_crossing_frequency", "mode": meta["mode_j"]},
        "plus": {"Ez_key": "Ez_plus", "Ex_key": "Ex_plus", "Ey_key": "Ey_plus", "length_m": mixed_L, "frequency_Hz": f_cross, "frequency_source": "crossing_degenerate_frequency", "mode": "plus"},
        "minus": {"Ez_key": "Ez_minus", "Ex_key": "Ex_minus", "Ey_key": "Ey_minus", "length_m": mixed_L, "frequency_Hz": f_cross, "frequency_source": "crossing_degenerate_frequency", "mode": "minus"},
    }

    fields = {}
    for name, job in jobs.items():
        fields[name] = analyse_field_Ez(
            field_data[job["Ez_key"]],
            length_m=job["length_m"],
            frequency_Hz=job["frequency_Hz"],
            radius_m=radius_m,
            config=config,
            Ex_xyz=field_data.get(job["Ex_key"]),
            Ey_xyz=field_data.get(job["Ey_key"]),
        )
        fields[name]["mode"] = job["mode"]
        fields[name]["Ez_key"] = job["Ez_key"]
        fields[name]["Ex_key"] = job["Ex_key"]
        fields[name]["Ey_key"] = job["Ey_key"]
        fields[name]["frequency_source"] = job["frequency_source"]

    comparison = compare_parent_and_mixed_figures(fields)

    family_data_sources = {}
    if family_data_by_m is not None:
        for m, data in family_data_by_m.items():
            family_data_sources[int(m)] = data.get("metadata", {}).get("source_file", "unknown")

    out = {
        "crossing_folder": str(folder),
        "crossing": crossing,
        "mode_i": meta["mode_i"],
        "mode_j": meta["mode_j"],
        "family_data_sources": family_data_sources,
        "units": {
            "V0": "V/C",
            "k_parallel": "V/pC",
            "K_parallel": "V/pC/m_z",
            "k_perp": "V/pC/m_perp",
            "K_perp": "V/pC/m_perp/m_z",
            "k_Q": "V/pC/m_perp^2",
            "K_Q": "V/pC/m_perp^2/m_z",
            "focusing_matrix_raw": "V/C/m_perp^2",
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
    """Return magnitudes used for parent/mixed enhancement comparisons.

    Primary comparison entries use the agreed reported, length-normalised
    uppercase metrics.  Signed quadrupole components are compared by magnitude.
    """
    f = field_analysis["figures_of_merit"]
    return {
        "K_parallel": float(abs(f.get("K_parallel_V_per_pC_per_m", float("nan")))),
        "K_perp": float(abs(f.get("K_perp_V_per_pC_per_m2", float("nan")))),
        "K_Q": float(abs(f.get("K_Q_V_per_pC_per_m3", f.get("KQ_V_per_pC_per_m3", float("nan"))))),
        "K_xx": float(abs(f.get("K_xx_V_per_pC_per_m3", f.get("Kxx_V_per_pC_per_m3", float("nan"))))),
        "K_xy": float(abs(f.get("K_xy_V_per_pC_per_m3", f.get("Kxy_V_per_pC_per_m3", float("nan"))))),
        "K_yy": float(abs(f.get("K_yy_V_per_pC_per_m3", f.get("Kyy_V_per_pC_per_m3", float("nan"))))),
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


def _fmt_complex(z: complex) -> str:
    z = complex(z)
    return f"{z.real:.6e}{z.imag:+.6e}j"


def write_summary_txt(result: dict[str, Any], filename: str | Path) -> None:
    """Write a detailed diagnostic summary for consistency checks.

    This intentionally prints redundant quantities so discrepancies between
    appendices can be traced to phase convention, frequency/length choice,
    stored-energy convention, unit conversion, or axis-vs-fit voltage choice.
    """
    lines: list[str] = []
    lines.append(f"{result['crossing'].get('pair_type', 'crossing')}: {result['mode_i']} -- {result['mode_j']}")
    lines.append(f"ell = {result['crossing']['length_factor']:.12g}")
    lines.append(f"f_cross = {result['crossing']['frequency_Hz']:.12e} Hz")
    lines.append("")

    cfg = result.get("analysis_config", {})
    lines.append("ANALYSIS CONFIGURATION")
    lines.append(f"  beta                         = {cfg.get('beta')}")
    lines.append(f"  fit_pixels                   = {cfg.get('fit_pixels')}")
    lines.append(f"  fit_pixels_dipole            = {cfg.get('fit_pixels_dipole', cfg.get('fit_pixels'))}")
    lines.append(f"  fit_pixels_quadrupole        = {cfg.get('fit_pixels_quadrupole', cfg.get('fit_pixels'))}")
    lines.append(f"  charge_C                     = {cfg.get('charge_C')}")
    lines.append(f"  centred_z                    = {cfg.get('centred_z')}  (False means z in [0,L]; True means z in [-L/2,L/2])")
    lines.append("  primary stored energy U      = U_CST_equiv = 0.5 eps0 int|E|^2 dV")
    lines.append("  legacy time-average E-only U = diagnostic only; not used for k_parallel")
    lines.append("")

    lines.append("PARENT FAMILY DATA SOURCES")
    # These are stored in analysis output when available.
    for m, src in sorted(result.get("family_data_sources", {}).items()):
        lines.append(f"  m={m}: {src}")
    if not result.get("family_data_sources"):
        lines.append("  none supplied; parent frequencies used fallback crossing frequency")
    lines.append("")

    lines.append("IMPORTANT UNIT CONVENTIONS")
    lines.append("  raw loss_like in the old table = |V0|^2/4, not U-normalised")
    lines.append("  U-normalised k_parallel        = |V|^2/(4 U_CST) [V/C]")
    lines.append("  conversion to V/pC             = [V/C] * 1e-12")
    lines.append("  per-metre value                = [V/pC] / length_m")
    lines.append("")

    for name in ("E1", "E2", "plus", "minus"):
        field = result["fields"][name]
        fa = field["figures_of_merit"]
        fit = field["fit"]
        fit_dipole = field.get("fit_dipole", fit)
        fit_quadrupole = field.get("fit_quadrupole", fit)
        en = field.get("energy_diagnostics", {})
        kp = field.get("kparallel_diagnostics", {})

        lines.append(f"{name} ({field['mode']}):")
        lines.append(f"  Ez_key / Ex_key / Ey_key       = {field.get('Ez_key')} / {field.get('Ex_key')} / {field.get('Ey_key')}")
        lines.append(f"  frequency_Hz                  = {field['frequency_Hz']:.12e}")
        lines.append(f"  frequency_source              = {field.get('frequency_source', 'unknown')}")
        lines.append(f"  length_m                      = {field['length_m']:.12e}")
        lines.append(f"  z_start_m, z_stop_m           = {field.get('z_start_m', float('nan')):.12e}, {field.get('z_stop_m', float('nan')):.12e}")
        lines.append(f"  centred_z                     = {field.get('centred_z')}")
        lines.append(f"  axis_indices_xy               = {field.get('axis_indices_xy')}")
        lines.append(f"  transverse_pixel_x_m          = {field['transverse_pixel_x_m']:.12e}")
        lines.append(f"  transverse_pixel_y_m          = {field['transverse_pixel_y_m']:.12e}")
        lines.append(f"  longitudinal_pixel_m          = {field['longitudinal_pixel_m']:.12e}")
        lines.append("")
        lines.append("  VOLTAGE DIAGNOSTICS")
        lines.append(f"    V_axis complex              = {_fmt_complex(field.get('Vz_axis_complex_V_per_C', 0.0))} V/C")
        lines.append(f"    |V_axis|                    = {field.get('Vz_axis_abs_V_per_C', float('nan')):.12e} V/C")
        lines.append(f"    V0_fit complex              = {_fmt_complex(fa['V0_V_per_C'])} V/C")
        lines.append(f"    |V0_fit|                    = {abs(fa['V0_V_per_C']):.12e} V/C")
        lines.append(f"    old_loss_like=|V0|^2/4      = {fa['loss_like_V2_per_C2']:.12e} V^2/C^2")
        lines.append("")
        lines.append("  STORED-ENERGY DIAGNOSTICS")
        lines.append(f"    int_Ez2_dV                  = {en.get('int_Ez2_dV', float('nan')):.12e}")
        lines.append(f"    int_Etot2_dV                = {en.get('int_Etot2_dV', float('nan')):.12e}")
        lines.append(f"    U_Ez_only_time_average_J    = {en.get('U_Ez_only_time_average_J', float('nan')):.12e}")
        lines.append(f"    U_Etotal_time_average_J     = {en.get('U_Etotal_time_average_J', float('nan')):.12e}")
        lines.append(f"    U_Ez_only_peak_J            = {en.get('U_Ez_only_peak_J', float('nan')):.12e}")
        lines.append(f"    U_Etotal_peak_J             = {en.get('U_Etotal_peak_J', float('nan')):.12e}")
        lines.append(f"    U_CST_equiv_J               = {en.get('U_CST_equiv_J', float('nan')):.12e}")
        lines.append(f"    U_used_label                = {en.get('U_used_label', 'unknown')}")
        lines.append(f"    U_Ez_only_used_J [peak diag]= {en.get('U_Ez_only_used_J', float('nan')):.12e}")
        lines.append(f"    U_Etotal_used_J [=U_CST]    = {en.get('U_Etotal_used_J', float('nan')):.12e}")
        lines.append("")
        lines.append("  U-NORMALISED k_parallel DIAGNOSTICS")
        for key in ("axis_U_CST", "fit_V0_U_CST", "axis_Ez_only_U", "axis_Etotal_used_U_CST_alias", "fit_V0_Ez_only_U", "fit_V0_Etotal_used_U_CST_alias"):
            row = kp.get(key, {})
            lines.append(f"    {key:18s}: k={row.get('k_V_per_C', float('nan')):.12e} V/C, "
                         f"{row.get('k_V_per_pC', float('nan')):.12e} V/pC, "
                         f"{row.get('k_V_per_pC_per_m', float('nan')):.12e} V/pC/m")
        lines.append("")
        lines.append("  AGREED INTEGRATED AND LENGTH-NORMALISED METRICS")
        lines.append(f"    k_parallel                   = {fa.get('k_parallel_V_per_pC', float('nan')):.12e} V/pC")
        lines.append(f"    K_parallel                   = {fa.get('K_parallel_V_per_pC_per_m', float('nan')):.12e} V/pC/m_z")
        lines.append(f"    k_perp                       = {fa.get('k_perp_V_per_pC_per_m', float('nan')):.12e} V/pC/m_perp")
        lines.append(f"    K_perp                       = {fa.get('K_perp_V_per_pC_per_m2', float('nan')):.12e} V/pC/m_perp/m_z")
        lines.append(f"    k_Q                          = {fa.get('k_Q_V_per_pC_per_m2', float('nan')):.12e} V/pC/m_perp^2")
        lines.append(f"    K_Q                          = {fa.get('K_Q_V_per_pC_per_m3', float('nan')):.12e} V/pC/m_perp^2/m_z")
        lines.append("")
        lines.append("  MULTIPOLE FIT DIAGNOSTICS")
        lines.append(f"    |kick| raw PW               = {fa['kick_magnitude_V_per_C_per_m']:.12e} V/C/m")
        lines.append(f"    legacy kick alias (=K_perp) = {fa.get('kick_magnitude_V_per_C_per_m2', float('nan')):.12e} V/C/m_perp/m_z")
        lines.append(f"    legacy kick alias (=K_perp) = {fa.get('kick_magnitude_V_per_pC_per_m2', float('nan')):.12e} V/pC/m_perp/m_z")
        lines.append(f"    |K|_F raw                   = {fa['quadrupole_matrix_norm_V_per_C_per_m_per_m']:.12e} V/C/m/m")
        lines.append(f"    U_CST |K|_F^2/(4U)          = {fa.get('quadrupole_matrix_norm_V_per_pC_per_m3', float('nan')):.12e} V/pC/m^3")
        lines.append(f"    Kxx,Kxy,Kyy raw magnitudes  = {abs(fa['Kxx_V_per_C_per_m_per_m']):.12e}, {abs(fa['Kxy_V_per_C_per_m_per_m']):.12e}, {abs(fa['Kyy_V_per_C_per_m_per_m']):.12e}")
        lines.append(f"    Kxx,Kxy,Kyy reported        = {abs(fa.get('Kxx_V_per_pC_per_m3', float('nan'))):.12e}, {abs(fa.get('Kxy_V_per_pC_per_m3', float('nan'))):.12e}, {abs(fa.get('Kyy_V_per_pC_per_m3', float('nan'))):.12e} V/pC/m^3")
        lines.append(f"    dipole fit relative RMS     = {fit_dipole['relative_rms_residual']:.12e}")
        lines.append(f"    dipole fit n_points/rank    = {fit_dipole['n_points']} / {fit_dipole['rank']}")
        lines.append(f"    dipole fit x_range_m        = {fit_dipole['x_fit_range_m']}")
        lines.append(f"    dipole fit y_range_m        = {fit_dipole['y_fit_range_m']}")
        lines.append(f"    quadrupole fit relative RMS = {fit_quadrupole['relative_rms_residual']:.12e}")
        lines.append(f"    quadrupole fit n_points/rank= {fit_quadrupole['n_points']} / {fit_quadrupole['rank']}")
        lines.append(f"    quadrupole fit x_range_m    = {fit_quadrupole['x_fit_range_m']}")
        lines.append(f"    quadrupole fit y_range_m    = {fit_quadrupole['y_fit_range_m']}")
        lines.append("")

    lines.append("Enhancement relative to max(parent), using existing non-U-normalised magnitudes:")
    for metric, row in result["comparison"]["enhancement"].items():
        lines.append(f"  {metric:10s}: plus={row['plus_over_parent_max']:.4g}, minus={row['minus_over_parent_max']:.4g}")

    lines.append("")
    lines.append("LIKELY CONSISTENCY CHECKS IF APPENDICES DISAGREE")
    lines.append("  1. Compare frequency_source: parents should use parent design frequencies, not f_cross fallback.")
    lines.append("  2. Compare length_m: parents should use design length; plus/minus should use design length * ell.")
    lines.append("  3. Compare centred_z and whether other scripts use real cos() only or abs(complex voltage).")
    lines.append("  4. Compare U_CST_equiv_J against Ez-only and time-average diagnostics.")
    lines.append("  5. Compare k_V_per_pC_per_m, not raw V/C or old_loss_like.")
    lines.append("  6. Compare V_axis and V0_fit; large differences indicate axis/centering/fit issues.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Batch helpers
# -----------------------------------------------------------------------------

def find_crossing_folders(root: str | Path) -> list[Path]:
    """Find crossing folders that have both metadata and a valid field_data.npz.

    Invalid/corrupt field_data.npz files are skipped with a warning rather than
    causing a hard-to-diagnose zipfile.BadZipFile later. Delete/regenerate any
    listed corrupt file.
    """
    root = Path(root)
    folders: list[Path] = []
    for p in sorted(root.rglob("heterotypic_crossing_analysis.pkl")):
        folder = p.parent
        field_file = folder / "field_data.npz"
        if is_valid_npz_file(field_file):
            folders.append(folder)
        elif field_file.exists():
            size = field_file.stat().st_size if field_file.exists() else 0
            print(
                f"WARNING: skipping {folder} because {field_file.name} is not "
                f"a valid .npz archive (size={size} bytes). Delete/regenerate it."
            )
    return folders


def load_family_data_files(*files: str | Path) -> dict[int, dict]:
    """Load parent-family pickle files keyed by azimuthal index m.

    The source filename is stored in metadata and printed so stale root-level
    pickle files are easy to spot.
    """
    out: dict[int, dict] = {}
    for f in files:
        f = Path(f)
        print(f"Loading family data: {f}")
        data = pickle_load(f)
        # Prefer metadata; otherwise infer m from first mnp key.
        if "metadata" in data and "family_m" in data["metadata"]:
            m = int(data["metadata"]["family_m"])
        else:
            first = next(iter(data["TM"].keys()))
            m = int(first[0])
            data.setdefault("metadata", {})["family_m"] = m

        data.setdefault("metadata", {})["source_file"] = str(f)
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

    # IMPORTANT: family pickles are written by the crossing scripts to the
    # ``data`` subfolder.  The previous version pointed at the project root,
    # which allowed stale root-level pickles to be loaded and could give parent
    # modes, e.g. TM113, the wrong design frequency.
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    voxel_res = 151

    family_files = [
        datapath / f"TMm0_TMm0_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
    ]
    missing = [f for f in family_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required family data files:\n"
            + "\n".join(str(f) for f in missing)
            + "\nRegenerate them or correct datapath."
        )
    family_data = load_family_data_files(*family_files)

    cfg = FieldAnalysisConfig(
        f_010=1.3e9,
        fit_pixels=4,
        fit_pixels_dipole=4,
        fit_pixels_quadrupole=8,
        charge_C=1.0,
    )
    analyse_all_heterotypic_crossings(
        heterotypic_root,
        family_data_by_m=family_data,
        config=cfg,
        save=True,
    )
