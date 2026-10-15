"""RF/Fourier multipole analysis of heterotypic mixed fields.

This module is a drop-in-style companion to the heterotypic Hessian/Taylor
analysis.  It keeps the same folder workflow:

    field_data.npz
    heterotypic_crossing_analysis.pkl

but replaces the Taylor/Hessian fit with an azimuthal Fourier/RF multipole
projection of the complex longitudinal voltage

    Vz(x,y) = integral Ez(x,y,z) exp(i omega z / beta c) dz.

The local RF multipole expansion is written as

    Vz(rho,phi) = c0
                + rho  (c1 cos phi  + s1 sin phi)
                + rho^2(c2 cos 2phi + s2 sin 2phi) + ...

The extracted coefficients are converted into the same primary beam-dynamics
figures used in the Hessian/Taylor workflow:

    k_parallel = |c0|^2/(4 U)

    k_perp     = (c/(4 U omega)) * (|c1|^2 + |s1|^2)
    K_perp     = k_perp/L

    k_Q        = 4(c/omega) * sqrt(|c2|^2 + |s2|^2)/sqrt(4 U)
    K_Q        = k_Q/L

Notes
-----
* U is the CST-equivalent stored energy estimated from the saved E-fields,
  U_CST_equiv = 0.5 eps0 int |E|^2 dV, matching the latest U_CST convention.
* Parent fields use their parent design frequency when family data are supplied;
  mixed fields use the crossing frequency.
* Parent fields use L=lambda_010/2; mixed fields use L=lambda_010/2 * ell.
* Outputs are saved with the word ``multipole`` in the filename so they remain
  separate from the Hessian/Taylor analysis results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import zipfile
from typing import Any

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover - fallback only used if scipy is absent
    RegularGridInterpolator = None

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
    filename = Path(filename)
    try:
        with np.load(filename, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(
            f"{filename} is not a valid .npz zip archive. Delete/regenerate this "
            "crossing folder's field_data.npz with the field-mixing script."
        ) from exc


def is_valid_npz_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.exists() and filename.stat().st_size > 0 and zipfile.is_zipfile(filename)


# -----------------------------------------------------------------------------
# Coordinates, voltages, energy
# -----------------------------------------------------------------------------

def centred_transverse_coords(nx: int, ny: int, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-float(radius_m), float(radius_m), int(nx))
    y = np.linspace(-float(radius_m), float(radius_m), int(ny))
    return x, y


def longitudinal_coords(nz: int, length_m: float, *, centred: bool = False) -> np.ndarray:
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
    centred_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    Ez = np.nan_to_num(np.asarray(Ez_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Ez.ndim != 3:
        raise ValueError(f"Ez_xyz must be 3D, got shape {Ez.shape}")
    z_m = longitudinal_coords(Ez.shape[2], length_m, centred=centred_z)
    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = np.exp(1j * omega * z_m / (float(beta) * C0))
    Vz_xy = np.trapezoid(Ez * phase[None, None, :], z_m, axis=2)
    return Vz_xy, z_m


def _trapezoid3(values: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray) -> float:
    a = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapezoid(np.trapezoid(np.trapezoid(a, z_m, axis=2), y_m, axis=1), x_m, axis=0))


def electric_energy_diagnostics_from_components(
    *,
    Ex_xyz: np.ndarray | None,
    Ey_xyz: np.ndarray | None,
    Ez_xyz: np.ndarray,
    radius_m: float,
    length_m: float,
) -> dict[str, float]:
    Ez = np.nan_to_num(np.asarray(Ez_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nx, ny, nz = Ez.shape
    x_m, y_m = centred_transverse_coords(nx, ny, radius_m)
    z_m = np.linspace(0.0, float(length_m), nz)

    Ex = np.zeros_like(Ez) if Ex_xyz is None else np.nan_to_num(np.asarray(Ex_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ey = np.zeros_like(Ez) if Ey_xyz is None else np.nan_to_num(np.asarray(Ey_xyz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    int_Ez2 = _trapezoid3(Ez * Ez, x_m, y_m, z_m)
    int_Etot2 = _trapezoid3(Ex * Ex + Ey * Ey + Ez * Ez, x_m, y_m, z_m)

    U_Ez_only_time = 0.25 * EPS0 * int_Ez2
    U_Etotal_time = 0.25 * EPS0 * int_Etot2
    U_Ez_only_peak = 0.5 * EPS0 * int_Ez2
    U_Etotal_peak = 0.5 * EPS0 * int_Etot2

    return {
        "int_Ez2_dV": int_Ez2,
        "int_Etot2_dV": int_Etot2,
        "U_Ez_only_time_average_J": U_Ez_only_time,
        "U_Etotal_time_average_J": U_Etotal_time,
        "U_Ez_only_peak_J": U_Ez_only_peak,
        "U_Etotal_peak_J": U_Etotal_peak,
        "U_CST_equiv_J": U_Etotal_peak,
        "U_used_J": U_Etotal_peak,
        "U_used_label": "U_CST_equiv = 0.5*eps0*int(|E|^2)dV",
        "energy_note": "Primary U is CST-equivalent from E only; H not present in field_data.npz.",
    }


# -----------------------------------------------------------------------------
# Fourier/RF multipole extraction
# -----------------------------------------------------------------------------

def _interp_complex_on_circle(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    rho_m: float,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return phi and Vz(rho,phi) from bilinear interpolation on Vz[x,y]."""
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    points = np.column_stack((rho_m * np.cos(phi), rho_m * np.sin(phi)))

    if RegularGridInterpolator is not None:
        interp_re = RegularGridInterpolator((x_m, y_m), np.asarray(Vz_xy).real, bounds_error=False, fill_value=np.nan)
        interp_im = RegularGridInterpolator((x_m, y_m), np.asarray(Vz_xy).imag, bounds_error=False, fill_value=np.nan)
        vals = interp_re(points) + 1j * interp_im(points)
    else:
        vals = _manual_bilinear_complex(Vz_xy, x_m, y_m, points)

    if not np.all(np.isfinite(vals.real) & np.isfinite(vals.imag)):
        raise ValueError(
            f"Non-finite interpolation values at rho={rho_m:.6e} m. "
            "Check that the sampling radius lies inside the field map."
        )
    return phi, vals


def _manual_bilinear_complex(V: np.ndarray, x: np.ndarray, y: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # Minimal fallback for environments without scipy.
    out = np.empty(len(pts), dtype=complex)
    for i, (xp, yp) in enumerate(pts):
        ix = np.searchsorted(x, xp) - 1
        iy = np.searchsorted(y, yp) - 1
        ix = int(np.clip(ix, 0, len(x) - 2))
        iy = int(np.clip(iy, 0, len(y) - 2))
        x0, x1 = x[ix], x[ix + 1]
        y0, y1 = y[iy], y[iy + 1]
        tx = 0.0 if x1 == x0 else (xp - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yp - y0) / (y1 - y0)
        out[i] = (
            (1 - tx) * (1 - ty) * V[ix, iy]
            + tx * (1 - ty) * V[ix + 1, iy]
            + (1 - tx) * ty * V[ix, iy + 1]
            + tx * ty * V[ix + 1, iy + 1]
        )
    return out


def extract_rf_multipoles_from_Vz(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    rho_dipole_m: float,
    rho_quadrupole_m: float,
    n_phi: int = 720,
    charge_C: float = 1.0,
) -> dict[str, Any]:
    """Extract c0, c1/s1 and c2/s2 from azimuthal Fourier projections.

    c0 is taken from the on-axis voltage for direct consistency with the
    standard loss-factor definition.  c0_circle diagnostics are also recorded.
    The dipole coefficients use rho_dipole_m; the quadrupole coefficients use
    rho_quadrupole_m.  This allows the validated fit radii used previously
    (4 pixels for dipole-like terms and 8 pixels for quadrupole-like terms) to
    be retained in the RF multipole projection.
    """
    V = np.asarray(Vz_xy, dtype=complex) / float(charge_C)
    nx, ny = V.shape
    ix0, iy0 = nx // 2, ny // 2
    c0_axis = V[ix0, iy0]

    phi1, V1 = _interp_complex_on_circle(V, x_m, y_m, rho_m=rho_dipole_m, n_phi=n_phi)
    phi2, V2 = _interp_complex_on_circle(V, x_m, y_m, rho_m=rho_quadrupole_m, n_phi=n_phi)

    c0_dipole_circle = np.mean(V1)
    c0_quadrupole_circle = np.mean(V2)

    # Standard Fourier coefficient normalisation:
    # a_n = (1/pi) int f cos(n phi) dphi ~= 2/N sum f cos(n phi).
    c1 = (2.0 / len(phi1)) * np.sum(V1 * np.cos(phi1)) / rho_dipole_m
    s1 = (2.0 / len(phi1)) * np.sum(V1 * np.sin(phi1)) / rho_dipole_m
    c2 = (2.0 / len(phi2)) * np.sum(V2 * np.cos(2.0 * phi2)) / (rho_quadrupole_m ** 2)
    s2 = (2.0 / len(phi2)) * np.sum(V2 * np.sin(2.0 * phi2)) / (rho_quadrupole_m ** 2)

    return {
        "method": "azimuthal_fourier_rf_multipole",
        "n_phi": int(n_phi),
        "charge_C": float(charge_C),
        "axis_indices_xy": (int(ix0), int(iy0)),
        "rho_dipole_m": float(rho_dipole_m),
        "rho_quadrupole_m": float(rho_quadrupole_m),
        "c0_axis_V_per_C": c0_axis,
        "c0_dipole_circle_mean_V_per_C": c0_dipole_circle,
        "c0_quadrupole_circle_mean_V_per_C": c0_quadrupole_circle,
        "c1_cos_phi_V_per_C_per_m": c1,
        "s1_sin_phi_V_per_C_per_m": s1,
        "c2_cos_2phi_V_per_C_per_m2": c2,
        "s2_sin_2phi_V_per_C_per_m2": s2,
        "normal_quadrupole_coeff_c2_V_per_C_per_m2": c2,
        "skew_quadrupole_coeff_s2_V_per_C_per_m2": s2,
    }


def figures_of_merit_from_rf_multipoles(
    multipoles: dict[str, Any],
    *,
    frequency_Hz: float,
    U_J: float,
    length_m: float,
) -> dict[str, Any]:
    """Convert RF multipole coefficients into U_CST-normalised metrics."""
    omega = 2.0 * np.pi * float(frequency_Hz)
    if U_J is None or not np.isfinite(U_J) or U_J <= 0.0:
        U_J = float("nan")

    c0 = complex(multipoles["c0_axis_V_per_C"])
    c1 = complex(multipoles["c1_cos_phi_V_per_C_per_m"])
    s1 = complex(multipoles["s1_sin_phi_V_per_C_per_m"])
    c2 = complex(multipoles["c2_cos_2phi_V_per_C_per_m2"])
    s2 = complex(multipoles["s2_sin_2phi_V_per_C_per_m2"])

    pw = C0 / omega
    Vperp_x_raw = pw * c1
    Vperp_y_raw = pw * s1
    Vperp_raw_mag = float(np.sqrt(abs(Vperp_x_raw) ** 2 + abs(Vperp_y_raw) ** 2))

    K_normal_raw = 2.0 * pw * c2
    K_skew_raw = 2.0 * pw * s2

    # The RF multipole gradient components above are the normal/skew entries
    # of the transverse-voltage gradient matrix.  The Hessian/Taylor workflow
    # reports the scalar quadrupole strength as
    #
    #     K_Q = sqrt((Kxx - Kyy)^2 + 4 Kxy^2).
    #
    # For Vz_Q = c2(x^2-y^2)+2s2xy this is
    #
    #     K_Q = 4(c/omega) sqrt(|c2|^2 + |s2|^2),
    #
    # i.e. a factor of two larger than sqrt(K_normal^2 + K_skew^2).
    KQ_raw_component_magnitude = float(2.0 * pw * np.sqrt(abs(c2) ** 2 + abs(s2) ** 2))
    KQ_raw = float(2.0 * KQ_raw_component_magnitude)

    out: dict[str, Any] = {
        "frequency_Hz": float(frequency_Hz),
        "omega_rad_s": float(omega),
        "V0_V_per_C": c0,
        "c1_cos_phi_V_per_C_per_m": c1,
        "s1_sin_phi_V_per_C_per_m": s1,
        "c2_cos_2phi_V_per_C_per_m2": c2,
        "s2_sin_2phi_V_per_C_per_m2": s2,
        "normal_quadrupole_coeff_c2_V_per_C_per_m2": c2,
        "skew_quadrupole_coeff_s2_V_per_C_per_m2": s2,
        "kick_x_raw_V_per_C_per_m": Vperp_x_raw,
        "kick_y_raw_V_per_C_per_m": Vperp_y_raw,
        "kick_magnitude_raw_V_per_C_per_m": Vperp_raw_mag,
        "K_normal_raw_V_per_C_per_m_per_m": K_normal_raw,
        "K_skew_raw_V_per_C_per_m_per_m": K_skew_raw,
        "KQ_component_magnitude_raw_V_per_C_per_m_per_m": KQ_raw_component_magnitude,
        "KQ_raw_V_per_C_per_m_per_m": KQ_raw,
    }

    if np.isfinite(U_J) and U_J > 0.0:
        denom = 4.0 * U_J
        length_ok = np.isfinite(length_m) and length_m > 0.0

        # Integrated quantities, normalised to stored energy U.
        kpar = abs(c0) ** 2 / denom
        dipole_coeff_sq = abs(c1) ** 2 + abs(s1) ** 2
        kperp = (C0 / (omega * denom)) * dipole_coeff_sq
        kQ = KQ_raw / np.sqrt(denom)

        # Reported quantities, additionally normalised by analysed structure length d.
        Kpar = kpar / float(length_m) if length_ok else float("nan")
        Kperp = kperp / float(length_m) if length_ok else float("nan")
        KQ = kQ / float(length_m) if length_ok else float("nan")

        # Optional normal/skew integrated and length-normalised quadrupole components.
        k_normal = abs(K_normal_raw) / np.sqrt(denom)
        k_skew = abs(K_skew_raw) / np.sqrt(denom)
        K_normal = k_normal / float(length_m) if length_ok else float("nan")
        K_skew = k_skew / float(length_m) if length_ok else float("nan")

        out.update({
            # Integrated monopole quantity: k_parallel [V/C].
            "k_parallel_V_per_C": float(kpar),
            "k_parallel_V_per_pC": float(kpar * PC),

            # Length-normalised monopole metric: K_parallel [V/C/m_z].
            "K_parallel_V_per_C_per_m": float(Kpar),
            "K_parallel_V_per_pC_per_m": float(Kpar * PC),

            # Integrated dipole quantity: k_perp [V/C/m_perp].
            "k_perp_V_per_C_per_m": float(kperp),
            "k_perp_V_per_pC_per_m": float(kperp * PC),

            # Length-normalised dipole metric: K_perp [V/C/m_perp/m_z].
            "K_perp_V_per_C_per_m2": float(Kperp),
            "K_perp_V_per_pC_per_m2": float(Kperp * PC),

            # Integrated quadrupole quantity: k_Q [V/C/m_perp^2].
            "k_Q_V_per_C_per_m2": float(kQ),
            "k_Q_V_per_pC_per_m2": float(kQ * PC),

            # Length-normalised quadrupole metric: K_Q [V/C/m_perp^2/m_z].
            "K_Q_V_per_C_per_m3": float(KQ),
            "K_Q_V_per_pC_per_m3": float(KQ * PC),

            # Normal/skew quadrupole components at both normalisation levels.
            "k_normal_V_per_C_per_m2": float(k_normal),
            "k_skew_V_per_C_per_m2": float(k_skew),
            "k_normal_V_per_pC_per_m2": float(k_normal * PC),
            "k_skew_V_per_pC_per_m2": float(k_skew * PC),
            "K_normal_V_per_C_per_m3": float(K_normal),
            "K_skew_V_per_C_per_m3": float(K_skew),
            "K_normal_V_per_pC_per_m3": float(K_normal * PC),
            "K_skew_V_per_pC_per_m3": float(K_skew * PC),

            "U_CST_normalisation_J": float(U_J),
            "U_CST_normalisation_note": (
                "Agreed RF multipole convention: "
                "k_parallel=|c0|^2/(4U), K_parallel=k_parallel/d; "
                "k_perp=(c/(4U omega))(|c1|^2+|s1|^2), K_perp=k_perp/d; "
                "k_Q=4(c/omega)sqrt(|c2|^2+|s2|^2)/sqrt(4U), K_Q=k_Q/d."
            ),
        })
    return out


# -----------------------------------------------------------------------------
# Workflow helpers
# -----------------------------------------------------------------------------

@dataclass
class FieldAnalysisConfig:
    f_010: float = 1.3e9
    radius_m: float | None = None
    design_length_m: float | None = None
    beta: float = 1.0
    # Pixel radii retained from the validated Hessian/Taylor comparisons.
    fit_pixels: int = 4
    fit_pixels_dipole: int = 4
    fit_pixels_quadrupole: int = 8
    n_phi: int = 720
    charge_C: float = 1.0
    centred_z: bool = False


def pillbox_radius_from_f010(f_010: float) -> float:
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
    """Return the parent design frequency when available.

    This multipole diagnostic workflow is allowed to continue when the parent
    family dictionaries are incomplete.  If a family/mode is missing, use the
    crossing frequency and print a clear warning rather than raising.  This keeps
    the batch workflow running and makes the frequency source visible in the
    saved output/summary.
    """
    if family_data_by_m is None:
        print(
            f"WARNING: no family_data_by_m supplied; using fallback/crossing "
            f"frequency {fallback_Hz:.12e} Hz for {mode_name}."
        )
        return float(fallback_Hz)

    try:
        _, mnp = mode_name.split("_", 1)
        m = int(mnp[0])
    except Exception as exc:
        print(
            f"WARNING: could not parse mode_name={mode_name!r}; using "
            f"fallback/crossing frequency {fallback_Hz:.12e} Hz."
        )
        return float(fallback_Hz)

    if m not in family_data_by_m:
        print(
            f"WARNING: no family data supplied for m={m} while looking up "
            f"{mode_name}; available families are {sorted(family_data_by_m)}. "
            f"Using fallback/crossing frequency {fallback_Hz:.12e} Hz."
        )
        return float(fallback_Hz)

    tm_table = family_data_by_m[m].get("TM", {})
    if mnp not in tm_table:
        print(
            f"WARNING: mode {mnp!r} not found in family m={m} while looking up "
            f"{mode_name}; available modes are {sorted(tm_table)}. Using "
            f"fallback/crossing frequency {fallback_Hz:.12e} Hz."
        )
        return float(fallback_Hz)

    return float(tm_table[mnp]["design_frequency_Hz"])


def analyse_field_Ez_multipole(
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
    dx = float(x_m[1] - x_m[0]) if len(x_m) > 1 else float("nan")
    dy = float(y_m[1] - y_m[0]) if len(y_m) > 1 else float("nan")
    pixel = 0.5 * (abs(dx) + abs(dy))

    rho_d = float(config.fit_pixels_dipole) * pixel
    rho_q = float(config.fit_pixels_quadrupole) * pixel
    if rho_q >= 0.95 * radius_m:
        raise ValueError("Quadrupole sampling radius is too close to the map boundary.")

    Vz_xy, z_m = complex_voltage_map_from_Ez(
        Ez_xyz,
        length_m=length_m,
        frequency_Hz=frequency_Hz,
        beta=config.beta,
        centred_z=config.centred_z,
    )

    multipoles = extract_rf_multipoles_from_Vz(
        Vz_xy,
        x_m,
        y_m,
        rho_dipole_m=rho_d,
        rho_quadrupole_m=rho_q,
        n_phi=config.n_phi,
        charge_C=config.charge_C,
    )

    energy = electric_energy_diagnostics_from_components(
        Ex_xyz=Ex_xyz,
        Ey_xyz=Ey_xyz,
        Ez_xyz=Ez_xyz,
        radius_m=radius_m,
        length_m=length_m,
    )
    fom = figures_of_merit_from_rf_multipoles(
        multipoles,
        frequency_Hz=frequency_Hz,
        U_J=energy["U_CST_equiv_J"],
        length_m=length_m,
    )

    ix0, iy0 = nx // 2, ny // 2
    return {
        "analysis_method": "rf_fourier_multipole",
        "length_m": float(length_m),
        "frequency_Hz": float(frequency_Hz),
        "transverse_pixel_x_m": dx,
        "transverse_pixel_y_m": dy,
        "axis_indices_xy": (int(ix0), int(iy0)),
        "longitudinal_pixel_m": float(z_m[1] - z_m[0]) if len(z_m) > 1 else float("nan"),
        "z_start_m": float(z_m[0]),
        "z_stop_m": float(z_m[-1]),
        "centred_z": bool(config.centred_z),
        "sampling_radii": {
            "dipole_pixels": int(config.fit_pixels_dipole),
            "quadrupole_pixels": int(config.fit_pixels_quadrupole),
            "rho_dipole_m": rho_d,
            "rho_quadrupole_m": rho_q,
            "n_phi": int(config.n_phi),
        },
        "multipole_coefficients": multipoles,
        "figures_of_merit": fom,
        "energy_diagnostics": energy,
    }


def _metric(fields: dict[str, Any], field: str, key: str) -> float:
    return float(fields[field]["figures_of_merit"].get(key, float("nan")))


def _ratio_max(fields: dict[str, Any], key: str) -> float:
    parent = max(abs(_metric(fields, "E1", key)), abs(_metric(fields, "E2", key)))
    mixed = max(abs(_metric(fields, "plus", key)), abs(_metric(fields, "minus", key)))
    return float(mixed / parent) if np.isfinite(parent) and parent > 0 else float("nan")


def compare_parent_and_mixed_figures(fields: dict[str, Any]) -> dict[str, Any]:
    metric_keys = {
        "K_parallel_V_per_pC_per_m": "K_parallel_V_per_pC_per_m",
        "K_perp_V_per_pC_per_m2": "K_perp_V_per_pC_per_m2",
        "K_Q_V_per_pC_per_m3": "K_Q_V_per_pC_per_m3",
    }
    out: dict[str, Any] = {}
    for label, key in metric_keys.items():
        out[label] = {
            "E1": _metric(fields, "E1", key),
            "E2": _metric(fields, "E2", key),
            "plus": _metric(fields, "plus", key),
            "minus": _metric(fields, "minus", key),
            "Rmax": _ratio_max(fields, key),
            "source_key": key,
        }
    return out


def analyse_heterotypic_crossing_folder_multipole(
    crossing_folder: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None = None,
    config: FieldAnalysisConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
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
        fields[name] = analyse_field_Ez_multipole(
            field_data[job["Ez_key"]],
            length_m=job["length_m"],
            frequency_Hz=job["frequency_Hz"],
            radius_m=radius_m,
            config=config,
            Ex_xyz=field_data.get(job["Ex_key"]),
            Ey_xyz=field_data.get(job["Ey_key"]),
        )
        fields[name].update({
            "mode": job["mode"],
            "Ez_key": job["Ez_key"],
            "Ex_key": job["Ex_key"],
            "Ey_key": job["Ey_key"],
            "frequency_source": job["frequency_source"],
        })

    comparison = compare_parent_and_mixed_figures(fields)

    family_data_sources = {}
    if family_data_by_m is not None:
        for m, data in family_data_by_m.items():
            family_data_sources[int(m)] = data.get("metadata", {}).get("source_file", "unknown")

    out = {
        "analysis_method": "rf_fourier_multipole",
        "crossing_folder": str(folder),
        "crossing": crossing,
        "mode_i": meta["mode_i"],
        "mode_j": meta["mode_j"],
        "family_data_sources": family_data_sources,
        "units": {
            "c0": "V/C",
            "c1_s1": "V/C/m",
            "c2_s2": "V/C/m^2",
            "k_parallel": "V/pC",
            "K_parallel": "V/pC/m_z",
            "k_perp": "V/pC/m_perp",
            "K_perp": "V/pC/m_perp/m_z",
            "k_Q": "V/pC/m_perp^2",
            "K_Q": "V/pC/m_perp^2/m_z",
            "U": "J",
        },
        "configuration": {
            "beta": float(config.beta),
            "charge_C": float(config.charge_C),
            "centred_z": bool(config.centred_z),
            "n_phi": int(config.n_phi),
            "fit_pixels_dipole_as_sampling_radius": int(config.fit_pixels_dipole),
            "fit_pixels_quadrupole_as_sampling_radius": int(config.fit_pixels_quadrupole),
            "primary_stored_energy_U": "U_CST_equiv = 0.5 eps0 int|E|^2 dV",
            "method_note": "Azimuthal Fourier/RF multipole projection; no Hessian/Taylor fit used for primary metrics.",
        },
        "fields": fields,
        "comparison": comparison,
    }

    if save:
        pickle_save(out, folder / "heterotypic_rf_multipole_analysis.pkl")
        write_text_summary(out, folder / "heterotypic_rf_multipole_summary.txt")
    return out


def write_text_summary(result: dict[str, Any], filename: str | Path) -> None:
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("RF/Fourier multipole heterotypic analysis")
    lines.append(f"crossing: {result.get('mode_i')} -- {result.get('mode_j')}")
    crossing = result.get("crossing", {})
    lines.append(f"ell = {crossing.get('length_factor', 'unknown')}")
    lines.append(f"f_cross = {crossing.get('frequency_Hz', 'unknown')} Hz")
    lines.append("")
    lines.append("ANALYSIS CONFIGURATION")
    for k, v in result.get("configuration", {}).items():
        lines.append(f"  {k:36s} = {v}")
    lines.append("")
    for name in ("E1", "E2", "plus", "minus"):
        f = result["fields"][name]
        fom = f["figures_of_merit"]
        mp = f["multipole_coefficients"]
        lines.append(f"{name:5s} {f.get('mode','')}")
        lines.append(f"  frequency_Hz            = {f['frequency_Hz']:.12e} ({f.get('frequency_source')})")
        lines.append(f"  length_m                = {f['length_m']:.12e}")
        lines.append(f"  rho_dipole_m            = {mp['rho_dipole_m']:.12e}")
        lines.append(f"  rho_quadrupole_m        = {mp['rho_quadrupole_m']:.12e}")
        lines.append(f"  c0_axis                 = {mp['c0_axis_V_per_C']:.12e}")
        lines.append(f"  c1                      = {mp['c1_cos_phi_V_per_C_per_m']:.12e}")
        lines.append(f"  s1                      = {mp['s1_sin_phi_V_per_C_per_m']:.12e}")
        lines.append(f"  c2                      = {mp['c2_cos_2phi_V_per_C_per_m2']:.12e}")
        lines.append(f"  s2                      = {mp['s2_sin_2phi_V_per_C_per_m2']:.12e}")
        lines.append(f"  k_parallel              = {fom.get('k_parallel_V_per_pC', float('nan')):.12e} V/pC")
        lines.append(f"  K_parallel              = {fom.get('K_parallel_V_per_pC_per_m', float('nan')):.12e} V/pC/m_z")
        lines.append(f"  k_perp                  = {fom.get('k_perp_V_per_pC_per_m', float('nan')):.12e} V/pC/m_perp")
        lines.append(f"  K_perp                  = {fom.get('K_perp_V_per_pC_per_m2', float('nan')):.12e} V/pC/m_perp/m_z")
        lines.append(f"  k_Q                     = {fom.get('k_Q_V_per_pC_per_m2', float('nan')):.12e} V/pC/m_perp^2")
        lines.append(f"  K_Q                     = {fom.get('K_Q_V_per_pC_per_m3', float('nan')):.12e} V/pC/m_perp^2/m_z")
        lines.append("")
    lines.append("COMPARISON Rmax")
    for label, row in result.get("comparison", {}).items():
        lines.append(f"  {label:30s} Rmax = {row.get('Rmax', float('nan')):.12g}")
    p.write_text("\n".join(lines), encoding="utf-8")


def find_crossing_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    folders = []
    for p in root.rglob("field_data.npz"):
        folder = p.parent
        if (folder / "heterotypic_crossing_analysis.pkl").exists() and is_valid_npz_file(p):
            folders.append(folder)
    return sorted(set(folders))


def load_family_data_files(*filenames: str | Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for filename in filenames:
        filename = Path(filename)
        if not filename.exists():
            continue
        data = pickle_load(filename)
        # Infer m from first TM key if metadata does not contain it.
        m_val = data.get("metadata", {}).get("m", None) if isinstance(data, dict) else None
        if m_val is None:
            keys = list(data.get("TM", {}).keys())
            if not keys:
                continue
            m_val = int(str(keys[0])[0])
        data.setdefault("metadata", {})["source_file"] = str(filename)
        out[int(m_val)] = data
    return out


def analyse_all_heterotypic_crossings_multipole(
    root: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None = None,
    config: FieldAnalysisConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    folders = find_crossing_folders(root)
    results: dict[str, Any] = {}
    for folder in folders:
        print(f"Analysing RF multipoles: {folder}")
        results[str(folder)] = analyse_heterotypic_crossing_folder_multipole(
            folder,
            family_data_by_m=family_data_by_m,
            config=config,
            save=save,
        )
    if save:
        pickle_save(results, Path(root) / "all_heterotypic_rf_multipole_analyses.pkl")
    return results


# Backwards-friendly aliases with explicit multipole filenames/results.
analyse_heterotypic_crossing_folder = analyse_heterotypic_crossing_folder_multipole
analyse_all_heterotypic_crossings = analyse_all_heterotypic_crossings_multipole


# -----------------------------------------------------------------------------
# Example command-line use
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit these paths for your machine.
    heterotypic_root = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings")
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    voxel_res = 151

    family_files = [
        datapath / f"TMm0_TMm0_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
    ]
    existing = [f for f in family_files if f.exists()]
    family_data = load_family_data_files(*existing) if existing else None

    missing = [f for f in family_files if not f.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(f) for f in missing))

    family_data = load_family_data_files(*family_files)
    print(f"Loaded family keys: {family_data.keys()}")

    cfg = FieldAnalysisConfig(
        f_010=1.3e9,
        fit_pixels_dipole=4,
        fit_pixels_quadrupole=8,
        n_phi=720,
        charge_C=1.0,
        centred_z=False,
    )
    analyse_all_heterotypic_crossings_multipole(
        heterotypic_root,
        family_data_by_m=family_data,
        config=cfg,
        save=True,
    )
