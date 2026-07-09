#!/usr/bin/env python3
"""
brett_quadrupole_KQ_with_pkl_frequencies.py

Apply David Brett's azimuthal RF-multipole method to Ez field maps stored in
field_data.npz and print K_Q for the four fields E1, E2, plus and minus.

This version also reads crossing_analysis.pkl from the same directory as the
.npz file and uses the appropriate frequency metadata for each field where it is
available. If the pickle only contains one crossing frequency, that value is
used as the fallback for all four fields.

Method
------
1. Load Ez(x,y,z) for E1, E2, plus and minus.
2. Sample Ez on a transverse circle of radius rho.
3. Integrate longitudinally:

       Vz(rho,phi) = int Ez(rho,phi,z) exp(i omega z / beta c) dz.

4. Extract the n=2 azimuthal Fourier coefficients from

       Vz(rho,phi) ~= rho^2 [ b2 cos(2 phi) + a2 sin(2 phi) ].

5. Convert the quadrupole coefficient to the local normal/skew quadrupole
   gradient magnitude

       K_grad = (2 c / omega) sqrt(|b2|^2 + |a2|^2).

   The signed-K Hessian scripts define their scalar quadrupole strength as

       K_Q = sqrt((Kxx - Kyy)^2 + 4 Kxy^2),

   so for an ideal quadrupole K_Q = 2 K_grad.  This script therefore prints
   both K_grad and the Hessian-equivalent K_Q, plus the same reported
   normalisation used by the signed-K quadrupole analysis:

       K_Q_reported = K_Q / sqrt(4 U_CST) / L * 1e-12  [V/pC/m^3].

Arrays are assumed to be indexed as Ez[x, y, z].
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle as pkl
from typing import Any

import numpy as np
from scipy.special import jn_zeros

C0 = 299_792_458.0
FIELD_NAMES = ("E1", "E2", "plus", "minus")


def tm_root_v_mn(m: int, n: int) -> float:
    """Return the nth zero of J_m used for TM_mnp pillbox modes."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(jn_zeros(int(m), int(n))[-1])


def pillbox_radius_from_f010(f_010_Hz: float) -> float:
    """Pillbox radius fixed by the TM010 frequency."""
    return float(tm_root_v_mn(0, 1) * C0 / (2.0 * np.pi * float(f_010_Hz)))


def design_length_from_f010(f_010_Hz: float) -> float:
    """Design half-wavelength length used for parent fields."""
    return float(C0 / float(f_010_Hz) / 2.0)


def f_tm(m: int, n: int, p: int, R_m: float, L_m: float) -> float:
    """Analytical TM_mnp pillbox frequency."""
    v = tm_root_v_mn(m, n)
    return float((C0 / (2.0 * np.pi)) * np.sqrt((v / float(R_m)) ** 2 + (int(p) * np.pi / float(L_m)) ** 2))


def parse_tm_mnp(mode_name: str) -> tuple[int, int, int]:
    """Parse strings such as 'TM_213', 'TM213' or '213'."""
    text = str(mode_name).strip().replace("TM_", "").replace("TM", "")
    if len(text) != 3 or not text.isdigit():
        raise ValueError(f"Could not parse TM mnp mode from {mode_name!r}")
    return int(text[0]), int(text[1]), int(text[2])


def parent_design_frequency_from_meta(meta: dict[str, Any], field_name: str, f_010_Hz: float) -> tuple[float | None, str | None]:
    """Derive E1/E2 parent design frequencies from mode_i/mode_j metadata.

    This matches the signed-K Hessian workflow: E1 and E2 are parent fields
    evaluated at their design length L=lambda_010/2, while plus/minus use the
    crossing frequency and crossing length.
    """
    if field_name not in ("E1", "E2"):
        return None, None
    mode_key = "mode_i" if field_name == "E1" else "mode_j"
    mode_name = None
    for path in [(mode_key,), ("crossing", mode_key)]:
        cur = meta
        try:
            for key in path:
                cur = cur[key]
            mode_name = str(cur)
            break
        except Exception:
            pass
    if mode_name is None:
        return None, None
    try:
        m, n, p = parse_tm_mnp(mode_name)
        R = pillbox_radius_from_f010(f_010_Hz)
        L = design_length_from_f010(f_010_Hz)
        return f_tm(m, n, p, R, L), f"derived from {'.'.join(path)} at L=lambda_010/2"
    except Exception:
        return None, None


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def pickle_load(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pkl.load(handle)


def pickle_save(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_Ez_fields(field_data_fname: str | Path) -> dict[str, np.ndarray]:
    """Load Ez fields from field_data.npz."""
    field_data_fname = Path(field_data_fname)

    with np.load(field_data_fname, allow_pickle=True) as data:
        def get_one(candidates: tuple[str, ...], label: str) -> np.ndarray:
            for key in candidates:
                if key in data:
                    return np.asarray(data[key], dtype=float)
            raise KeyError(f"Could not find {label} Ez field in {field_data_fname}")

        return {
            "E1": get_one(("E1_Ez", "Ez1"), "E1"),
            "E2": get_one(("E2_Ez", "Ez2"), "E2"),
            "plus": get_one(("Ez_plus", "plus_Ez"), "plus"),
            "minus": get_one(("Ez_minus", "minus_Ez"), "minus"),
        }


def _scalar_or_none(value: Any) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            val = float(arr.reshape(-1)[0])
            if np.isfinite(val):
                return val
    except Exception:
        pass
    return None


def _nested_get_scalar(obj: Any, key_path: tuple[str, ...]) -> float | None:
    cur = obj
    try:
        for key in key_path:
            cur = cur[key]
        return _scalar_or_none(cur)
    except Exception:
        return None


def _first_present_scalar(obj: Any, paths: list[tuple[str, ...]]) -> tuple[float | None, str | None]:
    for path in paths:
        val = _nested_get_scalar(obj, path)
        if val is not None:
            return val, ".".join(path)
    return None, None


def load_npz_coordinate_arrays(field_data_fname: str | Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return x, y, z arrays if present in field_data.npz, otherwise None."""
    with np.load(field_data_fname, allow_pickle=True) as data:
        def arr_or_none(candidates: tuple[str, ...]) -> np.ndarray | None:
            for key in candidates:
                if key in data:
                    arr = np.asarray(data[key], dtype=float)
                    if arr.ndim == 1:
                        return arr
            return None

        x_arr = arr_or_none(("x_m", "x", "xs", "x_grid_m"))
        y_arr = arr_or_none(("y_m", "y", "ys", "y_grid_m"))
        z_arr = arr_or_none(("z_m", "z", "zs", "z_grid_m"))

    return x_arr, y_arr, z_arr


# -----------------------------------------------------------------------------
# Metadata from crossing_analysis.pkl
# -----------------------------------------------------------------------------

def load_crossing_metadata(crossing_pkl: str | Path | None) -> dict[str, Any]:
    if crossing_pkl is None:
        return {}
    crossing_pkl = Path(crossing_pkl)
    if not crossing_pkl.exists():
        return {}
    return pickle_load(crossing_pkl)


def frequency_for_field(meta: dict[str, Any], field_name: str, cli_frequency_Hz: float | None, f_010_Hz: float) -> tuple[float, str]:
    """
    Return frequency_Hz for E1/E2/plus/minus and a short source description.

    The loader is deliberately permissive because crossing_analysis.pkl files
    have evolved during the analysis. It first tries field-specific entries and
    then falls back to the crossing/global frequency.
    """
    if cli_frequency_Hz is not None:
        return float(cli_frequency_Hz), "--frequency-Hz"

    aliases = {
        "E1": ("E1", "field_E1", "parent1", "parent_1", "mode_i", "i"),
        "E2": ("E2", "field_E2", "parent2", "parent_2", "mode_j", "j"),
        "plus": ("plus", "Eplus", "E_plus"),
        "minus": ("minus", "Eminus", "E_minus"),
    }[field_name]

    # Field-specific keys first.
    field_paths: list[tuple[str, ...]] = []
    for alias in aliases:
        field_paths += [
            ("focusing", alias, "frequency_Hz"),
            ("focusing", alias, "freq_Hz"),
            ("focusing", alias, "f_Hz"),
            (alias, "frequency_Hz"),
            (alias, "freq_Hz"),
            (f"{alias}_frequency_Hz",),
            (f"frequency_{alias}_Hz",),
            (f"freq_{alias}_Hz",),
        ]

    val, src = _first_present_scalar(meta, field_paths)
    if val is not None:
        return val, f"crossing_analysis.pkl:{src}"

    # Parent aliases that are common in crossing dictionaries.  If no explicit
    # parent frequency is present, derive it analytically at the design length
    # L=lambda_010/2.  Do this BEFORE falling back to the crossing frequency.
    if field_name == "E1":
        parent_paths = [
            ("crossing", "mode_i_frequency_Hz"),
            ("crossing", "frequency_i_Hz"),
            ("crossing", "freq_i_Hz"),
            ("mode_i_frequency_Hz",),
            ("frequency_i_Hz",),
        ]
        val, src = _first_present_scalar(meta, parent_paths)
        if val is not None:
            return val, f"crossing_analysis.pkl:{src}"
        val, src = parent_design_frequency_from_meta(meta, field_name, f_010_Hz)
        if val is not None:
            return val, src or "derived parent design frequency"

    if field_name == "E2":
        parent_paths = [
            ("crossing", "mode_j_frequency_Hz"),
            ("crossing", "frequency_j_Hz"),
            ("crossing", "freq_j_Hz"),
            ("mode_j_frequency_Hz",),
            ("frequency_j_Hz",),
        ]
        val, src = _first_present_scalar(meta, parent_paths)
        if val is not None:
            return val, f"crossing_analysis.pkl:{src}"
        val, src = parent_design_frequency_from_meta(meta, field_name, f_010_Hz)
        if val is not None:
            return val, src or "derived parent design frequency"

    # Mixed/crossing/global frequency fallback.  This is appropriate for plus
    # and minus.  For E1/E2, reaching here means there was not enough metadata
    # to derive the parent mode frequency.
    fallback_paths = [
        ("crossing", "frequency_Hz"),
        ("crossing", "freq_Hz"),
        ("frequency_Hz",),
        ("freq_Hz",),
        ("f_Hz",),
    ]
    val, src = _first_present_scalar(meta, fallback_paths)
    if val is not None:
        return val, f"crossing_analysis.pkl:{src}"

    raise ValueError(
        f"No frequency found for {field_name}. Pass --frequency-Hz or add frequency metadata to crossing_analysis.pkl."
    )


def pixel_spacing_from_meta(meta: dict[str, Any], field_name: str, axis: str) -> tuple[float | None, str | None]:
    """axis is 'transverse' or 'longitudinal'."""
    key = f"{axis}_pixel_m"
    aliases = {
        "E1": ("E1", "field_E1", "parent1", "parent_1", "mode_i", "i"),
        "E2": ("E2", "field_E2", "parent2", "parent_2", "mode_j", "j"),
        "plus": ("plus", "Eplus", "E_plus"),
        "minus": ("minus", "Eminus", "E_minus"),
    }[field_name]

    paths: list[tuple[str, ...]] = []
    for alias in aliases:
        paths += [("focusing", alias, key), (alias, key), (f"{alias}_{key}",)]
    paths += [(key,),]

    return _first_present_scalar(meta, paths)


def construct_centered_axis(n: int, step_m: float) -> np.ndarray:
    """Axis centred on zero, matching arrays indexed with the beam axis at the middle pixel."""
    mid = (n - 1) / 2.0
    return (np.arange(n, dtype=float) - mid) * float(step_m)


def construct_z_axis(n: int, dz_m: float) -> np.ndarray:
    """Longitudinal axis from 0 to (n-1) dz."""
    return np.arange(n, dtype=float) * float(dz_m)


def coordinates_for_field(
    field: np.ndarray,
    field_name: str,
    *,
    meta: dict[str, Any],
    npz_x: np.ndarray | None,
    npz_y: np.ndarray | None,
    npz_z: np.ndarray | None,
    cli_cavity_radius_m: float | None,
    cli_length_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Return x, y, z coordinate arrays for one field."""
    nx, ny, nz = field.shape
    sources: dict[str, str] = {}

    if npz_x is not None:
        x_arr = npz_x
        sources["x"] = "field_data.npz:x"
    else:
        dx, dx_src = pixel_spacing_from_meta(meta, field_name, "transverse")
        if dx is not None:
            x_arr = construct_centered_axis(nx, dx)
            sources["x"] = f"crossing_analysis.pkl:{dx_src}"
        elif cli_cavity_radius_m is not None:
            x_arr = np.linspace(-cli_cavity_radius_m, cli_cavity_radius_m, nx)
            sources["x"] = "--cavity-radius-m"
        else:
            raise ValueError("No x-coordinate metadata found. Pass --cavity-radius-m.")

    if npz_y is not None:
        y_arr = npz_y
        sources["y"] = "field_data.npz:y"
    else:
        dy, dy_src = pixel_spacing_from_meta(meta, field_name, "transverse")
        if dy is not None:
            y_arr = construct_centered_axis(ny, dy)
            sources["y"] = f"crossing_analysis.pkl:{dy_src}"
        elif cli_cavity_radius_m is not None:
            y_arr = np.linspace(-cli_cavity_radius_m, cli_cavity_radius_m, ny)
            sources["y"] = "--cavity-radius-m"
        else:
            raise ValueError("No y-coordinate metadata found. Pass --cavity-radius-m.")

    if npz_z is not None and len(npz_z) == nz:
        z_arr = npz_z
        sources["z"] = "field_data.npz:z"
    else:
        dz, dz_src = pixel_spacing_from_meta(meta, field_name, "longitudinal")
        if dz is not None:
            z_arr = construct_z_axis(nz, dz)
            sources["z"] = f"crossing_analysis.pkl:{dz_src}"
        elif cli_length_m is not None:
            z_arr = np.linspace(0.0, cli_length_m, nz)
            sources["z"] = "--length-m"
        else:
            raise ValueError(f"No z-coordinate metadata found for {field_name}. Pass --length-m.")

    if len(x_arr) != nx or len(y_arr) != ny or len(z_arr) != nz:
        raise ValueError(
            f"Coordinate lengths do not match {field_name} shape {field.shape}: "
            f"len(x)={len(x_arr)}, len(y)={len(y_arr)}, len(z)={len(z_arr)}"
        )

    return x_arr, y_arr, z_arr, sources


# -----------------------------------------------------------------------------
# Brett-style azimuthal quadrupole extraction
# -----------------------------------------------------------------------------

def bilinear_interpolate_xy_all_z(
    field_xyz: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate field[x,y,z] at many (xq,yq), returning [n_phi,n_z]."""
    if field_xyz.ndim != 3:
        raise ValueError(f"Expected 3D field array, got {field_xyz.shape}")
    if not (np.all(np.diff(x_arr) > 0.0) and np.all(np.diff(y_arr) > 0.0)):
        raise ValueError("x_arr and y_arr must be strictly increasing.")

    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    if xq.shape != yq.shape:
        raise ValueError("xq and yq must have the same shape.")

    ix = np.searchsorted(x_arr, xq) - 1
    iy = np.searchsorted(y_arr, yq) - 1
    ix = np.clip(ix, 0, len(x_arr) - 2)
    iy = np.clip(iy, 0, len(y_arr) - 2)

    x0 = x_arr[ix]
    x1 = x_arr[ix + 1]
    y0 = y_arr[iy]
    y1 = y_arr[iy + 1]
    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    f00 = field_xyz[ix, iy, :]
    f10 = field_xyz[ix + 1, iy, :]
    f01 = field_xyz[ix, iy + 1, :]
    f11 = field_xyz[ix + 1, iy + 1, :]

    return (
        (1.0 - tx)[:, None] * (1.0 - ty)[:, None] * f00
        + tx[:, None] * (1.0 - ty)[:, None] * f10
        + (1.0 - tx)[:, None] * ty[:, None] * f01
        + tx[:, None] * ty[:, None] * f11
    )


def longitudinal_voltage_on_circle(
    Ez_xyz: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    *,
    sample_radius_m: float,
    frequency_Hz: float,
    beta: float,
    n_phi: int,
    phase_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if phase_sign not in (-1, +1):
        raise ValueError("phase_sign must be +1 or -1.")

    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    xq = sample_radius_m * np.cos(phi)
    yq = sample_radius_m * np.sin(phi)

    if np.any(xq < x_arr[0]) or np.any(xq > x_arr[-1]) or np.any(yq < y_arr[0]) or np.any(yq > y_arr[-1]):
        raise ValueError("sample_radius_m is outside the available transverse grid.")

    Ez_phi_z = bilinear_interpolate_xy_all_z(Ez_xyz, x_arr, y_arr, xq, yq)
    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = np.exp(1j * phase_sign * omega * z_arr / (beta * C0))
    Vz_phi = np.trapezoid(Ez_phi_z * phase[None, :], z_arr, axis=1)
    return phi, Vz_phi


def brett_quadrupole_coefficients(phi: np.ndarray, Vz_phi: np.ndarray, *, sample_radius_m: float) -> tuple[complex, complex]:
    """Extract b2, a2 from Vz(phi) ~= rho^2[b2 cos(2phi)+a2 sin(2phi)]."""
    rho = float(sample_radius_m)
    if rho <= 0.0:
        raise ValueError("sample_radius_m must be positive.")
    b2 = (2.0 / len(phi)) * np.sum(Vz_phi * np.cos(2.0 * phi)) / rho**2
    a2 = (2.0 / len(phi)) * np.sum(Vz_phi * np.sin(2.0 * phi)) / rho**2
    return complex(b2), complex(a2)


def KQ_from_brett_coefficients(b2: complex, a2: complex, *, frequency_Hz: float) -> float:
    omega = 2.0 * np.pi * float(frequency_Hz)
    return float((2.0 * C0 / omega) * np.sqrt(abs(b2) ** 2 + abs(a2) ** 2))


def analyse_field_KQ(
    Ez_xyz: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    *,
    sample_radius_m: float,
    frequency_Hz: float,
    beta: float,
    n_phi: int,
    phase_sign: int,
) -> dict[str, complex | float]:
    phi, Vz_phi = longitudinal_voltage_on_circle(
        Ez_xyz,
        x_arr,
        y_arr,
        z_arr,
        sample_radius_m=sample_radius_m,
        frequency_Hz=frequency_Hz,
        beta=beta,
        n_phi=n_phi,
        phase_sign=phase_sign,
    )
    b2, a2 = brett_quadrupole_coefficients(phi, Vz_phi, sample_radius_m=sample_radius_m)
    return {
        "b2": b2,
        "a2": a2,
        "K_Q_raw": KQ_from_brett_coefficients(b2, a2, frequency_Hz=frequency_Hz),
    }


def energy_for_field(meta: dict[str, Any], field_name: str) -> float | None:
    paths = [
        ("focusing", field_name, "U_CST_J"),
        ("focusing", field_name, "energy_diagnostics", "U_CST_J"),
        ("focusing", field_name, "energy_diagnostics", "U_used_J"),
        (f"{field_name}_U_CST_J",),
        (f"U_{field_name}_J",),
        ("U_CST_J",),
        ("U_used_J",),
        ("U_CST_equiv_J",),
    ]
    val, _ = _first_present_scalar(meta, paths)
    return val


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Brett-style quadrupole K values for E1, E2, plus and minus, using parent design frequencies for E1/E2 and crossing frequency for plus/minus."
    )
    parser.add_argument(
        "--field-data",
        type=Path,
        default=None,
        help="Path to field_data.npz. If omitted, uses DIRECTORY/field_data.npz.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles\TM220_TM213"),
        help="Directory containing field_data.npz and crossing_analysis.pkl.",
    )
    parser.add_argument(
        "--crossing-analysis",
        type=Path,
        default=None,
        help="Path to crossing_analysis.pkl. If omitted, uses the same directory as field_data.npz.",
    )
    parser.add_argument("--frequency-Hz", type=float, default=None, help="Override frequency in Hz for all four fields.")
    parser.add_argument("--frequency-010-Hz", type=float, default=1.3e9, help="TM010 reference frequency used to derive parent design frequencies and lengths.")
    parser.add_argument("--length-m", type=float, default=None, help="Fallback integration length if z metadata is absent.")
    parser.add_argument("--cavity-radius-m", type=float, default=None, help="Fallback transverse half-width if x/y metadata is absent.")
    parser.add_argument("--sample-radius-m", type=float, default=None, help="Azimuthal probe radius rho in m.")
    parser.add_argument("--sample-radius-fraction", type=float, default=0.10, help="rho / transverse half-range if --sample-radius-m is not supplied.")
    parser.add_argument("--beta", type=float, default=1.0, help="Particle beta used in phase factor.")
    parser.add_argument("--n-phi", type=int, default=720, help="Number of azimuthal samples.")
    parser.add_argument("--phase-sign", type=int, choices=(-1, 1), default=1, help="Sign in exp(+-i omega z/beta c).")
    parser.add_argument("--save-pkl", action="store_true", help="Save results to brett_KQ_results.pkl.")
    args = parser.parse_args()

    field_data_fname = args.field_data if args.field_data is not None else args.directory / "field_data.npz"
    field_data_fname = Path(field_data_fname)
    directory = field_data_fname.parent

    crossing_pkl = args.crossing_analysis if args.crossing_analysis is not None else directory / "crossing_analysis.pkl"
    meta = load_crossing_metadata(crossing_pkl)

    fields = load_Ez_fields(field_data_fname)
    npz_x, npz_y, npz_z = load_npz_coordinate_arrays(field_data_fname)

    print("Brett-style azimuthal quadrupole extraction")
    print(f"field_data        = {field_data_fname}")
    print(f"crossing_analysis = {crossing_pkl if Path(crossing_pkl).exists() else 'not found'}")
    print(f"n_phi             = {args.n_phi}")
    print(f"phase_sign        = {args.phase_sign:+d}")
    print(f"beta              = {args.beta:.12g}")
    print(f"frequency_010_Hz  = {args.frequency_010_Hz:.12e}")
    print("")

    results: dict[str, dict[str, Any]] = {}

    for name in FIELD_NAMES:
        field = fields[name]
        frequency_Hz, freq_source = frequency_for_field(meta, name, args.frequency_Hz, args.frequency_010_Hz)
        x_arr, y_arr, z_arr, coord_sources = coordinates_for_field(
            field,
            name,
            meta=meta,
            npz_x=npz_x,
            npz_y=npz_y,
            npz_z=npz_z,
            cli_cavity_radius_m=args.cavity_radius_m,
            cli_length_m=args.length_m,
        )

        # input(f"{args.sample_radius_m = }")

        if args.sample_radius_m is not None:
            sample_radius_m = float(args.sample_radius_m)
        else:
            half_range = min(abs(x_arr[0]), abs(x_arr[-1]), abs(y_arr[0]), abs(y_arr[-1]))
            sample_radius_m = float(args.sample_radius_fraction) * half_range
            print(f"{args.sample_radius_fraction = }")
            print(f"{sample_radius_m = }")

            # sample_radius_m = 1.e-3  # 4.953120376861e-01
            # sample_radius_m = 2.e-3  # 5.088597035976e-01
            # sample_radius_m = 3.e-3  # 4.994739061058e-01
            # sample_radius_m = 4.e-3  # 4.943865729787e-01
            # sample_radius_m = 5.e-3  # 4.914824374413e-01

        res = analyse_field_KQ(
            field,
            x_arr,
            y_arr,
            z_arr,
            sample_radius_m=sample_radius_m,
            frequency_Hz=frequency_Hz,
            beta=args.beta,
            n_phi=args.n_phi,
            phase_sign=args.phase_sign,
        )

        U = energy_for_field(meta, name)
        result_record: dict[str, Any] = {
            **res,
            "frequency_Hz": frequency_Hz,
            "frequency_source": freq_source,
            "sample_radius_m": sample_radius_m,
            "z_start_m": float(z_arr[0]),
            "z_end_m": float(z_arr[-1]),
            "x_source": coord_sources["x"],
            "y_source": coord_sources["y"],
            "z_source": coord_sources["z"],
        }
        length_used_m = float(z_arr[-1] - z_arr[0])
        result_record["length_used_m"] = length_used_m
        if U is not None and U > 0.0:
            K_grad_raw = float(res["K_Q_raw"])
            KQ_hessian_equiv_raw = 2.0 * K_grad_raw
            result_record["U_CST_J"] = U
            result_record["K_grad_over_sqrt_4U"] = K_grad_raw / float(np.sqrt(4.0 * U))
            result_record["K_Q_hessian_equiv_raw"] = KQ_hessian_equiv_raw
            result_record["K_Q_hessian_equiv_over_sqrt_4U"] = KQ_hessian_equiv_raw / float(np.sqrt(4.0 * U))
            result_record["K_Q_reported_V_per_pC_per_m3"] = (
                KQ_hessian_equiv_raw / float(np.sqrt(4.0 * U)) / length_used_m * 1.0e-12
            )

        results[name] = result_record

        b2 = res["b2"]
        a2 = res["a2"]
        KQ = float(res["K_Q_raw"])

        print(f"{name:5s}")
        print(f"  frequency_Hz = {frequency_Hz:.12e}  ({freq_source})")
        print(f"  z range      = {z_arr[0]:.12e} to {z_arr[-1]:.12e} m  ({coord_sources['z']})")
        print(f"  rho          = {sample_radius_m:.12e} m")
        print(f"  b2           = {b2.real:.12e} {b2.imag:+.12e}j")
        print(f"  a2           = {a2.real:.12e} {a2.imag:+.12e}j")
        length_used_m = float(z_arr[-1] - z_arr[0])
        KQ_hessian_equiv = 2.0 * KQ
        print(f"  K_grad_raw_Brett            = {KQ:.12e}")
        print(f"  K_Q_raw_hessian_equiv       = {KQ_hessian_equiv:.12e}  (= 2*K_grad_raw_Brett)")
        print(f"  length_used_m               = {length_used_m:.12e}")
        if U is not None and U > 0.0:
            print(f"  U_CST_J                     = {U:.12e}")
            print(f"  K_grad/sqrt(4U)             = {KQ / np.sqrt(4.0 * U):.12e}")
            print(f"  K_Q_hessian_equiv/sqrt(4U)  = {KQ_hessian_equiv / np.sqrt(4.0 * U):.12e}")
            print(
                f"  K_Q_reported                = "
                f"{KQ_hessian_equiv / np.sqrt(4.0 * U) / length_used_m * 1.0e-12:.12e} V/pC/m^3"
            )
            hess_ref = _nested_get_scalar(meta, ("focusing", name, "K_Q_V_per_pC_per_m3"))
            if hess_ref is not None:
                print(f"  Hessian K_Q reference        = {hess_ref:.12e} V/pC/m^3")
        print("")

    if args.save_pkl:
        out = directory / "brett_KQ_results.pkl"
        pickle_save(results, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
