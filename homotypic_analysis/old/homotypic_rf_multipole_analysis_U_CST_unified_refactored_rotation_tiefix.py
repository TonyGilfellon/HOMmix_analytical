"""Unified homotypic RF/Fourier multipole analysis for TM m=0, 1 and 2.

This standalone module keeps the established analytical homotypic workflow:

1. Build/load parent-family data for TM_0np, TM_1np and TM_2np.
2. Find like-family crossings as the pillbox length factor is varied.
3. For m=1 and m=2, rotate each parent so the global |E| maximum lies in
   the vertical longitudinal section |E|[mid_x, :, :], then form E+ and E-.
4. Save ``field_data.npz`` in each crossing folder.
5. Calculate the complex longitudinal-voltage map

       Vz(x,y) = integral Ez(x,y,z) exp(i omega z/(beta c)) dz.

6. Extract c0, c1/s1 and c2/s2 by azimuthal Fourier projection.
7. Store both integrated and structure-length-normalised metrics:

       k_parallel,  K_parallel = k_parallel/d
       k_perp,      K_perp     = k_perp/d
       k_Q,         K_Q        = k_Q/d

The quadrupole (m=2) extraction is the same second-harmonic method used in the
Brett-style analysis.  The same general RF multipole decomposition is used for
all three homotypic families.

Array convention: field[x_index, y_index, z_index].
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import jn_zeros

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1.0e-12


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def pickle_save(obj: Any, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename: str | Path) -> Any:
    with Path(filename).open("rb") as f:
        return pickle.load(f)


def save_npz_dict(filename: str | Path, data: dict[str, np.ndarray]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, **data)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# Analytical pillbox frequencies and fields
# -----------------------------------------------------------------------------

def tm_root_v_mn(m: int, n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(jn_zeros(int(m), int(n))[-1])


def pillbox_radius_from_freq(f_Hz: float) -> float:
    return float(tm_root_v_mn(0, 1) * C0 / (2.0 * np.pi * float(f_Hz)))


def f_tm(m: int, n: int, p: int, R: float, L: float) -> float:
    if R <= 0.0 or L <= 0.0:
        raise ValueError("R and L must be positive")
    if p < 0:
        raise ValueError("p must be >= 0")
    v = tm_root_v_mn(m, n)
    return float((C0 / (2.0 * np.pi)) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2))


def _tm_e_field_cylindrical(
    r: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    *,
    m: int,
    n: int,
    p: int,
    R: float,
    L: float,
    E0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical TM_mnp field shape with dimensionally consistent E_perp."""
    kc = tm_root_v_mn(m, n) / R
    kz = p * np.pi / L
    arg = kc * r

    Jm = special.jv(m, arg)
    Jmp = special.jvp(m, arg, 1)
    cos_m = np.cos(m * theta)
    sin_m = np.sin(m * theta)
    cos_z = np.cos(kz * z)
    sin_z = np.sin(kz * z)

    Ez = E0 * Jm * cos_m * cos_z
    if p == 0:
        return np.zeros_like(Ez), np.zeros_like(Ez), Ez

    Er = -E0 * (kz / kc) * Jmp * cos_m * sin_z
    with np.errstate(divide="ignore", invalid="ignore"):
        J_over_r = Jm / r
        if m == 1:
            J_over_r = np.where(r == 0.0, kc / 2.0, J_over_r)
        else:
            J_over_r = np.where(r == 0.0, 0.0, J_over_r)
        Etheta = E0 * (kz / kc**2) * m * J_over_r * sin_m * sin_z
    return Er, Etheta, Ez


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
    E0: float = 1.0,
    dtype=np.float32,
) -> dict[str, np.ndarray]:
    x = np.linspace(-R, R, int(x_res))
    y = np.linspace(-R, R, int(y_res))
    z = np.linspace(0.0, L, int(z_res))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    mask = r <= R

    Er, Etheta, Ez = _tm_e_field_cylindrical(
        r, theta, Z, m=m, n=n, p=p, R=R, L=L, E0=E0
    )
    Ex = Er * np.cos(theta) - Etheta * np.sin(theta)
    Ey = Er * np.sin(theta) + Etheta * np.cos(theta)
    Eperp = np.hypot(Ex, Ey)
    Eabs = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(mask, a, np.nan).astype(dtype, copy=False)

    return {
        "Ex": masked(Ex),
        "Ey": masked(Ey),
        "Ez": masked(Ez),
        "Eperp": masked(Eperp),
        "|E|": masked(Eabs),
    }


def parse_mnp(mnp: str) -> tuple[int, int, int]:
    s = str(mnp).replace("TM_", "").replace("TM", "")
    if len(s) != 3 or not s.isdigit():
        raise ValueError(f"Expected three-digit mnp mode key, got {mnp!r}")
    return int(s[0]), int(s[1]), int(s[2])


# -----------------------------------------------------------------------------
# Family-data assembly and crossing search
# -----------------------------------------------------------------------------

def assemble_family_data(
    m: int,
    *,
    n_max: int,
    p_max: int,
    frequency_010: float,
    LF_start: float,
    LF_stop: float,
    param_sweep_resolution: int,
    voxel_res: int,
) -> dict[str, Any]:
    d0 = C0 / float(frequency_010) / 2.0
    R = pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, int(param_sweep_resolution))

    data: dict[str, Any] = {
        "TM": {},
        "length_factor_vector": length_factors.tolist(),
        "metadata": {
            "family_m": int(m),
            "frequency_010_Hz": float(frequency_010),
            "lambda_010_m": float(C0 / frequency_010),
            "design_length_m": float(d0),
            "pillbox_radius_m": float(R),
            "voxel_res": int(voxel_res),
        },
    }

    for n in range(1, int(n_max) + 1):
        for p in range(int(p_max) + 1):
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = pillbox_field_voxel_grid_xyz(
                R, d0, m, n, p, voxel_res, voxel_res, voxel_res, E0=1.0
            )
            freqs = np.asarray([f_tm(m, n, p, R, ell * d0) for ell in length_factors])
            f_design = f_tm(m, n, p, R, d0)
            data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": freqs.tolist(),
                "frequency_normalised": (freqs / frequency_010).tolist(),
                "design_frequency_Hz": float(f_design),
                "design_frequency_normalised": float(f_design / frequency_010),
            }
    return data


def find_like_family_crossings(family_data: dict[str, Any]) -> dict[str, Any]:
    ell = np.asarray(family_data["length_factor_vector"], dtype=float)
    modes = sorted(family_data["TM"])
    crossings: dict[str, Any] = {}
    crossed: set[str] = set()

    for i, mode_i in enumerate(modes):
        fi = np.asarray(family_data["TM"][mode_i]["frequency_Hz"], dtype=float)
        for mode_j in modes[i + 1 :]:
            fj = np.asarray(family_data["TM"][mode_j]["frequency_Hz"], dtype=float)
            diff = fi - fj
            brackets = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
            for idx in brackets:
                if np.isclose(diff[idx], 0.0):
                    ell_cross = float(ell[idx])
                elif np.isclose(diff[idx + 1], 0.0):
                    ell_cross = float(ell[idx + 1])
                else:
                    ell_cross = float(
                        brentq(
                            lambda x: np.interp(x, ell, fi) - np.interp(x, ell, fj),
                            float(ell[idx]),
                            float(ell[idx + 1]),
                        )
                    )
                f_cross = float(np.interp(ell_cross, ell, fi))
                key = f"TM_{mode_i}--TM_{mode_j}@{ell_cross:.8g}"
                if key in crossings:
                    continue
                crossings[key] = {
                    "mode_i": f"TM_{mode_i}",
                    "mode_j": f"TM_{mode_j}",
                    "length_factor": ell_cross,
                    "frequency_Hz": f_cross,
                    "pair_type": f"homotypic_m{mode_i[0]}",
                }
                crossed.update((f"TM_{mode_i}", f"TM_{mode_j}"))

    return {"TM": {"crossings": crossings, "modes_that_cross": sorted(crossed)}}



# -----------------------------------------------------------------------------
# Field rotation/alignment
# -----------------------------------------------------------------------------

def eabs_from_components(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
) -> np.ndarray:
    """Return |E|, treating NaNs outside the pillbox as zero."""
    return np.sqrt(
        np.nan_to_num(np.asarray(Ex, dtype=float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ey, dtype=float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ez, dtype=float), nan=0.0) ** 2
    )


def rotation_angle_to_vertical_plane(
    Eabs: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[float, tuple[int, int, int]]:
    """Return the active z-rotation that moves the brightest voxel to x=0."""
    peak = tuple(
        int(v)
        for v in np.unravel_index(
            np.nanargmax(Eabs),
            Eabs.shape,
        )
    )
    ix, iy, _ = peak
    phi = np.arctan2(float(y_m[iy]), float(x_m[ix]))

    # After active rotation by alpha: x' = r cos(phi + alpha).
    # Select the smallest-magnitude solution of x'=0.
    candidates = np.asarray([
        np.pi / 2.0 - phi,
        -np.pi / 2.0 - phi,
    ])
    candidates = (candidates + np.pi) % (2.0 * np.pi) - np.pi
    alpha = float(candidates[np.argmin(np.abs(candidates))])
    return float(np.degrees(alpha)), peak


def rotate_vector_field_about_z(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    angle_deg: float,
    *,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Actively rotate both the field map and vector components about z."""
    Ex0 = np.nan_to_num(np.asarray(Ex, dtype=float), nan=fill_value)
    Ey0 = np.nan_to_num(np.asarray(Ey, dtype=float), nan=fill_value)
    Ez0 = np.nan_to_num(np.asarray(Ez, dtype=float), nan=fill_value)

    interpolators = [
        RegularGridInterpolator(
            (x_m, y_m, z_m),
            component,
            bounds_error=False,
            fill_value=fill_value,
        )
        for component in (Ex0, Ey0, Ez0)
    ]

    Xout, Yout = np.meshgrid(x_m, y_m, indexing='ij')
    angle_rad = np.radians(float(angle_deg))

    # Inverse spatial map for active rotation.
    Xsource = Xout * np.cos(angle_rad) + Yout * np.sin(angle_rad)
    Ysource = -Xout * np.sin(angle_rad) + Yout * np.cos(angle_rad)
    source_xy = np.column_stack([Xsource.ravel(), Ysource.ravel()])

    sampled_components: list[np.ndarray] = []
    for interpolator in interpolators:
        sampled = np.empty_like(Ex0)
        for k, z_value in enumerate(z_m):
            points = np.column_stack([
                source_xy,
                np.full(source_xy.shape[0], float(z_value)),
            ])
            sampled[:, :, k] = interpolator(points).reshape(len(x_m), len(y_m))
        sampled_components.append(sampled)

    Ex_spatial, Ey_spatial, Ez_rotated = sampled_components

    # Rotate the Cartesian vector components by the same active angle.
    Ex_rotated = Ex_spatial * np.cos(angle_rad) - Ey_spatial * np.sin(angle_rad)
    Ey_rotated = Ex_spatial * np.sin(angle_rad) + Ey_spatial * np.cos(angle_rad)
    Eabs_rotated = eabs_from_components(Ex_rotated, Ey_rotated, Ez_rotated)

    return {
        'Ex': Ex_rotated,
        'Ey': Ey_rotated,
        'Ez': Ez_rotated,
        'Eperp': np.hypot(Ex_rotated, Ey_rotated),
        '|E|': Eabs_rotated,
        'angle_deg': float(angle_deg),
    }


def align_field_to_vertical_plane(
    field: dict[str, np.ndarray],
    *,
    radius_m: float,
    length_m: float,
    label: str = '',
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Align the global |E| maximum with ``|E|[mid_x, :, :]``.

    This is the validated method used by the earlier homotypic m=1/m=2
    analyses: locate the brightest voxel, calculate the active z rotation that
    maps its azimuth onto x=0, rotate the spatial arrays and vector components,
    and then verify the result.
    """
    Ex = np.asarray(field['Ex'])
    Ey = np.asarray(field['Ey'])
    Ez = np.asarray(field['Ez'])
    nx, ny, nz = Ex.shape

    x_m = np.linspace(-float(radius_m), float(radius_m), nx)
    y_m = np.linspace(-float(radius_m), float(radius_m), ny)
    z_m = np.linspace(0.0, float(length_m), nz)

    Eabs_before = eabs_from_components(Ex, Ey, Ez)
    angle_deg, peak_before = rotation_angle_to_vertical_plane(Eabs_before, x_m, y_m)
    rotated = rotate_vector_field_about_z(
        Ex, Ey, Ez, x_m, y_m, z_m, angle_deg
    )

    Eabs_after = np.asarray(rotated['|E|'])
    mid_x = nx // 2

    # Quadrupole fields can contain several symmetry-related voxels with the
    # same global |E| value.  np.nanargmax() returns only the first such voxel,
    # which may lie outside x=mid even though another exactly equal maximum is
    # correctly present in the vertical longitudinal plane.  Therefore the
    # physically relevant validation is that the plane maximum equals the
    # global maximum, not that the arbitrary first argmax index lies there.
    global_peak_first = tuple(
        int(v)
        for v in np.unravel_index(
            np.nanargmax(Eabs_after),
            Eabs_after.shape,
        )
    )
    global_max = float(np.nanmax(Eabs_after))
    vertical_plane = Eabs_after[mid_x, :, :]
    vertical_plane_max = float(np.nanmax(vertical_plane))

    if not np.isclose(
        vertical_plane_max,
        global_max,
        rtol=1.0e-8,
        atol=max(1.0e-14, 1.0e-10 * max(global_max, 1.0)),
    ):
        raise RuntimeError(
            f'{label}: rotation did not place a global |E| maximum in '
            f'Eabs[{mid_x}, :, :]; global={global_max:.12e}, '
            f'vertical-plane={vertical_plane_max:.12e}, '
            f'first_global_peak={global_peak_first}, '
            f'angle={angle_deg:.12g} deg.'
        )

    # Record the representative global maximum that is explicitly in the
    # requested vertical plane.  This avoids misleading diagnostics for
    # degenerate m=2 maxima.
    iy_plane, iz_plane = (
        int(v)
        for v in np.unravel_index(
            np.nanargmax(vertical_plane),
            vertical_plane.shape,
        )
    )
    peak_after = (int(mid_x), iy_plane, iz_plane)

    aligned = {
        'Ex': np.asarray(rotated['Ex']),
        'Ey': np.asarray(rotated['Ey']),
        'Ez': np.asarray(rotated['Ez']),
        'Eperp': np.asarray(rotated['Eperp']),
        '|E|': Eabs_after,
    }
    diagnostics = {
        'rotation_applied': True,
        'rotation_angle_deg': float(angle_deg),
        'peak_before_xyz': peak_before,
        'peak_after_xyz': peak_after,
        'first_global_peak_after_xyz': global_peak_first,
        'mid_x_index': int(mid_x),
        'global_max_Eabs_after': global_max,
        'vertical_plane_max_Eabs_after': vertical_plane_max,
        'vertical_plane_contains_global_max': True,
    }

    print(
        f'{label}: rotated by {angle_deg:.6f} deg; '
        f'peak {peak_before} -> vertical-plane peak {peak_after}; '
        f'first global argmax={global_peak_first}; '
        f'Eabs[{mid_x}, :, :] contains a global |E| maximum.'
    )
    return aligned, diagnostics


# -----------------------------------------------------------------------------
# Crossing fields and plotting
# -----------------------------------------------------------------------------

def _finite(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def combine_crossing_fields(E1: dict[str, np.ndarray], E2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for comp in ("Ex", "Ey", "Ez"):
        a = _finite(E1[comp])
        b = _finite(E2[comp])
        out[f"E1_{comp}"] = a
        out[f"E2_{comp}"] = b
        out[f"{comp}_plus"] = a + b
        out[f"{comp}_minus"] = a - b

    for label, prefix in (("E1", "E1_"), ("E2", "E2_"), ("plus", ""), ("minus", "")):
        if label in ("plus", "minus"):
            ex, ey, ez = out[f"Ex_{label}"], out[f"Ey_{label}"], out[f"Ez_{label}"]
        else:
            ex, ey, ez = out[f"{prefix}Ex"], out[f"{prefix}Ey"], out[f"{prefix}Ez"]
        out[f"abs_{label}"] = np.sqrt(ex**2 + ey**2 + ez**2)
    return out


def get_or_create_field_data(
    E1: dict[str, np.ndarray],
    E2: dict[str, np.ndarray],
    folder: Path,
    *,
    create_fields: bool,
    family_m: int,
    rotation_E1: dict[str, Any] | None = None,
    rotation_E2: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Create/load crossing fields and invalidate old unrotated m=1/m=2 data."""
    filename = folder / 'field_data.npz'
    recreate = bool(create_fields or not filename.exists())

    if not recreate and int(family_m) in (1, 2):
        existing = load_npz_dict(filename)
        rotation_flag = existing.get('vertical_plane_rotation_applied')
        if rotation_flag is None or not bool(np.asarray(rotation_flag).item()):
            print(
                f'{filename}: old field data has no validated vertical-plane '
                f'rotation metadata; regenerating.'
            )
            recreate = True
        else:
            return existing

    if recreate:
        data = combine_crossing_fields(E1, E2)
        data['vertical_plane_rotation_applied'] = np.asarray(
            int(family_m) in (1, 2), dtype=np.bool_
        )

        for prefix, diagnostics in (
            ('E1', rotation_E1),
            ('E2', rotation_E2),
        ):
            if diagnostics is None:
                continue
            data[f'{prefix}_rotation_angle_deg'] = np.asarray(
                diagnostics['rotation_angle_deg'], dtype=float
            )
            data[f'{prefix}_peak_before_xyz'] = np.asarray(
                diagnostics['peak_before_xyz'], dtype=int
            )
            data[f'{prefix}_peak_after_xyz'] = np.asarray(
                diagnostics['peak_after_xyz'], dtype=int
            )

        save_npz_dict(filename, data)
        return data

    return load_npz_dict(filename)


def extract_slices(field_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Extract the four 2D slices used by the compact appendix summary PDF.

    For every three-dimensional field array ``F[x, y, z]`` this creates:

        ``*_iris_1``             -> ``F[:, :, 0]``
        ``*_iris_2``             -> ``F[:, :, -1]``
        ``*_transverse_mid``     -> ``F[:, :, mid_z]``
        ``*_longitudinal_mid``   -> ``F[mid_x, :, :]``

    Plot orientation is applied later so the underlying saved arrays retain
    their normal field-map indexing.
    """
    slices: dict[str, np.ndarray] = {}

    for key, field in field_data.items():
        if not isinstance(field, np.ndarray) or field.ndim != 3:
            continue

        mid_x = field.shape[0] // 2
        mid_z = field.shape[2] // 2

        slices[f"{key}_iris_1"] = field[:, :, 0]
        slices[f"{key}_iris_2"] = field[:, :, -1]
        slices[f"{key}_transverse_mid"] = field[:, :, mid_z]
        slices[f"{key}_longitudinal_mid"] = field[mid_x, :, :]

    return slices


def _save_one_2x4_slice_on_gridspec(
    *,
    fig: plt.Figure,
    grid_slot,
    slice_dict: dict[str, np.ndarray],
    slice_type: str,
    block_title: str,
) -> None:
    """Draw one 2x4 ``[Ez, |E|] x [E1, E2, E-, E+]`` slice block.

    This reproduces the established appendix figure layout used by the
    heterotypic analysis:

        top row    : normalised Ez
        bottom row : normalised |E|
        columns    : E1, E2, E-, E+

    Each row has one shared colour bar.  The numerical annotation in each panel
    gives the maximum absolute field after normalisation to the largest parent
    field for that row.
    """
    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]
    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]

    def real_image(array: np.ndarray) -> np.ndarray:
        return np.real(np.asarray(array))

    def orient_for_plot(
        array: np.ndarray,
        current_slice_type: str,
    ) -> np.ndarray:
        if (
            current_slice_type.startswith("iris")
            or current_slice_type == "transverse_mid"
        ):
            return array.T
        return array

    def safe_reference(arrays: list[np.ndarray]) -> float:
        reference = max(
            float(np.nanmax(np.abs(array)))
            for array in arrays
        )
        if not np.isfinite(reference) or reference <= 0.0:
            return 1.0
        return reference

    def safe_vmax(
        arrays: list[np.ndarray],
        reference: float,
    ) -> float:
        scaled_maximum = max(
            float(np.nanmax(np.abs(array / reference)))
            for array in arrays
        )
        if not np.isfinite(scaled_maximum) or scaled_maximum <= 0.0:
            return 1.0
        return max(1.0, scaled_maximum)

    grid = grid_slot.subgridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.04,
    )

    parent_ez_reference = safe_reference([
        real_image(slice_dict[f"E1_Ez_{slice_type}"]),
        real_image(slice_dict[f"E2_Ez_{slice_type}"]),
    ])
    parent_abs_reference = safe_reference([
        real_image(slice_dict[f"abs_E1_{slice_type}"]),
        real_image(slice_dict[f"abs_E2_{slice_type}"]),
    ])

    first_axis = None

    for row_index, row_keys in enumerate(rows):
        raw_row_data = [
            real_image(slice_dict[f"{key}_{slice_type}"])
            for key in row_keys
        ]

        if row_index == 0:
            reference = parent_ez_reference
            scaled_row_data = [
                array / reference
                for array in raw_row_data
            ]
            vmax = safe_vmax(raw_row_data, reference)
            vmin = -vmax
            colour_map = "RdBu_r"
        else:
            reference = parent_abs_reference
            scaled_row_data = [
                array / reference
                for array in raw_row_data
            ]
            vmin = 0.0
            vmax = safe_vmax(raw_row_data, reference)
            colour_map = "viridis"

        image_handle = None

        for column_index, (raw_array, scaled_array) in enumerate(
            zip(raw_row_data, scaled_row_data)
        ):
            axis = fig.add_subplot(grid[row_index, column_index])
            if first_axis is None:
                first_axis = axis

            plot_array = orient_for_plot(
                scaled_array,
                slice_type,
            )

            image_handle = axis.imshow(
                plot_array,
                origin="lower",
                cmap=colour_map,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

            n_y, n_x = plot_array.shape
            axis.set_box_aspect(n_y / n_x)
            axis.margins(0.0)
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(
                    column_titles[column_index],
                    fontsize=10,
                    fontweight="bold",
                    pad=2,
                )

            if column_index == 0:
                axis.set_ylabel(
                    row_titles[row_index],
                    fontsize=9,
                    rotation=90,
                    labelpad=7,
                )

            axis.text(
                0.04,
                0.96,
                f"{np.nanmax(np.abs(scaled_array)):.2f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.2,
                },
            )

        colour_bar_axis = fig.add_subplot(grid[row_index, 4])
        colour_bar = fig.colorbar(
            image_handle,
            cax=colour_bar_axis,
        )
        colour_bar.ax.tick_params(
            labelsize=7,
            length=2,
            pad=1,
        )

    if first_axis is not None:
        first_axis.text(
            0.0,
            1.04,
            block_title,
            transform=first_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 0.2,
            },
        )


def save_four_slice_pdfs_and_merge(
    slice_dict: dict[str, np.ndarray],
    output_directory: str | Path,
    merged_pdf_name: str = "combined_four_slice_summary.pdf",
) -> Path:
    """Write the established single-page, four-block appendix PDF.

    The output is one tall PDF page rather than four separate PDF pages.  It
    contains the following blocks, from top to bottom:

        1. Transverse iris 1
        2. Transverse iris 2
        3. Longitudinal vertical mid-plane
        4. Transverse mid-plane

    Each block is a 2x4 grid with rows ``Ez`` and ``|E|`` and columns
    ``E1``, ``E2``, ``E-`` and ``E+``.  This is the same layout as the
    established heterotypic appendix field-summary PDFs.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    specifications = [
        ("iris_1", "Transverse iris 1"),
        ("iris_2", "Transverse iris 2"),
        (
            "longitudinal_mid",
            "Longitudinal vertical mid-plane",
        ),
        ("transverse_mid", "Transverse mid-plane"),
    ]

    figure = plt.figure(
        figsize=(7.2, 12.0),
        constrained_layout=False,
    )
    outer_grid = figure.add_gridspec(
        4,
        1,
        left=0.075,
        right=0.965,
        bottom=0.025,
        top=0.975,
        hspace=0.20,
    )

    for block_index, (
        slice_type,
        block_title,
    ) in enumerate(specifications):
        _save_one_2x4_slice_on_gridspec(
            fig=figure,
            grid_slot=outer_grid[block_index, 0],
            slice_dict=slice_dict,
            slice_type=slice_type,
            block_title=block_title,
        )

    output_file = (
        output_directory / merged_pdf_name
    )
    figure.savefig(
        output_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)

    print(
        f"Saved single-page appendix field-summary PDF: "
        f"{output_file}"
    )
    return output_file


def plot_field_slices_combined(
    field_data: dict[str, np.ndarray],
    output_directory: str | Path,
    title: str = "",
) -> Path:
    """Create the publication-style appendix field-summary PDF.

    The ``title`` argument is retained for workflow compatibility.  The compact
    appendix PDF itself follows the established clean layout and therefore does
    not add a crossing title above the four slice blocks; the crossing title is
    supplied by the LaTeX appendix page.

    The PDF is written to:

        ``<crossing>/slice_summary_pdfs/``
        ``<crossing-name>_field_summary.pdf``
    """
    output_directory = Path(output_directory)
    summary_directory = (
        output_directory / "slice_summary_pdfs"
    )
    summary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_dict = extract_slices(field_data)

    return save_four_slice_pdfs_and_merge(
        slice_dict,
        summary_directory,
        merged_pdf_name=(
            f"{output_directory.name}_field_summary.pdf"
        ),
    )


# -----------------------------------------------------------------------------
# Voltage, energy and RF multipole extraction
# -----------------------------------------------------------------------------

def centred_transverse_coords(nx: int, ny: int, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(-float(radius_m), float(radius_m), int(nx)),
        np.linspace(-float(radius_m), float(radius_m), int(ny)),
    )


def longitudinal_coords(nz: int, length_m: float, *, centred: bool = False) -> np.ndarray:
    z = np.linspace(0.0, float(length_m), int(nz))
    return z - 0.5 * float(length_m) if centred else z


def complex_voltage_map_from_Ez(
    Ez_xyz: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    beta: float,
    centred_z: bool,
) -> tuple[np.ndarray, np.ndarray]:
    Ez = _finite(Ez_xyz)
    z_m = longitudinal_coords(Ez.shape[2], length_m, centred=centred_z)
    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = np.exp(1j * omega * z_m / (float(beta) * C0))
    return np.trapezoid(Ez * phase[None, None, :], z_m, axis=2), z_m


def _trapezoid3(a: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(np.trapezoid(_finite(a), z, axis=2), y, axis=1), x, axis=0))


def electric_energy_diagnostics(
    Ex_xyz: np.ndarray,
    Ey_xyz: np.ndarray,
    Ez_xyz: np.ndarray,
    *,
    radius_m: float,
    length_m: float,
) -> dict[str, Any]:
    nx, ny, nz = np.asarray(Ez_xyz).shape
    x, y = centred_transverse_coords(nx, ny, radius_m)
    z = np.linspace(0.0, float(length_m), nz)
    int_Ez2 = _trapezoid3(_finite(Ez_xyz) ** 2, x, y, z)
    int_Etot2 = _trapezoid3(_finite(Ex_xyz) ** 2 + _finite(Ey_xyz) ** 2 + _finite(Ez_xyz) ** 2, x, y, z)
    U_CST = 0.5 * EPS0 * int_Etot2
    return {
        "int_Ez2_dV": int_Ez2,
        "int_Etot2_dV": int_Etot2,
        "U_Ez_only_time_average_J": 0.25 * EPS0 * int_Ez2,
        "U_Etotal_time_average_J": 0.25 * EPS0 * int_Etot2,
        "U_Ez_only_peak_J": 0.5 * EPS0 * int_Ez2,
        "U_Etotal_peak_J": U_CST,
        "U_CST_equiv_J": U_CST,
        "U_used_J": U_CST,
        "U_used_label": "U_CST_equiv = 0.5*eps0*int(|E|^2)dV",
    }


def _interp_complex_circle(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    rho_m: float,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    points = np.column_stack((rho_m * np.cos(phi), rho_m * np.sin(phi)))
    re = RegularGridInterpolator((x_m, y_m), np.asarray(Vz_xy).real, bounds_error=False, fill_value=np.nan)
    im = RegularGridInterpolator((x_m, y_m), np.asarray(Vz_xy).imag, bounds_error=False, fill_value=np.nan)
    vals = re(points) + 1j * im(points)
    if not np.all(np.isfinite(vals.real) & np.isfinite(vals.imag)):
        raise ValueError(f"Non-finite circle interpolation at rho={rho_m:.6e} m")
    return phi, vals


def extract_rf_multipoles(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    rho_dipole_m: float,
    rho_quadrupole_m: float,
    n_phi: int,
    charge_C: float,
) -> dict[str, Any]:
    V = np.asarray(Vz_xy, dtype=complex) / float(charge_C)
    ix0, iy0 = V.shape[0] // 2, V.shape[1] // 2
    phi1, V1 = _interp_complex_circle(V, x_m, y_m, rho_m=rho_dipole_m, n_phi=n_phi)
    phi2, V2 = _interp_complex_circle(V, x_m, y_m, rho_m=rho_quadrupole_m, n_phi=n_phi)

    c0 = V[ix0, iy0]
    c1 = (2.0 / len(phi1)) * np.sum(V1 * np.cos(phi1)) / rho_dipole_m
    s1 = (2.0 / len(phi1)) * np.sum(V1 * np.sin(phi1)) / rho_dipole_m
    c2 = (2.0 / len(phi2)) * np.sum(V2 * np.cos(2.0 * phi2)) / rho_quadrupole_m**2
    s2 = (2.0 / len(phi2)) * np.sum(V2 * np.sin(2.0 * phi2)) / rho_quadrupole_m**2

    return {
        "method": "azimuthal_fourier_rf_multipole",
        "n_phi": int(n_phi),
        "charge_C": float(charge_C),
        "axis_indices_xy": (int(ix0), int(iy0)),
        "rho_dipole_m": float(rho_dipole_m),
        "rho_quadrupole_m": float(rho_quadrupole_m),
        "c0_axis_V_per_C": c0,
        "c0_dipole_circle_mean_V_per_C": np.mean(V1),
        "c0_quadrupole_circle_mean_V_per_C": np.mean(V2),
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
    """Store agreed integrated k and length-normalised K quantities."""
    omega = 2.0 * np.pi * float(frequency_Hz)
    c0 = complex(multipoles["c0_axis_V_per_C"])
    c1 = complex(multipoles["c1_cos_phi_V_per_C_per_m"])
    s1 = complex(multipoles["s1_sin_phi_V_per_C_per_m"])
    c2 = complex(multipoles["c2_cos_2phi_V_per_C_per_m2"])
    s2 = complex(multipoles["s2_sin_2phi_V_per_C_per_m2"])

    if not np.isfinite(U_J) or U_J <= 0.0 or length_m <= 0.0:
        raise ValueError("U_J and length_m must be positive")

    # Agreed definitions.
    k_parallel = abs(c0) ** 2 / (4.0 * U_J)
    K_parallel = k_parallel / length_m

    dipole_coeff_sq = abs(c1) ** 2 + abs(s1) ** 2
    k_perp = (C0 / (4.0 * U_J * omega)) * dipole_coeff_sq
    K_perp = k_perp / length_m

    # Factor 4 matches the Hessian scalar convention
    # sqrt((Kxx-Kyy)^2 + 4 Kxy^2).
    k_Q = (4.0 * C0 / omega) * np.sqrt(abs(c2) ** 2 + abs(s2) ** 2) / np.sqrt(4.0 * U_J)
    K_Q = k_Q / length_m

    dipole_total = np.sqrt(dipole_coeff_sq)
    dipole_x_fraction = abs(c1) ** 2 / dipole_coeff_sq if dipole_coeff_sq > 0.0 else 0.0
    dipole_y_fraction = abs(s1) ** 2 / dipole_coeff_sq if dipole_coeff_sq > 0.0 else 0.0

    k_perp_x = k_perp * dipole_x_fraction
    k_perp_y = k_perp * dipole_y_fraction
    K_perp_x = K_perp * dipole_x_fraction
    K_perp_y = K_perp * dipole_y_fraction

    return {
        "frequency_Hz": float(frequency_Hz),
        "omega_rad_s": float(omega),
        "length_m": float(length_m),
        "V0_V_per_C": c0,
        "dipole_coefficient_magnitude_V_per_C_per_m": float(dipole_total),
        "quadrupole_coefficient_magnitude_V_per_C_per_m2": float(np.sqrt(abs(c2) ** 2 + abs(s2) ** 2)),

        # Explicit agreed integrated quantities.
        "k_parallel_V_per_C": float(k_parallel),
        "k_parallel_V_per_pC": float(k_parallel * PC),
        "k_perp_V_per_C_per_m": float(k_perp),
        "k_perp_V_per_pC_per_m": float(k_perp * PC),
        "k_perp_x_V_per_C_per_m": float(k_perp_x),
        "k_perp_y_V_per_C_per_m": float(k_perp_y),
        "k_perp_x_V_per_pC_per_m": float(k_perp_x * PC),
        "k_perp_y_V_per_pC_per_m": float(k_perp_y * PC),
        "k_Q_V_per_C_per_m2": float(k_Q),
        "k_Q_V_per_pC_per_m2": float(k_Q * PC),

        # Explicit agreed length-normalised reported quantities.
        "K_parallel_V_per_C_per_m": float(K_parallel),
        "K_parallel_V_per_pC_per_m": float(K_parallel * PC),
        "K_perp_V_per_C_per_m2": float(K_perp),
        "K_perp_V_per_pC_per_m2": float(K_perp * PC),
        "K_perp_x_V_per_C_per_m2": float(K_perp_x),
        "K_perp_y_V_per_C_per_m2": float(K_perp_y),
        "K_perp_x_V_per_pC_per_m2": float(K_perp_x * PC),
        "K_perp_y_V_per_pC_per_m2": float(K_perp_y * PC),
        "K_Q_V_per_C_per_m3": float(K_Q),
        "K_Q_V_per_pC_per_m3": float(K_Q * PC),

        # Backwards-compatible aliases used by earlier appendix/report code.
        "loss_like_V_per_C": float(k_parallel),
        "loss_like_V_per_pC": float(k_parallel * PC),
        "loss_like_V_per_C_per_m": float(K_parallel),
        "loss_like_V_per_pC_per_m": float(K_parallel * PC),
        "kick_magnitude_V_per_C_per_m2": float(K_perp),
        "kick_magnitude_V_per_pC_per_m2": float(K_perp * PC),
        "kick_x_V_per_C_per_m2": float(K_perp_x),
        "kick_y_V_per_C_per_m2": float(K_perp_y),
        "kick_x_V_per_pC_per_m2": float(K_perp_x * PC),
        "kick_y_V_per_pC_per_m2": float(K_perp_y * PC),
        "KQ_V_per_C_per_m3": float(K_Q),
        "KQ_V_per_pC_per_m3": float(K_Q * PC),

        "U_CST_normalisation_J": float(U_J),
        "normalisation_note": (
            "k_parallel=|c0|^2/(4U); K_parallel=k_parallel/d; "
            "k_perp=c(|c1|^2+|s1|^2)/(4U omega); K_perp=k_perp/d; "
            "k_Q=4(c/omega)sqrt(|c2|^2+|s2|^2)/sqrt(4U); K_Q=k_Q/d."
        ),
    }


# -----------------------------------------------------------------------------
# One field and one crossing
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    analysis_root: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_rf_multipole")
    data_root: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    frequency_010_Hz: float = 1.3e9
    n_max: int = 3
    p_max: int = 3
    LF_start: float = 0.7
    LF_stop: float = 1.3
    param_sweep_resolution: int = 1000
    voxel_res: int = 151
    beta: float = 1.0
    charge_C: float = 1.0
    centred_z: bool = False
    fit_pixels_dipole: int = 4
    fit_pixels_quadrupole: int = 8
    n_phi: int = 720
    create_family_data: bool = False
    create_fields: bool = False
    make_plots: bool = True


def analyse_field(
    Ex_xyz: np.ndarray,
    Ey_xyz: np.ndarray,
    Ez_xyz: np.ndarray,
    *,
    frequency_Hz: float,
    length_m: float,
    radius_m: float,
    cfg: RunConfig,
) -> dict[str, Any]:
    nx, ny, _ = np.asarray(Ez_xyz).shape
    x_m, y_m = centred_transverse_coords(nx, ny, radius_m)
    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    pixel = 0.5 * (abs(dx) + abs(dy))
    rho_d = float(cfg.fit_pixels_dipole) * pixel
    rho_q = float(cfg.fit_pixels_quadrupole) * pixel
    if rho_q >= 0.95 * radius_m:
        raise ValueError("Quadrupole sampling radius too close to map boundary")

    Vz_xy, z_m = complex_voltage_map_from_Ez(
        Ez_xyz,
        length_m=length_m,
        frequency_Hz=frequency_Hz,
        beta=cfg.beta,
        centred_z=cfg.centred_z,
    )
    multipoles = extract_rf_multipoles(
        Vz_xy,
        x_m,
        y_m,
        rho_dipole_m=rho_d,
        rho_quadrupole_m=rho_q,
        n_phi=cfg.n_phi,
        charge_C=cfg.charge_C,
    )
    energy = electric_energy_diagnostics(
        Ex_xyz, Ey_xyz, Ez_xyz, radius_m=radius_m, length_m=length_m
    )
    fom = figures_of_merit_from_rf_multipoles(
        multipoles,
        frequency_Hz=frequency_Hz,
        U_J=energy["U_CST_equiv_J"],
        length_m=length_m,
    )
    return {
        "analysis_method": "rf_fourier_multipole",
        "length_m": float(length_m),
        "frequency_Hz": float(frequency_Hz),
        "transverse_pixel_x_m": dx,
        "transverse_pixel_y_m": dy,
        "longitudinal_pixel_m": float(z_m[1] - z_m[0]),
        "z_start_m": float(z_m[0]),
        "z_stop_m": float(z_m[-1]),
        "centred_z": bool(cfg.centred_z),
        "sampling_radii": {
            "dipole_pixels": int(cfg.fit_pixels_dipole),
            "quadrupole_pixels": int(cfg.fit_pixels_quadrupole),
            "rho_dipole_m": rho_d,
            "rho_quadrupole_m": rho_q,
            "n_phi": int(cfg.n_phi),
        },
        "multipole_coefficients": multipoles,
        "figures_of_merit": fom,
        "energy_diagnostics": energy,
    }


def _metric(fields: dict[str, Any], field: str, key: str) -> float:
    return float(fields[field]["figures_of_merit"].get(key, np.nan))


def _rmax(fields: dict[str, Any], key: str) -> float:
    parent = max(abs(_metric(fields, "E1", key)), abs(_metric(fields, "E2", key)))
    mixed = max(abs(_metric(fields, "plus", key)), abs(_metric(fields, "minus", key)))
    return float(mixed / parent) if np.isfinite(parent) and parent > 0.0 else float("nan")


def compare_parent_and_mixed(fields: dict[str, Any]) -> dict[str, Any]:
    metric_keys = {
        "k_parallel_V_per_pC": "k_parallel_V_per_pC",
        "K_parallel_V_per_pC_per_m": "K_parallel_V_per_pC_per_m",
        "k_perp_V_per_pC_per_m": "k_perp_V_per_pC_per_m",
        "K_perp_V_per_pC_per_m2": "K_perp_V_per_pC_per_m2",
        "k_Q_V_per_pC_per_m2": "k_Q_V_per_pC_per_m2",
        "K_Q_V_per_pC_per_m3": "K_Q_V_per_pC_per_m3",
    }
    out: dict[str, Any] = {}
    for label, key in metric_keys.items():
        out[label] = {
            "E1": _metric(fields, "E1", key),
            "E2": _metric(fields, "E2", key),
            "plus": _metric(fields, "plus", key),
            "minus": _metric(fields, "minus", key),
            "Rmax": _rmax(fields, key),
            "source_key": key,
        }
    return out


def crossing_folder_name(mode_i: str, mode_j: str) -> str:
    return f"{mode_i}_{mode_j}".replace("TM_", "TM")


def analyse_crossing(
    crossing_key: str,
    crossing: dict[str, Any],
    family_data: dict[str, Any],
    family_root: Path,
    cfg: RunConfig,
) -> dict[str, Any]:
    mode_i = crossing["mode_i"].split("_", 1)[1]
    mode_j = crossing["mode_j"].split("_", 1)[1]
    m_i, _, _ = parse_mnp(mode_i)
    m_j, _, _ = parse_mnp(mode_j)
    if m_i != m_j:
        raise ValueError("Unified homotypic workflow only accepts like-m crossings")

    folder = family_root / crossing_folder_name(crossing["mode_i"], crossing["mode_j"])
    folder.mkdir(parents=True, exist_ok=True)

    E1_original = family_data['TM'][mode_i]['3D_Efield']
    E2_original = family_data['TM'][mode_j]['3D_Efield']

    d0 = float(family_data['metadata']['design_length_m'])
    R = float(family_data['metadata']['pillbox_radius_m'])

    rotation_E1: dict[str, Any] | None = None
    rotation_E2: dict[str, Any] | None = None

    # m=0 is azimuthally symmetric. For m=1 and m=2, align each parent before
    # forming E+ and E-, so all parent and mixed fields inherit the same vertical
    # longitudinal orientation used by the Fourier extraction and appendix plots.
    if m_i in (1, 2):
        E1, rotation_E1 = align_field_to_vertical_plane(
            E1_original,
            radius_m=R,
            length_m=d0,
            label=f'TM_{mode_i}',
        )
        E2, rotation_E2 = align_field_to_vertical_plane(
            E2_original,
            radius_m=R,
            length_m=d0,
            label=f'TM_{mode_j}',
        )
    else:
        E1 = E1_original
        E2 = E2_original

    field_data = get_or_create_field_data(
        E1,
        E2,
        folder,
        create_fields=cfg.create_fields,
        family_m=int(m_i),
        rotation_E1=rotation_E1,
        rotation_E2=rotation_E2,
    )
    mixed_d = d0 * float(crossing["length_factor"])
    f1 = float(family_data["TM"][mode_i]["design_frequency_Hz"])
    f2 = float(family_data["TM"][mode_j]["design_frequency_Hz"])
    f_cross = float(crossing["frequency_Hz"])

    jobs = {
        "E1": ("E1_Ex", "E1_Ey", "E1_Ez", d0, f1, f"TM_{mode_i}", "parent_design_frequency"),
        "E2": ("E2_Ex", "E2_Ey", "E2_Ez", d0, f2, f"TM_{mode_j}", "parent_design_frequency"),
        "plus": ("Ex_plus", "Ey_plus", "Ez_plus", mixed_d, f_cross, "plus", "crossing_degenerate_frequency"),
        "minus": ("Ex_minus", "Ey_minus", "Ez_minus", mixed_d, f_cross, "minus", "crossing_degenerate_frequency"),
    }

    fields: dict[str, Any] = {}
    for name, (exk, eyk, ezk, length, freq, mode, source) in jobs.items():
        fields[name] = analyse_field(
            field_data[exk], field_data[eyk], field_data[ezk],
            frequency_Hz=freq, length_m=length, radius_m=R, cfg=cfg,
        )
        fields[name].update({
            "mode": mode,
            "Ex_key": exk,
            "Ey_key": eyk,
            "Ez_key": ezk,
            "frequency_source": source,
        })

    result = {
        "analysis_method": "homotypic_rf_fourier_multipole",
        "crossing_key": crossing_key,
        "crossing": crossing,
        "mode_i": f"TM_{mode_i}",
        "mode_j": f"TM_{mode_j}",
        "family_m": int(m_i),
        "rotation_diagnostics": {
            "E1": rotation_E1,
            "E2": rotation_E2,
            "applied_to_family": bool(m_i in (1, 2)),
            "target_plane": "Eabs[mid_x, :, :]",
        },
        "fields": fields,
        "comparison": compare_parent_and_mixed(fields),
        "configuration": asdict(cfg),
        "units": {
            "k_parallel": "V/pC",
            "K_parallel": "V/pC/m_z",
            "k_perp": "V/pC/m_perp",
            "K_perp": "V/pC/m_perp/m_z",
            "k_Q": "V/pC/m_perp^2",
            "K_Q": "V/pC/m_perp^2/m_z",
        },
    }

    pickle_save(result, folder / "crossing_analysis.pkl")
    pickle_save(result, folder / "homotypic_rf_multipole_analysis.pkl")
    write_summary(result, folder / "homotypic_rf_multipole_summary.txt")
    if cfg.make_plots:
        plot_field_slices_combined(field_data, folder, f"TM{mode_i} -- TM{mode_j}")
    return result


def write_summary(result: dict[str, Any], filename: str | Path) -> None:
    lines = [
        "Unified homotypic RF/Fourier multipole analysis",
        f"family m = {result['family_m']}",
        f"crossing = {result['mode_i']} -- {result['mode_j']}",
        f"ell = {result['crossing']['length_factor']:.12e}",
        f"f_cross_Hz = {result['crossing']['frequency_Hz']:.12e}",
        "",
        "FIELD ROTATION",
        f"  applied_to_family = {result.get('rotation_diagnostics', {}).get('applied_to_family')}",
        f"  target_plane = {result.get('rotation_diagnostics', {}).get('target_plane')}",
    ]
    for parent_name in ("E1", "E2"):
        diag = result.get("rotation_diagnostics", {}).get(parent_name)
        if diag:
            lines.extend([
                f"  {parent_name} angle_deg = {diag['rotation_angle_deg']:.12e}",
                f"  {parent_name} peak_before = {diag['peak_before_xyz']}",
                f"  {parent_name} peak_after = {diag['peak_after_xyz']}",
                f"  {parent_name} vertical_plane_contains_global_max = {diag['vertical_plane_contains_global_max']}",
            ])
    lines.append("")
    for name in ("E1", "E2", "plus", "minus"):
        field = result["fields"][name]
        fom = field["figures_of_merit"]
        mp = field["multipole_coefficients"]
        lines += [
            f"{name} ({field['mode']}):",
            f"  frequency_Hz = {field['frequency_Hz']:.12e} ({field['frequency_source']})",
            f"  length_m = {field['length_m']:.12e}",
            f"  U_CST_J = {field['energy_diagnostics']['U_CST_equiv_J']:.12e}",
            f"  c0 = {mp['c0_axis_V_per_C']:.12e}",
            f"  c1 = {mp['c1_cos_phi_V_per_C_per_m']:.12e}",
            f"  s1 = {mp['s1_sin_phi_V_per_C_per_m']:.12e}",
            f"  c2 = {mp['c2_cos_2phi_V_per_C_per_m2']:.12e}",
            f"  s2 = {mp['s2_sin_2phi_V_per_C_per_m2']:.12e}",
            f"  k_parallel = {fom['k_parallel_V_per_pC']:.12e} V/pC",
            f"  K_parallel = {fom['K_parallel_V_per_pC_per_m']:.12e} V/pC/m",
            f"  k_perp = {fom['k_perp_V_per_pC_per_m']:.12e} V/pC/m",
            f"  K_perp = {fom['K_perp_V_per_pC_per_m2']:.12e} V/pC/m^2",
            f"  k_Q = {fom['k_Q_V_per_pC_per_m2']:.12e} V/pC/m^2",
            f"  K_Q = {fom['K_Q_V_per_pC_per_m3']:.12e} V/pC/m^3",
            "",
        ]
    lines.append("Rmax values")
    for label, row in result["comparison"].items():
        lines.append(f"  {label:32s} = {row['Rmax']:.12e}")
    Path(filename).write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Batch workflow
# -----------------------------------------------------------------------------

def family_data_filename(cfg: RunConfig, m: int) -> Path:
    v = cfg.voxel_res
    return cfg.data_root / f"TMm{m}_TMm{m}_data_dict_{v}x{v}x{v}.pkl"


def load_or_create_family_data(cfg: RunConfig, m: int) -> dict[str, Any]:
    filename = family_data_filename(cfg, m)
    if cfg.create_family_data or not filename.exists():
        data = assemble_family_data(
            m,
            n_max=cfg.n_max,
            p_max=cfg.p_max,
            frequency_010=cfg.frequency_010_Hz,
            LF_start=cfg.LF_start,
            LF_stop=cfg.LF_stop,
            param_sweep_resolution=cfg.param_sweep_resolution,
            voxel_res=cfg.voxel_res,
        )
        data.setdefault("metadata", {})["source_file"] = str(filename)
        pickle_save(data, filename)
        return data
    data = pickle_load(filename)
    data.setdefault("metadata", {})["source_file"] = str(filename)
    return data


def analyse_family(cfg: RunConfig, m: int) -> dict[str, Any]:
    family_data = load_or_create_family_data(cfg, m)
    family_root = cfg.analysis_root / {0: "monopole_monopole", 1: "dipole_dipole", 2: "quadrupole_quadrupole"}[m]
    family_root.mkdir(parents=True, exist_ok=True)

    crossings = find_like_family_crossings(family_data)
    pickle_save(crossings, family_root / "crossing_results.pkl")

    analyses: dict[str, Any] = {}
    for key, crossing in crossings["TM"]["crossings"].items():
        print(
            f"Analysing m={m}: {key}; ell={crossing['length_factor']:.8g}; "
            f"f={crossing['frequency_Hz'] / 1e9:.6g} GHz"
        )
        analyses[key] = analyse_crossing(key, crossing, family_data, family_root, cfg)

    pickle_save(analyses, family_root / "all_crossing_analyses.pkl")
    pickle_save(analyses, family_root / "all_homotypic_rf_multipole_analyses.pkl")
    return analyses


def main(cfg: RunConfig = RunConfig()) -> dict[int, dict[str, Any]]:
    cfg.analysis_root.mkdir(parents=True, exist_ok=True)
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    all_results = {m: analyse_family(cfg, m) for m in (0, 1, 2)}
    # all_results = {2: analyse_family(cfg, 2)}
    pickle_save(all_results, cfg.analysis_root / "all_homotypic_rf_multipole_analyses.pkl")
    return all_results


if __name__ == "__main__":
    config = RunConfig(
        frequency_010_Hz=1.3e9,
        n_max=3,
        p_max=3,
        LF_start=0.7,
        LF_stop=1.3,
        param_sweep_resolution=1000,
        voxel_res=151,
        beta=1.0,
        charge_C=1.0,
        centred_z=False,
        fit_pixels_dipole=4,
        fit_pixels_quadrupole=8,
        n_phi=720,
        create_family_data=False,
        create_fields=False,
        make_plots=True,
    )
    main(config)
