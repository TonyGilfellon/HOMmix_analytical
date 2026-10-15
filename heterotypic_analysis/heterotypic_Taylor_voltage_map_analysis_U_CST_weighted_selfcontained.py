"""Self-contained heterotypic TM pillbox field generation and Taylor voltage-map analysis.

This module is based on ``heterotypic_Taylor_voltage_map_analysis_U_CST_agreed_metrics_summary_layout.py``
and incorporates only the analytical TM field generation and field-rotation machinery needed from
``heterotypic_crossing_field_mixing_method2_U_CST_consistent_quadrupole_rotation.py``.

Workflow
--------
1. Build or load TM monopole (m=0), dipole (m=1), and quadrupole (m=2) family data.
2. Locate the M-D, M-Q, and D-Q frequency-degenerate crossings over the requested length range.
3. Rotate dipole and quadrupole parent fields using the existing heterotypic conventions.
4. Construct equal-weight mixed fields using

       E_plus  = (E1 + E2) / sqrt(2)
       E_minus = (E1 - E2) / sqrt(2)

   component by component for Ex, Ey, and Ez.
5. Save ``field_data.npz`` and ``heterotypic_crossing_analysis.pkl`` for each crossing.
6. Analyse E1, E2, E+, and E- using the agreed Taylor voltage-map decomposition and U_CST normalisation.
7. Save the per-crossing analysis pickle/text summary and the single-page appendix field-summary PDF.

The parent fields are not rescaled.  Only the mixed fields carry the 1/sqrt(2) equal-weight factor.
The saved field data contain an explicit mixing-convention marker; old unweighted cached fields are
regenerated automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import zipfile
from typing import Any, Iterable

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
# Self-contained analytical field generation, rotation, and heterotypic mixing
# -----------------------------------------------------------------------------

MIXING_SCALE = 1.0 / np.sqrt(2.0)
MIXING_CONVENTION = "equal_weight_1_over_sqrt2"

FAMILY_LABEL = {
    0: "monopole",
    1: "dipole",
    2: "quadrupole",
}

PAIR_TYPES = {
    "monopole_dipole": (0, 1),
    "dipole_quadrupole": (1, 2),
    "monopole_quadrupole": (0, 2),
}


def tm_root_v_mn(m: int, n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(jn_zeros(int(m), int(n))[-1])


def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    if R <= 0.0 or L <= 0.0:
        raise ValueError("R and L must be positive")
    if p < 0:
        raise ValueError("p must be >= 0")
    v = tm_root_v_mn(m, n)
    return float((c / (2.0 * np.pi)) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2))


def pillbox_radius_from_freq(f_Hz: float) -> float:
    return float(tm_root_v_mn(0, 1) * C0 / (2.0 * np.pi * float(f_Hz)))


def _field_coords(R: float, L: float, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    return (
        np.linspace(-float(R), float(R), int(nx)),
        np.linspace(-float(R), float(R), int(ny)),
        np.linspace(0.0, float(L), int(nz)),
    )


def _E_field_cyl_TM(
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
    """Analytical TM_mnp electric field in cylindrical components."""
    kc = tm_root_v_mn(m, n) / float(R)
    kz = np.pi * int(p) / float(L)
    arg = kc * r
    Jm = special.jv(m, arg)
    Jmp = special.jvp(m, arg, 1)
    cos_m = np.cos(m * theta)
    sin_m = np.sin(m * theta)
    cos_z = np.cos(kz * z)
    sin_z = np.sin(kz * z)

    Ez = float(E0) * Jm * cos_m * cos_z
    if p == 0:
        return np.zeros_like(Ez), np.zeros_like(Ez), Ez

    Er = -float(E0) * (kz / kc) * Jmp * cos_m * sin_z
    with np.errstate(divide="ignore", invalid="ignore"):
        J_over_r = Jm / r
        if m == 1:
            J_over_r = np.where(r == 0.0, kc / 2.0, J_over_r)
        else:
            J_over_r = np.where(r == 0.0, 0.0, J_over_r)
        Etheta = float(E0) * (kz / kc**2) * m * J_over_r * sin_m * sin_z
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
    x, y, z = _field_coords(R, L, (x_res, y_res, z_res))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    mask = r <= float(R)
    Er, Etheta, Ez = _E_field_cyl_TM(r, theta, Z, m, n, p, R, L, E0=E0)
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
        "x_m": x,
        "y_m": y,
        "z_m": z,
    }


def eabs_from_components(Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.nan_to_num(np.asarray(Ex, dtype=float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ey, dtype=float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ez, dtype=float), nan=0.0) ** 2
    )


def rotate_vector_field_about_z(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    angle_deg: float,
    *,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Actively rotate the spatial map and Cartesian vector components about z."""
    Ex0 = np.nan_to_num(np.asarray(Ex, dtype=float), nan=fill_value)
    Ey0 = np.nan_to_num(np.asarray(Ey, dtype=float), nan=fill_value)
    Ez0 = np.nan_to_num(np.asarray(Ez, dtype=float), nan=fill_value)
    interpolators = [
        RegularGridInterpolator((x, y, z), component, bounds_error=False, fill_value=fill_value)
        for component in (Ex0, Ey0, Ez0)
    ]
    Xout, Yout = np.meshgrid(x, y, indexing="ij")
    angle_rad = np.radians(float(angle_deg))
    Xsource = Xout * np.cos(angle_rad) + Yout * np.sin(angle_rad)
    Ysource = -Xout * np.sin(angle_rad) + Yout * np.cos(angle_rad)
    source_xy = np.column_stack([Xsource.ravel(), Ysource.ravel()])

    sampled_components: list[np.ndarray] = []
    for interpolator in interpolators:
        sampled = np.empty_like(Ex0)
        for k, z_value in enumerate(z):
            points = np.column_stack([
                source_xy,
                np.full(source_xy.shape[0], float(z_value)),
            ])
            sampled[:, :, k] = interpolator(points).reshape(len(x), len(y))
        sampled_components.append(sampled)

    Ex_spatial, Ey_spatial, Ez_rotated = sampled_components
    Ex_rotated = Ex_spatial * np.cos(angle_rad) - Ey_spatial * np.sin(angle_rad)
    Ey_rotated = Ex_spatial * np.sin(angle_rad) + Ey_spatial * np.cos(angle_rad)
    return {
        "Ex": Ex_rotated,
        "Ey": Ey_rotated,
        "Ez": Ez_rotated,
        "Eabs": eabs_from_components(Ex_rotated, Ey_rotated, Ez_rotated),
        "angle_deg": float(angle_deg),
    }


def align_field_to_vertical_plane_peak(field: dict[str, np.ndarray], *, label: str = "") -> dict[str, Any]:
    """Dipole alignment retained from the established heterotypic generator."""
    Ex, Ey, Ez = field["Ex"], field["Ey"], field["Ez"]
    x = np.asarray(field.get("x_m", np.linspace(-1.0, 1.0, Ex.shape[0])), dtype=float)
    y = np.asarray(field.get("y_m", np.linspace(-1.0, 1.0, Ex.shape[1])), dtype=float)
    z = np.asarray(field.get("z_m", np.linspace(0.0, 1.0, Ex.shape[2])), dtype=float)
    Eabs0 = eabs_from_components(Ex, Ey, Ez)
    ix, iy, iz = np.unravel_index(np.nanargmax(Eabs0), Eabs0.shape)
    peak0 = (int(ix), int(iy), int(iz))
    phi = np.arctan2(y[iy], x[ix])
    candidates = np.asarray([np.pi / 2.0 - phi, -np.pi / 2.0 - phi])
    candidates = (candidates + np.pi) % (2.0 * np.pi) - np.pi
    mid = len(x) // 2

    best = None
    for candidate in candidates:
        angle = float(np.degrees(candidate))
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle)
        peak = tuple(int(v) for v in np.unravel_index(np.nanargmax(trial["Eabs"]), trial["Eabs"].shape))
        global_max = float(np.nanmax(trial["Eabs"]))
        midplane_max = float(np.nanmax(trial["Eabs"][mid, :, :]))
        score = (
            abs(peak[0] - mid),
            abs(global_max - midplane_max) / (global_max if global_max else 1.0),
        )
        if best is None or score < best[0]:
            best = (score, angle, trial, peak, global_max, midplane_max)

    _, angle, rotated, peak1, global_max, midplane_max = best
    if abs(peak1[0] - mid) > 1 and not np.isclose(
        midplane_max, global_max, rtol=2.0e-3, atol=1.0e-12
    ):
        raise RuntimeError(
            f"{label}: dipole rotation failed; peak_after={peak1}, expected x index {mid}, "
            f"angle={angle}, global_max={global_max:.6e}, midplane_max={midplane_max:.6e}"
        )
    rotated.update({
        "rotation_angle_deg": angle,
        "rotation_method": "peak_to_vertical_plane",
        "peak_before": peak0,
        "peak_after": peak1,
        "global_max_after": global_max,
        "vertical_midplane_max_after": midplane_max,
    })
    return rotated


def align_quadrupole_principal_axes(
    field: dict[str, np.ndarray],
    *,
    label: str = "",
    fit_pixels: int = 12,
    z_index: int | None = None,
    target: str = "principal_axes_xy",
) -> dict[str, Any]:
    """Quadrupole Ez-principal-axis alignment from the established generator."""
    Ex, Ey, Ez = field["Ex"], field["Ey"], field["Ez"]
    x = np.asarray(field.get("x_m", np.linspace(-1.0, 1.0, Ex.shape[0])), dtype=float)
    y = np.asarray(field.get("y_m", np.linspace(-1.0, 1.0, Ex.shape[1])), dtype=float)
    z = np.asarray(field.get("z_m", np.linspace(0.0, 1.0, Ex.shape[2])), dtype=float)
    Eabs0 = eabs_from_components(Ex, Ey, Ez)
    peak0 = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs0), Eabs0.shape))
    nx, ny, _ = Ez.shape
    ix0, iy0 = nx // 2, ny // 2

    if z_index is None:
        rpx = min(int(fit_pixels), ix0 - 1, iy0 - 1)
        near = Ez[ix0-rpx:ix0+rpx+1, iy0-rpx:iy0+rpx+1, :]
        z_index = int(np.nanargmax(np.nanmax(np.abs(near), axis=(0, 1))))

    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    max_px = min(int(fit_pixels), ix0 - 1, iy0 - 1)

    pts: list[tuple[float, float]] = []
    vals: list[float] = []
    for i in range(ix0 - max_px, ix0 + max_px + 1):
        for j in range(iy0 - max_px, iy0 + max_px + 1):
            xx = (i - ix0) * dx
            yy = (j - iy0) * dy
            if np.hypot(xx, yy) <= max_px * min(abs(dx), abs(dy)):
                pts.append((xx, yy))
                vals.append(float(Ez[i, j, z_index]))

    pts_arr = np.asarray(pts, dtype=float)
    vals_arr = np.asarray(vals, dtype=float)
    X = pts_arr[:, 0]
    Y = pts_arr[:, 1]
    A = np.column_stack([np.ones_like(X), X, Y, X*X, X*Y, Y*Y])
    coeff, *_ = np.linalg.lstsq(A, vals_arr, rcond=None)
    _, _, _, axx, axy, ayy = coeff
    H = np.array([[2.0*axx, axy], [axy, 2.0*ayy]], dtype=float)
    theta_rad = 0.5 * np.arctan2(2.0 * H[0, 1], H[0, 0] - H[1, 1])
    theta_deg = float(np.degrees(theta_rad))

    if target == "principal_axes_xy":
        base = -theta_deg
        candidate_angles = [base, base + 90.0, base - 90.0, base + 180.0]
    elif target == "diagonal_lobes_vertical":
        base1 = 45.0 - theta_deg
        base2 = -45.0 - theta_deg
        candidate_angles = [
            base1, base1 + 90.0, base1 - 90.0,
            base2, base2 + 90.0, base2 - 90.0,
        ]
    else:
        raise ValueError("target must be 'principal_axes_xy' or 'diagonal_lobes_vertical'")

    candidate_angles = [float((a + 180.0) % 360.0 - 180.0) for a in candidate_angles]

    def fitted_H_after_rotation(angle_deg: float) -> np.ndarray:
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle_deg)
        Ezr = np.asarray(trial["Ez"])
        vals_r = []
        for xx, yy in pts_arr:
            i = int(round(ix0 + xx / dx))
            j = int(round(iy0 + yy / dy))
            vals_r.append(Ezr[i, j, z_index])
        coeff_r, *_ = np.linalg.lstsq(A, np.asarray(vals_r, dtype=float), rcond=None)
        _, _, _, axx_r, axy_r, ayy_r = coeff_r
        return np.array([[2.0*axx_r, axy_r], [axy_r, 2.0*ayy_r]], dtype=float)

    best = None
    for angle in candidate_angles:
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle)
        Eabs = np.asarray(trial["Eabs"])
        peak = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs), Eabs.shape))
        global_max = float(np.nanmax(Eabs))
        midplane_max = float(np.nanmax(Eabs[ix0, :, :]))
        Hr = fitted_H_after_rotation(angle)
        Hxx, Hxy, Hyy = Hr[0, 0], Hr[0, 1], Hr[1, 1]
        diag_scale = max(abs(Hxx), abs(Hyy), 1.0e-300)
        score = (
            abs(Hxy) / diag_scale,
            0.0 if Hxx * Hyy < 0.0 else 1.0,
            abs(global_max - midplane_max) / (global_max if global_max else 1.0),
            abs(peak[0] - ix0),
        )
        if best is None or score < best[0]:
            best = (score, angle, trial, peak, global_max, midplane_max, Hr)

    _, angle, rotated, peak1, global_max, midplane_max, H_after = best
    rotated.update({
        "rotation_angle_deg": angle,
        "rotation_method": "quadrupole_Ez_principal_axis",
        "rotation_target": target,
        "quadrupole_angle_before_deg": theta_deg,
        "quadrupole_fit_z_index": int(z_index),
        "quadrupole_fit_pixels": int(max_px),
        "quadrupole_H_before": H,
        "quadrupole_H_after": H_after,
        "quadrupole_cross_ratio_after": float(
            abs(H_after[0, 1]) / max(abs(H_after[0, 0]), abs(H_after[1, 1]), 1.0e-300)
        ),
        "peak_before": peak0,
        "peak_after": peak1,
        "global_max_after": global_max,
        "vertical_midplane_max_after": midplane_max,
    })
    return rotated


def prepare_field_for_mixing(
    field: dict[str, np.ndarray],
    *,
    m: int,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    """Apply the established heterotypic parent-field orientation convention."""
    if int(m) == 0:
        return {
            "Ex": np.asarray(field["Ex"]),
            "Ey": np.asarray(field["Ey"]),
            "Ez": np.asarray(field["Ez"]),
        }, None
    if int(m) == 2:
        rotated = align_quadrupole_principal_axes(field, label=label)
    else:
        rotated = align_field_to_vertical_plane_peak(field, label=label)
    return {
        "Ex": np.asarray(rotated["Ex"]),
        "Ey": np.asarray(rotated["Ey"]),
        "Ez": np.asarray(rotated["Ez"]),
    }, rotated


def combine_fields_equal_weight(E1: dict[str, np.ndarray], E2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct E+/-=(E1 +/- E2)/sqrt(2), preserving unscaled parents."""
    out: dict[str, np.ndarray] = {}
    for prefix, field in (("E1", E1), ("E2", E2)):
        for component in ("Ex", "Ey", "Ez"):
            out[f"{prefix}_{component}"] = np.asarray(field[component])
        out[f"abs_{prefix}"] = eabs_from_components(field["Ex"], field["Ey"], field["Ez"])

    for label, sign in (("plus", 1.0), ("minus", -1.0)):
        Ex = MIXING_SCALE * (np.asarray(E1["Ex"]) + sign * np.asarray(E2["Ex"]))
        Ey = MIXING_SCALE * (np.asarray(E1["Ey"]) + sign * np.asarray(E2["Ey"]))
        Ez = MIXING_SCALE * (np.asarray(E1["Ez"]) + sign * np.asarray(E2["Ez"]))
        out[f"Ex_{label}"] = Ex
        out[f"Ey_{label}"] = Ey
        out[f"Ez_{label}"] = Ez
        out[f"abs_{label}"] = eabs_from_components(Ex, Ey, Ez)

    # Scalar metadata are intentionally stored in the npz so stale unweighted
    # field caches can be detected and rebuilt automatically.
    out["mixing_scale"] = np.asarray(MIXING_SCALE, dtype=float)
    out["mixing_convention"] = np.asarray(MIXING_CONVENTION)
    return out


def field_data_uses_current_mixing(field_data: dict[str, np.ndarray]) -> bool:
    convention = field_data.get("mixing_convention")
    scale = field_data.get("mixing_scale")
    if convention is None or scale is None:
        return False
    try:
        convention_text = str(np.asarray(convention).item())
        scale_value = float(np.asarray(scale).item())
    except Exception:
        return False
    return convention_text == MIXING_CONVENTION and np.isclose(scale_value, MIXING_SCALE)


def family_data_filename(datapath: Path, m: int, voxel_res: int) -> Path:
    return datapath / f"TMm{m}_TMm{m}_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl"


def assemble_family_data(
    *,
    m: int,
    n_max: int,
    p_max: int,
    frequency_010: float,
    LF_start: float,
    LF_stop: float,
    param_sweep_resolution: int,
    voxel_res: int,
) -> dict[str, Any]:
    design_L = C0 / float(frequency_010) / 2.0
    radius = pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(float(LF_start), float(LF_stop), int(param_sweep_resolution))
    data: dict[str, Any] = {
        "TM": {},
        "length_factor_vector": length_factors.tolist(),
        "metadata": {
            "family_m": int(m),
            "family_label": FAMILY_LABEL[int(m)],
            "frequency_010_Hz": float(frequency_010),
            "R_m": float(radius),
            "design_L_m": float(design_L),
            "voxel_res": int(voxel_res),
        },
    }
    for p in range(int(p_max) + 1):
        for n in range(1, int(n_max) + 1):
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = pillbox_field_voxel_grid_xyz(
                radius, design_L, m, n, p,
                voxel_res, voxel_res, voxel_res,
                E0=1.0,
            )
            frequencies = np.asarray(
                [f_tm(m, n, p, radius, ell * design_L) for ell in length_factors],
                dtype=float,
            )
            design_frequency = f_tm(m, n, p, radius, design_L)
            data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": frequencies.tolist(),
                "frequency_normalised": (frequencies / float(frequency_010)).tolist(),
                "design_frequency_Hz": float(design_frequency),
                "design_frequency_normalised": float(design_frequency / frequency_010),
            }
    return data


def load_or_create_family_data(
    datapath: Path,
    *,
    m: int,
    n_max: int,
    p_max: int,
    f_010: float,
    LF_start: float,
    LF_stop: float,
    param_sweep_resolution: int,
    voxel_res: int,
    create_family_data: bool,
) -> dict[str, Any]:
    datapath.mkdir(parents=True, exist_ok=True)
    filename = family_data_filename(datapath, m, voxel_res)
    if not create_family_data and filename.exists():
        print(f"Loading TM m={m} family data from {filename}")
        data = pickle_load(filename)
    else:
        print(f"Creating TM m={m} family data")
        data = assemble_family_data(
            m=m,
            n_max=n_max,
            p_max=p_max,
            frequency_010=f_010,
            LF_start=LF_start,
            LF_stop=LF_stop,
            param_sweep_resolution=param_sweep_resolution,
            voxel_res=voxel_res,
        )
        pickle_save(data, filename)
    data.setdefault("metadata", {})["source_file"] = str(filename)
    return data


def assemble_sweep_table(family_data: dict[int, dict], f_010: float) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for m, data in family_data.items():
        ell = np.asarray(data["length_factor_vector"], dtype=float)
        for mnp, entry in data["TM"].items():
            frequencies = np.asarray(entry["frequency_Hz"], dtype=float)
            table[f"TM_{mnp}"] = {
                "family_m": int(m),
                "mnp": mnp,
                "length_factor": ell,
                "frequency_Hz": frequencies,
                "frequency_normalised": frequencies / float(f_010),
                "design_frequency_Hz": float(entry["design_frequency_Hz"]),
            }
    return table


def detect_pair_crossings(
    sweep_table: dict[str, dict[str, Any]],
    *,
    m_a: int,
    m_b: int,
    pair_name: str,
) -> dict[str, dict[str, Any]]:
    modes_a = [name for name, row in sweep_table.items() if row["family_m"] == int(m_a)]
    modes_b = [name for name, row in sweep_table.items() if row["family_m"] == int(m_b)]
    crossings: dict[str, dict[str, Any]] = {}
    for name_a in modes_a:
        ell = np.asarray(sweep_table[name_a]["length_factor"], dtype=float)
        fa = np.asarray(sweep_table[name_a]["frequency_Hz"], dtype=float)
        for name_b in modes_b:
            fb = np.asarray(sweep_table[name_b]["frequency_Hz"], dtype=float)
            difference = fa - fb
            for idx in np.where(difference[:-1] * difference[1:] <= 0.0)[0]:
                if np.isclose(difference[idx], 0.0) and idx > 0 and np.isclose(difference[idx-1], 0.0):
                    continue
                if np.isclose(difference[idx], 0.0):
                    ell_cross = float(ell[idx])
                elif np.isclose(difference[idx+1], 0.0):
                    ell_cross = float(ell[idx+1])
                else:
                    ell_cross = float(brentq(
                        lambda x: np.interp(x, ell, fa) - np.interp(x, ell, fb),
                        float(ell[idx]),
                        float(ell[idx+1]),
                    ))
                f_cross = float(np.interp(ell_cross, ell, fa))
                fhat_cross = float(np.interp(
                    ell_cross,
                    ell,
                    np.asarray(sweep_table[name_a]["frequency_normalised"], dtype=float),
                ))
                key = f"{pair_name}:{name_a}--{name_b}@{ell_cross:.8g}"
                crossings[key] = {
                    "pair_type": pair_name,
                    "mode_i": name_a,
                    "mode_j": name_b,
                    "m_i": int(m_a),
                    "m_j": int(m_b),
                    "length_factor": ell_cross,
                    "frequency_Hz": f_cross,
                    "frequency_normalised": fhat_cross,
                }
    return crossings


def find_heterotypic_crossings(sweep_table: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for pair_name, (m_a, m_b) in PAIR_TYPES.items():
        out[pair_name] = detect_pair_crossings(
            sweep_table,
            m_a=m_a,
            m_b=m_b,
            pair_name=pair_name,
        )
        print(f"{pair_name}: found {len(out[pair_name])} crossings")
    return out


def parse_mode_name(mode_name: str) -> tuple[int, str]:
    _, mnp = str(mode_name).split("_", 1)
    return int(mnp[0]), mnp


def _rotation_metadata(rotation: dict[str, Any] | None) -> dict[str, Any] | None:
    if rotation is None:
        return None
    keep = (
        "rotation_angle_deg",
        "rotation_method",
        "rotation_target",
        "peak_before",
        "peak_after",
        "global_max_after",
        "vertical_midplane_max_after",
        "quadrupole_angle_before_deg",
        "quadrupole_fit_z_index",
        "quadrupole_fit_pixels",
        "quadrupole_cross_ratio_after",
    )
    return {key: rotation.get(key) for key in keep if key in rotation}


def safe_folder_name(crossing_key: str) -> str:
    value = str(crossing_key).replace(":", "__").replace("@", "__ell_")
    return value.replace("--", "__").replace(".", "p")


def generate_crossing_field_data(
    crossing_key: str,
    crossing: dict[str, Any],
    family_data: dict[int, dict],
    out_dir: Path,
    *,
    create_fields: bool,
) -> dict[str, Any]:
    """Generate/reuse one crossing's weighted mixed fields and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    field_file = out_dir / "field_data.npz"
    metadata_file = out_dir / "heterotypic_crossing_analysis.pkl"

    recreate = bool(create_fields or not field_file.exists() or not metadata_file.exists())
    if not recreate:
        try:
            existing = load_npz_dict(field_file)
            if not field_data_uses_current_mixing(existing):
                print(f"{field_file}: old/unmarked mixing convention detected; regenerating.")
                recreate = True
        except Exception:
            recreate = True

    if not recreate:
        return pickle_load(metadata_file)

    m_i, mnp_i = parse_mode_name(crossing["mode_i"])
    m_j, mnp_j = parse_mode_name(crossing["mode_j"])
    raw_i = family_data[m_i]["TM"][mnp_i]["3D_Efield"]
    raw_j = family_data[m_j]["TM"][mnp_j]["3D_Efield"]
    E1, rotation_i = prepare_field_for_mixing(raw_i, m=m_i, label=f"TM{mnp_i}")
    E2, rotation_j = prepare_field_for_mixing(raw_j, m=m_j, label=f"TM{mnp_j}")
    field_data = combine_fields_equal_weight(E1, E2)
    np.savez_compressed(field_file, **field_data)

    metadata = {
        "crossing_key": crossing_key,
        "crossing": crossing,
        "mode_i": crossing["mode_i"],
        "mode_j": crossing["mode_j"],
        "m_i": int(m_i),
        "m_j": int(m_j),
        "mnp_i": mnp_i,
        "mnp_j": mnp_j,
        "rotation_i": _rotation_metadata(rotation_i),
        "rotation_j": _rotation_metadata(rotation_j),
        "mixing_convention": MIXING_CONVENTION,
        "mixing_scale": float(MIXING_SCALE),
        "mixing_equations": {
            "plus": "E_plus=(E1+E2)/sqrt(2)",
            "minus": "E_minus=(E1-E2)/sqrt(2)",
        },
        "files": {"field_data_npz": str(field_file)},
    }
    pickle_save(metadata, metadata_file)
    return metadata


def generate_all_heterotypic_fields(
    crossings_by_pair: dict[str, dict[str, dict[str, Any]]],
    family_data: dict[int, dict],
    root: Path,
    *,
    create_fields: bool,
) -> dict[str, Any]:
    generated: dict[str, Any] = {}
    for pair_name, crossings in crossings_by_pair.items():
        generated[pair_name] = {}
        pair_dir = root / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        for key, crossing in crossings.items():
            print(f"Generating heterotypic fields: {key}")
            folder = pair_dir / safe_folder_name(key)
            generated[pair_name][key] = generate_crossing_field_data(
                key,
                crossing,
                family_data,
                folder,
                create_fields=create_fields,
            )
    pickle_save(generated, root / "all_heterotypic_crossing_analyses.pkl")
    return generated


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
# Appendix field-summary PDF plotting
# -----------------------------------------------------------------------------

def _finite_plot_array(array: np.ndarray) -> np.ndarray:
    """Return a finite real plotting array without modifying stored field data."""
    return np.nan_to_num(
        np.real(np.asarray(array)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def ensure_field_magnitudes(
    field_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a plotting dictionary containing all required ``abs_*`` arrays."""
    out = dict(field_data)

    component_keys = {
        "E1": ("E1_Ex", "E1_Ey", "E1_Ez"),
        "E2": ("E2_Ex", "E2_Ey", "E2_Ez"),
        "plus": ("Ex_plus", "Ey_plus", "Ez_plus"),
        "minus": ("Ex_minus", "Ey_minus", "Ez_minus"),
    }

    for label, (ex_key, ey_key, ez_key) in component_keys.items():
        magnitude_key = f"abs_{label}"
        if magnitude_key in out:
            continue

        missing = [
            key for key in (ex_key, ey_key, ez_key)
            if key not in out
        ]
        if missing:
            raise KeyError(
                f"Cannot construct {magnitude_key!r}; "
                f"missing field component(s): {missing}"
            )

        ex = _finite_plot_array(out[ex_key])
        ey = _finite_plot_array(out[ey_key])
        ez = _finite_plot_array(out[ez_key])
        out[magnitude_key] = np.sqrt(ex**2 + ey**2 + ez**2)

    return out


def extract_summary_slices(
    field_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract the four two-dimensional slices used in the appendix PDF."""
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


def _draw_summary_slice_block(
    *,
    fig: plt.Figure,
    grid_slot,
    slice_dict: dict[str, np.ndarray],
    slice_type: str,
    block_title: str,
    config: "FieldAnalysisConfig",
) -> None:
    """Draw one non-overlapping 2x4 ``[Ez, |E|] x [E1,E2,E-,E+]`` block."""
    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]
    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]

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
        return reference if np.isfinite(reference) and reference > 0.0 else 1.0

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
        3,
        5,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            1.0,
            config.summary_colourbar_width_ratio,
        ],
        height_ratios=[
            config.summary_block_title_height_ratio,
            1.0,
            1.0,
        ],
        wspace=config.summary_panel_wspace,
        hspace=config.summary_panel_hspace,
    )

    title_axis = fig.add_subplot(grid[0, :4])
    title_axis.set_axis_off()
    title_axis.text(
        0.0,
        0.5,
        block_title,
        transform=title_axis.transAxes,
        ha="left",
        va="center",
        fontsize=config.summary_block_title_fontsize,
        fontweight="bold",
        clip_on=False,
    )

    parent_ez_reference = safe_reference([
        _finite_plot_array(slice_dict[f"E1_Ez_{slice_type}"]),
        _finite_plot_array(slice_dict[f"E2_Ez_{slice_type}"]),
    ])
    parent_abs_reference = safe_reference([
        _finite_plot_array(slice_dict[f"abs_E1_{slice_type}"]),
        _finite_plot_array(slice_dict[f"abs_E2_{slice_type}"]),
    ])

    for row_index, row_keys in enumerate(rows):
        raw_row_data = [
            _finite_plot_array(slice_dict[f"{key}_{slice_type}"])
            for key in row_keys
        ]

        if row_index == 0:
            reference = parent_ez_reference
            scaled_row_data = [array / reference for array in raw_row_data]
            vmax = safe_vmax(raw_row_data, reference)
            vmin = -vmax
            colour_map = "RdBu_r"
        else:
            reference = parent_abs_reference
            scaled_row_data = [array / reference for array in raw_row_data]
            vmin = 0.0
            vmax = safe_vmax(raw_row_data, reference)
            colour_map = "viridis"

        image_handle = None
        plot_row = row_index + 1

        for column_index, scaled_array in enumerate(scaled_row_data):
            axis = fig.add_subplot(grid[plot_row, column_index])
            plot_array = orient_for_plot(scaled_array, slice_type)

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
                    fontsize=config.summary_column_title_fontsize,
                    fontweight="bold",
                    pad=config.summary_column_title_pad,
                )

            if column_index == 0:
                axis.set_ylabel(
                    row_titles[row_index],
                    fontsize=config.summary_row_label_fontsize,
                    rotation=90,
                    labelpad=config.summary_row_label_pad,
                )

            axis.text(
                config.summary_annotation_x,
                config.summary_annotation_y,
                f"{np.nanmax(np.abs(scaled_array)):.2f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=config.summary_annotation_fontsize,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": config.summary_annotation_box_alpha,
                    "pad": config.summary_annotation_box_pad,
                },
            )

        if image_handle is None:
            raise RuntimeError(f"No image was generated for {block_title!r}.")

        colour_bar_axis = fig.add_subplot(grid[plot_row, 4])
        colour_bar = fig.colorbar(image_handle, cax=colour_bar_axis)
        colour_bar.ax.tick_params(
            labelsize=config.summary_colourbar_tick_fontsize,
            length=config.summary_colourbar_tick_length,
            pad=config.summary_colourbar_tick_pad,
        )


def save_heterotypic_field_summary_pdf(
    field_data: dict[str, np.ndarray],
    crossing_folder: str | Path,
    *,
    config: "FieldAnalysisConfig",
) -> Path:
    """Create the single-page appendix field-summary PDF for one crossing."""
    crossing_folder = Path(crossing_folder)
    summary_directory = crossing_folder / "slice_summary_pdfs"
    summary_directory.mkdir(parents=True, exist_ok=True)

    plotting_fields = ensure_field_magnitudes(field_data)
    slice_dict = extract_summary_slices(plotting_fields)

    specifications = [
        ("iris_1", "Transverse iris 1"),
        ("iris_2", "Transverse iris 2"),
        ("longitudinal_mid", "Longitudinal vertical mid-plane"),
        ("transverse_mid", "Transverse mid-plane"),
    ]

    figure = plt.figure(
        figsize=(
            config.summary_figure_width,
            config.summary_figure_height,
        ),
        constrained_layout=False,
    )
    outer_grid = figure.add_gridspec(
        4,
        1,
        left=config.summary_margin_left,
        right=config.summary_margin_right,
        bottom=config.summary_margin_bottom,
        top=config.summary_margin_top,
        hspace=config.summary_block_hspace,
    )

    for block_index, (slice_type, block_title) in enumerate(specifications):
        _draw_summary_slice_block(
            fig=figure,
            grid_slot=outer_grid[block_index, 0],
            slice_dict=slice_dict,
            slice_type=slice_type,
            block_title=block_title,
            config=config,
        )

    output_file = (
        summary_directory
        / f"{crossing_folder.name}_field_summary.pdf"
    )
    figure.savefig(
        output_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=config.summary_save_pad_inches,
    )
    plt.close(figure)

    print(f"Saved heterotypic appendix field-summary PDF: {output_file}")
    return output_file


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

    # Appendix field-summary PDF layout and typography. Increase or decrease
    # these values manually without editing the plotting functions.
    summary_figure_width: float = 8.6
    summary_figure_height: float = 19.0
    summary_block_title_fontsize: float = 20.0
    summary_column_title_fontsize: float = 20.0
    summary_row_label_fontsize: float = 20.0
    summary_annotation_fontsize: float = 18.0
    summary_colourbar_tick_fontsize: float = 12.0
    summary_column_title_pad: float = 2.0
    summary_row_label_pad: float = 9.0
    summary_annotation_x: float = 0.035
    summary_annotation_y: float = 0.965
    summary_annotation_box_alpha: float = 0.82
    summary_annotation_box_pad: float = 1.5
    summary_colourbar_tick_length: float = 3.0
    summary_colourbar_tick_pad: float = 2.0
    summary_colourbar_width_ratio: float = 0.055
    summary_block_title_height_ratio: float = 0.55
    summary_panel_wspace: float = 0.10
    summary_panel_hspace: float = 0.12
    summary_block_hspace: float = 0.01
    summary_margin_left: float = 0.10
    summary_margin_right: float = 0.965
    summary_margin_bottom: float = 0.025
    summary_margin_top: float = 0.985
    summary_save_pad_inches: float = 0.04

    # End-to-end field-generation/sweep configuration.
    datapath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    analysis_root: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings")
    n_max: int = 3
    p_max: int = 3
    LF_start: float = 0.7
    LF_stop: float = 1.3
    param_sweep_resolution: int = 1000
    voxel_res: int = 151

    create_family_data: bool = False
    create_fields: bool = False
    make_plots: bool = True


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

    if config.make_plots:
        save_heterotypic_field_summary_pdf(
            field_data,
            folder,
            config=config,
        )


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
# End-to-end command-line workflow
# -----------------------------------------------------------------------------

def run_full_heterotypic_taylor_analysis(config: FieldAnalysisConfig | None = None) -> dict[str, Any]:
    """Generate weighted fields and run the complete Taylor analysis."""
    if config is None:
        config = FieldAnalysisConfig()

    config.datapath.mkdir(parents=True, exist_ok=True)
    config.analysis_root.mkdir(parents=True, exist_ok=True)

    family_data: dict[int, dict] = {}
    for m in (0, 1, 2):
        family_data[m] = load_or_create_family_data(
            config.datapath,
            m=m,
            n_max=config.n_max,
            p_max=config.p_max,
            f_010=config.f_010,
            LF_start=config.LF_start,
            LF_stop=config.LF_stop,
            param_sweep_resolution=config.param_sweep_resolution,
            voxel_res=config.voxel_res,
            create_family_data=config.create_family_data,
        )

    sweep_table = assemble_sweep_table(family_data, config.f_010)
    crossings_by_pair = find_heterotypic_crossings(sweep_table)
    pickle_save(crossings_by_pair, config.analysis_root / "all_heterotypic_crossings.pkl")

    generate_all_heterotypic_fields(
        crossings_by_pair,
        family_data,
        config.analysis_root,
        create_fields=config.create_fields,
    )

    results = analyse_all_heterotypic_crossings(
        config.analysis_root,
        family_data_by_m=family_data,
        config=config,
        save=True,
    )

    print("\nCompleted self-contained heterotypic Taylor voltage-map analysis.")
    print(f"Mixing convention: {MIXING_CONVENTION}")
    print(f"Mixing scale:      {MIXING_SCALE:.16g}")
    print(f"Crossings analysed: {len(results)}")
    return results


if __name__ == "__main__":
    cfg = FieldAnalysisConfig(
        f_010=1.3e9,
        fit_pixels=4,
        fit_pixels_dipole=4,
        fit_pixels_quadrupole=8,
        charge_C=1.0,
        centred_z=False,
        create_family_data=True,
        create_fields=True,
        make_plots=True,
    )
    run_full_heterotypic_taylor_analysis(cfg)
