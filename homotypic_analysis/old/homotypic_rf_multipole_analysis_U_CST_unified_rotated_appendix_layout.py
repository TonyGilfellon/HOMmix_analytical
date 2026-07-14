"""Unified homotypic RF/Fourier multipole analysis for TM m=0, 1 and 2.

This standalone module keeps the established analytical homotypic workflow:

1. Build/load parent-family data for TM_0np, TM_1np and TM_2np.
2. Find like-family crossings as the pillbox length factor is varied.
3. Form E1, E2, E+ and E- field maps for every crossing.
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
# Field alignment
# -----------------------------------------------------------------------------

def eabs_from_components(Ex, Ey, Ez):
    return np.sqrt(
        np.nan_to_num(np.asarray(Ex, float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ey, float), nan=0.0) ** 2
        + np.nan_to_num(np.asarray(Ez, float), nan=0.0) ** 2
    )


def rotation_angle_to_vertical_plane(Eabs, x_m, y_m):
    peak = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs), Eabs.shape))
    ix, iy, _ = peak
    phi = np.arctan2(y_m[iy], x_m[ix])
    candidates = np.asarray([np.pi / 2.0 - phi, -np.pi / 2.0 - phi])
    candidates = (candidates + np.pi) % (2.0 * np.pi) - np.pi
    angle = float(candidates[np.argmin(np.abs(candidates))])
    return float(np.degrees(angle)), peak


def rotate_vector_field_about_z(Ex, Ey, Ez, x_m, y_m, z_m, angle_deg, fill_value=0.0):
    Ex0 = np.nan_to_num(np.asarray(Ex, float), nan=fill_value)
    Ey0 = np.nan_to_num(np.asarray(Ey, float), nan=fill_value)
    Ez0 = np.nan_to_num(np.asarray(Ez, float), nan=fill_value)

    interpolators = [
        RegularGridInterpolator(
            (x_m, y_m, z_m),
            component,
            bounds_error=False,
            fill_value=fill_value,
        )
        for component in (Ex0, Ey0, Ez0)
    ]

    Xg, Yg = np.meshgrid(x_m, y_m, indexing="ij")
    angle = np.radians(float(angle_deg))
    Xs = Xg * np.cos(angle) + Yg * np.sin(angle)
    Ys = -Xg * np.sin(angle) + Yg * np.cos(angle)
    source_xy = np.column_stack([Xs.ravel(), Ys.ravel()])

    sampled = []
    for interpolator in interpolators:
        component_out = np.empty_like(Ex0)
        for k, z_value in enumerate(z_m):
            points = np.column_stack([
                source_xy,
                np.full(source_xy.shape[0], float(z_value)),
            ])
            component_out[:, :, k] = interpolator(points).reshape(len(x_m), len(y_m))
        sampled.append(component_out)

    Ex_s, Ey_s, Ez_r = sampled
    Ex_r = Ex_s * np.cos(angle) - Ey_s * np.sin(angle)
    Ey_r = Ex_s * np.sin(angle) + Ey_s * np.cos(angle)
    Eabs_r = eabs_from_components(Ex_r, Ey_r, Ez_r)

    return {
        "Ex": Ex_r,
        "Ey": Ey_r,
        "Ez": Ez_r,
        "Eperp": np.hypot(Ex_r, Ey_r),
        "|E|": Eabs_r,
        "angle_deg": float(angle_deg),
    }


def align_field_to_vertical_plane(field, *, radius_m, length_m, label=""):
    Ex = np.asarray(field["Ex"])
    Ey = np.asarray(field["Ey"])
    Ez = np.asarray(field["Ez"])
    nx, ny, nz = Ex.shape

    x_m = np.linspace(-float(radius_m), float(radius_m), nx)
    y_m = np.linspace(-float(radius_m), float(radius_m), ny)
    z_m = np.linspace(0.0, float(length_m), nz)

    Eabs_before = eabs_from_components(Ex, Ey, Ez)
    angle_deg, peak_before = rotation_angle_to_vertical_plane(Eabs_before, x_m, y_m)
    rotated = rotate_vector_field_about_z(Ex, Ey, Ez, x_m, y_m, z_m, angle_deg)

    Eabs_after = np.asarray(rotated["|E|"])
    peak_after = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs_after), Eabs_after.shape))
    mid_x = nx // 2
    global_max = float(np.nanmax(Eabs_after))
    vertical_max = float(np.nanmax(Eabs_after[mid_x, :, :]))

    if abs(peak_after[0] - mid_x) > 1:
        raise RuntimeError(
            f"{label}: rotation failed; peak_after={peak_after}, expected x index {mid_x}, "
            f"angle={angle_deg:.12g} deg."
        )
    if not np.isclose(vertical_max, global_max, rtol=1e-10, atol=max(1e-14, 1e-12 * global_max)):
        raise RuntimeError(
            f"{label}: Eabs[mid_x,:,:] does not contain the global maximum after rotation."
        )

    aligned = {
        "Ex": np.asarray(rotated["Ex"]),
        "Ey": np.asarray(rotated["Ey"]),
        "Ez": np.asarray(rotated["Ez"]),
        "Eperp": np.asarray(rotated["Eperp"]),
        "|E|": Eabs_after,
    }
    diagnostics = {
        "rotation_applied": True,
        "rotation_angle_deg": float(angle_deg),
        "peak_before_xyz": peak_before,
        "peak_after_xyz": peak_after,
        "mid_x_index": int(mid_x),
        "global_max_Eabs_after": global_max,
        "vertical_plane_max_Eabs_after": vertical_max,
        "vertical_plane_contains_global_max": True,
    }
    print(
        f"{label}: rotated by {angle_deg:.6f} deg; peak {peak_before} -> {peak_after}; "
        f"Eabs[{mid_x},:,:] contains the global |E| maximum."
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
    E1,
    E2,
    folder,
    *,
    create_fields,
    family_m,
    rotation_E1=None,
    rotation_E2=None,
):
    filename = Path(folder) / "field_data.npz"
    recreate = bool(create_fields or not filename.exists())

    if not recreate and int(family_m) in (1, 2):
        existing = load_npz_dict(filename)
        flag = existing.get("vertical_plane_rotation_applied")
        if flag is None or not bool(np.asarray(flag).item()):
            print(f"{filename}: old unrotated field data detected; regenerating.")
            recreate = True
        else:
            return existing

    if recreate:
        data = combine_crossing_fields(E1, E2)
        data["vertical_plane_rotation_applied"] = np.asarray(
            int(family_m) in (1, 2),
            dtype=np.bool_,
        )
        for prefix, diag in (("E1", rotation_E1), ("E2", rotation_E2)):
            if diag:
                data[f"{prefix}_rotation_angle_deg"] = np.asarray(diag["rotation_angle_deg"], float)
                data[f"{prefix}_peak_before_xyz"] = np.asarray(diag["peak_before_xyz"], int)
                data[f"{prefix}_peak_after_xyz"] = np.asarray(diag["peak_after_xyz"], int)
        save_npz_dict(filename, data)
        return data

    return load_npz_dict(filename)


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

    E1_original = family_data["TM"][mode_i]["3D_Efield"]
    E2_original = family_data["TM"][mode_j]["3D_Efield"]

    d0 = float(family_data["metadata"]["design_length_m"])
    R = float(family_data["metadata"]["pillbox_radius_m"])

    rotation_E1 = None
    rotation_E2 = None
    if m_i in (1, 2):
        E1, rotation_E1 = align_field_to_vertical_plane(
            E1_original,
            radius_m=R,
            length_m=d0,
            label=f"TM_{mode_i}",
        )
        E2, rotation_E2 = align_field_to_vertical_plane(
            E2_original,
            radius_m=R,
            length_m=d0,
            label=f"TM_{mode_j}",
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
    ]
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
        create_family_data=True,
        create_fields=True,
        make_plots=True,
    )
    main(config)
