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
from matplotlib.backends.backend_pdf import PdfPages
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
) -> dict[str, np.ndarray]:
    filename = folder / "field_data.npz"
    if create_fields or not filename.exists():
        data = combine_crossing_fields(E1, E2)
        save_npz_dict(filename, data)
        return data
    return load_npz_dict(filename)


def plot_field_slices_combined(
    field_data: dict[str, np.ndarray],
    out_dir: Path,
    title: str,
) -> Path:
    """Create the established four field-slice figures and appendix PDF.

    Outputs
    -------
    In ``out_dir``:
        iris_1.png
        iris_2.png
        transverse_mid.png
        longitudinal_mid.png

    In ``out_dir / "slice_summary_pdfs"``:
        iris_1.pdf
        iris_2.pdf
        transverse_mid.pdf
        longitudinal_mid.pdf
        <crossing-folder-name>_field_summary.pdf

    The final file is a four-page PDF used by the PRAB appendix compiler.
    Columns are [E1, E2, E-, E+] so the plots and tables use the same order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir = out_dir / "slice_summary_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    merged_pdf = pdf_dir / f"{out_dir.name}_field_summary.pdf"

    slice_specs = {
        "iris_1": lambda a: np.asarray(a)[:, :, 0].T,
        "iris_2": lambda a: np.asarray(a)[:, :, -1].T,
        "transverse_mid": (
            lambda a: np.asarray(a)[
                :,
                :,
                np.asarray(a).shape[2] // 2,
            ].T
        ),
        "longitudinal_mid": (
            lambda a: np.asarray(a)[
                np.asarray(a).shape[0] // 2,
                :,
                :,
            ]
        ),
    }

    # Deliberately use E-, E+ to match the appendix table columns.
    rows = [
        ("E1_Ex", "E2_Ex", "Ex_minus", "Ex_plus"),
        ("E1_Ey", "E2_Ey", "Ey_minus", "Ey_plus"),
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]
    row_titles = [
        r"$E_x/E_{z,\mathrm{ref}}$",
        r"$E_y/E_{z,\mathrm{ref}}$",
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]
    col_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]

    def safe_reference(arrays: list[np.ndarray]) -> float:
        reference = max(
            float(np.nanmax(np.abs(np.asarray(array))))
            for array in arrays
        )
        if not np.isfinite(reference) or reference <= 0.0:
            return 1.0
        return reference

    with PdfPages(merged_pdf) as summary:
        for slice_name, slicer in slice_specs.items():
            fig, axes = plt.subplots(
                4,
                4,
                figsize=(14, 10),
                constrained_layout=True,
            )
            fig.suptitle(f"{title}: {slice_name}")

            parent_ez_ref = safe_reference([
                slicer(field_data["E1_Ez"]),
                slicer(field_data["E2_Ez"]),
            ])
            parent_abs_ref = safe_reference([
                slicer(field_data["abs_E1"]),
                slicer(field_data["abs_E2"]),
            ])

            for row_index, keys in enumerate(rows):
                raw_images = [
                    np.real(slicer(field_data[key]))
                    for key in keys
                ]
                reference = (
                    parent_abs_ref
                    if row_index == 3
                    else parent_ez_ref
                )
                scaled_images = [
                    image / reference
                    for image in raw_images
                ]
                vmax = max(
                    1.0,
                    max(
                        float(np.nanmax(np.abs(image)))
                        for image in scaled_images
                    ),
                )

                row_mappable = None
                for column_index, image in enumerate(scaled_images):
                    axis = axes[row_index, column_index]

                    if row_index == 3:
                        row_mappable = axis.imshow(
                            image,
                            origin="lower",
                            aspect="auto",
                            vmin=0.0,
                            vmax=vmax,
                            cmap="viridis",
                        )
                    else:
                        row_mappable = axis.imshow(
                            image,
                            origin="lower",
                            aspect="auto",
                            vmin=-vmax,
                            vmax=vmax,
                            cmap="RdBu_r",
                        )

                    if row_index == 0:
                        axis.text(
                            0.5,
                            1.02,
                            col_titles[column_index],
                            transform=axis.transAxes,
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            fontweight="bold",
                            zorder=100,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.90,
                                "pad": 2.0,
                            },
                            clip_on=False,
                        )

                    if column_index == 0:
                        axis.text(
                            -0.12,
                            0.5,
                            row_titles[row_index],
                            transform=axis.transAxes,
                            rotation=90,
                            ha="center",
                            va="center",
                            fontsize=11,
                        )

                    axis.set_xticks([])
                    axis.set_yticks([])

                if row_mappable is not None:
                    fig.colorbar(
                        row_mappable,
                        ax=axes[row_index, :],
                        shrink=0.75,
                    )

            png_path = out_dir / f"{slice_name}.png"
            individual_pdf = pdf_dir / f"{slice_name}.pdf"

            fig.savefig(png_path, dpi=250)
            fig.savefig(individual_pdf)
            summary.savefig(fig)
            plt.close(fig)

    print(f"Saved appendix field-summary PDF: {merged_pdf}")
    return merged_pdf


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

    E1 = family_data["TM"][mode_i]["3D_Efield"]
    E2 = family_data["TM"][mode_j]["3D_Efield"]
    field_data = get_or_create_field_data(E1, E2, folder, create_fields=cfg.create_fields)

    d0 = float(family_data["metadata"]["design_length_m"])
    R = float(family_data["metadata"]["pillbox_radius_m"])
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
