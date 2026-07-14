"""Generate longitudinal-voltage, transverse-voltage and curvature maps.

This script reads the heterotypic ``field_data.npz`` and
``heterotypic_crossing_analysis.pkl`` files produced for each crossing folder.
For E1, E2, E+ and E- it calculates

    Vz(x,y) = integral Ez(x,y,z) exp(i omega z / beta c) dz

then applies Panofsky-Wenzel,

    Vx = (i c / omega) dVz/dx
    Vy = (i c / omega) dVz/dy

and differentiates the transverse voltage once more,

    Cxx = dVx/dx,  Cxy = dVx/dy
    Cyx = dVy/dx,  Cyy = dVy/dy.

For every field, three summary figures and one NPZ data file are written to

    <crossing folder>/voltage_derivative_maps/

The coordinate, length and frequency conventions follow
``heterotypic_Hessian_Taylor_analysis_U_CST.py``.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

C0 = 299_792_458.0


# -----------------------------------------------------------------------------
# I/O and coordinate helpers
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    filename = Path(filename)
    try:
        with np.load(filename, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(f"Invalid or corrupt NPZ file: {filename}") from exc


def centred_transverse_coords(
    nx: int,
    ny: int,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_m = np.linspace(-float(radius_m), float(radius_m), int(nx))
    y_m = np.linspace(-float(radius_m), float(radius_m), int(ny))
    return x_m, y_m


def longitudinal_coords(
    nz: int,
    length_m: float,
    *,
    centred: bool = False,
) -> np.ndarray:
    z_m = np.linspace(0.0, float(length_m), int(nz))
    if centred:
        z_m -= 0.5 * float(length_m)
    return z_m


def pillbox_radius_from_f010(f_010_Hz: float) -> float:
    v01 = 2.404825557695773
    return float(v01 * C0 / (2.0 * np.pi * float(f_010_Hz)))


def design_length_from_f010(f_010_Hz: float) -> float:
    return float(C0 / (2.0 * float(f_010_Hz)))


def lookup_parent_frequency(
    family_data_by_m: dict[int, dict] | None,
    *,
    mode_name: str,
    fallback_Hz: float,
) -> float:
    """Return the parent design frequency, or the crossing frequency fallback."""
    if family_data_by_m is None:
        return float(fallback_Hz)

    try:
        _, mnp = mode_name.split("_", 1)
        m = int(mnp[0])
        return float(family_data_by_m[m]["TM"][mnp]["design_frequency_Hz"])
    except (ValueError, KeyError, IndexError) as exc:
        raise KeyError(
            f"Could not obtain a design frequency for {mode_name!r} from family data."
        ) from exc


def load_family_data_files(*filenames: str | Path) -> dict[int, dict]:
    family_data: dict[int, dict] = {}
    for filename in filenames:
        filename = Path(filename)
        data = pickle_load(filename)
        if "metadata" in data and "family_m" in data["metadata"]:
            m = int(data["metadata"]["family_m"])
        else:
            first_mnp = next(iter(data["TM"]))
            m = int(first_mnp[0])
        family_data[m] = data
        print(f"Loaded m={m} parent-family data: {filename}")
    return family_data


# -----------------------------------------------------------------------------
# Voltage maps and numerical derivatives
# -----------------------------------------------------------------------------

def complex_longitudinal_voltage_map(
    Ez_xyz: np.ndarray,
    *,
    length_m: float,
    frequency_Hz: float,
    beta: float = 1.0,
    centred_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate Ez[x,y,z] along z, including the transit-time phase."""
    Ez = np.nan_to_num(
        np.asarray(Ez_xyz, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if Ez.ndim != 3:
        raise ValueError(f"Ez_xyz must be 3D; received shape {Ez.shape}.")

    z_m = longitudinal_coords(Ez.shape[2], length_m, centred=centred_z)
    omega = 2.0 * np.pi * float(frequency_Hz)
    phase = np.exp(1j * omega * z_m / (float(beta) * C0))
    Vz_xy = np.trapezoid(Ez * phase[None, None, :], z_m, axis=2)
    return Vz_xy, z_m


def transverse_voltage_from_longitudinal(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    frequency_Hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dVz/dx, dVz/dy, Vx and Vy using Panofsky-Wenzel."""
    edge_order = 2 if min(Vz_xy.shape) >= 3 else 1
    dVz_dx, dVz_dy = np.gradient(
        Vz_xy,
        x_m,
        y_m,
        edge_order=edge_order,
    )
    omega = 2.0 * np.pi * float(frequency_Hz)
    pw_factor = 1j * C0 / omega
    Vx_xy = pw_factor * dVz_dx
    Vy_xy = pw_factor * dVz_dy
    return dVz_dx, dVz_dy, Vx_xy, Vy_xy


def transverse_voltage_curvature(
    Vx_xy: np.ndarray,
    Vy_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the full Jacobian of the complex transverse-voltage map."""
    edge_order = 2 if min(Vx_xy.shape) >= 3 else 1
    dVx_dx, dVx_dy = np.gradient(
        Vx_xy,
        x_m,
        y_m,
        edge_order=edge_order,
    )
    dVy_dx, dVy_dy = np.gradient(
        Vy_xy,
        x_m,
        y_m,
        edge_order=edge_order,
    )
    return {
        "Cxx": dVx_dx,
        "Cxy": dVx_dy,
        "Cyx": dVy_dx,
        "Cyy": dVy_dy,
    }


def calculate_voltage_derivative_maps(
    Ez_xyz: np.ndarray,
    *,
    radius_m: float,
    length_m: float,
    frequency_Hz: float,
    beta: float = 1.0,
    centred_z: bool = False,
) -> dict[str, Any]:
    nx, ny, _ = np.asarray(Ez_xyz).shape
    x_m, y_m = centred_transverse_coords(nx, ny, radius_m)

    Vz_xy, z_m = complex_longitudinal_voltage_map(
        Ez_xyz,
        length_m=length_m,
        frequency_Hz=frequency_Hz,
        beta=beta,
        centred_z=centred_z,
    )
    dVz_dx, dVz_dy, Vx_xy, Vy_xy = transverse_voltage_from_longitudinal(
        Vz_xy,
        x_m,
        y_m,
        frequency_Hz=frequency_Hz,
    )
    curvature = transverse_voltage_curvature(Vx_xy, Vy_xy, x_m, y_m)

    return {
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "Vz": Vz_xy,
        "dVz_dx": dVz_dx,
        "dVz_dy": dVz_dy,
        "Vx": Vx_xy,
        "Vy": Vy_xy,
        **curvature,
    }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def phase_reference(*arrays: np.ndarray) -> float:
    """Use the largest complex sample across arrays as a common phase reference."""
    best = 0.0 + 0.0j
    for array in arrays:
        a = np.asarray(array, dtype=complex)
        finite = np.isfinite(a.real) & np.isfinite(a.imag)
        if not finite.any():
            continue
        values = a[finite]
        candidate = values[int(np.argmax(np.abs(values)))]
        if abs(candidate) > abs(best):
            best = candidate
    return float(np.angle(best)) if abs(best) > 0.0 else 0.0


def phase_aligned_real(array: np.ndarray, phase_rad: float) -> np.ndarray:
    return (np.asarray(array, dtype=complex) * np.exp(-1j * phase_rad)).real


def symmetric_limit(*arrays: np.ndarray) -> float:
    values = [np.nanmax(np.abs(a)) for a in arrays if np.asarray(a).size]
    limit = float(max(values)) if values else 1.0
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def add_map(
    ax: plt.Axes,
    data_xy: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    title: str,
    colourbar_label: str,
    signed: bool,
    limit: float | None = None,
) -> None:
    kwargs: dict[str, Any] = {}
    if signed:
        lim = symmetric_limit(data_xy) if limit is None else limit
        kwargs.update(vmin=-lim, vmax=lim, cmap="RdBu_r")
    else:
        kwargs.update(vmin=0.0, cmap="viridis")

    image = ax.imshow(
        np.asarray(data_xy).T,
        origin="lower",
        extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
        aspect="equal",
        interpolation="nearest",
        **kwargs,
    )
    ax.axvline(0.0, color="k", alpha=0.25, linewidth=0.7)
    ax.axhline(0.0, color="k", alpha=0.25, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cbar = ax.figure.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label(colourbar_label)


def plot_longitudinal_voltage(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
) -> None:
    x_mm = maps["x_m"] * 1.0e3
    y_mm = maps["y_m"] * 1.0e3
    Vz = maps["Vz"]
    phase = phase_reference(Vz)
    Vz_aligned = Vz * np.exp(-1j * phase)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    signed_lim = symmetric_limit(Vz_aligned.real, Vz_aligned.imag)
    add_map(
        axes[0], Vz_aligned.real, x_mm, y_mm,
        title="Phase-aligned Re($V_z$)", colourbar_label="V", signed=True,
        limit=signed_lim,
    )
    add_map(
        axes[1], Vz_aligned.imag, x_mm, y_mm,
        title="Phase-aligned Im($V_z$)", colourbar_label="V", signed=True,
        limit=signed_lim,
    )
    add_map(
        axes[2], np.abs(Vz), x_mm, y_mm,
        title="$|V_z|$", colourbar_label="V", signed=False,
    )
    fig.suptitle(f"{title}\nLongitudinal voltage map")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transverse_voltage(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
) -> None:
    x_mm = maps["x_m"] * 1.0e3
    y_mm = maps["y_m"] * 1.0e3
    Vx, Vy = maps["Vx"], maps["Vy"]
    phase = phase_reference(Vx, Vy)
    Vx_real = phase_aligned_real(Vx, phase)
    Vy_real = phase_aligned_real(Vy, phase)
    signed_lim = symmetric_limit(Vx_real, Vy_real)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    add_map(
        axes[0, 0], Vx_real, x_mm, y_mm,
        title="Phase-aligned Re($V_x$)", colourbar_label="V", signed=True,
        limit=signed_lim,
    )
    add_map(
        axes[0, 1], Vy_real, x_mm, y_mm,
        title="Phase-aligned Re($V_y$)", colourbar_label="V", signed=True,
        limit=signed_lim,
    )
    add_map(
        axes[1, 0], np.abs(Vx), x_mm, y_mm,
        title="$|V_x|$", colourbar_label="V", signed=False,
    )
    add_map(
        axes[1, 1], np.abs(Vy), x_mm, y_mm,
        title="$|V_y|$", colourbar_label="V", signed=False,
    )
    fig.suptitle(f"{title}\nTransverse voltage from Panofsky-Wenzel")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transverse_curvature(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
) -> None:
    x_mm = maps["x_m"] * 1.0e3
    y_mm = maps["y_m"] * 1.0e3
    arrays = [maps["Cxx"], maps["Cxy"], maps["Cyx"], maps["Cyy"]]
    phase = phase_reference(*arrays)
    aligned = [phase_aligned_real(a, phase) for a in arrays]
    signed_lim = symmetric_limit(*aligned)

    labels = [
        r"$C_{xx}=\partial V_x/\partial x$",
        r"$C_{xy}=\partial V_x/\partial y$",
        r"$C_{yx}=\partial V_y/\partial x$",
        r"$C_{yy}=\partial V_y/\partial y$",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    for ax, array, label in zip(axes.flat, aligned, labels):
        add_map(
            ax, array, x_mm, y_mm,
            title=label, colourbar_label="V/m", signed=True,
            limit=signed_lim,
        )
    fig.suptitle(f"{title}\nCurvature/Jacobian of the transverse voltage")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_map_data(
    maps: dict[str, Any],
    *,
    outfile: Path,
    length_m: float,
    frequency_Hz: float,
    beta: float,
    centred_z: bool,
) -> None:
    np.savez_compressed(
        outfile,
        x_m=maps["x_m"],
        y_m=maps["y_m"],
        z_m=maps["z_m"],
        Vz=maps["Vz"],
        dVz_dx=maps["dVz_dx"],
        dVz_dy=maps["dVz_dy"],
        Vx=maps["Vx"],
        Vy=maps["Vy"],
        Cxx=maps["Cxx"],
        Cxy=maps["Cxy"],
        Cyx=maps["Cyx"],
        Cyy=maps["Cyy"],
        length_m=float(length_m),
        frequency_Hz=float(frequency_Hz),
        beta=float(beta),
        centred_z=bool(centred_z),
    )


# -----------------------------------------------------------------------------
# Crossing-folder and batch processing
# -----------------------------------------------------------------------------

def analyse_crossing_folder(
    crossing_folder: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None,
    f_010_Hz: float = 1.3e9,
    radius_m: float | None = None,
    design_length_m: float | None = None,
    beta: float = 1.0,
    centred_z: bool = False,
    output_subfolder: str = "voltage_derivative_maps",
) -> None:
    folder = Path(crossing_folder)
    field_data = load_npz_dict(folder / "field_data.npz")
    metadata = pickle_load(folder / "heterotypic_crossing_analysis.pkl")
    crossing = metadata["crossing"]

    radius = radius_m or pillbox_radius_from_f010(f_010_Hz)
    parent_length = design_length_m or design_length_from_f010(f_010_Hz)
    mixed_length = parent_length * float(crossing["length_factor"])
    crossing_frequency = float(crossing["frequency_Hz"])

    f_E1 = lookup_parent_frequency(
        family_data_by_m,
        mode_name=metadata["mode_i"],
        fallback_Hz=crossing_frequency,
    )
    f_E2 = lookup_parent_frequency(
        family_data_by_m,
        mode_name=metadata["mode_j"],
        fallback_Hz=crossing_frequency,
    )

    jobs = {
        "E1": {
            "Ez_key": "E1_Ez",
            "label": metadata["mode_i"],
            "length_m": parent_length,
            "frequency_Hz": f_E1,
        },
        "E2": {
            "Ez_key": "E2_Ez",
            "label": metadata["mode_j"],
            "length_m": parent_length,
            "frequency_Hz": f_E2,
        },
        "plus": {
            "Ez_key": "Ez_plus",
            "label": r"$E_+$",
            "length_m": mixed_length,
            "frequency_Hz": crossing_frequency,
        },
        "minus": {
            "Ez_key": "Ez_minus",
            "label": r"$E_-$",
            "length_m": mixed_length,
            "frequency_Hz": crossing_frequency,
        },
    }

    out_dir = folder / output_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    for field_name, job in jobs.items():
        Ez_key = job["Ez_key"]
        if Ez_key not in field_data:
            print(f"WARNING: {folder} has no {Ez_key}; skipping {field_name}.")
            continue

        maps = calculate_voltage_derivative_maps(
            field_data[Ez_key],
            radius_m=radius,
            length_m=float(job["length_m"]),
            frequency_Hz=float(job["frequency_Hz"]),
            beta=beta,
            centred_z=centred_z,
        )

        label = str(job["label"])
        title = (
            f"{label} | f = {float(job['frequency_Hz']) / 1e9:.6f} GHz | "
            f"L = {float(job['length_m']) * 1e3:.6f} mm"
        )
        stem = field_name.lower()
        plot_longitudinal_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_longitudinal_voltage_map.pdf",
        )
        plot_transverse_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_map.pdf",
        )
        plot_transverse_curvature(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_curvature_map.pdf",
        )
        save_map_data(
            maps,
            outfile=out_dir / f"{stem}_voltage_derivative_maps.npz",
            length_m=float(job["length_m"]),
            frequency_Hz=float(job["frequency_Hz"]),
            beta=beta,
            centred_z=centred_z,
        )
        print(f"Saved voltage maps for {field_name} in {out_dir}")


def find_crossing_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        metadata_file.parent
        for metadata_file in root.rglob("heterotypic_crossing_analysis.pkl")
        if (metadata_file.parent / "field_data.npz").exists()
    )


def analyse_all_crossings(
    root: str | Path,
    *,
    family_data_by_m: dict[int, dict] | None,
    **kwargs: Any,
) -> None:
    folders = find_crossing_folders(root)
    print(f"Found {len(folders)} crossing folders below {root}")
    for folder in folders:
        print(f"\nAnalysing {folder}")
        try:
            analyse_crossing_folder(
                folder,
                family_data_by_m=family_data_by_m,
                **kwargs,
            )
        except Exception as exc:
            print(f"ERROR while processing {folder}: {exc}")


# -----------------------------------------------------------------------------
# Edit-and-run configuration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    heterotypic_root = Path(
        r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings"
    )
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")

    voxel_res = 151
    family_files = [
        datapath / f"TMm0_TMm0_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
        datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
    ]

    missing = [filename for filename in family_files if not filename.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing parent-family data files:\n" + "\n".join(map(str, missing))
        )

    family_data = load_family_data_files(*family_files)

    # Batch mode: analyse every valid crossing folder beneath heterotypic_root.
    analyse_all_crossings(
        heterotypic_root,
        family_data_by_m=family_data,
        f_010_Hz=1.3e9,
        beta=1.0,
        centred_z=False,
        output_subfolder="voltage_derivative_maps",
    )

    # To analyse only one crossing, replace the analyse_all_crossings() call with:
    # analyse_crossing_folder(
    #     heterotypic_root
    #     / "dipole_quadrupole"
    #     / "dipole_quadrupole__TM_111__TM_210__ell_0p70327841",
    #     family_data_by_m=family_data,
    #     f_010_Hz=1.3e9,
    #     beta=1.0,
    #     centred_z=False,
    # )
