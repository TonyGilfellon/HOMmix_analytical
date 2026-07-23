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

SCRIPT_VERSION = "2026-07-23-publication-longitudinal-map-v8"

# Publication styling for the longitudinal-voltage map.
PUBLICATION_MAP_LIMIT_MM = 10.0
PUBLICATION_MAP_TICKS_MM = np.array(
    [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
    dtype=float,
)
PUBLICATION_AXIS_LABEL_FONTSIZE = 14.0
PUBLICATION_TICK_FONTSIZE = 12.0
PUBLICATION_COLORBAR_LABEL_FONTSIZE = 14.0
PUBLICATION_COLORBAR_TICK_FONTSIZE = 12.0


def analysis_circle_geometry(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    analysis_pixels: int,
) -> tuple[float, np.ndarray]:
    """Return the physical Taylor-analysis radius and a circular grid mask.

    ``analysis_pixels`` is interpreted in the same way as the Taylor fitting
    window: the radius is that number of transverse pixels from the axis.
    The smaller of dx and dy is used so the circle always lies inside the
    corresponding square pixel window.
    """
    if int(analysis_pixels) < 1:
        raise ValueError("analysis_pixels must be >= 1")
    dx = float(np.mean(np.diff(x_m)))
    dy = float(np.mean(np.diff(y_m)))
    radius_m = int(analysis_pixels) * min(abs(dx), abs(dy))
    X, Y = np.meshgrid(x_m, y_m, indexing="ij")
    mask = X**2 + Y**2 <= radius_m**2 + np.finfo(float).eps
    return radius_m, mask



def fit_quadratic_taylor_inside_circle(
    Vz_xy: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Fit a complex second-order Taylor model to Vz inside ``mask``.

    The coefficient convention is

        Vz = V0 + Vx*x + Vy*y
             + 0.5*Vxx*x**2 + Vxy*x*y + 0.5*Vyy*y**2.

    Fitting the complex voltage retains the phase information needed by the
    Panofsky-Wenzel factor.  Plotting and residual diagnostics use Re(...).
    """
    Vz = np.asarray(Vz_xy, dtype=complex)
    X, Y = np.meshgrid(x_m, y_m, indexing="ij")
    valid = mask & np.isfinite(Vz.real) & np.isfinite(Vz.imag)
    if np.count_nonzero(valid) < 6:
        raise ValueError("At least six valid samples are required for a quadratic Taylor fit.")

    A = np.column_stack(
        [
            np.ones(np.count_nonzero(valid)),
            X[valid],
            Y[valid],
            0.5 * X[valid] ** 2,
            X[valid] * Y[valid],
            0.5 * Y[valid] ** 2,
        ]
    )
    coeff, _, rank, singular_values = np.linalg.lstsq(
        A, Vz[valid], rcond=None
    )
    names = ("V0", "Vx", "Vy", "Vxx", "Vxy", "Vyy")
    coefficients = {name: coeff[i] for i, name in enumerate(names)}
    return {
        "coefficients": coefficients,
        "rank": int(rank),
        "singular_values": singular_values,
        "n_points": int(np.count_nonzero(valid)),
    }


def reconstruct_maps_from_taylor(
    fit: dict[str, Any],
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    frequency_Hz: float,
) -> dict[str, np.ndarray]:
    """Return Vz, transverse voltage and Jacobian from the Taylor model."""
    c = fit["coefficients"]
    X, Y = np.meshgrid(x_m, y_m, indexing="ij")

    Vz = (
        c["V0"]
        + c["Vx"] * X
        + c["Vy"] * Y
        + 0.5 * c["Vxx"] * X**2
        + c["Vxy"] * X * Y
        + 0.5 * c["Vyy"] * Y**2
    )
    dVz_dx = c["Vx"] + c["Vxx"] * X + c["Vxy"] * Y
    dVz_dy = c["Vy"] + c["Vxy"] * X + c["Vyy"] * Y

    omega = 2.0 * np.pi * float(frequency_Hz)
    pw_factor = 1j * C0 / omega
    Vx = pw_factor * dVz_dx
    Vy = pw_factor * dVz_dy

    shape = Vz.shape
    Cxx = np.full(shape, pw_factor * c["Vxx"], dtype=complex)
    Cxy = np.full(shape, pw_factor * c["Vxy"], dtype=complex)
    Cyx = np.full(shape, pw_factor * c["Vxy"], dtype=complex)
    Cyy = np.full(shape, pw_factor * c["Vyy"], dtype=complex)

    return {
        "Vz": Vz,
        "dVz_dx": dVz_dx,
        "dVz_dy": dVz_dy,
        "Vx": Vx,
        "Vy": Vy,
        "Cxx": Cxx,
        "Cxy": Cxy,
        "Cyx": Cyx,
        "Cyy": Cyy,
    }


def residual_statistics(
    data_xy: np.ndarray,
    reconstruction_xy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Return RMS and relative RMS for real-part residuals inside ``mask``."""
    data = real_part(data_xy)
    reconstruction = real_part(reconstruction_xy)
    valid = mask & np.isfinite(data) & np.isfinite(reconstruction)
    if not np.any(valid):
        return {"rms": float("nan"), "relative_rms": float("nan"), "max_abs": float("nan")}
    residual = data[valid] - reconstruction[valid]
    rms = float(np.sqrt(np.mean(residual**2)))
    data_rms = float(np.sqrt(np.mean(data[valid] ** 2)))
    return {
        "rms": rms,
        "relative_rms": rms / data_rms if data_rms > 0.0 else float("nan"),
        "max_abs": float(np.max(np.abs(residual))),
    }

def real_part(array: np.ndarray) -> np.ndarray:
    """Return Re(array), replacing non-finite values by NaN for plotting."""
    out = np.asarray(array, dtype=complex).real.astype(float, copy=False)
    out = np.array(out, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def symmetric_limit_inside_mask(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    mask: np.ndarray,
) -> float:
    """Return a shared symmetric colour limit from values inside ``mask``."""
    maxima: list[float] = []
    for array in arrays:
        a = np.asarray(array, dtype=float)
        values = a[mask & np.isfinite(a)]
        if values.size:
            maxima.append(float(np.max(np.abs(values))))
    limit = max(maxima, default=1.0)
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def masked_to_analysis_circle(data_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask all samples outside the circular Taylor-analysis region."""
    return np.ma.array(np.asarray(data_xy, dtype=float), mask=~mask)


def add_analysis_circle(ax: plt.Axes, radius_mm: float) -> None:
    from matplotlib.patches import Circle

    circle = Circle(
        (0.0, 0.0),
        radius_mm,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.2,
        zorder=10,
        label=f"Taylor radius = {radius_mm:.3f} mm",
    )
    ax.add_patch(circle)


def add_map(
    ax: plt.Axes,
    data_xy: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    title: str | None,
    colourbar_label: str,
    limit: float,
    analysis_radius_mm: float,
    zoom: bool,
    analysis_mask: np.ndarray,
    data_scale: float = 1.0,
    axis_limit_mm: float | None = None,
    axis_ticks_mm: np.ndarray | None = None,
    axis_label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    colorbar_label_fontsize: float | None = None,
    colorbar_tick_fontsize: float | None = None,
    show_analysis_circle: bool = True,
) -> None:
    plot_data = (
        masked_to_analysis_circle(data_xy, analysis_mask)
        if zoom
        else np.asarray(data_xy, dtype=float)
    )
    plot_data = plot_data * float(data_scale)
    scaled_limit = float(limit) * abs(float(data_scale))

    image = ax.imshow(
        plot_data.T,
        origin="lower",
        extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-scaled_limit,
        vmax=scaled_limit,
    )
    ax.axvline(0.0, color="k", alpha=0.25, linewidth=0.7, zorder=9)
    ax.axhline(0.0, color="k", alpha=0.25, linewidth=0.7, zorder=9)

    if show_analysis_circle:
        add_analysis_circle(ax, analysis_radius_mm)

    if axis_limit_mm is not None:
        ax.set_xlim(-float(axis_limit_mm), float(axis_limit_mm))
        ax.set_ylim(-float(axis_limit_mm), float(axis_limit_mm))
    elif zoom:
        pad = 1.05 * analysis_radius_mm
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad, pad)

    if axis_ticks_mm is not None:
        ticks = np.asarray(axis_ticks_mm, dtype=float)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    if title:
        ax.set_title(title)

    ax.set_xlabel("x [mm]", fontsize=axis_label_fontsize)
    ax.set_ylabel("y [mm]", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    cbar = ax.figure.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label(colourbar_label, fontsize=colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)


def plot_longitudinal_voltage(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
    zoom: bool,
) -> None:
    """Plot the transverse map of Re(Vz) using publication styling.

    The plotted voltage is converted from volts to millivolts using

        1 V = 1000 mV.

    The displayed transverse range is fixed to -10 mm <= x,y <= 10 mm,
    with ticks every 2.5 mm. The title and figure-level suptitle are omitted.
    """
    x_m = maps["x_m"]
    y_m = maps["y_m"]
    x_mm = x_m * 1.0e3
    y_mm = y_m * 1.0e3

    radius_m, mask = analysis_circle_geometry(
        x_m,
        y_m,
        analysis_pixels=analysis_pixels,
    )
    Vz_real_V = real_part(maps["Vz"])
    limit_V = symmetric_limit_inside_mask([Vz_real_V], mask)

    fig, ax = plt.subplots(
        figsize=(6.2, 5.4),
        constrained_layout=True,
    )
    add_map(
        ax,
        Vz_real_V,
        x_mm,
        y_mm,
        title=None,
        colourbar_label="mV",
        limit=limit_V,
        analysis_radius_mm=radius_m * 1.0e3,
        zoom=zoom,
        analysis_mask=mask,
        data_scale=1.0e3,
        axis_limit_mm=PUBLICATION_MAP_LIMIT_MM,
        axis_ticks_mm=PUBLICATION_MAP_TICKS_MM,
        axis_label_fontsize=PUBLICATION_AXIS_LABEL_FONTSIZE,
        tick_fontsize=PUBLICATION_TICK_FONTSIZE,
        colorbar_label_fontsize=PUBLICATION_COLORBAR_LABEL_FONTSIZE,
        colorbar_tick_fontsize=PUBLICATION_COLORBAR_TICK_FONTSIZE,
        show_analysis_circle=True,
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_transverse_voltage(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
    zoom: bool,
) -> None:
    x_m = maps["x_m"]
    y_m = maps["y_m"]
    x_mm = x_m * 1.0e3
    y_mm = y_m * 1.0e3
    radius_m, mask = analysis_circle_geometry(
        x_m, y_m, analysis_pixels=analysis_pixels
    )
    Vx_real = real_part(maps["Vx"])
    Vy_real = real_part(maps["Vy"])
    limit = symmetric_limit_inside_mask([Vx_real, Vy_real], mask)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    for ax, data, panel_title in (
        (axes[0], Vx_real, r"$\mathrm{Re}(V_x)$"),
        (axes[1], Vy_real, r"$\mathrm{Re}(V_y)$"),
    ):
        add_map(
            ax,
            data,
            x_mm,
            y_mm,
            title=panel_title,
            colourbar_label="V",
            limit=limit,
            analysis_radius_mm=radius_m * 1.0e3,
            zoom=zoom,
            analysis_mask=mask,
        )
    scope = "within Taylor-analysis radius" if zoom else "full transverse domain"
    fig.suptitle(f"{title}\nTransverse voltage: {scope}")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transverse_jacobian(
    maps: dict[str, Any],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
    zoom: bool,
) -> None:
    x_m = maps["x_m"]
    y_m = maps["y_m"]
    x_mm = x_m * 1.0e3
    y_mm = y_m * 1.0e3
    radius_m, mask = analysis_circle_geometry(
        x_m, y_m, analysis_pixels=analysis_pixels
    )
    arrays = [
        real_part(maps["Cxx"]),
        real_part(maps["Cxy"]),
        real_part(maps["Cyx"]),
        real_part(maps["Cyy"]),
    ]
    limit = symmetric_limit_inside_mask(arrays, mask)
    labels = [
        r"$\mathrm{Re}(\partial V_x/\partial x)$",
        r"$\mathrm{Re}(\partial V_x/\partial y)$",
        r"$\mathrm{Re}(\partial V_y/\partial x)$",
        r"$\mathrm{Re}(\partial V_y/\partial y)$",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), constrained_layout=True)
    for ax, array, label in zip(axes.flat, arrays, labels):
        add_map(
            ax,
            array,
            x_mm,
            y_mm,
            title=label,
            colourbar_label="V/m",
            limit=limit,
            analysis_radius_mm=radius_m * 1.0e3,
            zoom=zoom,
            analysis_mask=mask,
        )
    scope = "within Taylor-analysis radius" if zoom else "full transverse domain"
    fig.suptitle(f"{title}\nJacobian of transverse voltage: {scope}")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_taylor_longitudinal_residual(
    maps: dict[str, Any],
    taylor_maps: dict[str, np.ndarray],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
) -> None:
    x_m, y_m = maps["x_m"], maps["y_m"]
    x_mm, y_mm = x_m * 1.0e3, y_m * 1.0e3
    radius_m, mask = analysis_circle_geometry(x_m, y_m, analysis_pixels=analysis_pixels)
    data = real_part(maps["Vz"])
    reconstruction = real_part(taylor_maps["Vz"])
    residual = data - reconstruction
    field_limit = symmetric_limit_inside_mask([data, reconstruction], mask)
    residual_limit = symmetric_limit_inside_mask([residual], mask)
    stats = residual_statistics(maps["Vz"], taylor_maps["Vz"], mask)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    panels = (
        (data, r"Numerical $\mathrm{Re}(V_z)$", field_limit),
        (reconstruction, r"Taylor $\mathrm{Re}(V_z^{(2)})$", field_limit),
        (residual, r"Residual $\mathrm{Re}(V_z-V_z^{(2)})$", residual_limit),
    )
    for ax, (array, panel_title, limit) in zip(axes, panels):
        add_map(
            ax, array, x_mm, y_mm, title=panel_title, colourbar_label="V",
            limit=limit, analysis_radius_mm=radius_m * 1.0e3, zoom=True,
            analysis_mask=mask,
        )
    fig.suptitle(
        f"{title}\nSecond-order Taylor reconstruction of longitudinal voltage | "
        f"RMS residual = {stats['rms']:.3e} V | relative RMS = {stats['relative_rms']:.3e}"
    )
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_taylor_transverse_residual(
    maps: dict[str, Any],
    taylor_maps: dict[str, np.ndarray],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
) -> None:
    x_m, y_m = maps["x_m"], maps["y_m"]
    x_mm, y_mm = x_m * 1.0e3, y_m * 1.0e3
    radius_m, mask = analysis_circle_geometry(x_m, y_m, analysis_pixels=analysis_pixels)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)

    stats_text = []
    for row, key in enumerate(("Vx", "Vy")):
        data = real_part(maps[key])
        reconstruction = real_part(taylor_maps[key])
        residual = data - reconstruction
        field_limit = symmetric_limit_inside_mask([data, reconstruction], mask)
        residual_limit = symmetric_limit_inside_mask([residual], mask)
        stats = residual_statistics(maps[key], taylor_maps[key], mask)
        stats_text.append(
            f"{key}: RMS={stats['rms']:.3e} V, rel={stats['relative_rms']:.3e}"
        )
        for col, (array, panel_title, limit) in enumerate(
            (
                (data, rf"Numerical $\mathrm{{Re}}({key})$", field_limit),
                (reconstruction, rf"Taylor $\mathrm{{Re}}({key}^{{(2)}})$", field_limit),
                (residual, rf"Residual $\mathrm{{Re}}({key}-{key}^{{(2)}})$", residual_limit),
            )
        ):
            add_map(
                axes[row, col], array, x_mm, y_mm, title=panel_title,
                colourbar_label="V", limit=limit,
                analysis_radius_mm=radius_m * 1.0e3, zoom=True,
                analysis_mask=mask,
            )
    fig.suptitle(
        f"{title}\nSecond-order Taylor reconstruction of transverse voltage | "
        + " | ".join(stats_text)
    )
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_taylor_jacobian_residual(
    maps: dict[str, Any],
    taylor_maps: dict[str, np.ndarray],
    *,
    title: str,
    outfile: Path,
    analysis_pixels: int,
) -> None:
    """Plot numerical, Taylor and residual Jacobian components in a 4x3 grid."""
    x_m, y_m = maps["x_m"], maps["y_m"]
    x_mm, y_mm = x_m * 1.0e3, y_m * 1.0e3
    radius_m, mask = analysis_circle_geometry(x_m, y_m, analysis_pixels=analysis_pixels)
    keys = ("Cxx", "Cxy", "Cyx", "Cyy")
    labels = (
        r"$\partial V_x/\partial x$",
        r"$\partial V_x/\partial y$",
        r"$\partial V_y/\partial x$",
        r"$\partial V_y/\partial y$",
    )
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 18.0), constrained_layout=True)
    stats_text = []
    for row, (key, label) in enumerate(zip(keys, labels)):
        data = real_part(maps[key])
        reconstruction = real_part(taylor_maps[key])
        residual = data - reconstruction
        field_limit = symmetric_limit_inside_mask([data, reconstruction], mask)
        residual_limit = symmetric_limit_inside_mask([residual], mask)
        stats = residual_statistics(maps[key], taylor_maps[key], mask)
        stats_text.append(f"{key}: rel={stats['relative_rms']:.3e}")
        for col, (array, prefix, limit) in enumerate(
            (
                (data, "Numerical", field_limit),
                (reconstruction, "Taylor", field_limit),
                (residual, "Residual", residual_limit),
            )
        ):
            add_map(
                axes[row, col], array, x_mm, y_mm,
                title=f"{prefix} Re({label})", colourbar_label="V/m",
                limit=limit, analysis_radius_mm=radius_m * 1.0e3, zoom=True,
                analysis_mask=mask,
            )
    fig.suptitle(
        f"{title}\nSecond-order Taylor reconstruction of transverse-voltage Jacobian | "
        + " | ".join(stats_text)
    )
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
    fit_pixels_dipole: int,
    fit_pixels_quadrupole: int,
    taylor_fit_dipole: dict[str, Any] | None = None,
    taylor_fit_quadrupole: dict[str, Any] | None = None,
    taylor_fit: dict[str, Any] | None = None,
    taylor_maps: dict[str, np.ndarray] | None = None,
) -> None:
    dipole_radius_m, dipole_mask = analysis_circle_geometry(
        maps["x_m"], maps["y_m"], analysis_pixels=fit_pixels_dipole
    )
    quadrupole_radius_m, quadrupole_mask = analysis_circle_geometry(
        maps["x_m"], maps["y_m"], analysis_pixels=fit_pixels_quadrupole
    )
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
        fit_pixels_dipole=int(fit_pixels_dipole),
        fit_pixels_quadrupole=int(fit_pixels_quadrupole),
        dipole_analysis_radius_m=float(dipole_radius_m),
        quadrupole_analysis_radius_m=float(quadrupole_radius_m),
        dipole_analysis_mask=dipole_mask,
        quadrupole_analysis_mask=quadrupole_mask,
        length_m=float(length_m),
        frequency_Hz=float(frequency_Hz),
        beta=float(beta),
        centred_z=bool(centred_z),
        plotted_component="real",
        taylor_coefficients_dipole=np.array(
            [taylor_fit_dipole["coefficients"][name] for name in ("V0", "Vx", "Vy")],
            dtype=complex,
        ) if taylor_fit_dipole is not None else np.array([], dtype=complex),
        taylor_coefficients_quadrupole=np.array(
            [taylor_fit_quadrupole["coefficients"][name] for name in ("Vxx", "Vxy", "Vyy")],
            dtype=complex,
        ) if taylor_fit_quadrupole is not None else np.array([], dtype=complex),
        taylor_coefficients=np.array(
            [
                taylor_fit["coefficients"][name]
                for name in ("V0", "Vx", "Vy", "Vxx", "Vxy", "Vyy")
            ],
            dtype=complex,
        ) if taylor_fit is not None else np.array([], dtype=complex),
        taylor_coefficient_names=np.array(
            ["V0", "Vx", "Vy", "Vxx", "Vxy", "Vyy"]
        ),
        Vz_taylor=taylor_maps["Vz"] if taylor_maps is not None else np.array([]),
        Vx_taylor=taylor_maps["Vx"] if taylor_maps is not None else np.array([]),
        Vy_taylor=taylor_maps["Vy"] if taylor_maps is not None else np.array([]),
        Cxx_taylor=taylor_maps["Cxx"] if taylor_maps is not None else np.array([]),
        Cxy_taylor=taylor_maps["Cxy"] if taylor_maps is not None else np.array([]),
        Cyx_taylor=taylor_maps["Cyx"] if taylor_maps is not None else np.array([]),
        Cyy_taylor=taylor_maps["Cyy"] if taylor_maps is not None else np.array([]),
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
    fit_pixels_dipole: int = 4,
    fit_pixels_quadrupole: int = 8,
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
        _, dipole_mask = analysis_circle_geometry(
            maps["x_m"], maps["y_m"], analysis_pixels=fit_pixels_dipole
        )
        _, quadrupole_mask = analysis_circle_geometry(
            maps["x_m"], maps["y_m"], analysis_pixels=fit_pixels_quadrupole
        )
        taylor_fit_dipole = fit_quadratic_taylor_inside_circle(
            maps["Vz"], maps["x_m"], maps["y_m"], dipole_mask
        )
        taylor_fit_quadrupole = fit_quadratic_taylor_inside_circle(
            maps["Vz"], maps["x_m"], maps["y_m"], quadrupole_mask
        )

        # Match the heterotypic Hessian/Taylor analysis exactly:
        # V0, Vx and Vy come from the dipole window (4 pixels), while
        # Vxx, Vxy and Vyy come from the quadrupole window (8 pixels).
        hybrid_coefficients = {
            "V0": taylor_fit_dipole["coefficients"]["V0"],
            "Vx": taylor_fit_dipole["coefficients"]["Vx"],
            "Vy": taylor_fit_dipole["coefficients"]["Vy"],
            "Vxx": taylor_fit_quadrupole["coefficients"]["Vxx"],
            "Vxy": taylor_fit_quadrupole["coefficients"]["Vxy"],
            "Vyy": taylor_fit_quadrupole["coefficients"]["Vyy"],
        }
        taylor_fit = {
            "coefficients": hybrid_coefficients,
            "fit_pixels_dipole": int(fit_pixels_dipole),
            "fit_pixels_quadrupole": int(fit_pixels_quadrupole),
        }
        taylor_maps = reconstruct_maps_from_taylor(
            taylor_fit, maps["x_m"], maps["y_m"],
            frequency_Hz=float(job["frequency_Hz"]),
        )

        label = str(job["label"])
        title = (
            f"{label} | f = {float(job['frequency_Hz']) / 1e9:.6f} GHz | "
            f"L = {float(job['length_m']) * 1e3:.6f} mm"
        )
        stem = field_name.lower()
        # Full-domain plots, each with the Taylor-analysis circle overlaid.
        plot_longitudinal_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_longitudinal_voltage.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=False,
        )
        plot_transverse_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage.pdf",
            analysis_pixels=fit_pixels_dipole,
            zoom=False,
        )
        plot_transverse_jacobian(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_jacobian.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=False,
        )

        # Circularly masked, radius-cropped plots.  Their colour limits use only
        # values inside the Taylor-analysis circle, exactly as for the full maps.
        plot_longitudinal_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_longitudinal_voltage_zoom.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=True,
        )
        plot_transverse_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_zoom.pdf",
            analysis_pixels=fit_pixels_dipole,
            zoom=True,
        )
        plot_transverse_jacobian(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_jacobian_zoom.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=True,
        )

        # Duplicate the zoomed figures using the exact analysis-radius filenames
        # originally requested.
        plot_longitudinal_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_longitudinal_voltage_map_analysis_radius.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=True,
        )
        plot_transverse_voltage(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_map_analysis_radius.pdf",
            analysis_pixels=fit_pixels_dipole,
            zoom=True,
        )
        plot_transverse_jacobian(
            maps,
            title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_curvature_map_analysis_radius.pdf",
            analysis_pixels=fit_pixels_quadrupole,
            zoom=True,
        )
        # Taylor reconstruction and residual diagnostics inside the analysis radius.
        plot_taylor_longitudinal_residual(
            maps, taylor_maps, title=title,
            outfile=out_dir / f"{stem}_longitudinal_voltage_taylor_residual.pdf",
            analysis_pixels=fit_pixels_quadrupole,
        )
        plot_taylor_transverse_residual(
            maps, taylor_maps, title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_taylor_residual.pdf",
            analysis_pixels=fit_pixels_dipole,
        )
        plot_taylor_jacobian_residual(
            maps, taylor_maps, title=title,
            outfile=out_dir / f"{stem}_transverse_voltage_jacobian_taylor_residual.pdf",
            analysis_pixels=fit_pixels_quadrupole,
        )

        save_map_data(
            maps,
            outfile=out_dir / f"{stem}_voltage_derivative_maps.npz",
            length_m=float(job["length_m"]),
            frequency_Hz=float(job["frequency_Hz"]),
            beta=beta,
            centred_z=centred_z,
            fit_pixels_dipole=fit_pixels_dipole,
            fit_pixels_quadrupole=fit_pixels_quadrupole,
            taylor_fit_dipole=taylor_fit_dipole,
            taylor_fit_quadrupole=taylor_fit_quadrupole,
            taylor_fit=taylor_fit,
            taylor_maps=taylor_maps,
        )
        print(
            f"Saved 12 PDFs plus NPZ for {field_name} in {out_dir} "
            f"(fit_pixels_dipole={fit_pixels_dipole}, "
            f"fit_pixels_quadrupole={fit_pixels_quadrupole})"
        )
        print(f"  created: {stem}_longitudinal_voltage_zoom.pdf")
        print(f"  created: {stem}_transverse_voltage_zoom.pdf")
        print(f"  created: {stem}_transverse_voltage_jacobian_zoom.pdf")
        print(f"  created: {stem}_longitudinal_voltage_map_analysis_radius.pdf")
        print(f"  created: {stem}_transverse_voltage_map_analysis_radius.pdf")
        print(f"  created: {stem}_transverse_voltage_curvature_map_analysis_radius.pdf")
        print(f"  created: {stem}_longitudinal_voltage_taylor_residual.pdf")
        print(f"  created: {stem}_transverse_voltage_taylor_residual.pdf")
        print(f"  created: {stem}_transverse_voltage_jacobian_taylor_residual.pdf")


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
    print(f"Running heterotypic_voltage_map_derivatives.py [{SCRIPT_VERSION}]")
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
        fit_pixels_dipole=4,
        fit_pixels_quadrupole=8,
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
