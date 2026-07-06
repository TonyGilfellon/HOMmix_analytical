"""Standalone heterotypic TM pillbox mode-crossing and field-mixing study.

This updated version is self-contained: it no longer imports the badly named
HOMmix_analytical_master_module_quadrupole helper module.  The analytical TM
field builder, rotation/alignment helpers, field-combination helpers, slice
extraction, combined field-slice plotting, dipole kick helper, and quadrupole
focusing helper are included directly here.

Array convention throughout: field[x_index, y_index, z_index].
Plot convention:
    - transverse/iris images show x horizontal and y vertical
    - longitudinal images show z horizontal and y vertical

Crossing types
--------------
    monopole-dipole      : TM_0np with TM_1np
    dipole-quadrupole    : TM_1np with TM_2np
    monopole-quadrupole  : TM_0np with TM_2np

The main workflow assembles/loads TM_0np, TM_1np and TM_2np family sweeps, finds
heterotypic crossings, then saves parent/mixed field data and combined plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import jn_zeros

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1.0e-12


def quadrupole_reported_from_raw_K(
    K_raw_real: np.ndarray,
    *,
    U_CST_J: float,
    length_m: float,
) -> dict[str, float]:
    """Canonical reported quadrupole figures in V/pC/m^3.

    Values are square-normalised by stored energy, normalised per metre, and
    converted from per coulomb to per pC:

        |K_raw|^2 / (4 U_CST L) * 1e-12.
    """
    K = np.asarray(K_raw_real, dtype=float)
    U = float(U_CST_J)
    L = float(length_m)
    if not np.isfinite(U) or U <= 0.0:
        raise ValueError(f"U_CST_J must be positive and finite, got {U_CST_J!r}")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(f"length_m must be positive and finite, got {length_m!r}")
    scale = PC / (4.0 * U * L)
    return {
        "Kxx_V_per_pC_per_m3": float(abs(K[0, 0]) ** 2 * scale),
        "Kxy_V_per_pC_per_m3": float(abs(K[0, 1]) ** 2 * scale),
        "Kyy_V_per_pC_per_m3": float(abs(K[1, 1]) ** 2 * scale),
        "K_quad_strength_V_per_pC_per_m3": float(((K[0, 0] - K[1, 1]) ** 2 + 4.0 * K[0, 1] ** 2) * scale),
        "K_frobenius_V_per_pC_per_m3": float(np.sum(K * K) * scale),
    }


def pickle_save(obj, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def tm_root_v_mn(m: int, n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(jn_zeros(int(m), int(n))[-1])


def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive")
    v = tm_root_v_mn(m, n)
    return float((c / (2*np.pi))*np.sqrt((v/R)**2 + (p*np.pi/L)**2))


def pillbox_radius_from_freq(f_Hz: float) -> float:
    return float(tm_root_v_mn(0, 1) * C0 / (2*np.pi*float(f_Hz)))


def _field_coords(R: float, L: float, shape: tuple[int, int, int]):
    nx, ny, nz = shape
    x = np.linspace(-R, R, nx)
    y = np.linspace(-R, R, ny)
    z = np.linspace(0.0, L, nz)
    return x, y, z


def _E_field_cyl_TM(r, theta, z, m: int, n: int, p: int, R: float, L: float, E0: float = 1.0):
    """Analytical TM_mnp E-field shape in a pillbox.

    Ez = E0 J_m(kc r) cos(m theta) cos(kz z)
    Et = -(kz/kc^2) grad_t(Ez transverse part) sin(kz z)

    This includes the necessary kc factors. The previous p/r form can make
    Etheta artificially enormous because it misses the kc^-2 scaling.
    """
    kc = tm_root_v_mn(m, n) / R
    kz = np.pi * p / L
    x = kc * r
    Jm = special.jv(m, x)
    Jmp = special.jvp(m, x, 1)
    cos_m = np.cos(m*theta)
    sin_m = np.sin(m*theta)
    cos_z = np.cos(kz*z)
    sin_z = np.sin(kz*z)

    Ez = E0 * Jm * cos_m * cos_z
    if p == 0:
        return np.zeros_like(Ez), np.zeros_like(Ez), Ez

    Er = -E0 * (kz/kc) * Jmp * cos_m * sin_z
    with np.errstate(divide="ignore", invalid="ignore"):
        J_over_r = Jm / r
        # Finite on-axis limit for m=1: J1(kc r)/r -> kc/2. For m!=1 the
        # angular term makes the on-axis value zero in this analysis.
        if m == 1:
            J_over_r = np.where(r == 0.0, kc/2.0, J_over_r)
        else:
            J_over_r = np.where(r == 0.0, 0.0, J_over_r)
        Eth = E0 * (kz/(kc**2)) * m * J_over_r * sin_m * sin_z
    return Er, Eth, Ez


def pillbox_field_voxel_grid_xyz(R: float, L: float, m: int, n: int, p: int, x_res: int, y_res: int, z_res: int, *, E0=1.0, mode="TM", dtype=np.float32):
    if str(mode).upper() != "TM":
        raise NotImplementedError("This stripped module intentionally keeps only TM fields")
    x, y, z = _field_coords(R, L, (x_res, y_res, z_res))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = r <= R
    Er, Eth, Ez = _E_field_cyl_TM(r, theta, Z, m, n, p, R, L, E0=E0)
    Ex = Er*np.cos(theta) - Eth*np.sin(theta)
    Ey = Er*np.sin(theta) + Eth*np.cos(theta)
    Eperp = np.sqrt(Ex**2 + Ey**2)
    Eabs = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    def masked(a): return np.where(mask, a, np.nan).astype(dtype, copy=False)
    return {"Ex": masked(Ex), "Ey": masked(Ey), "Ez": masked(Ez), "Eperp": masked(Eperp), "|E|": masked(Eabs), "x_m": x, "y_m": y, "z_m": z}


def find_mode_crossings_from_all_data(all_data: dict, mode_type: str = "TM") -> dict:
    mode_type = mode_type.upper()
    if mode_type != "TM":
        raise NotImplementedError("Stripped quadrupole module only supports TM crossing searches")
    L = np.asarray(all_data["length_factor_vector"], dtype=float)
    modes = list(all_data["TM"].keys())
    modes.sort(key=lambda m: np.interp(1.0, L, np.asarray(all_data["TM"][m]["frequency_Hz"], dtype=float)))
    crossings = {}
    crossed = set()
    for ai, mi in enumerate(modes):
        fi = np.asarray(all_data["TM"][mi]["frequency_Hz"], dtype=float)
        for mj in modes[ai+1:]:
            fj = np.asarray(all_data["TM"][mj]["frequency_Hz"], dtype=float)
            g = fi - fj
            for idx in np.where(g[:-1]*g[1:] <= 0.0)[0]:
                if np.isclose(g[idx], 0.0) and idx > 0 and np.isclose(g[idx-1], 0.0):
                    continue
                if np.isclose(g[idx], 0.0):
                    Lc = L[idx]
                elif np.isclose(g[idx+1], 0.0):
                    Lc = L[idx+1]
                else:
                    Lc = brentq(lambda xx: np.interp(xx, L, fi)-np.interp(xx, L, fj), L[idx], L[idx+1])
                Fc = float(np.interp(Lc, L, fi))
                key = f"TM_{mi}--TM_{mj}@{Lc:.8g}"
                crossings[key] = {"mode_i": f"TM_{mi}", "mode_j": f"TM_{mj}", "length_factor": float(Lc), "frequency_Hz": Fc}
                crossed.update([f"TM_{mi}", f"TM_{mj}"])
    return {"TM": {"crossings": crossings, "modes_that_cross": sorted(crossed)}}


def eabs_from_components(Ex, Ey, Ez):
    return np.sqrt(np.nan_to_num(Ex, nan=0.0)**2 + np.nan_to_num(Ey, nan=0.0)**2 + np.nan_to_num(Ez, nan=0.0)**2)


def rotation_angle_to_vertical_plane(Eabs, x, y):
    ix, iy, iz = np.unravel_index(np.nanargmax(Eabs), Eabs.shape)
    phi = np.arctan2(y[iy], x[ix])
    # after active rotation by angle a, x' = r cos(phi+a). Want x'=0.
    candidates = np.array([np.pi/2 - phi, -np.pi/2 - phi])
    candidates = (candidates + np.pi) % (2*np.pi) - np.pi
    a = candidates[np.argmin(np.abs(candidates))]
    return float(np.degrees(a)), (int(ix), int(iy), int(iz))


def rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle_deg: float, fill_value=0.0):
    Ex = np.nan_to_num(np.asarray(Ex, float), nan=fill_value)
    Ey = np.nan_to_num(np.asarray(Ey, float), nan=fill_value)
    Ez = np.nan_to_num(np.asarray(Ez, float), nan=fill_value)
    interp = [RegularGridInterpolator((x, y, z), A, bounds_error=False, fill_value=fill_value) for A in (Ex, Ey, Ez)]
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    a = np.radians(angle_deg)
    # inverse map: source coordinates that land on output grid after active rotation
    Xs = Xg*np.cos(a) + Yg*np.sin(a)
    Ys = -Xg*np.sin(a) + Yg*np.cos(a)
    xy = np.column_stack([Xs.ravel(), Ys.ravel()])
    out = []
    for itp in interp:
        A = np.empty_like(Ex)
        for k, z0 in enumerate(z):
            pts = np.column_stack([xy, np.full(xy.shape[0], z0)])
            A[:, :, k] = itp(pts).reshape(len(x), len(y))
        out.append(A)
    Ex_s, Ey_s, Ez_rot = out
    # rotate vector components by same active rotation
    Ex_rot = Ex_s*np.cos(a) - Ey_s*np.sin(a)
    Ey_rot = Ex_s*np.sin(a) + Ey_s*np.cos(a)
    return {"x": x, "y": y, "z": z, "Ex": Ex_rot, "Ey": Ey_rot, "Ez": Ez_rot, "Eabs": eabs_from_components(Ex_rot, Ey_rot, Ez_rot), "angle_deg": float(angle_deg)}


def cylindrical_theta_r_map(Eabs, x, y, z_index=None, n_r=120, n_theta=361):
    if z_index is None:
        z_index = Eabs.shape[2]//2
    rmax = min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))
    r = np.linspace(0.0, rmax, n_r)
    theta_deg = np.linspace(-180.0, 180.0, n_theta)
    th = np.radians(theta_deg)
    Rg, Tg = np.meshgrid(r, th, indexing="ij")
    X = Rg*np.cos(Tg); Y = Rg*np.sin(Tg)
    z = np.arange(Eabs.shape[2], dtype=float)
    interp = RegularGridInterpolator((x, y, z), np.nan_to_num(Eabs, nan=0.0), bounds_error=False, fill_value=0.0)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_index))])
    vals = interp(pts).reshape(n_r, n_theta)
    # roll so theta=0 is central in x-axis
    return theta_deg, r, vals


def plot_theta_r_before_after(before_Eabs, after_Eabs, x, y, out_png, title=""):
    mid = before_Eabs.shape[2]//2
    th, r, B = cylindrical_theta_r_map(before_Eabs, x, y, mid)
    _, _, A = cylindrical_theta_r_map(after_Eabs, x, y, mid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    vmax = max(np.nanmax(B), np.nanmax(A))
    for ax, data, name in [(axes[0], B, "before rotation"), (axes[1], A, "after rotation")]:
        im = ax.imshow(data, origin="lower", aspect="auto", extent=[th[0], th[-1], r[0], r[-1]], vmin=0, vmax=vmax)
        ax.axvline(0.0, color="w", lw=1, alpha=0.8)
        ax.set_xlabel(r"$\theta$ [deg]")
        ax.set_ylabel("r [m]")
        ax.set_title(name)
    fig.colorbar(im, ax=axes, label=r"$|E|$")
    fig.suptitle(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def align_field_to_vertical_plane_peak(field: dict, out_plot: str | None = None, label="") -> dict:
    """Rotate a field so at least one global |E| maximum lies in |E|[mid,:,:].

    Quadrupole modes often have several equal |E| maxima.  We therefore test
    both mathematically valid rotations that put the selected maximum onto the
    vertical plane and choose the one whose global argmax is closest to x=mid.
    The final verification also checks that the vertical mid-plane contains a
    value equal to the global maximum within numerical tolerance.
    """
    Ex, Ey, Ez = field["Ex"], field["Ey"], field["Ez"]
    x = field.get("x_m", np.linspace(-1, 1, Ex.shape[0]))
    y = field.get("y_m", np.linspace(-1, 1, Ex.shape[1]))
    z = field.get("z_m", np.linspace(0, 1, Ex.shape[2]))
    Eabs0 = eabs_from_components(Ex, Ey, Ez)
    ix, iy, iz = np.unravel_index(np.nanargmax(Eabs0), Eabs0.shape)
    peak0 = (int(ix), int(iy), int(iz))
    phi = np.arctan2(y[iy], x[ix])
    candidates = np.array([np.pi/2 - phi, -np.pi/2 - phi])
    candidates = (candidates + np.pi) % (2*np.pi) - np.pi
    mid = len(x)//2

    best = None
    for a in candidates:
        angle = float(np.degrees(a))
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle)
        peak = tuple(int(v) for v in np.unravel_index(np.nanargmax(trial["Eabs"]), trial["Eabs"].shape))
        global_max = float(np.nanmax(trial["Eabs"]))
        midplane_max = float(np.nanmax(trial["Eabs"][mid, :, :]))
        score = (abs(peak[0] - mid), abs(global_max - midplane_max) / (global_max if global_max else 1.0))
        if best is None or score < best[0]:
            best = (score, angle, trial, peak, global_max, midplane_max)

    score, angle, rot, peak1, global_max, midplane_max = best
    if abs(peak1[0] - mid) > 1 and not np.isclose(midplane_max, global_max, rtol=2e-3, atol=1e-12):
        raise RuntimeError(
            f"{label}: rotation failed; peak_after={peak1}, expected x index {mid}, "
            f"angle={angle}, global_max={global_max:.6e}, midplane_max={midplane_max:.6e}"
        )
    if out_plot:
        plot_theta_r_before_after(Eabs0, rot["Eabs"], x, y, out_plot, title=f"{label}: angle={angle:.3f} deg")
    rot.update({
        "rotation_angle_deg": angle,
        "peak_before": peak0,
        "peak_after": peak1,
        "global_max_after": global_max,
        "vertical_midplane_max_after": midplane_max,
    })
    return rot

def align_field_to_vertical_plane(
    field: dict,
    out_plot: str | None = None,
    label: str = "",
    *,
    fit_pixels: int = 12,
    z_index: int | None = None,
    target: str = "principal_axes_xy",
) -> dict:
    """
    Rotate a quadrupole-like field using the principal-axis angle from Ez.

    Drop-in replacement for the old peak-based align_field_to_vertical_plane().
    It returns the same style of dict as rotate_vector_field_about_z(), plus
    metadata keys used by the current workflow.

    Parameters
    ----------
    field
        Dict containing Ex, Ey, Ez and optionally x_m, y_m, z_m.
    out_plot
        Optional diagnostic theta-r plot path.
    label
        Label for diagnostics.
    fit_pixels
        Radius in pixels for near-axis Ez quadratic fit.
    z_index
        Longitudinal slice used for determining quadrupole angle.
        If None, uses the slice where near-axis |Ez| is largest.
    target
        "principal_axes_xy":
            rotate so fitted Kxy is minimised; quadrupole axes align with x/y.
        "diagonal_lobes_vertical":
            rotate an Ez quadrupole with diagonal lobes onto the visual
            vertical/horizontal convention commonly expected in iris plots.
            This is usually the better option for matching your screenshot.
    """
    Ex, Ey, Ez = field["Ex"], field["Ey"], field["Ez"]

    x = field.get("x_m", np.linspace(-1.0, 1.0, Ex.shape[0]))
    y = field.get("y_m", np.linspace(-1.0, 1.0, Ex.shape[1]))
    z = field.get("z_m", np.linspace(0.0, 1.0, Ex.shape[2]))

    Eabs0 = eabs_from_components(Ex, Ey, Ez)
    peak0 = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs0), Eabs0.shape))

    nx, ny, nz = Ez.shape
    ix0, iy0 = nx // 2, ny // 2

    if z_index is None:
        # Choose the z slice with the largest near-axis Ez signal.
        r = min(int(fit_pixels), ix0 - 1, iy0 - 1)
        near = Ez[ix0 - r: ix0 + r + 1, iy0 - r: iy0 + r + 1, :]
        z_index = int(np.nanargmax(np.nanmax(np.abs(near), axis=(0, 1))))

    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0

    max_px = min(int(fit_pixels), ix0 - 1, iy0 - 1)

    pts = []
    vals = []

    for i in range(ix0 - max_px, ix0 + max_px + 1):
        for j in range(iy0 - max_px, iy0 + max_px + 1):
            xx = (i - ix0) * dx
            yy = (j - iy0) * dy

            if np.hypot(xx, yy) <= max_px * min(abs(dx), abs(dy)):
                pts.append((xx, yy))
                vals.append(Ez[i, j, z_index])

    pts = np.asarray(pts, dtype=float)
    vals = np.asarray(vals, dtype=float)

    X = pts[:, 0]
    Y = pts[:, 1]

    A = np.column_stack([
        np.ones_like(X),
        X,
        Y,
        X * X,
        X * Y,
        Y * Y,
    ])

    coeff, *_ = np.linalg.lstsq(A, vals, rcond=None)
    _, _, _, axx, axy, ayy = coeff

    H = np.array([
        [2.0 * axx, axy],
        [axy, 2.0 * ayy],
    ], dtype=float)

    # Principal-axis angle of the fitted quadrupole tensor.
    theta_rad = 0.5 * np.arctan2(2.0 * H[0, 1], H[0, 0] - H[1, 1])
    theta_deg = float(np.degrees(theta_rad))

    candidate_angles = []

    if target == "principal_axes_xy":
        # Put principal axes onto x/y.
        base = -theta_deg
        candidate_angles = [base, base + 90.0, base - 90.0, base + 180.0]

    elif target == "diagonal_lobes_vertical":
        # For an Ez quadrupole pattern, the visible high-field lobes may lie
        # 45 degrees from the principal axes. Test both 45-degree conventions.
        base1 = 45.0 - theta_deg
        base2 = -45.0 - theta_deg
        candidate_angles = [
            base1, base1 + 90.0, base1 - 90.0,
            base2, base2 + 90.0, base2 - 90.0,
        ]

    else:
        raise ValueError(
            "target must be 'principal_axes_xy' or 'diagonal_lobes_vertical'"
        )

    def wrap_angle_deg(a: float) -> float:
        return float((a + 180.0) % 360.0 - 180.0)

    candidate_angles = [wrap_angle_deg(a) for a in candidate_angles]

    def fitted_H_after_rotation(angle_deg: float) -> np.ndarray:
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle_deg)
        Ezr = trial["Ez"]

        vals_r = []
        for xx, yy in pts:
            i = int(round(ix0 + xx / dx))
            j = int(round(iy0 + yy / dy))
            vals_r.append(Ezr[i, j, z_index])

        vals_r = np.asarray(vals_r, dtype=float)
        coeff_r, *_ = np.linalg.lstsq(A, vals_r, rcond=None)
        _, _, _, axx_r, axy_r, ayy_r = coeff_r

        return np.array([
            [2.0 * axx_r, axy_r],
            [axy_r, 2.0 * ayy_r],
        ], dtype=float)

    best = None

    for angle in candidate_angles:
        trial = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle)
        Eabs = trial["Eabs"]

        peak = tuple(int(v) for v in np.unravel_index(np.nanargmax(Eabs), Eabs.shape))
        global_max = float(np.nanmax(Eabs))
        midplane_max = float(np.nanmax(Eabs[ix0, :, :]))

        Hr = fitted_H_after_rotation(angle)

        Hxx = Hr[0, 0]
        Hxy = Hr[0, 1]
        Hyy = Hr[1, 1]

        diag_scale = max(abs(Hxx), abs(Hyy), 1e-300)
        cross_ratio = abs(Hxy) / diag_scale

        # Prefer:
        #   1. small cross term after rotation,
        #   2. opposite signs of Hxx/Hyy,
        #   3. strong mid-plane maximum,
        #   4. global peak close to vertical mid-plane.
        opposite_sign_penalty = 0.0 if Hxx * Hyy < 0.0 else 1.0
        midplane_error = abs(global_max - midplane_max) / (global_max if global_max else 1.0)
        peak_x_error = abs(peak[0] - ix0)

        score = (
            cross_ratio,
            opposite_sign_penalty,
            midplane_error,
            peak_x_error,
        )

        if best is None or score < best[0]:
            best = (score, angle, trial, peak, global_max, midplane_max, Hr)

    score, angle, rot, peak1, global_max, midplane_max, H_after = best

    if out_plot:
        plot_theta_r_before_after(
            Eabs0,
            rot["Eabs"],
            x,
            y,
            out_plot,
            title=f"{label}: quadrupole-axis angle={angle:.3f} deg",
        )

    rot.update({
        "rotation_angle_deg": angle,
        "rotation_method": "quadrupole_Ez_principal_axis",
        "rotation_target": target,
        "quadrupole_angle_before_deg": theta_deg,
        "quadrupole_fit_z_index": int(z_index),
        "quadrupole_fit_pixels": int(max_px),
        "quadrupole_H_before": H,
        "quadrupole_H_after": H_after,
        "quadrupole_cross_ratio_after": float(
            abs(H_after[0, 1]) / max(abs(H_after[0, 0]), abs(H_after[1, 1]), 1e-300)
        ),
        "peak_before": peak0,
        "peak_after": peak1,
        "global_max_after": global_max,
        "vertical_midplane_max_after": midplane_max,
    })

    return rot


def combine_fields(E1: dict, E2: dict) -> dict:
    out = {}
    for prefix, F in [("E1", E1), ("E2", E2)]:
        for c in ("Ex", "Ey", "Ez"):
            out[f"{prefix}_{c}"] = np.asarray(F[c])
        out[f"abs_{prefix}"] = eabs_from_components(F["Ex"], F["Ey"], F["Ez"])
        out[f"trans_{prefix}"] = np.sqrt(F["Ex"]**2 + F["Ey"]**2)
    for op, sign in [("plus", 1.0), ("minus", -1.0)]:
        Ex = E1["Ex"] + sign*E2["Ex"]; Ey = E1["Ey"] + sign*E2["Ey"]; Ez = E1["Ez"] + sign*E2["Ez"]
        out[f"Ex_{op}"] = Ex; out[f"Ey_{op}"] = Ey; out[f"Ez_{op}"] = Ez
        out[f"abs_{op}"] = eabs_from_components(Ex, Ey, Ez)
        out[f"trans_{op}"] = np.sqrt(Ex**2 + Ey**2)
    return out


def save_field_data_npz(field_data: dict, filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, **field_data)


def extract_slices(field_data: dict) -> dict:
    """Extract the four 2D slices used by the compact summary PDFs.

    Generated keys include, for every 3D field F in field_data:
        *_iris_1             F[:, :, 0]
        *_iris_2             F[:, :, -1]
        *_transverse_mid     F[:, :, mid_z]
        *_longitudinal_mid   F[mid_x, :, :]
    """
    slices = {}
    for k, F in field_data.items():
        if isinstance(F, np.ndarray) and F.ndim == 3:
            midx, midz = F.shape[0] // 2, F.shape[2] // 2
            slices[f"{k}_iris_1"] = F[:, :, 0]
            slices[f"{k}_iris_2"] = F[:, :, -1]
            slices[f"{k}_transverse_mid"] = F[:, :, midz]
            slices[f"{k}_longitudinal_mid"] = F[midx, :, :]
    return slices


def plot_field_slices(field_data: dict, out_dir: str, title: str = ""):
    """Save monopole-style 4x3 field-slice plots for plus and minus fields.

    Produces eight files per crossing:
      plus_iris_1.png, plus_iris_2.png, plus_transverse_mid.png,
      plus_longitudinal_mid.png, and the equivalent four minus_*.png files.

    Array convention: field[x, y, z].
    Plot convention:
      - iris/transverse: x horizontal, y vertical, using F[:, :, z_index].T
      - longitudinal: z horizontal, y vertical, using F[x_mid, :, :]
      - Ex/Ey/Ez rows: RdBu_r, symmetric around zero
      - |E| row: viridis, 0 to row max
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_specs = {
        "iris_1": lambda F: np.asarray(F)[:, :, 0].T,
        "iris_2": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] - 1].T,
        "transverse_mid": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] // 2].T,
        "longitudinal_mid": lambda F: np.asarray(F)[np.asarray(F).shape[0] // 2, :, :],
    }

    op_rows = {
        "plus": [
            ("E1_Ex", "E2_Ex", "Ex_plus"),
            ("E1_Ey", "E2_Ey", "Ey_plus"),
            ("E1_Ez", "E2_Ez", "Ez_plus"),
            ("abs_E1", "abs_E2", "abs_plus"),
        ],
        "minus": [
            ("E1_Ex", "E2_Ex", "Ex_minus"),
            ("E1_Ey", "E2_Ey", "Ey_minus"),
            ("E1_Ez", "E2_Ez", "Ez_minus"),
            ("abs_E1", "abs_E2", "abs_minus"),
        ],
    }

    for op, rows in op_rows.items():
        for stype, slicer in slice_specs.items():
            fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
            fig.suptitle(f"{title} : {op} : {stype}")

            for r, row_keys in enumerate(rows):
                row_data = [slicer(field_data[k]) for k in row_keys]

                if r == 3:  # |E| row
                    vmax = max(float(np.nanmax(arr)) for arr in row_data)
                    vmin = 0.0
                    cmap = "viridis"
                else:
                    vmax = max(float(np.nanmax(np.abs(arr))) for arr in row_data)
                    vmin = -vmax
                    cmap = "RdBu_r"

                if not np.isfinite(vmax) or vmax == 0.0:
                    vmax = 1.0
                    if r != 3:
                        vmin = -1.0

                for c, (key, arr) in enumerate(zip(row_keys, row_data)):
                    ax = axes[r, c]
                    im = ax.imshow(
                        arr,
                        origin="lower",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                    )
                    ax.set_title(key)

                    if stype in ("iris_1", "iris_2", "transverse_mid"):
                        ax.set_xlabel("x pixel")
                        ax.set_ylabel("y pixel")
                    else:
                        ax.set_xlabel("z pixel")
                        ax.set_ylabel("y pixel")

                    ax.text(
                        0.02,
                        0.98,
                        f"max={np.nanmax(np.abs(arr)):.2e}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
                    )

                fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)

            fig.savefig(out_dir / f"{op}_{stype}.png", dpi=300)
            plt.close(fig)



def _slice_2d_for_plot(F: np.ndarray, slice_type: str) -> np.ndarray:
    """Return a 2D slice with plotting orientation already applied."""
    F = np.asarray(F)
    if F.ndim != 3:
        raise ValueError("Expected a 3D field array")
    if slice_type == "iris_1":
        return F[:, :, 0].T
    if slice_type == "iris_2":
        return F[:, :, F.shape[2] - 1].T
    if slice_type == "transverse_mid":
        return F[:, :, F.shape[2] // 2].T
    if slice_type == "longitudinal_mid":
        return F[F.shape[0] // 2, :, :]
    raise ValueError(f"Unknown slice_type: {slice_type}")


def plot_combined_field_slices_4x4(
    field_data: dict,
    out_dir: str | Path,
    title: str = "",
) -> None:
    """Save 4x4 combined plots with columns [E1, E2, E+, E-].

    Saved files:
        iris_1.png
        iris_2.png
        transverse_mid.png
        longitudinal_mid.png

    Rows:
        Ex, Ey, Ez, |E|

    Colour meaning:
        - Ex, Ey and Ez rows are divided by max(abs(E1_Ez), abs(E2_Ez)).
          Therefore colourbar value 1 means the parent Ez reference amplitude.

        - |E| row is divided by max(|E1|, |E2|).
          Therefore colourbar value 1 means the parent |E| reference amplitude.

        - Colourbar limits extend beyond +/-1 or 1 if E+ or E- exceed the
          parent reference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_specs = {
        "iris_1": lambda F: np.asarray(F)[:, :, 0].T,
        "iris_2": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] - 1].T,
        "transverse_mid": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] // 2].T,
        "longitudinal_mid": lambda F: np.asarray(F)[np.asarray(F).shape[0] // 2, :, :],
    }

    rows = [
        ("E1_Ex", "E2_Ex", "Ex_plus", "Ex_minus"),
        ("E1_Ey", "E2_Ey", "Ey_plus", "Ey_minus"),
        ("E1_Ez", "E2_Ez", "Ez_plus", "Ez_minus"),
        ("abs_E1", "abs_E2", "abs_plus", "abs_minus"),
    ]

    column_titles = ["E₁", "E₂", "E₊", "E₋"]
    row_titles = [
        "Eₓ / Ez ref",
        "Eᵧ / Ez ref",
        "Ez / Ez ref",
        "|E| / |E| ref",
    ]

    def _real_image(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        return np.real(a) if np.iscomplexobj(a) else a

    def _safe_ref(arrays: list[np.ndarray]) -> float:
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        if not np.isfinite(ref) or ref <= 0.0:
            ref = 1.0
        return ref

    def _safe_vmax(arrays: list[np.ndarray], ref: float) -> float:
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        if not np.isfinite(scaled_max) or scaled_max <= 0.0:
            scaled_max = 1.0
        return max(1.0, scaled_max)

    for stype, slicer in slice_specs.items():
        fig, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"{title} : plus/minus comparison : {stype}")

        parent_ez_ref = _safe_ref([
            slicer(_real_image(field_data["E1_Ez"])),
            slicer(_real_image(field_data["E2_Ez"])),
        ])

        parent_abs_ref = _safe_ref([
            slicer(_real_image(field_data["abs_E1"])),
            slicer(_real_image(field_data["abs_E2"])),
        ])

        for r, row_keys in enumerate(rows):
            raw_row_data = [slicer(_real_image(field_data[k])) for k in row_keys]

            is_abs_row = r == 3

            if is_abs_row:
                ref = parent_abs_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmin = 0.0
                vmax = _safe_vmax(raw_row_data, ref)
                cmap = "viridis"
            else:
                ref = parent_ez_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmax = _safe_vmax(raw_row_data, ref)
                vmin = -vmax
                cmap = "RdBu_r"

            for c, (key, arr_raw, arr_scaled) in enumerate(
                zip(row_keys, raw_row_data, row_data)
            ):
                ax = axes[r, c]
                im = ax.imshow(
                    arr_scaled,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )

                if r == 0:
                    ax.text(
                        0.5,
                        1.02,
                        column_titles[c],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontstyle="normal",
                        fontweight="bold",
                        zorder=100,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.90,
                            pad=2.0,
                        ),
                        clip_on=False,
                    )

                if c == 0:
                    ax.text(
                        -0.12,
                        0.5,
                        row_titles[r],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=11,
                        zorder=100,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.90,
                            pad=2.0,
                        ),
                        clip_on=False,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

                ax.text(
                    0.02,
                    0.98,
                    f"max={np.nanmax(np.abs(arr_raw)):.2e}\n"
                    f"norm={np.nanmax(np.abs(arr_scaled)):.2g}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.65,
                        edgecolor="none",
                    ),
                )

            fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)

        fig.savefig(out_dir / f"{stype}.png", dpi=300)
        plt.close(fig)


def _save_one_2x4_slice_on_gridspec(
    fig: plt.Figure,
    grid_slot,
    slice_dict: dict[str, np.ndarray],
    stype: str,
    block_title: str,
) -> None:
    """Draw one 2x4 [Ez, |E|] x [E1, E2, E-, E+] slice block."""
    rows = [
        ("E1_Ez", "E2_Ez", "Ez_minus", "Ez_plus"),
        ("abs_E1", "abs_E2", "abs_minus", "abs_plus"),
    ]
    column_titles = [r"$E_1$", r"$E_2$", r"$E_-$", r"$E_+$"]
    row_titles = [
        r"$E_z/E_{z,\mathrm{ref}}$",
        r"$|E|/|E|_{\mathrm{ref}}$",
    ]

    def _real_image(arr):
        return np.real(np.asarray(arr))

    def _plot_orient(arr, slice_type):
        return arr.T if slice_type.startswith("iris") or slice_type == "transverse_mid" else arr

    def _safe_ref(arrays):
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        return ref if np.isfinite(ref) and ref > 0.0 else 1.0

    def _safe_vmax(arrays, ref):
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        return max(1.0, scaled_max) if np.isfinite(scaled_max) and scaled_max > 0.0 else 1.0

    gs = grid_slot.subgridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 0.045],
        height_ratios=[1, 1],
        wspace=0.0,
        hspace=0.04,
    )

    parent_ez_ref = _safe_ref([
        _real_image(slice_dict[f"E1_Ez_{stype}"]),
        _real_image(slice_dict[f"E2_Ez_{stype}"]),
    ])
    parent_abs_ref = _safe_ref([
        _real_image(slice_dict[f"abs_E1_{stype}"]),
        _real_image(slice_dict[f"abs_E2_{stype}"]),
    ])

    first_ax = None

    for r, row_keys in enumerate(rows):
        raw_row_data = [_real_image(slice_dict[f"{k}_{stype}"]) for k in row_keys]

        if r == 0:
            ref = parent_ez_ref
            row_data = [arr / ref for arr in raw_row_data]
            vmax = _safe_vmax(raw_row_data, ref)
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            ref = parent_abs_ref
            row_data = [arr / ref for arr in raw_row_data]
            vmin = 0.0
            vmax = _safe_vmax(raw_row_data, ref)
            cmap = "viridis"

        im = None

        for c, (arr_raw, arr_scaled) in enumerate(zip(raw_row_data, row_data)):
            ax = fig.add_subplot(gs[r, c])
            if first_ax is None:
                first_ax = ax

            ax.set_anchor("C")
            plot_arr = _plot_orient(arr_scaled, stype)

            im = ax.imshow(
                plot_arr,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ny, nx = plot_arr.shape
            ax.set_box_aspect(ny / nx)
            ax.margins(0)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(column_titles[c], fontsize=10, fontweight="bold", pad=2)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9, rotation=90, labelpad=7)

            ax.text(
                0.04,
                0.96,
                f"{np.nanmax(np.abs(arr_scaled)):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
            )

        cax = fig.add_subplot(gs[r, 4])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7, length=2, pad=1)

    if first_ax is not None:
        first_ax.text(
            0.0,
            1.04,
            block_title,
            transform=first_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.2),
        )


def save_four_slice_pdfs_and_merge(
    slice_dict: dict[str, np.ndarray],
    out_dir: str | Path,
    merged_pdf_name: str = "combined_four_slice_summary.pdf",
) -> Path:
    """Save one tall single-page PDF containing all four 2x4 slice summaries.

    The name is retained for compatibility with the homotypic scripts, but this
    implementation no longer creates four separate pages and then merges them.
    It writes a single tall PDF directly, suitable for one \includegraphics call
    in the PRAB appendix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("iris_1", "Transverse iris 1"),
        ("iris_2", "Transverse iris 2"),
        ("longitudinal_mid", "Longitudinal vertical mid-plane"),
        ("transverse_mid", "Transverse mid-plane"),
    ]

    fig = plt.figure(figsize=(7.2, 12.0), constrained_layout=False)
    outer = fig.add_gridspec(
        4,
        1,
        left=0.075,
        right=0.965,
        bottom=0.025,
        top=0.975,
        hspace=0.20,
    )

    for block_idx, (stype, block_title) in enumerate(specs):
        _save_one_2x4_slice_on_gridspec(
            fig=fig,
            grid_slot=outer[block_idx, 0],
            slice_dict=slice_dict,
            stype=stype,
            block_title=block_title,
        )

    out_file = out_dir / merged_pdf_name
    fig.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote single-page summary PDF {out_file}")
    return out_file

def accelerating_voltage_complex(Ez_line, z_m, omega, beta=1.0):
    zc = np.asarray(z_m, float) - 0.5*(z_m[0] + z_m[-1])
    Ez_line = np.asarray(Ez_line, float)
    return np.trapezoid(Ez_line * np.exp(1j*omega*zc/(beta*C0)), zc)


def kick_from_Ez_field(Ez, f_010, f_mnp, l_factor, Req_m, *, axis="y", fit_pixels=8, beta=1.0):
    Ez = np.nan_to_num(np.asarray(Ez, float), nan=0.0)
    nx, ny, nz = Ez.shape
    ix0, iy0 = nx//2, ny//2
    L = (C0/f_010)/2.0 * float(l_factor)
    z_m = np.linspace(0.0, L, nz)
    dx = 2.0*Req_m/(nx-1); dy = 2.0*Req_m/(ny-1)
    omega = 2*np.pi*f_mnp
    vals = []
    max_pix = min(fit_pixels, ix0-1 if axis == "x" else iy0-1)
    for dp in range(-max_pix, max_pix+1):
        if dp == 0: continue
        if axis == "y":
            r = dp*dy; line = Ez[ix0, iy0+dp, :]
        else:
            r = dp*dx; line = Ez[ix0+dp, iy0, :]
        V = accelerating_voltage_complex(line, z_m, omega, beta=beta)
        vals.append((r, V))
    r = np.array([v[0] for v in vals], float)
    Vc = np.array([v[1] for v in vals], complex)
    # Fit real and imaginary separately; gradient magnitude is phase-invariant.
    gr = np.polyfit(r, Vc.real, 1)[0]
    gi = np.polyfit(r, Vc.imag, 1)[0]
    dVdr = gr + 1j*gi
    Vperp_per_m_offset = (C0/omega) * dVdr  # V/m_offset
    # Requested units: V/C/m/m == V/C/m^2. This is the PW transverse voltage
    # gradient per unit charge for a 1 C normalisation of the field map.
    kick_V_per_C_per_m2 = abs(Vperp_per_m_offset)
    return {"r_m": r, "Vz_complex_V": Vc, "dVz_dr_V_per_m": dVdr, "Vperp_per_m_offset_V_per_m": Vperp_per_m_offset, "kick_V_per_C_per_m_per_m": kick_V_per_C_per_m2, "transverse_pixel_m": dx if axis == "x" else dy, "longitudinal_pixel_m": L/(nz-1), "axis": axis}


# -----------------------------------------------------------------------------
# Quadrupole focusing / defocusing analysis
# -----------------------------------------------------------------------------

def _phase_align_complex_matrix(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Return M * exp(-i*phase) using the largest element as phase reference."""
    M = np.asarray(M, complex)
    idx = np.unravel_index(int(np.nanargmax(np.abs(M))), M.shape)
    ref = M[idx]
    phase = float(np.angle(ref)) if np.abs(ref) > 0 else 0.0
    return M * np.exp(-1j * phase), phase


def quadrupole_focusing_from_Ez_field(
    Ez,
    f_010: float,
    f_mnp: float,
    l_factor: float,
    Req_m: float,
    *,
    fit_pixels: int = 8,
    beta: float = 1.0,
) -> dict:
    """Estimate quadrupole focusing/defocusing from the complex Vz(x,y) map.

    Method
    ------
    1. Integrate each near-axis Ez(x,y,z) line to complex longitudinal voltage Vz.
    2. Fit the near-axis voltage map to

           Vz = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2

    3. Use Panofsky-Wenzel in the same convention as the dipole script:

           V_perp = (c / omega) grad_perp(Vz)

       so the local transverse-voltage gradient matrix is

           K = (c / omega) Hessian(Vz)

       Units are V/C/m/m for a 1 C field-map normalisation.

    Sign convention
    ---------------
    The returned phased matrix is phase-aligned to its largest element. Positive
    Kxx means a positive test charge at +x receives +x transverse voltage. An
    electron sees the opposite force. Therefore the returned electron labels
    reverse the voltage-gradient labels.
    """
    Ez = np.nan_to_num(np.asarray(Ez, float), nan=0.0)
    nx, ny, nz = Ez.shape
    ix0, iy0 = nx // 2, ny // 2
    L = (C0 / float(f_010)) / 2.0 * float(l_factor)
    z_m = np.linspace(0.0, L, nz)
    dx = 2.0 * float(Req_m) / (nx - 1)
    dy = 2.0 * float(Req_m) / (ny - 1)
    omega = 2.0 * np.pi * float(f_mnp)

    max_px = min(int(fit_pixels), ix0 - 1, iy0 - 1)
    points = []
    values = []
    for i in range(ix0 - max_px, ix0 + max_px + 1):
        for j in range(iy0 - max_px, iy0 + max_px + 1):
            x = (i - ix0) * dx
            y = (j - iy0) * dy
            # Use a circular fitting aperture to reduce square-corner bias.
            if np.hypot(x, y) <= max_px * min(dx, dy):
                V = accelerating_voltage_complex(Ez[i, j, :], z_m, omega, beta=beta)
                points.append((x, y))
                values.append(V)

    pts = np.asarray(points, float)
    Vc = np.asarray(values, complex)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([np.ones_like(x), x, y, x*x, x*y, y*y])

    coeff_real, *_ = np.linalg.lstsq(A, Vc.real, rcond=None)
    coeff_imag, *_ = np.linalg.lstsq(A, Vc.imag, rcond=None)
    coeff = coeff_real + 1j * coeff_imag
    a0, ax, ay, axx, axy, ayy = coeff

    H = np.array([[2.0 * axx, axy], [axy, 2.0 * ayy]], dtype=complex)  # d2Vz/dx_i dx_j, V/m^2
    K = (C0 / omega) * H  # transverse-voltage gradient, V/m^2 == V/C/m/m for 1 C map norm
    K_phase, phase_rad = _phase_align_complex_matrix(K)
    K_real = K_phase.real
    evals, evecs = np.linalg.eigh(K_real)

    # Axis labels for voltage-gradient convention. Electron labels are reversed.
    Kxx = float(K_real[0, 0])
    Kyy = float(K_real[1, 1])
    voltage_x = "defocusing" if Kxx > 0 else "focusing" if Kxx < 0 else "neutral"
    voltage_y = "defocusing" if Kyy > 0 else "focusing" if Kyy < 0 else "neutral"
    electron_x = "focusing" if Kxx > 0 else "defocusing" if Kxx < 0 else "neutral"
    electron_y = "focusing" if Kyy > 0 else "defocusing" if Kyy < 0 else "neutral"

    return {
        "fit_points_xy_m": pts,
        "Vz_complex_V": Vc,
        "poly_coefficients_complex": {
            "a0": a0, "ax": ax, "ay": ay, "axx": axx, "axy": axy, "ayy": ayy,
        },
        "hessian_V_per_m2_complex": H,
        "gradient_matrix_V_per_C_per_m_per_m_complex": K,
        "phase_reference_rad": phase_rad,
        "gradient_matrix_phase_aligned_real_V_per_C_per_m_per_m": K_real,
        "eigenvalues_phase_aligned_V_per_C_per_m_per_m": evals,
        "eigenvectors_columns_xy": evecs,
        "Kxx_V_per_C_per_m_per_m": Kxx,
        "Kxy_V_per_C_per_m_per_m": float(K_real[0, 1]),
        "Kyy_V_per_C_per_m_per_m": Kyy,
        "trace_V_per_C_per_m_per_m": float(np.trace(K_real)),
        "determinant": float(np.linalg.det(K_real)),
        "voltage_gradient_classification": {"x": voltage_x, "y": voltage_y},
        "electron_force_classification": {"x": electron_x, "y": electron_y},
        "transverse_pixel_m": dx,
        "longitudinal_pixel_m": L / (nz - 1),
        "fit_pixels": max_px,
        "units": "V/C/m/m",
    }


def plot_quadrupole_voltage_fit(focus_result: dict, out_png: str, title: str = ""):
    """Plot fitted near-axis complex Vz samples and fitted real quadratic map."""
    pts = focus_result["fit_points_xy_m"]
    Vc = focus_result["Vz_complex_V"]
    coeff = focus_result["poly_coefficients_complex"]
    phase = focus_result["phase_reference_rad"]
    Vp = Vc * np.exp(-1j * phase)
    x = pts[:, 0]
    y = pts[:, 1]
    n = 121
    xr = np.linspace(np.min(x), np.max(x), n)
    yr = np.linspace(np.min(y), np.max(y), n)
    X, Y = np.meshgrid(xr, yr, indexing="xy")
    c = {k: v * np.exp(-1j * phase) for k, v in coeff.items()}
    Z = (c["a0"] + c["ax"]*X + c["ay"]*Y + c["axx"]*X**2 + c["axy"]*X*Y + c["ayy"]*Y**2).real
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sc = axes[0].scatter(x, y, c=Vp.real, s=18)
    axes[0].set_title("phase-aligned Vz samples")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[0], label="Re(Vz) [V]")
    vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z))) or 1.0
    im = axes[1].imshow(Z, origin="lower", extent=[xr[0], xr[-1], yr[0], yr[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title("quadratic fit")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes[1], label="Re(fit Vz) [V]")
    fig.suptitle(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Heterotypic Method 2 scalar metric analysis
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricParams:
    """Parameters for local quadratic Method 2 metric extraction.

    The transverse grid is interpreted with the beam/cylinder axis at
    array[axis_i, axis_j, :] and the physical cylinder radius at radius_pixels
    pixels from that axis.  For the default 151 x 151 maps this gives
    axis_i = axis_j = 75 and radius_pixels = 75.
    """
    frequency_Hz: float
    length_m: float
    Req_m: float
    beta: float = 1.0
    axis_i: float = 75.0
    axis_j: float = 75.0
    radius_pixels: float = 75.0
    local_fit_pixels: int = 3
    centred_z: bool = False  # Monopole/diagnostic-aligned convention: z in [0,L].
    U_J: float | None = None

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * float(self.frequency_Hz)

    @property
    def dx_m(self) -> float:
        return float(self.Req_m) / float(self.radius_pixels)

    @property
    def dy_m(self) -> float:
        return self.dx_m


def _metric_xy_arrays(shape: tuple[int, int, int], params: MetricParams) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, _ = shape
    x = (np.arange(nx, dtype=float) - float(params.axis_i)) * params.dx_m
    y = (np.arange(ny, dtype=float) - float(params.axis_j)) * params.dy_m
    return x, y


def stored_energy_from_E_components(Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray, params: MetricParams) -> float:
    """CST-equivalent stored energy calculated separately for each field.

    The analytical field_data contains E but not H. For a lossless resonant
    eigenmode, the CST stored energy,

        U_CST = 1/4 int (eps0 |E|^2 + mu0 |H|^2) dV,

    is equal to the peak electric energy when the available real E-field maps
    are peak-amplitude maps:

        U_CST = 0.5 eps0 int |E|^2 dV.

    This is integrated over the cylindrical aperture and is used for all
    k_parallel, k_perp and K normalisations.
    """
    Ex = np.nan_to_num(np.asarray(Ex, float), nan=0.0)
    Ey = np.nan_to_num(np.asarray(Ey, float), nan=0.0)
    Ez = np.nan_to_num(np.asarray(Ez, float), nan=0.0)
    nx, ny, nz = Ez.shape
    x, y = _metric_xy_arrays(Ez.shape, params)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask_xy = (X * X + Y * Y) <= params.Req_m * params.Req_m
    dz = params.length_m / (nz - 1)
    dV = params.dx_m * params.dy_m * dz
    E2 = Ex * Ex + Ey * Ey + Ez * Ez
    U = 0.5 * EPS0 * float(np.sum(E2 * mask_xy[:, :, None])) * dV
    if not np.isfinite(U) or U <= 0.0:
        raise ValueError(f"Calculated non-positive stored energy U={U!r}.")
    return U


def accelerating_voltage_map_from_Ez(Ez: np.ndarray, params: MetricParams) -> np.ndarray:
    """Return complex transit-time voltage map Vz(x,y) from Ez[x,y,z].

    The default follows the monopole on-axis and diagnostic convention,
    integrating over z in [0, L].  A centred coordinate only changes the global
    phase of Vz-derived quantities, but keeping the same convention makes the
    printed complex voltages directly comparable across scripts.
    """
    Ez = np.nan_to_num(np.asarray(Ez, float), nan=0.0)
    nz = Ez.shape[2]
    if params.centred_z:
        z = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, nz)
    else:
        z = np.linspace(0.0, params.length_m, nz)
    phase = np.exp(1j * params.omega * z / (float(params.beta) * C0))
    return np.trapezoid(Ez * phase[None, None, :], z, axis=2)


def _local_quadratic_derivatives(Vz_map: np.ndarray, params: MetricParams) -> dict:
    """Local Method 2 derivative extraction from a small quadratic LS stencil."""
    Vz_map = np.asarray(Vz_map, complex)
    nx, ny = Vz_map.shape
    r = int(params.local_fit_pixels)
    if r < 1:
        raise ValueError("local_fit_pixels must be >= 1")

    i0 = int(round(params.axis_i))
    j0 = int(round(params.axis_j))
    if i0 - r < 0 or i0 + r >= nx or j0 - r < 0 or j0 + r >= ny:
        raise ValueError(
            f"Local fit radius {r} around axis ({params.axis_i},{params.axis_j}) "
            f"does not fit inside Vz_map shape {Vz_map.shape}."
        )

    aperture = r * min(params.dx_m, params.dy_m)
    pts = []
    vals = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            x = di * params.dx_m
            y = dj * params.dy_m
            if np.hypot(x, y) <= aperture:
                pts.append((x, y))
                vals.append(Vz_map[i0 + di, j0 + dj])

    pts = np.asarray(pts, float)
    Vc = np.asarray(vals, complex)
    if len(Vc) < 6:
        raise ValueError(f"Need at least 6 local points for quadratic fit; got {len(Vc)}")

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    cr, *_ = np.linalg.lstsq(A, Vc.real, rcond=None)
    ci, *_ = np.linalg.lstsq(A, Vc.imag, rcond=None)
    coeff = cr + 1j * ci
    a0, ax, ay, axx, axy, ayy = coeff
    grad = np.array([ax, ay], dtype=complex)
    H = np.array([[2.0 * axx, axy], [axy, 2.0 * ayy]], dtype=complex)
    return {
        "V0": a0,
        "grad_V_per_m": grad,
        "hessian_V_per_m2": H,
        "coefficients": {
            "a0": a0, "ax": ax, "ay": ay, "axx": axx, "axy": axy, "ayy": ayy,
        },
        "local_fit_points": int(len(Vc)),
        "local_fit_pixels": r,
    }


def _phase_align_complex_vector(v: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.asarray(v, complex)
    idx = int(np.nanargmax(np.abs(v)))
    ref = v[idx]
    phase = float(np.angle(ref)) if abs(ref) > 0 else 0.0
    return v * np.exp(-1j * phase), phase


def method2_local_quadratic_metrics_from_components(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    params: MetricParams,
) -> dict:
    """Apply benchmarked Method 2 to one field's components."""
    U = stored_energy_from_E_components(Ex, Ey, Ez, params)
    params = MetricParams(
        frequency_Hz=params.frequency_Hz,
        length_m=params.length_m,
        Req_m=params.Req_m,
        beta=params.beta,
        axis_i=params.axis_i,
        axis_j=params.axis_j,
        radius_pixels=params.radius_pixels,
        local_fit_pixels=params.local_fit_pixels,
        centred_z=params.centred_z,
        U_J=U,
    )
    Vz_map = accelerating_voltage_map_from_Ez(Ez, params)
    deriv = _local_quadratic_derivatives(Vz_map, params)

    V0 = deriv["V0"]
    grad = deriv["grad_V_per_m"]
    H = deriv["hessian_V_per_m2"]

    k_parallel = abs(V0) ** 2 / (4.0 * U)
    grad_norm = float(np.sqrt(abs(grad[0]) ** 2 + abs(grad[1]) ** 2))
    Vperp_per_m = (C0 / params.omega) * grad_norm
    k_perp = abs(Vperp_per_m) ** 2 / (4.0 * U)

    K_complex_raw = (C0 / params.omega) * H
    K_raw_phase, phase_K_raw = _phase_align_complex_matrix(K_complex_raw)
    K_raw_real = K_raw_phase.real
    reported_K = quadrupole_reported_from_raw_K(K_raw_real, U_CST_J=U, length_m=params.length_m)
    K_complex_U_norm = K_complex_raw / np.sqrt(4.0 * U)
    K_phase, phase_K = _phase_align_complex_matrix(K_complex_U_norm)
    K_real = K_phase.real
    evals, evecs = np.linalg.eigh(K_real)

    lin_phase, phase_lin = _phase_align_complex_vector((C0 / params.omega) * grad / np.sqrt(4.0 * U))

    Kxx = float(K_real[0, 0])
    Kxy = float(K_real[0, 1])
    Kyy = float(K_real[1, 1])
    K_quad_strength = float(np.sqrt((Kxx - Kyy) ** 2 + 4.0 * Kxy ** 2))
    K_fro = float(np.linalg.norm(K_real, ord="fro"))

    return {
        "method": "2_local_quadratic_least_squares",
        "frequency_Hz": float(params.frequency_Hz),
        "length_m": float(params.length_m),
        "U_CST_J": float(U),
        "U_J_electric_proxy": float(U),  # backwards-compatible alias
        "U_normalisation": "U_CST = 0.5*eps0*int(|E|^2)dV",
        "centred_z": bool(params.centred_z),
        "k_parallel_V_per_C": float(k_parallel),
        "k_parallel_V_per_pC": float(k_parallel * 1e-12),
        "k_perp_V_per_C_per_m2": float(k_perp),
        "k_perp_V_per_pC_per_m2": float(k_perp * 1e-12),
        "gradient_matrix_raw_phase_aligned_real_V_per_C_per_m_per_m": K_raw_real,
        "Kxx_raw_V_per_C_per_m_per_m": float(K_raw_real[0, 0]),
        "Kxy_raw_V_per_C_per_m_per_m": float(K_raw_real[0, 1]),
        "Kyy_raw_V_per_C_per_m_per_m": float(K_raw_real[1, 1]),
        **reported_K,
        "Kxx_U_norm": Kxx,
        "Kxy_U_norm": Kxy,
        "Kyy_U_norm": Kyy,
        "trace_U_norm": float(np.trace(K_real)),
        "determinant_U_norm": float(np.linalg.det(K_real)),
        "K_eig_min_U_norm": float(evals[0]),
        "K_eig_max_U_norm": float(evals[1]),
        "K_quad_strength_U_norm": K_quad_strength,
        "K_frobenius_U_norm": K_fro,
        "phase_K_rad": float(phase_K),
        "phase_linear_rad": float(phase_lin),
        "local_fit_pixels": int(deriv["local_fit_pixels"]),
        "local_fit_points": int(deriv["local_fit_points"]),
        "transverse_pixel_m": params.dx_m,
        "longitudinal_pixel_m": params.length_m / (np.asarray(Ez).shape[2] - 1),
        "axis_i": float(params.axis_i),
        "axis_j": float(params.axis_j),
        "radius_pixels": float(params.radius_pixels),
    }


def heterotypic_method2_metrics_table(
    field_data: dict,
    *,
    f_E1: float,
    f_E2: float,
    f_degen: float,
    f_010: float,
    ell: float,
    Req_m: float,
    beta: float = 1.0,
    axis_i: float = 75.0,
    axis_j: float = 75.0,
    radius_pixels: float = 75.0,
    local_fit_pixels: int = 3,
) -> pd.DataFrame:
    """Calculate Method 2 metrics for E1, E2, E+ and E- in heterotypic mixing."""
    d0 = (C0 / float(f_010)) / 2.0
    d = d0 * float(ell)
    specs = {
        "E1": ("E1_Ex", "E1_Ey", "E1_Ez", f_E1, d0),
        "E2": ("E2_Ex", "E2_Ey", "E2_Ez", f_E2, d0),
        "plus": ("Ex_plus", "Ey_plus", "Ez_plus", f_degen, d),
        "minus": ("Ex_minus", "Ey_minus", "Ez_minus", f_degen, d),
    }

    # Frequency/length convention:
    #   E1, E2      : native/design frequencies at the design length d0.
    #   E+, E-      : degenerate crossing frequency at d=d0*ell.
    rows = []
    for name, (kx, ky, kz, freq, length_m) in specs.items():
        p = MetricParams(
            frequency_Hz=float(freq),
            length_m=float(length_m),
            Req_m=float(Req_m),
            beta=float(beta),
            axis_i=float(axis_i),
            axis_j=float(axis_j),
            radius_pixels=float(radius_pixels),
            local_fit_pixels=int(local_fit_pixels),
        )
        m = method2_local_quadratic_metrics_from_components(
            field_data[kx], field_data[ky], field_data[kz], p
        )
        m.update({"field": name})
        rows.append(m)

    ordered = [
        "field", "method", "frequency_Hz", "length_m", "U_CST_J",
        "k_parallel_V_per_pC", "k_perp_V_per_pC_per_m2",
        "Kxx_V_per_pC_per_m3", "Kxy_V_per_pC_per_m3", "Kyy_V_per_pC_per_m3",
        "K_quad_strength_V_per_pC_per_m3",
        "Kxx_U_norm", "Kxy_U_norm", "Kyy_U_norm",
        "K_eig_min_U_norm", "K_eig_max_U_norm", "K_quad_strength_U_norm",
        "local_fit_pixels", "local_fit_points", "transverse_pixel_m", "longitudinal_pixel_m",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]]

# Configuration
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    # Edit for your machine.
    datapath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    savepath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings")

    # Mode range: m is fixed by family; n and p match the previous studies.
    n_max: int = 3
    p_max: int = 3

    # Pillbox / sweep settings.
    f_010: float = 1.3e9
    LF_start: float = 0.7
    LF_stop: float = 1.3
    param_sweep_resolution: int = 1000
    voxel_res: int = 151

    # Save/load behaviour.
    create_data: bool = True     # False = load previous family pkl if present.
    create_fields: bool = True   # False = load previous field_data.npz if present.
    make_plots: bool = True

    # Families to analyse.  Do not change unless you know why.
    families: tuple[int, ...] = (0, 1, 2)


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


# -----------------------------------------------------------------------------
# Basic save/load helpers
# -----------------------------------------------------------------------------

def pickle_save(obj, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename: str | Path):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_npz_dict(filename: str | Path, data: dict[str, np.ndarray]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, **data)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# Family data: load previous sweeps if available, otherwise create again
# -----------------------------------------------------------------------------

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
) -> dict:
    """Build one TM_mnp family: frequency sweeps and design-length field maps."""
    lambda_010 = C0 / float(frequency_010)
    design_L = lambda_010 / 2.0
    R = pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, param_sweep_resolution)

    all_data = {
        "TM": {},
        "length_factor_vector": length_factors.tolist(),
        "metadata": {
            "family_m": int(m),
            "family_label": FAMILY_LABEL.get(int(m), f"m{m}"),
            "frequency_010_Hz": float(frequency_010),
            "R_m": float(R),
            "design_L_m": float(design_L),
            "voxel_res": int(voxel_res),
        },
    }

    for p in range(p_max + 1):
        for n in range(1, n_max + 1):
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = pillbox_field_voxel_grid_xyz(
                R, design_L, m, n, p,
                voxel_res, voxel_res, voxel_res,
                E0=1.0,
                mode="TM",
            )
            freqs = [f_tm(m, n, p, R, lf * design_L) for lf in length_factors]
            design_freq = f_tm(m, n, p, R, design_L)
            all_data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": list(map(float, freqs)),
                "frequency_normalised": (np.asarray(freqs) / float(frequency_010)).tolist(),
                "design_frequency_Hz": float(design_freq),
                "design_frequency_normalised": float(design_freq / frequency_010),
            }

    return all_data


def load_or_create_family_data(config: RunConfig, m: int) -> dict:
    config.datapath.mkdir(parents=True, exist_ok=True)
    fname = family_data_filename(config.datapath, m, config.voxel_res)

    if (not config.create_data) and fname.exists():
        print(f"Loading TM m={m} family data from {fname}")
        return pickle_load(fname)

    print(f"Creating TM m={m} family data")
    data = assemble_family_data(
        m=m,
        n_max=config.n_max,
        p_max=config.p_max,
        frequency_010=config.f_010,
        LF_start=config.LF_start,
        LF_stop=config.LF_stop,
        param_sweep_resolution=config.param_sweep_resolution,
        voxel_res=config.voxel_res,
    )
    pickle_save(data, fname)
    return data


# -----------------------------------------------------------------------------
# Sweep assembly and crossing detection
# -----------------------------------------------------------------------------

def assemble_sweep_table(family_data: dict[int, dict], f_010: float) -> dict[str, dict]:
    """Flatten all length-factor sweeps into one mode-indexed dictionary."""
    table: dict[str, dict] = {}

    for m, data in family_data.items():
        L = np.asarray(data["length_factor_vector"], dtype=float)
        for mnp, entry in data["TM"].items():
            f_Hz = np.asarray(entry["frequency_Hz"], dtype=float)
            table[f"TM_{mnp}"] = {
                "family_m": int(m),
                "family_label": FAMILY_LABEL.get(int(m), f"m{m}"),
                "mnp": mnp,
                "length_factor": L,
                "frequency_Hz": f_Hz,
                "frequency_normalised": f_Hz / float(f_010),
                "design_frequency_Hz": float(entry["design_frequency_Hz"]),
            }

    return table


def detect_pair_crossings(
    sweep_table: dict[str, dict],
    *,
    m_a: int,
    m_b: int,
    pair_name: str,
) -> dict[str, dict]:
    """Find crossings between every TM_m_a mode and every TM_m_b mode."""
    modes_a = [name for name, d in sweep_table.items() if d["family_m"] == m_a]
    modes_b = [name for name, d in sweep_table.items() if d["family_m"] == m_b]

    crossings: dict[str, dict] = {}

    for name_a in modes_a:
        L = np.asarray(sweep_table[name_a]["length_factor"], dtype=float)
        fa = np.asarray(sweep_table[name_a]["frequency_Hz"], dtype=float)
        for name_b in modes_b:
            fb = np.asarray(sweep_table[name_b]["frequency_Hz"], dtype=float)
            if fa.shape != fb.shape:
                raise ValueError(f"Sweep shape mismatch for {name_a} and {name_b}")

            g = fa - fb
            candidate_indices = np.where(g[:-1] * g[1:] <= 0.0)[0]

            for idx in candidate_indices:
                # Avoid counting an exactly-zero plateau more than once.
                if np.isclose(g[idx], 0.0) and idx > 0 and np.isclose(g[idx - 1], 0.0):
                    continue

                if np.isclose(g[idx], 0.0):
                    Lc = float(L[idx])
                elif np.isclose(g[idx + 1], 0.0):
                    Lc = float(L[idx + 1])
                else:
                    Lc = float(brentq(
                        lambda xx: np.interp(xx, L, fa) - np.interp(xx, L, fb),
                        float(L[idx]),
                        float(L[idx + 1]),
                    ))

                fc = float(np.interp(Lc, L, fa))
                key = f"{pair_name}:{name_a}--{name_b}@{Lc:.8g}"
                crossings[key] = {
                    "pair_type": pair_name,
                    "mode_i": name_a,
                    "mode_j": name_b,
                    "m_i": int(m_a),
                    "m_j": int(m_b),
                    "length_factor": Lc,
                    "frequency_Hz": fc,
                    "frequency_normalised": fc / float(sweep_table[name_a]["frequency_Hz"][0] * 0 + 1.0),  # overwritten below
                }
                # Use the same f010 used for the sweeps.  It is not stored here,
                # so infer it from normalised/current for mode_a at the crossing.
                fhat_a = float(np.interp(Lc, L, sweep_table[name_a]["frequency_normalised"]))
                crossings[key]["frequency_normalised"] = fhat_a

    return crossings


def find_heterotypic_crossings(sweep_table: dict[str, dict]) -> dict[str, dict]:
    all_crossings: dict[str, dict] = {}
    for pair_name, (m_a, m_b) in PAIR_TYPES.items():
        pair_crossings = detect_pair_crossings(
            sweep_table,
            m_a=m_a,
            m_b=m_b,
            pair_name=pair_name,
        )
        all_crossings[pair_name] = pair_crossings
        print(f"{pair_name}: found {len(pair_crossings)} crossings")
    return all_crossings


# -----------------------------------------------------------------------------
# Plot sweeps and crossings
# -----------------------------------------------------------------------------

def plot_pair_sweeps(
    sweep_table: dict[str, dict],
    crossings_by_pair: dict[str, dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, (m_a, m_b) in PAIR_TYPES.items():
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for name, d in sweep_table.items():
            if d["family_m"] not in (m_a, m_b):
                continue
            L = d["length_factor"]
            fhat = d["frequency_normalised"]
            ls = "-" if d["family_m"] == m_a else "--"
            ax.plot(L, fhat, ls=ls, lw=1.0, alpha=0.8, label=name)

        for c in crossings_by_pair[pair_name].values():
            ax.scatter(c["length_factor"], c["frequency_normalised"], s=50, facecolors="none", edgecolors="k")

        ax.set_xlabel(r"Length factor, $\ell=L/L_0$")
        ax.set_ylabel(r"Normalised frequency, $\hat{f}=f/f_{010}$")
        ax.set_title(pair_name.replace("_", "-"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
        fig.savefig(out_dir / f"{pair_name}_sweeps.png", dpi=300)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Field preparation, rotation, mixing, saving and plotting
# -----------------------------------------------------------------------------

def parse_mode_name(mode_name: str) -> tuple[int, str]:
    # mode name format: "TM_012"
    _, mnp = mode_name.split("_", 1)
    return int(mnp[0]), mnp


def rotation_label(rot: dict | None) -> dict | None:
    if rot is None:
        return None
    keys = ["rotation_angle_deg", "peak_before", "peak_after", "global_max_after", "vertical_midplane_max_after"]
    return {k: rot.get(k) for k in keys if k in rot}


def prepare_field_for_mixing(field: dict, *, m: int, label: str, out_dir: Path) -> tuple[dict, dict | None]:
    """Return {'Ex','Ey','Ez'} for mixing, rotating non-monopole families.

    The monopole field is axisymmetric and is left unrotated.  Dipoles keep the
    previous peak-based alignment.  Quadrupoles use the homotypic quadrupole
    Ez principal-axis rotation so the quadrupolar orientation follows the same
    convention as the standalone homotypic quadrupole analysis.
    """
    if m == 0:
        return {"Ex": field["Ex"], "Ey": field["Ey"], "Ez": field["Ez"]}, None

    if m == 2:
        # Use the homotypic quadrupole rotation method: fit the near-axis Ez
        # quadrupole tensor and rotate its principal axes into the required
        # vertical longitudinal mid-plane convention.
        rot = align_field_to_vertical_plane(
            field,
            out_plot=str(out_dir / f"{label}_theta_r_rotation.png"),
            label=label,
        )
    else:
        # Preserve the previous peak-based alignment for dipoles, for which the
        # quadrupole Ez principal-axis fit is not the appropriate diagnostic.
        rot = align_field_to_vertical_plane_peak(
            field,
            out_plot=str(out_dir / f"{label}_theta_r_rotation.png"),
            label=label,
        )

    return {"Ex": rot["Ex"], "Ey": rot["Ey"], "Ez": rot["Ez"]}, rot


def get_or_create_heterotypic_field_data(
    crossing_key: str,
    crossing: dict,
    family_data: dict[int, dict],
    out_dir: Path,
    *,
    create_fields: bool,
) -> dict:
    field_file = out_dir / "field_data.npz"
    analysis_file = out_dir / "heterotypic_crossing_analysis.pkl"

    if (not create_fields) and field_file.exists() and analysis_file.exists():
        analysis = pickle_load(analysis_file)
        summary_pdf = out_dir / "slice_summary_pdfs" / f"{out_dir.name}_field_summary.pdf"
        if not summary_pdf.exists():
            field_data = load_npz_dict(field_file)
            slice_dict = extract_slices(field_data)
            pickle_save(slice_dict, out_dir / "slice_dict.pkl")
            summary_pdf = save_four_slice_pdfs_and_merge(
                slice_dict=slice_dict,
                out_dir=out_dir / "slice_summary_pdfs",
                merged_pdf_name=f"{out_dir.name}_field_summary.pdf",
            )
        analysis.setdefault("files", {})["slice_summary_dir"] = str(out_dir / "slice_summary_pdfs")
        analysis.setdefault("files", {})["merged_slice_pdf"] = str(summary_pdf)
        pickle_save(analysis, analysis_file)
        return analysis

    m_i, mnp_i = parse_mode_name(crossing["mode_i"])
    m_j, mnp_j = parse_mode_name(crossing["mode_j"])

    raw_i = family_data[m_i]["TM"][mnp_i]["3D_Efield"]
    raw_j = family_data[m_j]["TM"][mnp_j]["3D_Efield"]

    E1, rot_i = prepare_field_for_mixing(raw_i, m=m_i, label=f"TM{mnp_i}", out_dir=out_dir)
    E2, rot_j = prepare_field_for_mixing(raw_j, m=m_j, label=f"TM{mnp_j}", out_dir=out_dir)

    field_data = combine_fields(E1, E2)
    save_npz_dict(field_file, field_data)

    slice_dict = extract_slices(field_data)
    pickle_save(slice_dict, out_dir / "slice_dict.pkl")

    summary_pdf = save_four_slice_pdfs_and_merge(
        slice_dict=slice_dict,
        out_dir=out_dir / "slice_summary_pdfs",
        merged_pdf_name=f"{out_dir.name}_field_summary.pdf",
    )

    plots_dir = out_dir / "plots"
    plot_field_slices(
        field_data,
        str(plots_dir),
        title=f"{crossing['mode_i']} / {crossing['mode_j']}",
    )
    plot_combined_field_slices_4x4(
        field_data,
        plots_dir,
        title=f"{crossing['mode_i']} / {crossing['mode_j']}",
    )

    # Apply the benchmarked Method 2 local quadratic least-squares metric
    # extraction to all four fields.  For heterotypic crossings these scalar
    # metrics should be interpreted as local multipole components of the mixed
    # field, not as evidence that the mixed field is a pure monopole, dipole or
    # quadrupole.
    f_E1 = float(family_data[m_i]["TM"][mnp_i]["design_frequency_Hz"])
    f_E2 = float(family_data[m_j]["TM"][mnp_j]["design_frequency_Hz"])
    f_degen = float(crossing["frequency_Hz"])
    ell = float(crossing["length_factor"])
    f_010 = float(family_data[m_i]["metadata"]["frequency_010_Hz"])
    Req_m = float(family_data[m_i]["metadata"]["R_m"])

    method2_df = heterotypic_method2_metrics_table(
        field_data,
        f_E1=f_E1,
        f_E2=f_E2,
        f_degen=f_degen,
        f_010=f_010,
        ell=ell,
        Req_m=Req_m,
        beta=1.0,
        axis_i=field_data["E1_Ez"].shape[0] // 2,
        axis_j=field_data["E1_Ez"].shape[1] // 2,
        radius_pixels=(field_data["E1_Ez"].shape[0] - 1) / 2.0,
        local_fit_pixels=3,
    )
    metrics_csv = out_dir / "heterotypic_method2_local_quadratic_metrics.csv"
    metrics_pkl = out_dir / "heterotypic_method2_local_quadratic_metrics.pkl"
    method2_df.to_csv(metrics_csv, index=False)
    pickle_save(method2_df, metrics_pkl)


    print(f'\n{crossing["mode_i"] = }')
    print(f'{crossing["mode_j"] = }')
    print(f'{m_i = }')
    print(f'{m_j = }')
    print(f'{f_E1 = }')
    print(f'{f_E2 = }')
    print(f'{f_degen = }')
    print(f'{ell = }')
    print(f'design length = {(C0 / f_010) / 2.0}')
    print(f'crossing_length_m = {(C0 / f_010) / 2.0 *ell}')

    analysis = {
        "crossing_key": crossing_key,
        "crossing": crossing,
        "mode_i": crossing["mode_i"],
        "mode_j": crossing["mode_j"],
        "m_i": m_i,
        "m_j": m_j,
        "mnp_i": mnp_i,
        "mnp_j": mnp_j,
        "rotation_i": rotation_label(rot_i),
        "rotation_j": rotation_label(rot_j),
        "method2_frequency_rules": {
            "E1_frequency_Hz": f_E1,
            "E2_frequency_Hz": f_E2,
            "plus_minus_frequency_Hz": f_degen,
            "length_factor_ell": ell,
            "design_length_m": (C0 / f_010) / 2.0,
            "crossing_length_m": ((C0 / f_010) / 2.0) * ell,
        },
        "files": {
            "field_data_npz": str(field_file),
            "slice_dict_pkl": str(out_dir / "slice_dict.pkl"),
            "plots_dir": str(out_dir / "plots"),
            "slice_summary_dir": str(out_dir / "slice_summary_pdfs"),
            "merged_slice_pdf": str(summary_pdf),
            "method2_metrics_csv": str(metrics_csv),
            "method2_metrics_pkl": str(metrics_pkl),
        },
        "note": (
            "Method 2 local quadratic least-squares metrics have been calculated "
            "for E1, E2, E+ and E-. For heterotypic crossings these are local "
            "multipole components of a mixed field, not a classification of the "
            "mixed field as a pure monopole, dipole or quadrupole."
        ),
    }
    pickle_save(analysis, analysis_file)
    return analysis


def safe_folder_name(crossing_key: str) -> str:
    s = crossing_key.replace(":", "__").replace("@", "__ell_")
    s = s.replace("--", "__")
    s = s.replace(".", "p")
    return s


def process_heterotypic_crossings(
    crossings_by_pair: dict[str, dict],
    family_data: dict[int, dict],
    savepath: Path,
    *,
    create_fields: bool,
) -> dict[str, dict]:
    analyses: dict[str, dict] = {}

    for pair_name, pair_crossings in crossings_by_pair.items():
        pair_dir = savepath / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        analyses[pair_name] = {}

        for key, crossing in pair_crossings.items():
            print(f"Processing {key}")
            out_dir = pair_dir / safe_folder_name(key)
            out_dir.mkdir(parents=True, exist_ok=True)
            analyses[pair_name][key] = get_or_create_heterotypic_field_data(
                key,
                crossing,
                family_data,
                out_dir,
                create_fields=create_fields,
            )

    pickle_save(analyses, savepath / "all_heterotypic_crossing_analyses.pkl")
    return analyses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    config = RunConfig(create_data= False,
    create_fields= True,
    make_plots = True)
    config.datapath.mkdir(parents=True, exist_ok=True)
    config.savepath.mkdir(parents=True, exist_ok=True)

    family_data = {m: load_or_create_family_data(config, m) for m in config.families}

    for m, data in family_data.items():
        R = pillbox_radius_from_freq(config.f_010)
        L0 = (C0 / config.f_010) / 2.0

        for mnp, entry in data["TM"].items():
            mm, nn, pp = map(int, mnp)
            f_expected = f_tm(mm, nn, pp, R, L0)
            f_loaded = float(entry["design_frequency_Hz"])

            if not np.isclose(f_loaded, f_expected, rtol=1e-10, atol=1.0):
                raise ValueError(
                    f"Stale/inconsistent family data for TM{mnp}: "
                    f"loaded design_frequency_Hz={f_loaded:.12e}, "
                    f"expected={f_expected:.12e}. "
                    f"Regenerate data with create_data=True."
                )
            else:
                print(mm, nn, pp, f_loaded, f_expected)

    sweep_table = assemble_sweep_table(family_data, config.f_010)
    pickle_save(sweep_table, config.savepath / "heterotypic_sweep_table.pkl")

    crossings_by_pair = find_heterotypic_crossings(sweep_table)
    pickle_save(crossings_by_pair, config.savepath / "heterotypic_crossing_results.pkl")

    if config.make_plots:
        plot_pair_sweeps(sweep_table, crossings_by_pair, config.savepath / "sweep_plots")

    process_heterotypic_crossings(
        crossings_by_pair,
        family_data,
        config.savepath,
        create_fields=config.create_fields,
    )

    print("\nDone.  Heterotypic crossing fields, plots and Method 2 local metrics have been saved.")


if __name__ == "__main__":
    main()
