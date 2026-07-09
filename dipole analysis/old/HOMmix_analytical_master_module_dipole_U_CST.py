"""Stripped helper module for analytical TM m=1 dipole crossing analysis.

Array convention throughout: field[x_index, y_index, z_index].
Plot convention: transverse images show x horizontal and y vertical; longitudinal
images show z horizontal and y vertical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import jn_zeros

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1.0e-12


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
        raise NotImplementedError("Stripped dipole module only supports TM crossing searches")
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


def align_field_to_vertical_plane(field: dict, out_plot: str | None = None, label="") -> dict:
    Ex, Ey, Ez = field["Ex"], field["Ey"], field["Ez"]
    x = field.get("x_m", np.linspace(-1, 1, Ex.shape[0]))
    y = field.get("y_m", np.linspace(-1, 1, Ex.shape[1]))
    z = field.get("z_m", np.linspace(0, 1, Ex.shape[2]))
    Eabs0 = eabs_from_components(Ex, Ey, Ez)
    angle, peak0 = rotation_angle_to_vertical_plane(Eabs0, x, y)
    rot = rotate_vector_field_about_z(Ex, Ey, Ez, x, y, z, angle)
    peak1 = tuple(int(v) for v in np.unravel_index(np.nanargmax(rot["Eabs"]), rot["Eabs"].shape))
    mid = len(x)//2
    if abs(peak1[0] - mid) > 1:
        raise RuntimeError(f"{label}: rotation failed; peak_after={peak1}, expected x index {mid}, angle={angle}")
    if out_plot:
        plot_theta_r_before_after(Eabs0, rot["Eabs"], x, y, out_plot, title=f"{label}: angle={angle:.3f} deg")
    rot.update({"rotation_angle_deg": angle, "peak_before": peak0, "peak_after": peak1})
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


def extract_slices(field_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Extract commonly used 2D slices from each 3D field.

    Generated slices:
        *_iris_1            F[:, :, 0]
        *_iris_2            F[:, :, -1]
        *_transverse_mid    F[:, :, mid_z]
        *_longitudinal_mid  F[mid_x, :, :]
    """

    slices = {}

    for key, F in field_data.items():
        if not (isinstance(F, np.ndarray) and F.ndim == 3):
            continue

        midx = F.shape[0] // 2
        midz = F.shape[2] // 2

        slices[f"{key}_iris_1"] = F[:, :, 0]
        slices[f"{key}_iris_2"] = F[:, :, -1]
        slices[f"{key}_transverse_mid"] = F[:, :, midz]
        slices[f"{key}_longitudinal_mid"] = F[midx, :, :]

    return slices




def accelerating_voltage_complex(Ez_line, z_m, omega, beta=1.0, *, centre_z: bool = False):
    """Complex transit-time voltage using the shared convention z in [0,L].

    centre_z=False matches the current monopole/heterotypic Method-2 convention:

        Vz = integral_0^L Ez(z) exp(i omega z / beta c) dz.

    centre_z=True is retained only as a diagnostic/legacy option.
    """
    z = np.asarray(z_m, float)
    if centre_z:
        z = z - 0.5 * (z[0] + z[-1])
    Ez_line = np.asarray(Ez_line, float)
    return np.trapezoid(Ez_line * np.exp(1j * omega * z / (beta * C0)), z)

def _field_spacing_and_mask(
    shape: tuple[int, int, int],
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> tuple[float, float, float, np.ndarray, float, float, float]:
    """Return dx, dy, dz and the cylindrical transverse mask.

    For the standard 151^3 maps, the beam/cylinder axis is array[75,75,:]
    and radius_pixels=75, giving dx=dy=Req_m/75.
    """
    nx, ny, nz = shape
    if axis_i is None:
        axis_i = float(nx // 2)
    if axis_j is None:
        axis_j = float(ny // 2)
    if radius_pixels is None:
        radius_pixels = float(min(axis_i, axis_j, nx - 1 - axis_i, ny - 1 - axis_j))
    if radius_pixels <= 0.0:
        raise ValueError(f"radius_pixels must be positive, got {radius_pixels!r}")

    dx = float(Req_m) / float(radius_pixels)
    dy = dx
    dz = float(length_m) / (nz - 1)

    x = (np.arange(nx, dtype=float) - float(axis_i)) * dx
    y = (np.arange(ny, dtype=float) - float(axis_j)) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask_xy = (X * X + Y * Y) <= float(Req_m) * float(Req_m)
    return dx, dy, dz, mask_xy, float(axis_i), float(axis_j), float(radius_pixels)


def stored_energy_from_Etotal_CST_equivalent(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    *,
    Req_m: float,
    length_m: float,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
) -> dict[str, float]:
    """CST-equivalent stored energy from electric fields only.

    CST eigenmode stored energy is total time-averaged EM energy.  With only the
    analytical electric field available, use the lossless-resonator equivalence

        U_CST = 2 U_E,time = 0.5 eps0 integral |E|^2 dV.

    Diagnostics for older Ez-only/time-average conventions are also returned.
    """
    Ex = np.nan_to_num(np.asarray(Ex, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ey = np.nan_to_num(np.asarray(Ey, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Ez = np.nan_to_num(np.asarray(Ez, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Ex.shape != Ey.shape or Ex.shape != Ez.shape or Ez.ndim != 3:
        raise ValueError(f"Ex, Ey, Ez must be matching 3D arrays; got {Ex.shape}, {Ey.shape}, {Ez.shape}")

    dx, dy, dz, mask_xy, axis_i, axis_j, radius_pixels = _field_spacing_and_mask(
        Ez.shape,
        Req_m=Req_m,
        length_m=length_m,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )
    dV = dx * dy * dz
    mask = mask_xy[:, :, None]

    int_Ex2 = float(np.sum(Ex * Ex * mask) * dV)
    int_Ey2 = float(np.sum(Ey * Ey * mask) * dV)
    int_Ez2 = float(np.sum(Ez * Ez * mask) * dV)
    int_Etotal2 = int_Ex2 + int_Ey2 + int_Ez2

    U_E_time = 0.25 * EPS0 * int_Etotal2
    U_CST = 2.0 * U_E_time
    if not np.isfinite(U_CST) or U_CST <= 0.0:
        raise ValueError(f"Calculated non-positive CST-equivalent stored energy U={U_CST!r}")

    return {
        "U_CST_J": float(U_CST),
        "U_Etotal_time_average_J": float(U_E_time),
        "U_Etotal_peak_J": float(0.5 * EPS0 * int_Etotal2),
        "U_Ez_only_time_average_J": float(0.25 * EPS0 * int_Ez2),
        "U_Ez_only_peak_J": float(0.5 * EPS0 * int_Ez2),
        "int_Ex2_dV": int_Ex2,
        "int_Ey2_dV": int_Ey2,
        "int_Ez2_dV": int_Ez2,
        "int_Etotal2_dV": int_Etotal2,
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": float(dz),
        "axis_i": float(axis_i),
        "axis_j": float(axis_j),
        "radius_pixels": float(radius_pixels),
    }


def _write_kick_diagnostic_txt(
    filename: str | Path,
    *,
    label: str,
    axis: str,
    frequency_Hz: float,
    length_m: float,
    beta: float,
    fit_pixels: int,
    r_m: np.ndarray,
    Vc: np.ndarray,
    dVz_dr: complex,
    Vperp_per_m_offset: complex,
    kick_raw_V_per_C_per_m2: float,
    kick_U_norm_V_per_C_per_m2: float,
    kick_U_norm_V_per_pC_per_m2: float,
    kick_loss_equiv_U_norm: np.ndarray,
    kick_loss_equiv_raw: np.ndarray,
    energy: dict[str, float],
    transverse_pixel_m: float,
    longitudinal_pixel_m: float,
    centre_z: bool,
) -> None:
    lines: list[str] = []
    lines.append(f"{label}: dipole kick diagnostic")
    lines.append("")
    lines.append("CONVENTIONS")
    lines.append(f"  axis                         = {axis}")
    lines.append(f"  beta                         = {beta}")
    lines.append(f"  fit_pixels                   = {fit_pixels}")
    lines.append(f"  centred_z                    = {centre_z}  (False means z in [0,L])")
    lines.append(f"  frequency_Hz                 = {frequency_Hz:.12e}")
    lines.append(f"  length_m                     = {length_m:.12e}")
    lines.append(f"  transverse_pixel_m           = {transverse_pixel_m:.12e}")
    lines.append(f"  longitudinal_pixel_m         = {longitudinal_pixel_m:.12e}")
    lines.append(f"  axis_indices_xy              = ({energy['axis_i']:.6g}, {energy['axis_j']:.6g})")
    lines.append(f"  radius_pixels                = {energy['radius_pixels']:.6g}")
    lines.append("")
    lines.append("STORED ENERGY")
    lines.append("  Primary normalisation uses U_CST = 0.5 eps0 integral |E|^2 dV")
    for key in [
        "int_Ex2_dV", "int_Ey2_dV", "int_Ez2_dV", "int_Etotal2_dV",
        "U_Ez_only_time_average_J", "U_Ez_only_peak_J",
        "U_Etotal_time_average_J", "U_Etotal_peak_J", "U_CST_J",
    ]:
        lines.append(f"  {key:30s}= {energy[key]:.12e}")
    lines.append("")
    lines.append("PW GRADIENT METHOD")
    lines.append(f"  dVz_dr                       = {dVz_dr.real:.12e}{dVz_dr.imag:+.12e}j V/C/m")
    lines.append(f"  |dVz_dr|                     = {abs(dVz_dr):.12e} V/C/m")
    lines.append(f"  Vperp_per_m_offset           = {Vperp_per_m_offset.real:.12e}{Vperp_per_m_offset.imag:+.12e}j V/C/m")
    lines.append(f"  raw |Vperp|/m                = {kick_raw_V_per_C_per_m2:.12e} V/C/m/m")
    lines.append(f"  U-normalised                 = {kick_U_norm_V_per_C_per_m2:.12e} V/C/m/m")
    lines.append(f"  U-normalised                 = {kick_U_norm_V_per_pC_per_m2:.12e} V/pC/m/m")
    lines.append("")
    lines.append("OFFSET METHOD")
    lines.append("  k_perp(r) = |(c/omega) Vz(r)/r|^2/(4 U_CST)")
    finite = np.isfinite(kick_loss_equiv_U_norm)
    if np.any(finite):
        lines.append(f"  median U-normalised          = {np.nanmedian(kick_loss_equiv_U_norm):.12e} V/C/m/m")
        lines.append(f"  median U-normalised          = {np.nanmedian(kick_loss_equiv_U_norm) * PC:.12e} V/pC/m/m")
        lines.append(f"  median raw                   = {np.nanmedian(kick_loss_equiv_raw):.12e} V/C/m/m")
        rel = (np.nanmedian(kick_loss_equiv_U_norm) - kick_U_norm_V_per_C_per_m2) / kick_U_norm_V_per_C_per_m2 if kick_U_norm_V_per_C_per_m2 else np.nan
        lines.append(f"  median relative difference   = {rel:.12e}")
    lines.append("")
    lines.append("OFFSET TABLE")
    lines.append("  r_m, Re(Vz), Im(Vz), |Vz|, raw_kick, U_norm_kick_V_per_C_m_m, U_norm_kick_V_per_pC_m_m")
    for rr, vv, kr, ku in zip(r_m, Vc, kick_loss_equiv_raw, kick_loss_equiv_U_norm):
        lines.append(
            f"  {rr:.12e}, {vv.real:.12e}, {vv.imag:.12e}, {abs(vv):.12e}, "
            f"{kr:.12e}, {ku:.12e}, {ku * PC:.12e}"
        )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(lines))


def kick_from_Ez_field(
    Ex,
    Ey=None,
    Ez=None,
    f_010=None,
    f_mnp=None,
    l_factor=None,
    Req_m=None,
    *,
    axis="y",
    fit_pixels=8,
    beta=1.0,
    save_directory=None,
    label="",
    centre_z: bool = False,
    axis_i: float | None = None,
    axis_j: float | None = None,
    radius_pixels: float | None = None,
):
    """Dipole kick diagnostic from Ez, normalised to U_CST.

    New preferred call signature is

        kick_from_Ez_field(Ex, Ey, Ez, f_010=..., f_mnp=..., ...)

    A backwards-compatible Ez-only call is still accepted, but then Ex=Ey=0 and
    the stored energy is not CST-equivalent.  The homotypic driver has been
    updated to pass all three components.

    Reported primary kick factor:

        k_perp = |(c/omega) dVz/dr|^2 / (4 U_CST)

    in V/C/m/m and V/pC/m/m.
    """
    # Backwards-compatible support for the old positional Ez-only signature:
    # kick_from_Ez_field(Ez, f_010, f_mnp, l_factor, Req_m, ...)
    if Ez is None:
        Ez_arr = np.nan_to_num(np.asarray(Ex, float), nan=0.0)
        Ex_arr = np.zeros_like(Ez_arr)
        Ey_arr = np.zeros_like(Ez_arr)
    else:
        Ex_arr = np.nan_to_num(np.asarray(Ex, float), nan=0.0)
        Ey_arr = np.nan_to_num(np.asarray(Ey, float), nan=0.0)
        Ez_arr = np.nan_to_num(np.asarray(Ez, float), nan=0.0)

    if f_010 is None or f_mnp is None or l_factor is None or Req_m is None:
        raise ValueError("f_010, f_mnp, l_factor and Req_m must be supplied.")

    if Ex_arr.shape != Ey_arr.shape or Ex_arr.shape != Ez_arr.shape:
        raise ValueError(f"Ex, Ey, Ez must have matching shapes; got {Ex_arr.shape}, {Ey_arr.shape}, {Ez_arr.shape}")

    nx, ny, nz = Ez_arr.shape
    ix0, iy0 = nx // 2, ny // 2

    L = (C0 / float(f_010)) / 2.0 * float(l_factor)
    z_m = np.linspace(0.0, L, nz)

    energy = stored_energy_from_Etotal_CST_equivalent(
        Ex_arr, Ey_arr, Ez_arr,
        Req_m=float(Req_m),
        length_m=L,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
    )
    U_CST_J = energy["U_CST_J"]
    dx = energy["dx_m"]
    dy = energy["dy_m"]
    dr = dy if axis == "y" else dx

    omega = 2.0 * np.pi * float(f_mnp)
    max_pix = min(int(fit_pixels), iy0 - 1 if axis == "y" else ix0 - 1)

    vals = []
    for dp in range(-max_pix, max_pix + 1):
        if dp == 0:
            continue
        if axis == "y":
            r = dp * dy
            line = Ez_arr[ix0, iy0 + dp, :]
        elif axis == "x":
            r = dp * dx
            line = Ez_arr[ix0 + dp, iy0, :]
        else:
            raise ValueError("axis must be 'x' or 'y'")
        V = accelerating_voltage_complex(line, z_m, omega, beta=beta, centre_z=centre_z)
        vals.append((r, V))

    r = np.array([v[0] for v in vals], float)
    Vc = np.array([v[1] for v in vals], complex)
    order = np.argsort(r)
    r = r[order]
    Vc = Vc[order]

    # PW gradient fit.
    gr = np.polyfit(r, Vc.real, 1)[0]
    gi = np.polyfit(r, Vc.imag, 1)[0]
    dVz_dr = gr + 1j * gi
    Vperp_per_m_offset = (C0 / omega) * dVz_dr
    kick_pw_fit_raw = float(abs(Vperp_per_m_offset))
    kick_pw_fit_U = float(abs(Vperp_per_m_offset) ** 2 / (4.0 * U_CST_J))

    # Local PW estimates at each offset.
    dVz_dr_local = np.gradient(Vc, r)
    kick_pw_local_raw = np.abs((C0 / omega) * dVz_dr_local)
    kick_pw_local_U = kick_pw_local_raw**2 / (4.0 * U_CST_J)

    # Offset method: equivalent to |(c/omega) Vz/r|^2/(4U) near the axis.
    Vz_abs = np.abs(Vc)
    loss_like_raw = Vz_abs**2 / 4.0
    loss_U_norm_V_per_C = Vz_abs**2 / (4.0 * U_CST_J)

    kick_loss_equiv_raw = np.full_like(Vz_abs, np.nan, dtype=float)
    kick_loss_equiv_U = np.full_like(Vz_abs, np.nan, dtype=float)
    nonzero = ~np.isclose(r, 0.0)
    kick_loss_equiv_raw[nonzero] = (C0 / omega) * Vz_abs[nonzero] / np.abs(r[nonzero])
    kick_loss_equiv_U[nonzero] = kick_loss_equiv_raw[nonzero]**2 / (4.0 * U_CST_J)

    if save_directory is not None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        tag = f"{label}_" if label else ""

        np.savez_compressed(
            save_directory / f"{tag}kick_diagnostics.npz",
            r_m=r,
            Vz_complex=Vc,
            Vz_abs=Vz_abs,
            loss_like_raw=loss_like_raw,
            loss_U_norm_V_per_C=loss_U_norm_V_per_C,
            loss_U_norm_V_per_pC=loss_U_norm_V_per_C * PC,
            dVz_dr=dVz_dr,
            dVz_dr_local=dVz_dr_local,
            kick_pw_fit_raw_V_per_C_per_m_per_m=kick_pw_fit_raw,
            kick_pw_fit_U_norm_V_per_C_per_m_per_m=kick_pw_fit_U,
            kick_pw_fit_U_norm_V_per_pC_per_m_per_m=kick_pw_fit_U * PC,
            kick_pw_local_raw_V_per_C_per_m_per_m=kick_pw_local_raw,
            kick_pw_local_U_norm_V_per_C_per_m_per_m=kick_pw_local_U,
            kick_pw_local_U_norm_V_per_pC_per_m_per_m=kick_pw_local_U * PC,
            kick_loss_equiv_raw_V_per_C_per_m_per_m=kick_loss_equiv_raw,
            kick_loss_equiv_U_norm_V_per_C_per_m_per_m=kick_loss_equiv_U,
            kick_loss_equiv_U_norm_V_per_pC_per_m_per_m=kick_loss_equiv_U * PC,
            U_CST_J=U_CST_J,
            int_Ex2_dV=energy["int_Ex2_dV"],
            int_Ey2_dV=energy["int_Ey2_dV"],
            int_Ez2_dV=energy["int_Ez2_dV"],
            int_Etotal2_dV=energy["int_Etotal2_dV"],
            transverse_pixel_m=dr,
            longitudinal_pixel_m=L / (nz - 1),
            f_mnp=f_mnp,
            omega=omega,
            axis=axis,
        )

        _write_kick_diagnostic_txt(
            save_directory / f"{tag}diagnostic.txt",
            label=label or "field",
            axis=axis,
            frequency_Hz=float(f_mnp),
            length_m=L,
            beta=beta,
            fit_pixels=max_pix,
            r_m=r,
            Vc=Vc,
            dVz_dr=dVz_dr,
            Vperp_per_m_offset=Vperp_per_m_offset,
            kick_raw_V_per_C_per_m2=kick_pw_fit_raw,
            kick_U_norm_V_per_C_per_m2=kick_pw_fit_U,
            kick_U_norm_V_per_pC_per_m2=kick_pw_fit_U * PC,
            kick_loss_equiv_U_norm=kick_loss_equiv_U,
            kick_loss_equiv_raw=kick_loss_equiv_raw,
            energy=energy,
            transverse_pixel_m=dr,
            longitudinal_pixel_m=L / (nz - 1),
            centre_z=centre_z,
        )

        # Ez transverse and longitudinal diagnostic.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = axes[0].imshow(Ez_arr[:, :, nz // 2].T, origin="lower", aspect="equal")
        axes[0].set_title("Ez transverse mid")
        axes[0].set_xlabel("x pixel")
        axes[0].set_ylabel("y pixel")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(Ez_arr[ix0, :, :], origin="lower", aspect="auto")
        axes[1].set_title("Ez longitudinal mid")
        axes[1].set_xlabel("z pixel")
        axes[1].set_ylabel("y pixel")
        fig.colorbar(im1, ax=axes[1])
        fig.suptitle(f"{label}: Ez slices")
        fig.savefig(save_directory / f"{tag}Ez_slices.png", dpi=220)
        plt.close(fig)

        # Complex Vz and fit.
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(r * 1e3, Vc.real, "o-", label=r"Re$(V_z)$")
        ax.plot(r * 1e3, Vc.imag, "o-", label=r"Im$(V_z)$")
        ax.plot(r * 1e3, np.polyval(np.polyfit(r, Vc.real, 1), r), "--", label="Re fit")
        ax.plot(r * 1e3, np.polyval(np.polyfit(r, Vc.imag, 1), r), "--", label="Im fit")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"$V_z$ [V/C]")
        ax.set_title(f"{label}: complex Vz fit")
        ax.legend()
        fig.savefig(save_directory / f"{tag}Vz_complex_fit.png", dpi=220)
        plt.close(fig)

        # |Vz|
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(r * 1e3, Vz_abs, "o-")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"$|V_z|$ [V/C]")
        ax.set_title(f"{label}: offset voltage")
        fig.savefig(save_directory / f"{tag}r_vs_Vz_abs.png", dpi=220)
        plt.close(fig)

        # U-normalised loss-like offset scan.
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(r * 1e3, loss_U_norm_V_per_C * PC, "o-")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"$|V_z|^2/(4U_{CST})$ [V/pC]")
        ax.set_title(f"{label}: U-normalised offset voltage loss")
        fig.savefig(save_directory / f"{tag}r_vs_loss.png", dpi=220)
        plt.close(fig)

        # Kick comparison.
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(r * 1e3, kick_pw_local_U * PC, "o-", label="PW local gradient")
        ax.plot(r * 1e3, kick_loss_equiv_U * PC, "s-", label=r"$V_z/r$ offset equivalent")
        ax.axhline(kick_pw_fit_U * PC, color="k", ls="--", alpha=0.6, label="PW linear fit")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"Kick [$\mathrm{V/pC/m/m}$]")
        ax.set_title(f"{label}: U-normalised kick comparison")
        ax.legend()
        fig.savefig(save_directory / f"{tag}r_vs_kick_comparison.png", dpi=220)
        plt.close(fig)

    return {
        "r_m": r,
        "Vz_complex_V": Vc,
        "Vz_abs_V": Vz_abs,
        "loss": loss_U_norm_V_per_C * PC,
        "loss_like_raw_V2_per_C2": loss_like_raw,
        "loss_U_norm_V_per_C": loss_U_norm_V_per_C,
        "loss_U_norm_V_per_pC": loss_U_norm_V_per_C * PC,
        "U_CST_J": U_CST_J,
        "energy_diagnostics": energy,
        "dVz_dr_V_per_m": dVz_dr,
        "dVz_dr_local_V_per_m": dVz_dr_local,
        "Vperp_per_m_offset_V_per_m": Vperp_per_m_offset,
        "kick_raw_V_per_C_per_m_per_m": kick_pw_fit_raw,
        "kick_V_per_C_per_m_per_m": kick_pw_fit_U,
        "kick_V_per_pC_per_m_per_m": kick_pw_fit_U * PC,
        "kick_pw_local_raw_V_per_C_per_m_per_m": kick_pw_local_raw,
        "kick_pw_local_V_per_C_per_m_per_m": kick_pw_local_U,
        "kick_pw_local_V_per_pC_per_m_per_m": kick_pw_local_U * PC,
        "kick_loss_equiv_raw_V_per_C_per_m_per_m": kick_loss_equiv_raw,
        "kick_loss_equiv_V_per_C_per_m_per_m": kick_loss_equiv_U,
        "kick_loss_equiv_V_per_pC_per_m_per_m": kick_loss_equiv_U * PC,
        "transverse_pixel_m": dr,
        "longitudinal_pixel_m": L / (nz - 1),
        "axis": axis,
        "f_mnp_Hz": float(f_mnp),
        "omega_rad_s": omega,
        "centre_z": bool(centre_z),
        "normalisation": "U_CST = 0.5 eps0 integral |E|^2 dV; k_perp=|(c/omega)dVz/dr|^2/(4U_CST)",
    }


def load_field_data_npz(fname: str | Path) -> dict[str, np.ndarray]:
    """
    Load a field dictionary previously written by save_field_data_npz().

    Returns
    -------
    dict
        Dictionary containing the same keys as were passed to
        save_field_data_npz().
    """
    fname = Path(fname)

    with np.load(fname, allow_pickle=False) as data:
        field_data = {key: data[key] for key in data.files}

    return field_data