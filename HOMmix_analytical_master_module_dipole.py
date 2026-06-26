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




def accelerating_voltage_complex(Ez_line, z_m, omega, beta=1.0):
    zc = np.asarray(z_m, float) - 0.5*(z_m[0] + z_m[-1])
    Ez_line = np.asarray(Ez_line, float)
    return np.trapz(Ez_line * np.exp(1j*omega*zc/(beta*C0)), zc)

def kick_from_Ez_field(
    Ez,
    f_010,
    f_mnp,
    l_factor,
    Req_m,
    *,
    axis="y",
    fit_pixels=8,
    beta=1.0,
    save_directory=None,
    label="",
):
    """
    Dipole kick diagnostic from Ez.

    Uses f_mnp, not f_010, for the Panofsky-Wenzel factor:

        V_perp = (c / omega_mnp) dVz/dr

    Also compares with the offset method:

        Ez(r,z) -> Vz(r) -> loss(r) = |Vz(r)|^2 / 4
        kick_loss_equiv(r) = (c / omega_mnp) |Vz(r)| / |r|

    The loss-derived method is only expected to agree near the axis when
    Vz(r) is approximately linear in r.
    """

    Ez = np.nan_to_num(np.asarray(Ez, float), nan=0.0)

    nx, ny, nz = Ez.shape
    ix0, iy0 = nx // 2, ny // 2

    L = (C0 / f_010) / 2.0 * float(l_factor)
    z_m = np.linspace(0.0, L, nz)

    dx = 2.0 * Req_m / (nx - 1)
    dy = 2.0 * Req_m / (ny - 1)
    dr = dy if axis == "y" else dx

    omega = 2.0 * np.pi * float(f_mnp)

    max_pix = min(fit_pixels, iy0 - 1 if axis == "y" else ix0 - 1)

    vals = []

    for dp in range(-max_pix, max_pix + 1):
        if dp == 0:
            continue

        if axis == "y":
            r = dp * dy
            line = Ez[ix0, iy0 + dp, :]
        elif axis == "x":
            r = dp * dx
            line = Ez[ix0 + dp, iy0, :]
        else:
            raise ValueError("axis must be 'x' or 'y'")

        V = accelerating_voltage_complex(line, z_m, omega, beta=beta)
        vals.append((r, V))

    r = np.array([v[0] for v in vals], float)
    Vc = np.array([v[1] for v in vals], complex)

    order = np.argsort(r)
    r = r[order]
    Vc = Vc[order]

    # ------------------------------------------------------------------
    # Method 1: direct PW gradient fit
    # ------------------------------------------------------------------
    gr = np.polyfit(r, Vc.real, 1)[0]
    gi = np.polyfit(r, Vc.imag, 1)[0]

    dVz_dr = gr + 1j * gi

    Vperp_per_m_offset = (C0 / omega) * dVz_dr
    kick_pw_fit = abs(Vperp_per_m_offset)

    # Local PW estimate at each offset.
    dVz_dr_local = np.gradient(Vc, r)
    kick_pw_local = np.abs((C0 / omega) * dVz_dr_local)

    # ------------------------------------------------------------------
    # Method 2: Ez(r) -> Vz(r) -> loss -> kick-equivalent
    # ------------------------------------------------------------------
    Vz_abs = np.abs(Vc)

    loss = Vz_abs**2 / 4.0

    kick_loss_equiv = np.full_like(Vz_abs, np.nan, dtype=float)

    nonzero = ~np.isclose(r, 0.0)

    # Since loss = |Vz|^2 / 4, then |Vz| = 2 sqrt(loss).
    # For a dipole near-axis Vz ~ r dVz/dr, so:
    # kick = (c/omega) |dVz/dr| ~ (c/omega) |Vz|/|r|.
    kick_loss_equiv[nonzero] = (C0 / omega) * Vz_abs[nonzero] / np.abs(r[nonzero])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    if save_directory is not None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        tag = f"{label}_" if label else ""

        np.savez_compressed(
            save_directory / f"{tag}kick_diagnostics.npz",
            r_m=r,
            Vz_complex=Vc,
            Vz_abs=Vz_abs,
            loss=loss,
            dVz_dr=dVz_dr,
            dVz_dr_local=dVz_dr_local,
            kick_pw_fit=kick_pw_fit,
            kick_pw_local=kick_pw_local,
            kick_loss_equiv=kick_loss_equiv,
            transverse_pixel_m=dr,
            longitudinal_pixel_m=L / (nz - 1),
            f_mnp=f_mnp,
            omega=omega,
            axis=axis,
        )

        # Ez transverse and longitudinal diagnostic.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        im0 = axes[0].imshow(Ez[:, :, nz // 2].T, origin="lower", aspect="equal")
        axes[0].set_title("Ez transverse mid")
        axes[0].set_xlabel("x pixel")
        axes[0].set_ylabel("y pixel")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(Ez[ix0, :, :], origin="lower", aspect="auto")
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
        ax.set_ylabel(r"$V_z$ [V]")
        ax.set_title(f"{label}: complex Vz fit")
        ax.legend()

        fig.savefig(save_directory / f"{tag}Vz_complex_fit.png", dpi=220)
        plt.close(fig)

        # |Vz|
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

        ax.plot(r * 1e3, Vz_abs, "o-")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"$|V_z|$ [V]")
        ax.set_title(f"{label}: offset voltage")

        fig.savefig(save_directory / f"{tag}r_vs_Vz_abs.png", dpi=220)
        plt.close(fig)

        # Loss.
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

        ax.plot(r * 1e3, loss, "o-")
        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"$|V_z|^2/4$")
        ax.set_title(f"{label}: loss-like offset scan")

        fig.savefig(save_directory / f"{tag}r_vs_loss.png", dpi=220)
        plt.close(fig)

        # Kick comparison.
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

        ax.plot(r * 1e3, kick_pw_local, "o-", label="PW local gradient")
        ax.plot(r * 1e3, kick_loss_equiv, "s-", label=r"$V_z \rightarrow$ loss equivalent")
        ax.axhline(kick_pw_fit, color="k", ls="--", alpha=0.6, label="PW linear fit")

        ax.axvline(0.0, color="k", alpha=0.3)
        ax.set_xlabel("Offset [mm]")
        ax.set_ylabel(r"Kick [$\mathrm{V/C/m^2}$]")
        ax.set_title(f"{label}: kick comparison")
        ax.legend()

        fig.savefig(save_directory / f"{tag}r_vs_kick_comparison.png", dpi=220)
        plt.close(fig)

    return {
        "r_m": r,
        "Vz_complex_V": Vc,
        "Vz_abs_V": Vz_abs,
        "loss": loss,
        "dVz_dr_V_per_m": dVz_dr,
        "dVz_dr_local_V_per_m": dVz_dr_local,
        "Vperp_per_m_offset_V_per_m": Vperp_per_m_offset,
        "kick_V_per_C_per_m_per_m": kick_pw_fit,
        "kick_pw_local_V_per_C_per_m_per_m": kick_pw_local,
        "kick_loss_equiv_V_per_C_per_m_per_m": kick_loss_equiv,
        "transverse_pixel_m": dr,
        "longitudinal_pixel_m": L / (nz - 1),
        "axis": axis,
        "f_mnp_Hz": float(f_mnp),
        "omega_rad_s": omega,
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