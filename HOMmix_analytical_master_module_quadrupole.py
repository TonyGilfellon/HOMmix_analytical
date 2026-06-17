"""Stripped helper module for analytical TM m=2 quadrupole crossing analysis.

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


def align_field_to_vertical_plane(field: dict, out_plot: str | None = None, label="") -> dict:
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
    slices = {}
    for k, F in field_data.items():
        if isinstance(F, np.ndarray) and F.ndim == 3:
            midx, midz = F.shape[0]//2, F.shape[2]//2
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


def accelerating_voltage_complex(Ez_line, z_m, omega, beta=1.0):
    zc = np.asarray(z_m, float) - 0.5*(z_m[0] + z_m[-1])
    Ez_line = np.asarray(Ez_line, float)
    return np.trapz(Ez_line * np.exp(1j*omega*zc/(beta*C0)), zc)


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
