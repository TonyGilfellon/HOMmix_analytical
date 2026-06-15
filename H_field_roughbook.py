from pyparsing import results
import HOMmix_analytical_master_module as hamm
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.patches import Rectangle
import numpy as np
import pickle as pkl
from typing import Tuple
from scipy.special import jv, jvp, jn_zeros, jnp_zeros





C0 = 299_792_458.0
MU0 = 4e-7 * np.pi




def _safe_divide(num, den, fill=0.0):
    out = np.full_like(num, fill, dtype=float)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-30)
    return out


def _E_field_cyl_TM(r, theta, z, m, n, p, R, L, E0=1.0):
    """
    Ideal pillbox TM_mnp electric field in cylindrical coordinates.

    Returns:
        Er, Etheta, Ez

    Convention:
        Ez ~ J_m(k_r r) cos(m theta) cos(k_z z)
    """
    chi = jn_zeros(m, n)[-1]
    k_r = chi / R
    k_z = p * np.pi / L
    k_c2 = k_r**2

    Jm = jv(m, k_r * r)
    Jmp = jvp(m, k_r * r, 1)

    cos_m = np.cos(m * theta)
    sin_m = np.sin(m * theta)

    cos_z = np.cos(k_z * z)
    sin_z = np.sin(k_z * z)

    Ez = E0 * Jm * cos_m * cos_z

    if p == 0:
        Er = np.zeros_like(Ez)
        Eth = np.zeros_like(Ez)
    else:
        Er = -E0 * (k_z * k_r / k_c2) * Jmp * cos_m * sin_z
        Eth = E0 * (k_z * m / k_c2) * _safe_divide(Jm, r) * sin_m * sin_z

    return Er, Eth, Ez


def _E_field_cyl_TE(r, theta, z, m, n, p, R, L, E0=1.0):
    """
    Ideal pillbox TE_mnp electric field in cylindrical coordinates.

    Returns:
        Er, Etheta, Ez

    TE has Ez = 0.

    Here E0 is a transverse-field scale, not an accelerating-field scale.
    """
    chi_p = jnp_zeros(m, n)[-1]
    k_r = chi_p / R
    k_c2 = k_r**2
    k_z = p * np.pi / L

    Jm = jv(m, k_r * r)
    Jmp = jvp(m, k_r * r, 1)

    cos_m = np.cos(m * theta)
    sin_m = np.sin(m * theta)
    sin_z = np.sin(k_z * z)

    Ez = np.zeros_like(r)

    # A convenient TE transverse pattern.
    # For p=0 this gives zero transverse E in a closed pillbox TE mode.
    if p == 0:
        Er = np.zeros_like(r)
        Eth = np.zeros_like(r)
    else:
        Er = E0 * (m / k_c2) * _safe_divide(Jm, r) * sin_m * sin_z
        Eth = E0 * (k_r / k_c2) * Jmp * cos_m * sin_z

    return Er, Eth, Ez

def pillbox_mode_omega(
    R: float,
    L: float,
    m: int,
    n: int,
    p: int,
    mode: str = "TM",
) -> float:
    """
    Angular frequency for ideal pillbox TM_mnp / TE_mnp mode.

    TM: radial root is zero of J_m
    TE: radial root is zero of J_m'
    """
    mode = mode.upper()

    if mode == "TM":
        chi_mn = jn_zeros(m, n)[-1]
    elif mode == "TE":
        chi_mn = jnp_zeros(m, n)[-1]
    else:
        raise ValueError("mode must be 'TM' or 'TE'")

    k_r = chi_mn / R
    k_z = p * np.pi / L

    return C0 * np.sqrt(k_r**2 + k_z**2)


def curl_cartesian_from_grid(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
):
    """
    Curl of vector field F on grid indexed as F[x, y, z].

        curl F =
        (
            dFz/dy - dFy/dz,
            dFx/dz - dFz/dx,
            dFy/dx - dFx/dy
        )
    """
    edge_order = 2 if min(Fx.shape) >= 3 else 1

    dFx_dx, dFx_dy, dFx_dz = np.gradient(
        Fx, x, y, z, edge_order=edge_order
    )
    dFy_dx, dFy_dy, dFy_dz = np.gradient(
        Fy, x, y, z, edge_order=edge_order
    )
    dFz_dx, dFz_dy, dFz_dz = np.gradient(
        Fz, x, y, z, edge_order=edge_order
    )

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy

    return curl_x, curl_y, curl_z


def pillbox_field_voxel_grid_xyz(
    R: float,
    L: float,
    m: int,
    n: int,
    p: int,
    x_res: int,
    y_res: int,
    z_res: int,
    mode: str = "TM",
    E0: float = 1.0,
    z_range: Tuple[float, float] = (0.0, 1.0),
    dtype=np.float32,
    return_complex_H: bool = True,
    phasor_convention: str = "exp(+iwt)",
):
    """
    Returns Cartesian voxel grids indexed as out[x, y, z].

    Includes:
        Ex, Ey, Ez, Eperp, |E|
        Hx, Hy, Hz, Hperp, |H|

    H is obtained from Faraday's law:

        curl(E) = -i omega mu0 H       for exp(+i omega t)

    so:

        H = i curl(E) / (omega mu0)

    If using exp(-i omega t), the sign is reversed.

    Notes:
        - E fields are real-valued mode-shape amplitudes.
        - H fields are complex by default because they are 90 degrees out
          of phase with E in this phasor convention.
        - |H|, Hperp are magnitudes.
    """
    mode = mode.upper()
    if mode not in ("TM", "TE"):
        raise ValueError("mode must be 'TM' or 'TE'")

    if R <= 0 or L <= 0:
        raise ValueError("R and L must be > 0.")

    if any(k < 2 for k in (x_res, y_res, z_res)):
        raise ValueError("x_res, y_res, z_res must be >= 2.")

    z0 = float(z_range[0]) * L
    z1 = float(z_range[1]) * L

    if not (
        0.0 <= z_range[0] <= 1.0
        and 0.0 <= z_range[1] <= 1.0
        and z0 < z1
    ):
        raise ValueError(
            "z_range must be within [0,1] with z_range[0] < z_range[1]."
        )

    x_coords = np.linspace(-R, R, x_res, dtype=float)
    y_coords = np.linspace(-R, R, y_res, dtype=float)
    z_coords = np.linspace(z0, z1, z_res, dtype=float)

    X, Y, Z = np.meshgrid(
        x_coords,
        y_coords,
        z_coords,
        indexing="ij",
    )

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = r <= R

    if mode == "TM":
        Er, Eth, Ez = _E_field_cyl_TM(
            r, theta, Z, m, n, p, R, L, E0=E0
        )
    else:
        Er, Eth, Ez = _E_field_cyl_TE(
            r, theta, Z, m, n, p, R, L, E0=E0
        )

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    Ex = Er * cos_t - Eth * sin_t
    Ey = Er * sin_t + Eth * cos_t

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    omega = pillbox_mode_omega(R, L, m, n, p, mode=mode)
    freq_Hz = omega / (2.0 * np.pi)

    curl_Ex, curl_Ey, curl_Ez = curl_cartesian_from_grid(
        Ex, Ey, Ez, x_coords, y_coords, z_coords
    )

    if phasor_convention == "exp(+iwt)":
        h_factor = 1j / (omega * MU0)
    elif phasor_convention == "exp(-iwt)":
        h_factor = -1j / (omega * MU0)
    else:
        raise ValueError(
            "phasor_convention must be 'exp(+iwt)' or 'exp(-iwt)'"
        )

    Hx = h_factor * curl_Ex
    Hy = h_factor * curl_Ey
    Hz = h_factor * curl_Ez

    if not return_complex_H:
        Hx = np.abs(Hx)
        Hy = np.abs(Hy)
        Hz = np.abs(Hz)

    Hperp = np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2)
    Hmag = np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)

    nan = np.nan

    Exm = np.where(mask, Ex, nan).astype(dtype, copy=False)
    Eym = np.where(mask, Ey, nan).astype(dtype, copy=False)
    Ezm = np.where(mask, Ez, nan).astype(dtype, copy=False)
    Eperpm = np.where(mask, Eperp, nan).astype(dtype, copy=False)
    Emagm = np.where(mask, Emag, nan).astype(dtype, copy=False)

    if return_complex_H:
        complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
        Hxm = np.where(mask, Hx, nan + 0j).astype(complex_dtype, copy=False)
        Hym = np.where(mask, Hy, nan + 0j).astype(complex_dtype, copy=False)
        Hzm = np.where(mask, Hz, nan + 0j).astype(complex_dtype, copy=False)
    else:
        Hxm = np.where(mask, Hx, nan).astype(dtype, copy=False)
        Hym = np.where(mask, Hy, nan).astype(dtype, copy=False)
        Hzm = np.where(mask, Hz, nan).astype(dtype, copy=False)

    Hperpm = np.where(mask, Hperp, nan).astype(dtype, copy=False)
    Hmagm = np.where(mask, Hmag, nan).astype(dtype, copy=False)

    return {
        "Ex": Exm,
        "Ey": Eym,
        "Ez": Ezm,
        "Eperp": Eperpm,
        "|E|": Emagm,
        "Hx": Hxm,
        "Hy": Hym,
        "Hz": Hzm,
        "Hperp": Hperpm,
        "|H|": Hmagm,
        "omega_rad_s": omega,
        "frequency_Hz": freq_Hz,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "mask": mask,
    }

if __name__ == "__main__":


    savepath = r"D:\PhD\HOMmix\HOMmix_analytical\analysis\initial_quadrupole_analysis"
    csol = 299_792_458.0  # speed of light (m/s)
    frequency_010 = 1.3e9
    lambda_010 = csol / frequency_010
    R = hamm.pillbox_radius_from_freq(frequency_010)

    print(f"{R = } & {lambda_010/2. = }")

    m = 0
    n = 1
    p = 0
    x_res = 51
    y_res = 51
    z_res = 25

    field_maps = pillbox_field_voxel_grid_xyz(
        R=R,
        L=lambda_010 / 2.,
        m=m,
        n=n,
        p=p,
        x_res=x_res,
        y_res=y_res,
        z_res=z_res,
        E0=1.0,
        mode="TE",
        z_range=(0.0, 1.0),
        dtype=np.float32,
    )

    Hz_vert_sec = np.real(field_maps['Hz'][12, :, :])
    plt.imshow(Hz_vert_sec, cmap="gray")
    plt.show()

