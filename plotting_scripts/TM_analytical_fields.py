from functools import lru_cache
from typing import Tuple, Dict, Union
# from mode_identification import analytical_comparison as ac
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from functools import lru_cache

"""
SEE Wangler, page 30-31 FOR TM MODES

"""

C0 = 299_792_458.0  # speed of light (m/s)



# --- Roots ---
@lru_cache(maxsize=None)
def tm_root_v_mn(m: int, n: int) -> float:
    """
    TM: v_mn = n-th positive root of J_m(x) = 0  (n starts at 1)
    """
    if m < 0 or n < 1:
        raise ValueError("Require m >= 0 and n >= 1.")
    return special.jn_zeros(m, n)[-1]


@lru_cache(maxsize=None)
def te_root_vprime_mn(m: int, n: int) -> float:
    """
    TE: v'_mn = n-th positive root of J'_m(x) = 0  (n starts at 1)

    Implemented using SciPy's jnp_zeros (zeros of derivative of J_m).
    """
    if m < 0 or n < 1:
        raise ValueError("Require m >= 0 and n >= 1.")
    return special.jnp_zeros(m, n)[-1]


# --- Frequencies ---
def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    """
    TM_mnp frequency (Hz):
      f = (c / 2π) * sqrt( (v_mn / R)^2 + (pπ / L)^2 )
    where R is cavity radius.
    """
    if R <= 0 or L <= 0:
        raise ValueError("a and L must be > 0.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    v = tm_root_v_mn(m, n)
    return (c / (2 * np.pi)) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2)


def f_te(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    """
    TE_mnp frequency (Hz):
      f = (c / 2π) * sqrt( (v'_mn / R)^2 + (pπ / L)^2 )
    where R is cavity radius.
    """
    if R <= 0 or L <= 0:
        raise ValueError("a and L must be > 0.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    vprime = te_root_vprime_mn(m, n)
    return (c / (2 * np.pi)) * np.sqrt((vprime / R) ** 2 + (p * np.pi / L) ** 2)


# ---


@lru_cache(maxsize=None)
def besselj_zero(m: int, n: int) -> float:
    """
    n-th positive zero of J_m(x), with n starting at 1.
    """
    if m < 0 or n < 1:
        raise ValueError("Require m >= 0 and n >= 1.")
    return special.jn_zeros(m, n)[-1]

def k_mn(m: int, n: int, R: float) -> float:
    """
    Transverse wavenumber k_mn = x_{m,n} / R, where x_{m,n} is n-th root of J_m.
    """
    if R <= 0:
        raise ValueError("R must be > 0.")
    return besselj_zero(m, n) / R

def k_z(p: int, L_cav: float) -> float:
    """
    Longitudinal wavenumber k_z = pi * p / L_cav
    """
    if L_cav <= 0:
        raise ValueError("L_cav must be > 0.")
    return np.pi * p / L_cav

def omega(m: int, n: int, p: int, R: float, L_cav: float, c: float = C0) -> float:
    """
    Resonant angular frequency omega = c*sqrt(k_mn^2 + k_z^2)
    """
    kmn = k_mn(m, n, R)
    kz = k_z(p, L_cav)
    return c * np.sqrt(kmn**2 + kz**2)

def E_field_cyl(
    r, theta, z,
    m: int, n: int, p: int,
    R: float, L_cav: float,
    E0: float = 1.0
):
    """
    Returns (E_r, E_theta, E_z) in cylindrical coordinates for the mode (m,n,p),
    using the proportionalities exactly as in the provided equations.

    Inputs r, theta, z can be scalars or numpy arrays (broadcasting supported).
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)

    kmn = k_mn(m, n, R)
    kz = k_z(p, L_cav)

    # Bessel terms
    x = kmn * r
    Jm = special.jv(m, x)
    # derivative J'_m(x): scipy.special.jvp(m, x, 1)
    Jm_prime = special.jvp(m, x, 1)

    # Angular terms
    cos_mth = np.cos(m * theta)
    sin_mth = np.sin(m * theta)

    # z terms (as shown)
    cos_kzz = np.cos(kz * z)
    sin_kzz = np.sin(kz * z)

    # Field components (mode shape)
    Ez = E0 * (Jm * cos_mth * cos_kzz)
    Er = E0 * (p * Jm_prime * cos_mth * sin_kzz)

    # Handle p/r factor safely at r=0
    with np.errstate(divide="ignore", invalid="ignore"):
        Eth = E0 * ((p / r) * Jm * sin_mth * sin_kzz)
        Eth = np.where(r == 0.0, 0.0, Eth)  # finite, consistent limit

    return Er, Eth, Ez

def E_field_cartesian(
    x, y, z,
    m: int, n: int, p: int,
    R: float, L_cav: float,
    E0: float = 1.0
):
    """
    Convenience: compute E in Cartesian components (E_x, E_y, E_z) by converting
    cylindrical (E_r, E_theta, E_z).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    Er, Eth, Ez = E_field_cyl(r, theta, z, m, n, p, R, L_cav, E0=E0)

    # Cyl -> Cart:
    Ex = Er * np.cos(theta) - Eth * np.sin(theta)
    Ey = Er * np.sin(theta) + Eth * np.cos(theta)
    return Ex, Ey, Ez




@lru_cache(maxsize=None)
def _besselj_zero(m: int, n: int) -> float:
    """n-th positive zero of J_m(x), with n starting at 1."""
    return special.jn_zeros(m, n)[-1]

def _k_mn(m: int, n: int, R: float) -> float:
    return _besselj_zero(m, n) / R

def _k_z(p: int, L: float) -> float:
    return np.pi * p / L

def _E_field_cyl(r, theta, z, m: int, n: int, p: int, R: float, L: float, E0: float = 1.0):
    """
    Mode-shape fields (arb. units) matching the user's equations:
      Ez ∝ J_m(k_mn r) cos(mθ) cos(kz z)
      Er ∝ p J'_m(k_mn r) cos(mθ) sin(kz z)
      Eθ ∝ (p/r) J_m(k_mn r) sin(mθ) sin(kz z)
    """
    kmn = _k_mn(m, n, R)
    kz = _k_z(p, L)

    x = kmn * r
    Jm = special.jv(m, x)
    Jm_prime = special.jvp(m, x, 1)  # d/dx J_m(x)

    cos_mth = np.cos(m * theta)
    sin_mth = np.sin(m * theta)
    cos_kzz = np.cos(kz * z)
    sin_kzz = np.sin(kz * z)

    Ez = E0 * (Jm * cos_mth * cos_kzz)
    Er = E0 * (p * Jm_prime * cos_mth * sin_kzz)

    with np.errstate(divide="ignore", invalid="ignore"):
        Eth = E0 * ((p / r) * Jm * sin_mth * sin_kzz)
        Eth = np.where(r == 0.0, 0.0, Eth)

    return Er, Eth, Ez

def plot_yz_plane_field(
    m: int, n: int, p: int,
    R: float, L: float,
    x0: float = 0.0,
    Ny: int = 301,
    Nz: int = 301,
    E0: float = 1.0,
    what: str = "|E|",
    transpose=False,
):
    """
    Plot the YZ plane at a fixed x = x0 (default 0: central x coordinate).

    Default plotted quantity: |E| = sqrt(Ex^2 + Ey^2 + Ez^2)

    Parameters
    ----------
    m, n, p : int
        Mode indices.
    R, L : float
        Cavity radius and length (meters).
    x0 : float
        The x-coordinate for the YZ slice (meters). Default: 0.0.
    Ny, Nz : int
        Grid resolution along y and z.
    E0 : float
        Overall field scale (arb. units).
    what : str
        One of: "|E|" (default), "Ex", "Ey", "Ez", "Eperp"

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if abs(x0) > R:
        raise ValueError("x0 must satisfy |x0| <= R to intersect the cavity.")

    y = np.linspace(-R, R, Ny)
    z = np.linspace(0.0, L, Nz)
    Y, Z = np.meshgrid(y, z, indexing="xy")  # shapes (Nz, Ny)

    # Convert to cylindrical coordinates at each (x0, y)
    r = np.sqrt(x0**2 + Y**2)
    theta = np.arctan2(Y, x0)  # y,x

    # Mask points outside cylinder (r > R)
    mask = r <= R

    Er, Eth, Ez = _E_field_cyl(r, theta, Z, m, n, p, R, L, E0=E0)

    # Cyl -> Cart (at given theta)
    Ex = Er * np.cos(theta) - Eth * np.sin(theta)
    Ey = Er * np.sin(theta) + Eth * np.cos(theta)

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    key = what.strip().lower()
    if key in ["|e|", "e", "emag", "mag", "magnitude"]:
        F = Emag
        label = r"$|E|$ (arb. units)"
        title = r"$|E(y,z)|$"
    elif key == "ex":
        F = Ex
        label = r"$E_x$ (arb. units)"
        title = r"$E_x(y,z)$"
    elif key == "ey":
        F = Ey
        label = r"$E_y$ (arb. units)"
        title = r"$E_y(y,z)$"
    elif key == "ez":
        F = Ez
        label = r"$E_z$ (arb. units)"
        title = r"$E_z(y,z)$"
    elif key in ["eperp", "e_perp", "transverse"]:
        F = Eperp
        label = r"$|E_\perp|$ (arb. units)"
        title = r"$|E_\perp(y,z)|$"
    else:
        raise ValueError('what must be one of: "|E|", "Ex", "Ey", "Ez", "Eperp"')

    # Apply mask
    Fm = np.where(mask, F, np.nan)

    if transpose:
        Fm = Fm.T

    fig, ax = plt.subplots()
    im = ax.imshow(
        Fm,
        extent=[-R, R, 0.0, L],
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label=label)

    if transpose:
        ax.set_ylabel("y (m)")
        ax.set_xlabel("z (m)")
    else:
        ax.set_xlabel("y (m)")
        ax.set_ylabel("z (m)")

    ax.set_title(f"{title} at x={x0:.4g} m   (m={m}, n={n}, p={p})")

    # Draw cylinder boundary in YZ slice: |y| <= sqrt(R^2 - x0^2)
    y_lim = np.sqrt(max(R**2 - x0**2, 0.0))
    ax.plot([ y_lim,  y_lim], [0, L], linewidth=1)
    ax.plot([-y_lim, -y_lim], [0, L], linewidth=1)

    plt.tight_layout()
    return fig, ax, Fm


def plot_transverse_plane_field(
    m: int, n: int, p: int,
    R: float, L: float,
    z: float | None = None,
    N: int = 301,
    E0: float = 1.0,
    what: str = "|E|",
    vectors: bool = False,
    vector_stride: int = 12,
    quiver_scale: float | None = None,
    vector_what: str = "Eperp",
):
    """
    Plot the transverse (x,y) plane at z (default z=L/2).

    Plottable scalars via `what`:
      "|E|" (default), "Ex", "Ey", "Ez", "Eperp"

    Optional vector overlay (`vectors=True`) in the transverse plane:
      vector_what="Eperp" (default) -> arrows are (Ex, Ey)
      vector_what="ErEth"           -> arrows are (Er, Eθ) projected into (x,y) (same result as Ex,Ey),
                                       kept for explicitness.

    Returns (fig, ax).
    """
    if z is None:
        z = L / 2.0

    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    mask = r <= R

    Er, Eth, Ez = _E_field_cyl(r, theta, z, m, n, p, R, L, E0=E0)

    # Cyl -> Cart
    Ex = Er * np.cos(theta) - Eth * np.sin(theta)
    Ey = Er * np.sin(theta) + Eth * np.cos(theta)

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    key = what.strip().lower()
    if key in ["|e|", "e", "emag", "mag", "magnitude"]:
        F = Emag
        label = r"$|E|$ (arb. units)"
        title = r"$|E(x,y)|$"
    elif key == "ex":
        F = Ex
        label = r"$E_x$ (arb. units)"
        title = r"$E_x(x,y)$"
    elif key == "ey":
        F = Ey
        label = r"$E_y$ (arb. units)"
        title = r"$E_y(x,y)$"
    elif key == "ez":
        F = Ez
        label = r"$E_z$ (arb. units)"
        title = r"$E_z(x,y)$"
    elif key in ["eperp", "e_perp", "transverse"]:
        F = Eperp
        label = r"$|E_\perp|$ (arb. units)"
        title = r"$|E_\perp(x,y)|$"
    else:
        raise ValueError('what must be one of: "|E|", "Ex", "Ey", "Ez", "Eperp"')

    Fm = np.where(mask, F, np.nan)

    fig, ax = plt.subplots()
    im = ax.imshow(
        Fm,
        extent=[-R, R, -R, R],
        origin="lower",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label=label)

    # Optional vector overlay
    if vectors:
        s = vector_stride

        vkey = vector_what.strip().lower()
        if vkey in ["eperp", "exey", "xy", "cart"]:
            U = Ex
            V = Ey
        elif vkey in ["ereth", "cyl"]:
            # Convert (Er,Eθ) to (Ex,Ey) for plotting arrows in the xy plane
            U = Ex
            V = Ey
        else:
            raise ValueError('vector_what must be "Eperp" (Ex,Ey arrows) or "ErEth"')

        Um = np.where(mask, U, np.nan)
        Vm = np.where(mask, V, np.nan)

        ax.quiver(
            X[::s, ::s], Y[::s, ::s],
            Um[::s, ::s], Vm[::s, ::s],
            scale=quiver_scale
        )

    # Draw cavity boundary
    t = np.linspace(0, 2*np.pi, 600)
    ax.plot(R*np.cos(t), R*np.sin(t), linewidth=1)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title} at z={z:.4g} m   (m={m}, n={n}, p={p})")

    plt.tight_layout()
    return fig, ax, Fm

def pillbox_freq_from_radius(radius, GHz=False):
    """

    :param radius:
    :return:
    """
    radius = float(radius)
    csol = 299792458.0
    f_Hz = (2.4048 * csol) / (2.0 * np.pi * radius)
    f_GHz = f_Hz / 1e9
    if GHz:
        return f_GHz
    else:
        return f_Hz


def pillbox_radius_from_freq(f_Hz):
    f_Hz = float(f_Hz)
    csol = 299792458.0
    radius_m = (2.4048 * csol) / (2.0 * np.pi * f_Hz)

    return radius_m


def pillbox_field_voxel_grid_xyz(
    R: float, L: float,
    m: int, n: int, p: int,
    x_res: int, y_res: int, z_res: int,
    E0: float = 1.0,
    quantity: str = "|E|",
    z_range: Tuple[float, float] = (0.0, 1.0),
    return_components: bool = False,
    dtype=np.float32,
) -> Union[
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
]:
    """
    Returns a Cartesian voxel grid whose indexing matches: out[x, y, z].

    - Output array shape: (x_res, y_res, z_res)
    - Coordinate arrays: (x_coords, y_coords, z_coords) where
        x_coords[i] corresponds to out[i, :, :]
        y_coords[j] corresponds to out[:, j, :]
        z_coords[k] corresponds to out[:, :, k]

    Grid spans:
      x in [-R, +R]
      y in [-R, +R]
      z in [z_range[0]*L, z_range[1]*L]  (default 0..L)

    Points outside the cylinder (sqrt(x^2+y^2) > R) are set to NaN.
    """
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be > 0.")
    if any(k < 2 for k in (x_res, y_res, z_res)):
        raise ValueError("x_res, y_res, z_res must be >= 2.")

    z0 = float(z_range[0]) * L
    z1 = float(z_range[1]) * L
    if not (0.0 <= z_range[0] <= 1.0 and 0.0 <= z_range[1] <= 1.0 and z0 < z1):
        raise ValueError("z_range must be within [0,1] with z_range[0] < z_range[1].")

    # Voxel centers (1D coordinate arrays)
    x_coords = np.linspace(-R, R, x_res, dtype=float)
    y_coords = np.linspace(-R, R, y_res, dtype=float)
    z_coords = np.linspace(z0, z1, z_res, dtype=float)

    # IMPORTANT: indexing='ij' yields arrays shaped (x_res, y_res, z_res)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = r <= R

    Er, Eth, Ez = _E_field_cyl(r, theta, Z, m, n, p, R, L, E0=E0)

    # Cyl -> Cart
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Ex = Er * cos_t - Eth * sin_t
    Ey = Er * sin_t + Eth * cos_t

    Eperp = np.sqrt(Ex**2 + Ey**2)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # Mask outside cavity with NaN
    nan = np.nan
    Exm = np.where(mask, Ex, nan).astype(dtype, copy=False)
    Eym = np.where(mask, Ey, nan).astype(dtype, copy=False)
    Ezm = np.where(mask, Ez, nan).astype(dtype, copy=False)
    Eperpm = np.where(mask, Eperp, nan).astype(dtype, copy=False)
    Emagm = np.where(mask, Emag, nan).astype(dtype, copy=False)

    if return_components:
        data = {"Ex": Exm, "Ey": Eym, "Ez": Ezm, "Eperp": Eperpm, "|E|": Emagm}
        return data, (x_coords, y_coords, z_coords)

    key = quantity.strip().lower()
    if key in ["|e|", "e", "emag", "mag", "magnitude"]:
        out = Emagm
    elif key == "ex":
        out = Exm
    elif key == "ey":
        out = Eym
    elif key == "ez":
        out = Ezm
    elif key in ["eperp", "e_perp", "transverse"]:
        out = Eperpm
    else:
        raise ValueError('quantity must be one of: "|E|", "Ex", "Ey", "Ez", "Eperp"')

    return out, (x_coords, y_coords, z_coords)

def save_out_analytical_fields_for_CST_classification(m_max, n_max, p_max, savepath, cubic_res=21):
    for m in range(3,4):
        for n in range(1, n_max):
            for p in range(p_max):

                print(f"{m}{n}{p}")
                data, coords = pillbox_field_voxel_grid_xyz(
                    R=R,
                    L=L,
                    m=m, n=n, p=p,
                    x_res=cubic_res, y_res=cubic_res, z_res=cubic_res,
                    E0=1.0,
                    quantity="|E|",
                    z_range=(0.0, 1.0),
                    return_components=True,
                    dtype=np.float32,
                )
                # : Exm, "Ey": Eym, "Ez": Ezm, "Eperp": Eperpm, "|E|": Emagm}
                Ex = data["Ex"]
                Ey = data["Ey"]
                Ez = data["Ez"]
                Eabs = data["|E|"]
                xs = coords[0]
                ys = coords[1]
                zs = coords[2]

                Ex_rot, Ey_rot, Ez_rot, Eabs_rot, xs, ys, zs, angle_deg = ac.cylindrical_rotate_align_and_plot(
                    Ex, Ey, Ez, Eabs, xs, ys, zs)

                if m == 3 and n == 1 and p == 0:
                    fig, ax = plt.subplots(1, 2, sharey=True)
                    ax[0].imshow(Eabs_rot[:, :, 10], origin="lower")
                    ax[1].imshow(Eabs_rot[10, :, :], origin="lower")
                    plt.show()

                np.save(f"{savepath}\\TM{m}{n}{p}.npy", Eabs_rot)


if __name__ == "__main__":

    savepath = fr"D:\PhD\HOMmix\pillbox\analysis\ALL_modes\TM_analytical_study\TM012_TM020_degenerate_param_sweep"
    savepath_field_maps = fr"D:\PhD\HOMmix\pillbox\analysis\ALL_modes\TM_analytical_study\Eabs_analytical_fields_for_CST_classification"
    csol = 299792458.0
    frequency_010 = 1.3e9
    lambda_010 = csol/frequency_010
    R = pillbox_radius_from_freq(frequency_010)
    LF_start = 0.96
    LF_stop = 0.98
    LF_start_log = 1.e-5
    LF_stop_log = 1.e2
    param_sweep_resolution = 10000
    length_factor_vector = np.linspace(LF_start, LF_stop, param_sweep_resolution, endpoint=True)
    # length_factor_vector = np.logspace(LF_start_log, LF_stop_log, param_sweep_resolution, endpoint=True)

    # for lidx, length_factor in enumerate(length_factor_vector):
    length_factor = 1.0 # 0.969
    L = length_factor * lambda_010 / 2.




    # save_out_analytical_fields_for_CST_classification(m_max=4, n_max=3, p_max=3, savepath=savepath_field_maps, cubic_res=21)


    # exit()

    m = 1
    n = 2
    p = 1

    # Example usage:
    fig, ax, field_TM012 = plot_yz_plane_field(m=m, n=n, p=p, R=R, L=L, what="|E|", transpose=True)
    plt.show()
    # plt.savefig(f"{savepath}\\{lidx}_TM012.png")
    # plt.close("all")
    # default |E|
    # fig, ax, field_TM020 = plot_yz_plane_field(m=0, n=2, p=0, R=R, L=L, what="|E|", transpose=True)
    # plt.show()
    # plt.savefig(f"{savepath}\\{lidx}_TM020.png")
    # plt.close("all")

    # i, j = np.unravel_index(field_TM012.argmax(), field_TM012.shape)
    # print(f"{lidx+1}/{[param_sweep_resolution]} ({L = }):{i} {j} = {field_TM012[i, j]}")

    # # Example usage:
    fig, ax, FmT_TM012= plot_transverse_plane_field(m=m, n=n, p=p, R=R, L=L, what="|E|")
    # fig, ax, FmT_TM020 = plot_transverse_plane_field(m=0, n=2, p=0, R=R, L=L, what="|E|")
    # fig, ax = plot_transverse_plane_field(m=1, n=1, p=1, R=0.05, L=0.10, what="|E|", vectors=True)
    plt.show()

