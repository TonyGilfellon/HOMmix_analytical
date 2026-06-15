from pyparsing import results
import HOMmix_analytical_master_module as hamm
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.patches import Rectangle
import numpy as np
import pickle as pkl
from dataclasses import dataclass
from dataclasses import asdict
from scipy.interpolate import RegularGridInterpolator, CubicSpline

C_LIGHT = 299_792_458.0


@dataclass
class QuadFocusResult:
    kx_rad_per_m: float
    ky_rad_per_m: float
    gx_V_per_m2: float
    gy_V_per_m2: float
    cxx: float
    cyy: float
    cxy: float
    quad_balance: float
    fit_radius_m: float


def load_ez_pickle(path):
    with open(path, "rb") as f:
        Ez = pickle.load(f)

    Ez = np.asarray(Ez, dtype=float)

    if Ez.ndim != 3:
        raise ValueError(f"Expected 3D Ez array, got shape {Ez.shape}")

    return Ez


def make_pillbox_axes(Ez, R_m, L_m):
    """
    Assumes Ez shape is (nx, ny, nz), spanning:
        x = [-R, R]
        y = [-R, R]
        z = [-L/2, L/2]
    """
    nx, ny, nz = Ez.shape

    x = np.linspace(-R_m, R_m, nx)
    y = np.linspace(-R_m, R_m, ny)
    z = np.linspace(-L_m / 2, L_m / 2, nz)

    return x, y, z


def integrated_longitudinal_voltage(Ez, z_m):
    """
    Vz(x,y) = integral Ez(x,y,z) dz

    NaNs are treated as outside the cavity and ignored by setting them to zero.
    """
    Ez_clean = np.nan_to_num(Ez, nan=0.0)
    return np.trapezoid(Ez_clean, z_m, axis=2)


def fit_quadratic_vz_near_axis(Vz_xy, x_m, y_m, fit_radius_m):
    """
    Fits near-axis integrated voltage:

        Vz = c0 + cx x + cy y + cxx x^2 + cxy xy + cyy y^2

    For an ideal quadrupole:

        cxx ~= -cyy
        cxy ~= 0, depending on quadrupole orientation
    """
    X, Y = np.meshgrid(x_m, y_m, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    mask = (
        np.isfinite(Vz_xy)
        & (R <= fit_radius_m)
    )

    if np.count_nonzero(mask) < 6:
        raise ValueError("Not enough points inside fit radius for quadratic fit.")

    x = X[mask]
    y = Y[mask]
    v = Vz_xy[mask]

    A = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        x*y,
        y**2,
    ])

    coeffs, *_ = np.linalg.lstsq(A, v, rcond=None)

    return {
        "c0": coeffs[0],
        "cx": coeffs[1],
        "cy": coeffs[2],
        "cxx": coeffs[3],
        "cxy": coeffs[4],
        "cyy": coeffs[5],
    }


def extract_quad_angular_focusing_strengths(
    Ez,
    *,
    R_m=0.08826254491486922,
    L_m=0.11530479153846154,
    mode_frequency_Hz,
    beam_energy_eV,
    fit_radius_m=0.02,
):
    """
    Extracts quadrupole-like angular focusing strengths:

        x' = kx x
        y' = ky y

    Returned kx, ky units:
        rad / m

    Also returns integrated kick-gradient-like quantities:

        gx, gy in V / m^2

    Panofsky-Wenzel relation used:

        Vx = -(c / omega) dVz/dx
        Vy = -(c / omega) dVz/dy

    Therefore:

        gx = dVx/dx = -(c / omega) d2Vz/dx2
        gy = dVy/dy = -(c / omega) d2Vz/dy2

    and:

        kx = gx / beam_energy_eV
        ky = gy / beam_energy_eV
    """
    x, y, z = make_pillbox_axes(Ez, R_m, L_m)

    Vz_xy = integrated_longitudinal_voltage(Ez, z)

    coeffs = fit_quadratic_vz_near_axis(
        Vz_xy,
        x,
        y,
        fit_radius_m,
    )

    omega = 2 * np.pi * mode_frequency_Hz

    d2Vz_dx2 = 2.0 * coeffs["cxx"]
    d2Vz_dy2 = 2.0 * coeffs["cyy"]

    gx = -(C_LIGHT / omega) * d2Vz_dx2
    gy = -(C_LIGHT / omega) * d2Vz_dy2

    kx = gx / beam_energy_eV
    ky = gy / beam_energy_eV

    quad_balance = (kx + ky) / max(abs(kx), abs(ky), 1e-300)

    return QuadFocusResult(
        kx_rad_per_m=kx,
        ky_rad_per_m=ky,
        gx_V_per_m2=gx,
        gy_V_per_m2=gy,
        cxx=coeffs["cxx"],
        cyy=coeffs["cyy"],
        cxy=coeffs["cxy"],
        quad_balance=quad_balance,
        fit_radius_m=fit_radius_m,
    )



@dataclass
class RingMultipoleFit:
    radius_m: float
    a0: float
    a1_cos: float
    b1_sin: float
    a2_cos: float
    b2_sin: float
    theta_rad: np.ndarray
    vz_theta: np.ndarray

    def as_dict(self):
        return asdict(self)

def find_auto_ring_radius_from_peak_abs_vz(
    Ez: np.ndarray,
    *,
    R_m: float = 0.08826254491486922,
    L_m: float = 0.11530479153846154,
    r_max_search_m: float = 0.03,
    n_r: int = 80,
    n_theta: int = 180,
    n_z_interp: int = 101,
    min_radius_m: float | None = None,
):
    """
    Finds the radius where max_theta |Vz(r, theta)| is largest,
    searching only near the axis.

    This avoids choosing a large-radius peak near the cavity wall.
    """
    Ez = np.asarray(Ez, dtype=float)

    if min_radius_m is None:
        # Avoid r=0, where angular decomposition is ill-conditioned.
        dx = 2 * R_m / (Ez.shape[0] - 1)
        min_radius_m = dx

    r_max_search_m = min(r_max_search_m, 0.8 * R_m)

    candidate_radii = np.linspace(min_radius_m, r_max_search_m, n_r)

    best_radius = candidate_radii[0]
    best_value = -np.inf

    for r in candidate_radii:
        theta, vz_theta = sample_vz_on_ring(
            Ez,
            radius_m=r,
            n_theta=n_theta,
            n_z_interp=n_z_interp,
            R_m=R_m,
            L_m=L_m,
            diagnostic_plot=False,
            show=False,
            save_path=None,
        )

        peak_abs = np.nanmax(np.abs(vz_theta))

        if peak_abs > best_value:
            best_value = peak_abs
            best_radius = r

    return float(best_radius), float(best_value)

def load_ez_pickle(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        Ez = pickle.load(f)
    return np.asarray(Ez, dtype=float)


def make_axes_from_pillbox(
    Ez: np.ndarray,
    R_m: float = 0.08826254491486922,
    L_m: float = 0.11530479153846154,
):
    nx, ny, nz = Ez.shape
    x = np.linspace(-R_m, R_m, nx)
    y = np.linspace(-R_m, R_m, ny)
    z = np.linspace(-L_m / 2, L_m / 2, nz)
    return x, y, z


def sample_vz_on_ring(
    Ez: np.ndarray,
    *,
    radius_m: float,
    n_theta: int = 180,
    n_z_interp: int = 101,
    R_m: float = 0.08826254491486922,
    L_m: float = 0.11530479153846154,
    diagnostic_plot: bool = True,
    save_path: str | None = "ring_diagnostic_theta_r.png",
    show: bool = True,
):
    """
    Samples Ez on a transverse ring x=r cos(theta), y=r sin(theta),
    interpolates along z, then integrates to obtain Vz(theta).

    Uses cubic RegularGridInterpolator where possible.
    """
    Ez = np.asarray(Ez, dtype=float)

    if Ez.ndim != 3:
        raise ValueError(f"Expected Ez shape (nx, ny, nz), got {Ez.shape}")

    if radius_m <= 0 or radius_m >= R_m:
        raise ValueError("radius_m must be between 0 and R_m")

    x, y, z = make_axes_from_pillbox(Ez, R_m=R_m, L_m=L_m)

    # Replace NaNs outside pillbox with 0 so interpolation near boundary behaves.
    Ez_clean = np.nan_to_num(Ez, nan=0.0)

    interp = RegularGridInterpolator(
        (x, y, z),
        Ez_clean,
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_fine = np.linspace(z.min(), z.max(), n_z_interp)

    X_ring = radius_m * np.cos(theta)
    Y_ring = radius_m * np.sin(theta)

    # Points shape: (n_theta * n_z_interp, 3)
    TH, ZZ = np.meshgrid(theta, z_fine, indexing="ij")
    XX = radius_m * np.cos(TH)
    YY = radius_m * np.sin(TH)

    points = np.column_stack([
        XX.ravel(),
        YY.ravel(),
        ZZ.ravel(),
    ])

    Ez_ring_z = interp(points).reshape(n_theta, n_z_interp)

    # Vz(theta) = integral Ez dz
    vz_theta = np.trapezoid(Ez_ring_z, z_fine, axis=1)

    if diagnostic_plot:
        save_theta_r_diagnostic_plot(
            Ez,
            radius_m=radius_m,
            theta_samples=theta,
            R_m=R_m,
            L_m=L_m,
            save_path=save_path,
            show=show,
        )

    return theta, vz_theta


def fit_ring_multipoles_up_to_m2(theta_rad, vz_theta, *, radius_m):
    """
    Fits:

        Vz(theta) =
            a0
          + a1 cos(theta) + b1 sin(theta)
          + a2 cos(2 theta) + b2 sin(2 theta)

    m=0: monopole
    m=1: dipole
    m=2: quadrupole
    """
    theta_rad = np.asarray(theta_rad, dtype=float)
    vz_theta = np.asarray(vz_theta, dtype=float)

    A = np.column_stack([
        np.ones_like(theta_rad),
        np.cos(theta_rad),
        np.sin(theta_rad),
        np.cos(2.0 * theta_rad),
        np.sin(2.0 * theta_rad),
    ])

    coeffs, *_ = np.linalg.lstsq(A, vz_theta, rcond=None)

    return RingMultipoleFit(
        radius_m=radius_m,
        a0=coeffs[0],
        a1_cos=coeffs[1],
        b1_sin=coeffs[2],
        a2_cos=coeffs[3],
        b2_sin=coeffs[4],
        theta_rad=theta_rad,
        vz_theta=vz_theta,
    )


def save_theta_r_diagnostic_plot(
    Ez: np.ndarray,
    *,
    radius_m: float,
    theta_samples: np.ndarray,
    R_m: float,
    L_m: float,
    save_path: str | None,
    show: bool,
):
    """
    Diagnostic plot showing raw grid points in theta-r space
    and the interpolated ring sample locations.
    """
    nx, ny, nz = Ez.shape
    x, y, _ = make_axes_from_pillbox(Ez, R_m=R_m, L_m=L_m)

    X, Y = np.meshgrid(x, y, indexing="ij")
    r_raw = np.sqrt(X**2 + Y**2)
    theta_raw = np.mod(np.arctan2(Y, X), 2.0 * np.pi)

    valid_any_z = np.any(np.isfinite(Ez), axis=2)

    theta_ring = np.mod(theta_samples, 2.0 * np.pi)
    r_ring = np.full_like(theta_ring, radius_m)

    plt.figure(figsize=(8, 4.5))
    plt.scatter(
        theta_raw[valid_any_z],
        r_raw[valid_any_z],
        s=8,
        alpha=0.45,
        label="raw x-y grid points",
    )
    plt.scatter(
        theta_ring,
        r_ring,
        s=18,
        marker="x",
        label="interpolated ring samples",
    )
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$r$ [m]")
    plt.title("Raw grid and interpolated ring in theta-r plane")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

def extract_ring_multipole_coefficients_m0_m1_m2(
    Ez: np.ndarray,
    *,
    radius_m: float | str = "auto",
    n_theta: int = 180,
    n_z_interp: int = 101,
    R_m: float = 0.08826254491486922,
    L_m: float = 0.11530479153846154,
    auto_r_max_search_m: float = 0.03,
    auto_n_r: int = 80,
    diagnostic_plot: bool = True,
    save_path: str | None = "ring_diagnostic_theta_r.png",
    show: bool = True,
):
    """
    If radius_m="auto", finds a near-axis radius where |Vz| is largest
    after interpolation, then samples that ring.
    """
    n_theta = int(min(max(n_theta, 16), 720))
    n_z_interp = int(min(max(n_z_interp, 21), 401))

    if radius_m == "auto":
        radius_m, peak_abs_vz = find_auto_ring_radius_from_peak_abs_vz(
            Ez,
            R_m=R_m,
            L_m=L_m,
            r_max_search_m=auto_r_max_search_m,
            n_r=auto_n_r,
            n_theta=n_theta,
            n_z_interp=n_z_interp,
        )
        print(f"Auto-selected radius: {radius_m:.6e} m")
        print(f"Peak |Vz| on selected ring: {peak_abs_vz:.6e}")

    theta, vz_theta = sample_vz_on_ring(
        Ez,
        radius_m=float(radius_m),
        n_theta=n_theta,
        n_z_interp=n_z_interp,
        R_m=R_m,
        L_m=L_m,
        diagnostic_plot=diagnostic_plot,
        save_path=save_path,
        show=show,
    )

    return fit_ring_multipoles_up_to_m2(
        theta,
        vz_theta,
        radius_m=float(radius_m),
    )

if __name__ == "__main__":
    savepath = r"D:\PhD\HOMmix\HOMmix_analytical\analysis\initial_quadrupole_analysis"
    csol = 299_792_458.0  # speed of light (m/s)
    frequency_010 = 1.3e9
    lambda_010 = csol / frequency_010
    R = hamm.pillbox_radius_from_freq(frequency_010)

    print(f"{R = } & {lambda_010/2. = }")

    m = 1
    n = 1
    p = 1
    x_res = 51
    y_res = 51
    z_res = 26

    TM210_field_maps = hamm.pillbox_field_voxel_grid_xyz(
        R=R,
        L=lambda_010 / 2.,
        m=m,
        n=n,
        p=p,
        x_res=x_res,
        y_res=y_res,
        z_res=z_res,
        E0=1.0,
        mode="TM",
        z_range=(0.0, 1.0),
        dtype=np.float32,
    )

    """
    return {
        "Ex": Exm,
        "Ey": Eym,
        "Ez": Ezm,
        "Eperp": Eperpm,
        "|E|": Emagm,
    }
    """

    # Ez_vert_sec = TM210_field_maps["Ez"][21, :, :]
    Ez = TM210_field_maps["Ez"]

    # plt.imshow(Ez_vert_sec)
    # plt.show()

    # hamm.pickle_save(Ez_vert_sec, f"{savepath}\\Ez_vert_sec.pkl")
    # hamm.pickle_save(Ez, f"{savepath}\\Ez.pkl")

    result = extract_quad_angular_focusing_strengths(
        Ez,
        mode_frequency_Hz=1.3e9,  # replace with actual TM210 frequency
        beam_energy_eV=100e6,  # replace with beam energy
        fit_radius_m=0.01,  # 10 mm near-axis fit
    )

    for key, value in asdict(result).items():
        print(f"{key}: {value}")

    # Ez = load_ez_pickle("Ez.pkl")

    fit = extract_ring_multipole_coefficients_m0_m1_m2(
        Ez,
        radius_m="auto",
        auto_r_max_search_m=0.05,  # only search within 30 mm of axis
        auto_n_r=80,
        n_theta=90,
        n_z_interp=z_res,
        diagnostic_plot=False,
        save_path=f"{savepath}\\auto_radius_ring_diagnostic.png",
        show=False,
    )

    for key, value in fit.as_dict().items():
        if not isinstance(value, np.ndarray):
            print(f"{key:12s} = {value:.6e}")

    m0 = fit.a0
    m1_amplitude = np.hypot(fit.a1_cos, fit.b1_sin)
    m2_amplitude = np.hypot(fit.a2_cos, fit.b2_sin)

    norm = max([m0, m1_amplitude, m2_amplitude])
    m0_norm = m0/norm
    m1_norm = m1_amplitude/norm
    m2_norm = m2_amplitude/norm


    print("monopole amplitude   =", m0)
    print("dipole amplitude     =", m1_amplitude)
    print("quadrupole amplitude =", m2_amplitude)

    print("m0_norm   =", m0_norm)
    print("m1_norm     =", m1_norm)
    print("m2_norm =", m2_norm)
