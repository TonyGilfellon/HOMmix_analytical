import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl

def read_cst_data(path_fname):

    Z = []
    EZ = []

    with open(path_fname, "r") as f:
        lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # CST numerical data lines begin with a number,
            # minus sign, or plus sign.
            if line[0].isdigit() or line[0] in "+-":

                line_split = line.split()

                if len(line_split) < 2:
                    continue

                z = float(line_split[0])
                Ez = float(line_split[1])

                Z.append(z)
                EZ.append(Ez)

    return Z, EZ


def read_hommix_pkl(path_fname):

    with open(path_fname, "br") as f:
        data = pkl.load(f)

    Ex = data["TM"]["010"]["3D_Efield"]["Ex"]
    Ey = data["TM"]["010"]["3D_Efield"]["Ey"]
    Ez = data["TM"]["010"]["3D_Efield"]["Ez"]

    return Ex, Ey, Ez


def hommix_z_Ez(path_fname):

    EZ =  read_hommix_pkl(path_fname)[2]
    f_010 = 1.3e9
    csol = 299792458.0
    lambda_010 = csol/f_010
    d_m = lambda_010/2.
    # 151 field samples include both cavity endpoints, so span the full
    # physical cavity length with nz points.
    Z = np.linspace(0.0, d_m, EZ.shape[2])
    print(f"{len(Z) = }")
    print(f"{len(EZ) = }")
    print(f"{EZ.shape = }")
    # print(f"{Z.shape = }")
    return list(Z), list(np.abs(EZ[75,75,:]))



def Vacc_from_real_Ez(z_mm, Ez_Vperm, omega_rad_s, beta=1.0):
    """
    Compute accelerating voltage from real on-axis Ez for a standing-wave mode.

    Parameters
    ----------
    z_mm : array-like
        z coordinates in mm
    Ez_Vperm : array-like
        Real Ez field values in V/m
    omega_rad_s : float
        Angular frequency omega = 2*pi*f in rad/s
    beta : float, optional
        Particle velocity / c

    Returns
    -------
    float
        Accelerating voltage in V
    """
    c = 299792458.0

    z_m = np.asarray(z_mm, dtype=float) / 1000.0
    Ez = np.asarray(Ez_Vperm, dtype=float)

    # Optional: center z so that cavity center is at 0
    z_m = z_m - z_m[len(z_m) // 2]

    phase = omega_rad_s * z_m / (beta * c)
    integrand = Ez * np.cos(phase)

    Vacc = np.abs(np.trapezoid(integrand, z_m))

    return Vacc

def loss_from_Vz(Vz, U=1.):
    loss = Vz ** 2. / (4.*U)

    return loss

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
    EPS0 = 8.854_187_8128e-12
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

def main():
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    cst_tm010_data_path_fname = datapath / r"tm010_onaxis_Ez_cst.txt"

    z_cst_mm, Ez_cst = read_cst_data(cst_tm010_data_path_fname)

    # Sanity check: for the CST 1-J eigenmode normalisation used here,
    # the attached TM010 on-axis field is expected to be of order 1e7 V/m.
    print(f"CST Ez min  = {np.min(Ez_cst):.6e} V/m")
    print(f"CST Ez max  = {np.max(Ez_cst):.6e} V/m")
    print(f"CST Ez mean = {np.mean(Ez_cst):.6e} V/m")

    # hommix_tm010_data_path_fname = datapath / r"TMm0_TMm0_data_dict_151x151x151.pkl"
    hommix_tm010_data_path_fname = datapath / r"all_data_dict_151x151x151.pkl"
    z_hommix, Ez_hommix = hommix_z_Ez(hommix_tm010_data_path_fname)
    z_hommix_mm = [i*1e3 for i in z_hommix]



    # print(f"{len(z_hommix) = }")
    # print(f"{len(Ez_hommix) = }")
    #
    # plt.scatter(z_cst_mm, Ez_cst, label=f"CST ({np.mean(Ez_cst)})")
    # plt.scatter(z_hommix_mm, Ez_hommix, label=f"HOMmix ({np.mean(Ez_hommix)})")
    # plt.legend()
    # plt.show()

    f_010 = 1.3e9
    w_010 = f_010 * 2.0 * np.pi

    Vz_cst = Vacc_from_real_Ez(z_cst_mm, Ez_cst, w_010, beta=1.0)
    Vz_hommix = Vacc_from_real_Ez(z_hommix_mm, Ez_hommix, w_010, beta=1.0)

    Ex, Ey, Ez = read_hommix_pkl(hommix_tm010_data_path_fname)

    Ex_arr = np.array(Ex)
    Ey_arr = np.array(Ey)
    Ez_arr = np.array(Ez)

    f_010 = 1.3e9
    csol = 299792458.0
    lambda_010 = csol / f_010
    d_m = lambda_010 / 2.


    Req_m = (2.4048*csol)/(2.*np.pi*f_010)


    energy_hommix = stored_energy_from_Etotal_CST_equivalent(
        Ex_arr, Ey_arr, Ez_arr,
        Req_m=float(Req_m),
        length_m=d_m,
        radius_pixels=75,
    )
    # CST eigenmode fields are assumed to be normalised to 1 J stored energy.
    U_cst_J = 1.0

    # Stored energy associated with the analytical HOMmix field normalisation.
    U_hommix_J = energy_hommix["U_CST_J"]

    k_cst_V_per_C = loss_from_Vz(Vz_cst, U_cst_J)
    k_hommix_V_per_C = loss_from_Vz(Vz_hommix, U_hommix_J)

    k_cst_V_per_pC = k_cst_V_per_C * 1e-12
    k_hommix_V_per_pC = k_hommix_V_per_C * 1e-12

    percentage_difference = (
        (k_hommix_V_per_C - k_cst_V_per_C)
        / k_cst_V_per_C
        * 100.0
    )

    print()
    print("CST field diagnostics")
    print(f"CST Ez min  = {np.min(Ez_cst):.12e} V/m")
    print(f"CST Ez max  = {np.max(Ez_cst):.12e} V/m")
    print(f"CST Ez mean = {np.mean(Ez_cst):.12e} V/m")

    print()
    print("Stored energies")
    print(f"U_cst_J     = {U_cst_J:.12e} J")
    print(f"U_hommix_J  = {U_hommix_J:.12e} J")

    print()
    print("Transit-time voltages")
    print(f"Vz_cst      = {Vz_cst:.12e} V")
    print(f"Vz_hommix   = {Vz_hommix:.12e} V")

    print()
    print("Longitudinal loss factors")
    print(f"k_cst       = {k_cst_V_per_C:.12e} V/C")
    print(f"k_hommix    = {k_hommix_V_per_C:.12e} V/C")
    print(f"k_cst       = {k_cst_V_per_pC:.6f} V/pC")
    print(f"k_hommix    = {k_hommix_V_per_pC:.6f} V/pC")
    print(f"difference  = {percentage_difference:+.3f} %")


if __name__ == "__main__":
    main()