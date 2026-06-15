import pickle as pkl
import numpy as np
import HOMmix_Master_Module as hmm
import matplotlib.pyplot as plt
import pickle as pkl
import os
import numpy as np
from mode_maps.pip2_beta1_midcell_ALL_modes_dict import pip2_beta1_ALL_modes_dict as pip2_mm
from mode_maps.TESLA_midcell_ALL_modes_dict import TESLA_ALL_modes_dict as TESLA_mm
from mode_maps.SHINE_3HC_ALL_mode_dict import SHINE_3HC_mode_dict as shine_mm




def kick_from_panofsky_wenzel(
    Ez,
    f_010,
    f_mnp,
    l_factor,
    Req_m,
    Rir_div_Req,
    axis="y",
    beta=1.0,
    fit_pixels=5,
    center_index=None,
    plot=False,
):
    """
    Compute dipole kick factor from the transverse gradient of Vz
    using the Panofsky–Wenzel relation.

    Parameters
    ----------
    Ez : np.ndarray
        3D Ez field array with shape (Nx, Ny, Nz), real field in V/m.
    f_010 : float
        Fundamental frequency in Hz.
    f_mnp : float
        Mode frequency in Hz.
    l_factor : float
        Cell length scaling factor, so d = l_factor * lambda_010 / 2.
    Req_m : float
        Equator radius in metres.
    Rir_div_Req : float
        Iris radius divided by equator radius.
    axis : str, optional
        Direction along which to compute the transverse gradient.
        Must be "x" or "y". Default is "y".
    beta : float, optional
        Particle beta. Default is 1.0.
    fit_pixels : int, optional
        Number of pixels on each side of axis to use in linear fit.
        Default is 5.
    center_index : int or tuple, optional
        If int, use as center index in x and y.
        If tuple, interpreted as (ix0, iy0).
        If None, use geometric center.
    plot : bool, optional
        If True, plot Vz(r) and fitted slope.

    Returns
    -------
    dict
        {
            "r_m": radii used in fit (signed, m),
            "Vz_V": accelerating voltages at those radii,
            "dVz_dr_Vpm": fitted gradient dVz/dr at axis,
            "Vperp_V": transverse voltage from PW,
            "kick_Vpm2": kick factor,
            "fit_coeffs": np.polyfit coefficients,
            "center_indices": (ix0, iy0),
        }
    """

    c_sol = 299792458.0

    Ez = np.asarray(Ez, dtype=float)
    if Ez.ndim != 3:
        raise ValueError(f"Ez must be 3D, got shape {Ez.shape}")

    nx, ny, nz = Ez.shape

    # -----------------------------
    # Choose center pixel
    # -----------------------------
    if center_index is None:
        ix0 = nx // 2
        iy0 = ny // 2
    elif isinstance(center_index, int):
        ix0 = center_index
        iy0 = center_index
    else:
        ix0, iy0 = center_index

    # -----------------------------
    # Longitudinal geometry
    # -----------------------------
    w_mnp = 2.0 * np.pi * f_mnp
    lambda_010 = c_sol / f_010
    l_1_m = lambda_010 / 2.0
    d_m = l_factor * l_1_m

    n_pixels_longit = nz
    len_pixel_longit = d_m / float(n_pixels_longit - 1)
    z_mm = np.arange(n_pixels_longit) * len_pixel_longit * 1.0e3

    # -----------------------------
    # Transverse geometry
    # -----------------------------
    cav_diameter_m = 2.0 * Req_m
    n_pixel_trans = nx if axis == "x" else ny
    len_pixel_trans = cav_diameter_m / float(n_pixel_trans - 1)

    iris_radius_m = Req_m * Rir_div_Req
    iris_radius_pix = iris_radius_m / len_pixel_trans

    max_fit_pix = min(fit_pixels, int(np.floor(iris_radius_pix / 2.0)))
    if max_fit_pix < 1:
        raise ValueError("fit_pixels too small or iris radius too small in pixels.")

    # -----------------------------
    # Build symmetric Vz(r)
    # -----------------------------
    r_vals = []
    Vz_vals = []

    for dpix in range(-max_fit_pix, max_fit_pix + 1):
        if dpix == 0:
            continue

        if axis == "y":
            iy = iy0 + dpix
            if iy < 0 or iy >= ny:
                continue
            Ez_line = Ez[ix0, iy, :]
        elif axis == "x":
            ix = ix0 + dpix
            if ix < 0 or ix >= nx:
                continue
            Ez_line = Ez[ix, iy0, :]
        else:
            raise ValueError("axis must be 'x' or 'y'")

        r_m = dpix * len_pixel_trans
        Vacc = Vacc_from_real_Ez(z_mm, Ez_line, w_mnp, beta=beta)

        # Preserve dipole sign using the sign of offset
        # (Vacc_from_real_Ez returns abs(...), so we restore odd parity)
        Vacc_signed = np.sign(dpix) * Vacc

        r_vals.append(r_m)
        Vz_vals.append(Vacc_signed)

    r_vals = np.array(r_vals, dtype=float)
    Vz_vals = np.array(Vz_vals, dtype=float)

    # -----------------------------
    # Fit Vz(r) = G*r + b near axis
    # -----------------------------
    fit_coeffs = np.polyfit(r_vals, Vz_vals, 1)
    dVz_dr = fit_coeffs[0]   # V/m
    intercept = fit_coeffs[1]

    # -----------------------------
    # Panofsky–Wenzel
    # -----------------------------
    Vperp = (c_sol / w_mnp) * dVz_dr

    # Constant dipole kick factor
    kick = (c_sol / (4.0 * w_mnp * d_m)) * dVz_dr**2

    if plot:
        rr = np.linspace(r_vals.min(), r_vals.max(), 200)
        plt.figure()
        plt.plot(r_vals, Vz_vals, "o", label="Vz(r)")
        plt.plot(rr, dVz_dr * rr + intercept, "-", label="linear fit")
        plt.xlabel("r (m)")
        plt.ylabel("Vz (V)")
        plt.legend()
        plt.title(f"Panofsky–Wenzel fit along {axis}-axis")
        plt.show()

    return {
        "r_m": r_vals,
        "Vz_V": Vz_vals,
        "dVz_dr_Vpm": dVz_dr,
        "Vperp_V": Vperp,
        "kick_Vpm2": kick,
        "fit_coeffs": fit_coeffs,
        "center_indices": (ix0, iy0),
    }


def Vacc_from_Eacc(z_mm, Ez_Vperm, ang_freq_Hz):
    """
    A function that takes two lists (z coord & Ez)
    integrates them with the transit time factor and yields the accelerating voltage.
    :param z_mm: z-coordinates in mm
    :param Ez_Vperm: Ez in V/m
    :param ang_freq_Hz: 2*pi*f
    :return: V_acc in V
    """

    # force required datatypes on arguements
    z_mm = list(z_mm)
    Ez_Vperm = list(Ez_Vperm)
    ang_freq_Hz = float(ang_freq_Hz)

    # define constants
    c_sol = 299792458.0  # speed of light in a vacuum

    # convert z_mm to z_m (mm --> m) and set the central coordinate a z=0.0m
    z_central_mm = z_mm[int(float(len(z_mm)) / 2.0)]
    z_centered_mm = [i - z_central_mm for i in z_mm]
    z_centered_m = [i / 1000.0 for i in z_centered_mm]

    # combine Ez with the transit time factor
    #  (expressed here in explicit form [cos(x) + isin(x)])
    Ez_transit_time = [
        np.real(
            Ez_Vperm[i]
            * (
                    complex(np.cos((ang_freq_Hz * z_centered_m[i]) / c_sol))
                    - np.sin((ang_freq_Hz * z_centered_m[i]) / c_sol)
            )
        )
        for i in range(len(z_mm))
    ]

    # integrate over centralised zcoordinates in m
    Vacc = np.abs(
        sum(
            [
                Ez_transit_time[i + 1] * (z_centered_m[i + 1] - z_centered_m[i])
                for i in range(len(z_centered_m) - 1)
            ]
        )
    )

    return Vacc



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

def loss_from_Vz(Vz):

    loss = Vz**2. / 4.

    return loss

def kick_from_loss(loss, w, d, r):

    kick = (loss * 299792458.) / (w * d * r**2.)

    return kick

import numpy as np
import matplotlib.pyplot as plt


def _pair_symmetric_offsets(r, y, atol=None, rtol=1e-10):
    """
    Pair y(+r) with y(-r) for symmetric offset scans.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y)

    if r.ndim != 1 or y.ndim != 1:
        raise ValueError("r and y must both be 1D")
    if r.size != y.size:
        raise ValueError("r and y must have the same length")

    if atol is None:
        scale = max(1.0, np.max(np.abs(r)))
        atol = 1e-12 * scale

    center_candidates = np.where(np.isclose(r, 0.0, atol=atol, rtol=rtol))[0]
    center_index = int(center_candidates[0]) if center_candidates.size > 0 else None

    pos_indices = np.where(r > 0)[0]

    i_pos_list = []
    i_neg_list = []
    r_pos_list = []

    for ip in pos_indices:
        target = -r[ip]
        diffs = np.abs(r - target)
        j = np.argmin(diffs)

        if not np.isclose(r[j], target, atol=atol, rtol=rtol):
            continue
        if r[j] >= 0:
            continue

        i_pos_list.append(ip)
        i_neg_list.append(j)
        r_pos_list.append(r[ip])

    i_pos = np.asarray(i_pos_list, dtype=int)
    i_neg = np.asarray(i_neg_list, dtype=int)
    r_pos = np.asarray(r_pos_list, dtype=float)

    order = np.argsort(r_pos)
    i_pos = i_pos[order]
    i_neg = i_neg[order]
    r_pos = r_pos[order]

    return {
        "r_pos": r_pos,
        "i_pos": i_pos,
        "i_neg": i_neg,
        "center_index": center_index,
        "center_value": None if center_index is None else y[center_index],
    }


def _even_odd_decomposition(r, y, atol=None, rtol=1e-10):
    """
    Decompose y(r) into even and odd parts using paired +/- offsets.
    """
    pairs = _pair_symmetric_offsets(r, y, atol=atol, rtol=rtol)

    y = np.asarray(y)
    y_pos = y[pairs["i_pos"]]
    y_neg = y[pairs["i_neg"]]

    y_even = 0.5 * (y_pos + y_neg)
    y_odd = 0.5 * (y_pos - y_neg)

    return {
        "r_pos": pairs["r_pos"],
        "y_pos": y_pos,
        "y_neg": y_neg,
        "y_even": y_even,
        "y_odd": y_odd,
        "center_value": pairs["center_value"],
    }


def _fit_multipole_from_vz_scan(r, vz, atol=None, rtol=1e-10):
    """
    Fit:
        even(r) = a0 + a2 r^2
        odd(r)  = a1 r

    Returns:
        monopole   = a0
        dipole     = a1
        quadrupole = a2
    """
    r = np.asarray(r, dtype=float)
    vz = np.asarray(vz)

    decomp = _even_odd_decomposition(r, vz, atol=atol, rtol=rtol)

    rp = decomp["r_pos"]
    ve = decomp["y_even"]
    vo = decomp["y_odd"]

    if rp.size == 0:
        raise ValueError(
            "No +/- offset pairs found. Use symmetric offsets to do even/odd decomposition."
        )

    # Fit even part: ve = a2*r^2 + a0
    Xe = np.column_stack([rp**2, np.ones_like(rp)])
    beta_even, *_ = np.linalg.lstsq(Xe, ve, rcond=None)
    a2, a0 = beta_even

    # Fit odd part: vo = a1*r
    Xo = rp[:, None]
    beta_odd, *_ = np.linalg.lstsq(Xo, vo, rcond=None)
    a1 = beta_odd[0]

    r_ref = np.max(np.abs(r))
    M0 = abs(a0)
    M1 = abs(a1 * r_ref)
    M2 = abs(a2 * r_ref**2)
    denom = M0 + M1 + M2

    if denom > 0:
        monopole_fraction = M0 / denom
        dipole_fraction = M1 / denom
        quadrupole_fraction = M2 / denom
    else:
        monopole_fraction = np.nan
        dipole_fraction = np.nan
        quadrupole_fraction = np.nan

    even_mag = np.sum(np.abs(ve))
    odd_mag = np.sum(np.abs(vo))
    eo_denom = even_mag + odd_mag

    if eo_denom > 0:
        even_fraction = even_mag / eo_denom
        odd_fraction = odd_mag / eo_denom
    else:
        even_fraction = np.nan
        odd_fraction = np.nan

    return {
        "monopole": a0,
        "dipole": a1,
        "quadrupole": a2,
        "fractions": {
            "monopole_fraction": float(monopole_fraction),
            "dipole_fraction": float(dipole_fraction),
            "quadrupole_fraction": float(quadrupole_fraction),
            "even_fraction": float(even_fraction),
            "odd_fraction": float(odd_fraction),
        },
        "decomposition": decomp,
        "diagnostic_scale_m": float(r_ref),
    }


def Vz_loss_kick_from_field(
    datapath,
    field_saved_fname,
    f_010,
    f_mnp,
    l_factor,
    trans_res,
    Req_m,
    Rir_div_Req,
    plot=False,
    fit_multipoles=True,
    use_symmetric_offsets=True,
):
    """
    For 150x150x150 cavity data.

    Adds:
    - signed Vz scan from Ez
    - even/odd decomposition
    - monopole/dipole/quadrupole fit from Vz(r)

    Notes
    -----
    If use_symmetric_offsets=True, offsets are taken from negative to positive
    around the transverse mid pixel. This is required for even/odd decomposition.

    If use_symmetric_offsets=False, only positive offsets are used, matching the
    original behavior, but multipole fitting via even/odd decomposition is skipped.
    """

    # load field
    Ez = np.load(f"{datapath}\\{field_saved_fname}")
    print(f"{Ez.shape = }")

    pw = kick_from_panofsky_wenzel(
        Ez=Ez,
        f_010=f_010,
        f_mnp=f_mnp,
        l_factor=l_factor,
        Req_m=Req_m,
        Rir_div_Req=Rir_div_Req,
        axis="y",  # or "x", depending on polarization
        fit_pixels=5,
        plot=False,
    )

    kick = pw["kick_Vpm2"]
    print("kick =", kick)
    print("dVz/dr =", pw["dVz_dr_Vpm"])

    # manipulate variables
    w_mnp = f_mnp * 2.0 * np.pi
    lambda_010 = c_sol / f_010
    l_1_m = lambda_010 / 2.0
    d_m = l_factor * l_1_m

    # geometry / pixel conversion
    trans_mid_pixel = trans_res // 2
    longit_mid_pixel = 49
    print(f"{trans_mid_pixel = }\n{trans_res = }")

    n_pixels_longit = len(Ez[trans_mid_pixel, trans_mid_pixel, :])
    len_pixel_longit = d_m / float(n_pixels_longit - 1.0)

    z_mm = np.array([i * len_pixel_longit * 1.0e3 for i in range(n_pixels_longit)])

    cav_diameter_m = 2.0 * Req_m
    n_pixel_trans = len(Ez[trans_mid_pixel, :, longit_mid_pixel])
    len_pixel_trans = cav_diameter_m / float(n_pixel_trans - 1.0)

    iris_radius_m = Req_m * Rir_div_Req
    iris_radius_pix = iris_radius_m / len_pixel_trans
    quarter_iris_radius_pix = int(np.floor(iris_radius_pix / 4.0))

    print(f"{d_m = }")
    print(f"{n_pixel_trans = }")
    print(f"{len_pixel_trans = }")
    print(f"{iris_radius_m = }")
    print(f"{iris_radius_pix = }")
    print(f"{quarter_iris_radius_pix = }")

    # ------------------------------------------------------------------
    # Build transverse offsets
    # ------------------------------------------------------------------
    if use_symmetric_offsets:
        # include negative, zero, positive offsets
        pixel_offsets = np.arange(-quarter_iris_radius_pix, quarter_iris_radius_pix + 1, 1)
    else:
        # original style: positive offsets only
        pixel_offsets = np.arange(1, quarter_iris_radius_pix, 1)

    r_m_array = pixel_offsets * len_pixel_trans
    print(f"{r_m_array = }")

    # ------------------------------------------------------------------
    # Integrate Ez -> Vz at each offset
    # ------------------------------------------------------------------
    Vz_vert = []
    for dpix in pixel_offsets:
        Ez_vperm = Ez[trans_mid_pixel, trans_mid_pixel + dpix, :]
        Vacc = Vacc_from_real_Ez(z_mm, Ez_vperm, w_mnp)
        Vz_vert.append(Vacc)

    Vz_vert_array = np.asarray(Vz_vert)

    # loss / kick from Vz
    loss_vert_array = np.asarray([loss_from_Vz(v) for v in Vz_vert_array])

    kick_vert_array = np.asarray([
        kick_from_loss(loss_vert_array[i], w_mnp, d_m, r_m_array[i])
        if not np.isclose(r_m_array[i], 0.0) else np.nan
        for i in range(len(r_m_array))
    ])

    # ------------------------------------------------------------------
    # Multipole diagnosis from signed Vz(r)
    # ------------------------------------------------------------------
    multipole_fit = None
    if fit_multipoles:
        if use_symmetric_offsets:
            multipole_fit = _fit_multipole_from_vz_scan(r_m_array, Vz_vert_array)
            print("multipole fit:")
            print(f"  monopole   = {multipole_fit['monopole']}")
            print(f"  dipole     = {multipole_fit['dipole']}")
            print(f"  quadrupole = {multipole_fit['quadrupole']}")
            print(f"  fractions  = {multipole_fit['fractions']}")
        else:
            print("Skipping multipole fit: symmetric +/- offsets are required.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot:
        f, axs = plt.subplots(nrows=1, ncols=2)
        img = axs[0].imshow(Ez[:, :, longit_mid_pixel].T, origin="lower")
        img2 = axs[1].imshow(Ez[trans_mid_pixel, :, :], origin="lower")
        plt.colorbar(img)
        plt.colorbar(img2)
        plt.show()
        plt.savefig(f"{datapath}\\Ez_trans_longit.png")
        plt.close('all)')

        plt.plot(r_m_array, Vz_vert_array, "o-", label="$V_z$")
        plt.axvline(0.0, color="k", alpha=0.3)
        plt.legend()
        plt.show()
        plt.savefig(f"{datapath}\\r_vs_Vz.png")
        plt.close('all)')

        plt.plot(r_m_array, loss_vert_array, "o-", label="$loss$")
        plt.axvline(0.0, color="k", alpha=0.3)
        plt.legend()
        plt.show()
        plt.savefig(f"{datapath}\\r_vs_loss.png")
        plt.close('all)')

        plt.plot(r_m_array, kick_vert_array, "o-", label="$kick$")
        plt.axvline(0.0, color="k", alpha=0.3)
        plt.legend()
        plt.show()
        plt.savefig(f"{datapath}\\r_vs_kick.png")
        plt.close('all)')

        if multipole_fit is not None:
            decomp = multipole_fit["decomposition"]
            rp = decomp["r_pos"]
            ve = decomp["y_even"]
            vo = decomp["y_odd"]

            plt.plot(rp, ve, "o-", label="even($V_z$)")
            plt.plot(rp, vo, "o-", label="odd($V_z$)")
            plt.legend()
            plt.show()

    return (
        r_m_array,
        Vz_vert_array,
        loss_vert_array,
        kick_vert_array,
        pw,
        multipole_fit,
    )


def on_axis_Vz_loss_from_field(datapath, field_saved_fname, f_010, f_mnp, l_factor, trans_res,):
    """
    for 150x150x150 cavity data
    N.B mid-pixel is 75
    """


    Ez = np.load(f"{datapath}\\{field_saved_fname}")

    # manipulate variables
    w_mnp = f_mnp * 2. * np.pi  # angular_freq
    lambda_010 = c_sol / f_010  # wavelength
    l_1_m = lambda_010 / 2.  # design cell length

    d_m = l_factor * l_1_m  # degenerate cell length

    # calculate longitudinal pixel length in metres
    trans_mid_pixel = trans_res // 2
    print(f"{trans_mid_pixel = }\n{trans_res = }")
    n_pixels_longit = len(Ez[trans_mid_pixel, trans_mid_pixel, :])
    len_pixel_longit = d_m / float((n_pixels_longit) - 1.)

    # generate z-vector in mm
    z_mm = [i * len_pixel_longit * 1.e3 for i in range(n_pixels_longit)]


    print(f"{d_m = }")



    Ez_vperm = Ez[trans_mid_pixel, trans_mid_pixel, :]

    Vacc = Vacc_from_real_Ez(z_mm, Ez_vperm, w_mnp)


    loss= loss_from_Vz(Vacc)

    return Vacc, loss

# define functions
def loss_from_RQ(rq, angFreq, offset, divisor=4.0):
    """

    :param rq:
    :param angFreq:
    :param divisor:
    :return:
    """
    divisor = float(divisor)

    loss = angFreq * rq / (offset ** 2.0 * divisor)

    return loss


def kick_from_norm_loss(norm_loss, omega, period):
    csol = 299792458.0
    kick = norm_loss * csol / (omega * period)

    return kick

def load_pickles(base_dir, substring):
    results = {}

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        folder_path = os.path.join(folder_path, "study_data")

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if substring in file and file.endswith(".pkl"):
                file_path = os.path.join(folder_path, file)

                with open(file_path, "rb") as f:
                    results[folder] = pkl.load(f)

                break  # load first match per folder

    return results


def get_kick_at_offset(r_array, kick_array, r_query):
    """
    Return the kick at the nearest available radial offset.

    Parameters
    ----------
    r_array : array-like
        Radial offsets (in metres)
    kick_array : array-like
        Kick values corresponding to r_array
    r_query : float
        Desired offset (in metres)

    Returns
    -------
    dict
        {
            "requested_offset": r_query,
            "nearest_offset": r_nearest,
            "kick": kick_value,
            "index": idx
        }
    """
    r_array = np.asarray(r_array)
    kick_array = np.asarray(kick_array)

    idx = np.argmin(np.abs(r_array - r_query))
    r_nearest = r_array[idx]
    kick_value = kick_array[idx]

    return kick_value


def is_monopole(mode_str):
    """
    Returns True if the first index after 'TM' is 0 (monopole), else False.
    """
    if not mode_str.startswith("TM"):
        raise ValueError(f"Invalid mode string: {mode_str}")
    return mode_str[2] == "0"

if __name__ == "__main__":

    #
    # TESLA_TM121_220_dict = {
    #     "savepath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode043_mode046",
    #     "datapath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode043_mode046\study_data",
    #     "f_010": 1.3e9,
    #     "f_mnp": 3.8792e9,
    #     "f_E1": 3.897848e9,
    #     "f_E2": 4.066385e9,
    #     "l_factor": 1.18947,
    #     "Req_m": 103.3*1.e-3,
    #     "Rir_div_Req": 35 / 103.3,
    # }
    #
    # TESLA_TM022_220_dict = {
    #     "savepath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode039_mode043",
    #     "datapath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode039_mode043\study_data",
    #     "f_010": 1.3e9,
    #     "f_mnp": 3.88791e9,
    #     "f_E1": 3.755925e9,
    #     "f_E2": 3.897848e9,
    #     "l_factor": 0.9357,
    #     "Req_m": 103.3 * 1.e-3,
    #     "Rir_div_Req": 35 / 103.3,
    # }
    #
    #
    # pip2_Beta1_TM121_TM122_dict = {
    # "savepath": r"D:\PhD\HOMmix\pip2_Beta1\analysis\TM_modes\mode046_mode050",
    # "datapath": r"D:\PhD\HOMmix\pip2_Beta1\analysis\TM_modes\mode046_mode050\study_data",
    # "f_010": 650.e6,
    # "f_mnp": 2.14019e9,
    # "f_E1": 2.056233e9,
    # "f_E2": 2.110458e9,
    # "l_factor": 0.82153,
    # "Req_m": 200.138*1.e-3,
    # "Rir_div_Req": 58.884 / 200.138,
    # }
    #
    # pip2_Beta1_TM130_TM032_dict = {
    #     "savepath": r"D:\PhD\HOMmix\pip2_Beta1\analysis\TM_modes\mode077_mode079",
    #     "datapath": r"D:\PhD\HOMmix\pip2_Beta1\analysis\TM_modes\mode077_mode079\study_data",
    #     "f_010": 650.e6,
    #     "f_mnp": 2.4243223e9,
    #     "f_E1": 2.382291e9,
    #     "f_E2": 2.399643e9,
    #     "l_factor": 0.92660625,
    #     "Req_m": 200.138 * 1.e-3,
    #     "Rir_div_Req": 58.884 / 200.138,
    # }
    #
    # #
    # # #Test
    # # roverq_mode_50_15mmm_offset = 1.297981
    # # test_offset_mm = 15.
    # # test_offset_m = test_offset_mm*1.e-3
    # # f_test = f_E2
    # # w_test = 2. * np.pi * f_test
    # # period_test_m = 230.6096*1.e-3
    # # norm_loss_test = loss_from_RQ(roverq_mode_50_15mmm_offset, w_test, test_offset_m, divisor=2.0)
    # # kick_test = kick_from_norm_loss(norm_loss_test, w_test, period_test_m)
    # #
    # # print(f"R/Q route: {kick_test*1.e-12} $V/pC/m^2$")
    # # print(f"trad route: 3.930371942420379 $V/pC/m^2$")
    # # print(f"P-W route: 2.114205131856864e-07 $V/pC/m^2$")
    #
    # SHINE_3HC_TM122_TM130_dict = {
    # "savepath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode058_mode068",
    # "datapath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode058_mode068\study_data",
    # "f_010": 3.9e9,
    # "f_mnp": 13.27287e9,
    # "f_E1": 12.73139e9,
    # "f_E2": 13.21756e9,
    # "l_factor": 0.906026,
    # "Req_m": 35.79 * 1.e-3,
    # "Rir_div_Req": 15. / 35.79,
    # }
    #
    # SHINE_3HC_TM130_TM031_dict = {
    #     "savepath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode068_mode070",
    #     "datapath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode068_mode070\study_data",
    #     "f_010": 3.9e9,
    #     "f_mnp": 13.209818e9,
    #     "f_E1": 13.21756e9,
    #     "f_E2": 13.37416e9,
    #     "l_factor": 1.04039114758,
    #     "Req_m": 35.79 * 1.e-3,
    #     "Rir_div_Req": 15. / 35.79,
    # }
    #
    # SHINE_3HC_TM032_TM131_dict = {
    #     "savepath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode094_mode097",
    #     "datapath": r"D:\PhD\HOMmix\SHINE_3HC\analysis\TM_modes\mode094_mode097\study_data",
    #     "f_010": 3.9e9,
    #     "f_mnp": 15.0903071707e9,
    #     "f_E1": 14.732386e9,
    #     "f_E2": 14.99654e9,
    #     "l_factor": 0.96035175133,
    #     "Req_m": 35.79 * 1.e-3,
    #     "Rir_div_Req": 15. / 35.79,
    # }





    # # Test
    # roverq_mode_68_3p75mmm_offset = 6.471153
    # test_offset_mm =3.75
    # test_offset_m = test_offset_mm * 1.e-3
    # f_test = f_E2
    # w_test = 2. * np.pi * f_test
    # period_test_m = 38.44 * 1.e-3
    # norm_loss_test = loss_from_RQ(roverq_mode_68_3p75mmm_offset, w_test, test_offset_m, divisor=2.0)
    # kick_test = kick_from_norm_loss(norm_loss_test, w_test, period_test_m)
    #
    # print(f"R/Q route: {kick_test*1.e-12} $V/pC/m^2$")
    # print(f"trad route: 2461.0679119707056V/pC/m^2$")
    # print(f"P-W route: 4.6012858483914043e-07 $V/pC/m^2$")

    # dict_list = [
    #     # TESLA_TM121_220_dict,
    #     # TESLA_TM022_220_dict,
    #     # pip2_Beta1_TM121_TM122_dict,
    #     # pip2_Beta1_TM130_TM032_dict,
    #     # SHINE_3HC_TM122_TM130_dict,
    #     # SHINE_3HC_TM130_TM031_dict,
    #     SHINE_3HC_TM032_TM131_dict,
    # ]

    n_pix_TESLA = 295
    n_pix_pip2_Beta1 = 573
    n_pix_SHINE_3HC = 103
    longit_res = 100

    study_list = [
        "TESLA",
        "pip2_Beta1",
        # "3HC",
        "SHINE_3HC",
    ]

    for study in study_list:
    # study = "pip2_Beta1"

        if study == "TESLA":
            trans_res = n_pix_TESLA
            f_010 = 1300000000.
        elif study == "pip2_Beta1":
            trans_res = n_pix_pip2_Beta1
            f_010 = 650000000.
        elif study == "SHINE_3HC":
            trans_res = n_pix_SHINE_3HC
            f_010 = 3900000000.
        else:
            exit(f'study not recognized ({study})')

        base_dir = fr"D:\PhD\HOMmix\{study}\analysis\TM_modes"
        substring = f"{study}_mode"
        dict_of_dicts = load_pickles(base_dir, substring) # outputted from

        field_saved_fname_E1 = "E1_Ez.npy"
        field_saved_fname_E2 = "E2_Ez.npy"
        field_saved_fname_plus = "Ez_plus.npy"
        field_saved_fname_minus = "Ez_minus.npy"
        c_sol = 299792458.

        all_E1 = []
        all_E2 = []
        length_factors = []
        kick_Fhat = []
        E1_kicks = []
        E2_kicks = []
        Eplus_kicks = []
        Eminus_kicks = []
        E_pm_max_kick = []
        plus_kick_ratios = []
        minus_kick_ratios = []
        kick_ratios = []

        on_axis_mode_i = []
        on_axis_mode_j = []
        on_axis_length_factors = []
        on_axis_Fhat = []
        E1_on_axis_Vaccs = []
        E2_on_axis_Vaccs = []
        Eplus_on_axis_Vaccs = []
        Eminus_on_axis_Vaccs = []

        E1_on_axis_losses = []
        E2_on_axis_losses = []
        Eplus_on_axis_losses = []
        Eminus_on_axis_losses = []
        E_pm_max_loss = []
        plus_loss_ratios = []
        minus_loss_ratios = []
        loss_ratios = []

        for didx, d in enumerate(dict_of_dicts.keys()):
            name = dict_of_dicts[d]["name"]
            print(f"{didx}: {name}")
            E1_mode = dict_of_dicts[d]['E1_mode']
            E2_mode = dict_of_dicts[d]['E2_mode']
            savepath = dict_of_dicts[d]["savepath"]
            datapath = dict_of_dicts[d]["datapath"]
            f_010 = dict_of_dicts[d]["f_010"]
            f_mnp = dict_of_dicts[d]["f_mnp"]
            f_E1 = dict_of_dicts[d]["f_E1"]
            f_E2 = dict_of_dicts[d]["f_E2"]
            l_factor = dict_of_dicts[d]["l_factor"]
            Req_m = dict_of_dicts[d]["Req_m"]
            Rir_div_Req = dict_of_dicts[d]["Rir_div_Req"]
            calc_on_axis_loss = dict_of_dicts[d]["calc_on_axis_loss"]
            calc_kick = dict_of_dicts[d]["calc_kick"]



            w_mnp = f_mnp * 2. * np.pi
            lambda_010 = c_sol / f_010
            ell_1_m = lambda_010 / 2.

            if calc_kick:

                e600_sigma_m = 0.00023295536053072484
                n_sigma = 3.
                e600_3_sigma_m = n_sigma * e600_sigma_m

                E1_is_monopole = is_monopole(dict_of_dicts[d]['E1_mode'])
                E2_is_monopole = is_monopole(dict_of_dicts[d]['E2_mode'])



                r_m_E1, Vz_vert_E1, loss_vert_E1, kick_vert_E1, pw_E1, multipole_fit_E1 = Vz_loss_kick_from_field(
                    datapath,
                    field_saved_fname_E1,
                    f_E1,
                    f_mnp=f_mnp,
                    l_factor=1.,
                    trans_res = trans_res,
                    Req_m=Req_m,
                    Rir_div_Req=Rir_div_Req,
                    plot=True,
                    fit_multipoles=True,
                    use_symmetric_offsets=True,

                )

                print(multipole_fit_E1["monopole"])
                print(multipole_fit_E1["dipole"])
                print(multipole_fit_E1["quadrupole"])
                print(multipole_fit_E1["fractions"])

                r_m_E2, Vz_vert_E2, loss_vert_E2, kick_vert_E2, pw_E2, multipole_fit_E2 = Vz_loss_kick_from_field(
                    datapath,
                    field_saved_fname_E2,
                    f_E2,
                    f_mnp=f_mnp,
                    l_factor=1.,
                    trans_res=trans_res,
                    Req_m=Req_m,
                    Rir_div_Req=Rir_div_Req,
                    plot=True,
                    fit_multipoles=True,
                    use_symmetric_offsets=True,
                )

                print(multipole_fit_E2["monopole"])
                print(multipole_fit_E2["dipole"])
                print(multipole_fit_E2["quadrupole"])
                print(multipole_fit_E2["fractions"])

                r_m_plus, Vz_vert_plus, loss_vert_plus, kick_vert_plus, pw_plus, multipole_fit_plus = Vz_loss_kick_from_field(
                    datapath,
                    field_saved_fname_plus,
                    f_010,
                    f_mnp=f_mnp,
                    l_factor=l_factor,
                    trans_res=trans_res,
                    Req_m=Req_m,
                    Rir_div_Req = Rir_div_Req,
                    plot=True,
                    fit_multipoles=True,
                    use_symmetric_offsets=True,
                )

                print(multipole_fit_plus["monopole"])
                print(multipole_fit_plus["dipole"])
                print(multipole_fit_plus["quadrupole"])
                print(multipole_fit_plus["fractions"])

                r_m_minus, Vz_vert_minus, loss_vert_minus, kick_vert_minus, pw_minus, multipole_fit_minus = Vz_loss_kick_from_field(
                    datapath,
                    field_saved_fname_minus,
                    f_010,
                    f_mnp=f_mnp,
                    l_factor=l_factor,
                    trans_res=trans_res,
                    Req_m=Req_m,
                    Rir_div_Req=Rir_div_Req,
                    plot=True,
                    fit_multipoles=True,
                    use_symmetric_offsets=True,
                )

                print(multipole_fit_minus["monopole"])
                print(multipole_fit_minus["dipole"])
                print(multipole_fit_minus["quadrupole"])
                print(multipole_fit_minus["fractions"])

                print(f"\nE1\n{np.mean(kick_vert_E1)*1.e-12 = }")
                print(f"{kick_vert_E1[0] * 1.e-12 = }")
                print(f"{pw_E1['kick_Vpm2']*1.e-12 = }\n")

                print(f"\nE2\n{np.mean(kick_vert_E2)*1.e-12 = }")
                print(f"{kick_vert_E2[0]*1.e-12 = }")
                print(f"{pw_E2['kick_Vpm2']*1.e-12 = }\n")

                print(f"\nplus\n{np.mean(kick_vert_plus)*1.e-12 = }")
                print(f"{kick_vert_plus[0]*1.e-12 = }")
                print(f"{pw_plus['kick_Vpm2']*1.e-12 = }\n")

                print(f"\nminus\n{np.mean(kick_vert_minus)*1.e-12 = }")
                print(f"{kick_vert_minus[0]*1.e-12 = }")
                print(f"{pw_minus['kick_Vpm2']*1.e-12 = }\n")

                plt.plot(r_m_E1*1.e3, Vz_vert_E1*1.e-6, label="E1 $V_z$")
                plt.plot(r_m_E2*1.e3, Vz_vert_E2*1.e-6, label="E2 $V_z$")
                plt.plot(r_m_plus*1.e3, Vz_vert_plus*1.e-6, label="E+ $V_z$")
                plt.plot(r_m_minus*1.e3, Vz_vert_minus*1.e-6, label="E- $V_z$")
                plt.xlabel("Radial Offset [mm]")
                plt.ylabel("$V_z$ [MV]")
                plt.legend()
                # plt.show()
                plt.savefig(f"{savepath}\\r_vs_voltage.png")
                plt.close('all')

                plt.plot(r_m_E1*1.e3, loss_vert_E1*1.e-12, label="E1 loss")
                plt.plot(r_m_E2*1.e3, loss_vert_E2*1.e-12, label="E2 loss")
                plt.plot(r_m_plus*1.e3, loss_vert_plus*1.e-12, label="E+ loss")
                plt.plot(r_m_minus*1.e3, loss_vert_minus*1.e-12, label="E- loss")
                plt.xlabel("Radial Offset [mm]")
                plt.ylabel("Loss Factor [V/pC]")
                plt.legend()
                # plt.show()
                plt.savefig(f"{savepath}\\r_vs_loss.png")
                plt.close('all')

                plt.plot(r_m_E1*1.e3, kick_vert_E1*1.e-12, label="E1 kick")
                plt.plot(r_m_E2*1.e3, kick_vert_E2*1.e-12, label="E2 kick")
                plt.plot(r_m_plus*1.e3, kick_vert_plus*1.e-12, label="E+ kick")
                plt.plot(r_m_minus*1.e3, kick_vert_minus*1.e-12, label="E- kick")
                plt.xlabel("Radial Offset [mm]")
                plt.ylabel("Kick Factor [V/pC/$m^2$]")
                plt.legend()
                # plt.show()
                plt.savefig(f"{savepath}\\r_vs_kick.png")
                plt.close('all')
                
                # save_out_radial_data
                radial_dict_to_save = {
                    "r_m_E1": r_m_E1, 
                    "Vz_vert_E1": Vz_vert_E1, 
                    "loss_vert_E1": loss_vert_E1, 
                    "kick_vert_E1": kick_vert_E1,

                    "r_m_E2": r_m_E2,
                    "Vz_vert_E2": Vz_vert_E2,
                    "loss_vert_E2": loss_vert_E2,
                    "kick_vert_E2": kick_vert_E2,

                    "r_m_plus": r_m_plus,
                    "Vz_vert_plus": Vz_vert_plus,
                    "loss_vert_plus": loss_vert_plus,
                    "kick_vert_plus": kick_vert_plus,

                    "r_m_minus": r_m_minus,
                    "Vz_vert_minus": Vz_vert_minus,
                    "loss_vert_minus": loss_vert_minus,
                    "kick_vert_minus": kick_vert_minus,

                }

                hmm.pickle_dump(radial_dict_to_save, f"{datapath}\\radial_voltage_loss_kick_data.pkl")



                kick_vert_plus_3_sigma = get_kick_at_offset(r_m_plus, kick_vert_plus, e600_3_sigma_m) * e600_3_sigma_m
                kick_vert_minus_3_sigma = get_kick_at_offset(r_m_minus, kick_vert_minus, e600_3_sigma_m) * e600_3_sigma_m
                kick_vert_E1_3_sigma = get_kick_at_offset(r_m_E1, kick_vert_E1, e600_3_sigma_m) * e600_3_sigma_m
                kick_vert_E2_3_sigma = get_kick_at_offset(r_m_E2, kick_vert_E2, e600_3_sigma_m) * e600_3_sigma_m

                print(f"{kick_vert_plus_3_sigma*1.e-12 = }")
                print(f"{get_kick_at_offset(r_m_plus, kick_vert_plus, e600_3_sigma_m) *1.e-12 = }")
                E_pm_max_k = float(max(kick_vert_plus_3_sigma, kick_vert_minus_3_sigma))
                print(f"{E_pm_max_k = }")
                print(f"{type(E_pm_max_k) = }")
                E1_is_monopole = is_monopole(dict_of_dicts[d]['E1_mode'])
                E2_is_monopole = is_monopole(dict_of_dicts[d]['E2_mode'])


                plus_kick_ratio = min(
                    kick_vert_plus_3_sigma / kick_vert_E1_3_sigma,
                    kick_vert_plus_3_sigma / kick_vert_E2_3_sigma,
                )
                minus_kick_ratio = min(
                    kick_vert_minus_3_sigma / kick_vert_E1_3_sigma,
                    kick_vert_minus_3_sigma / kick_vert_E2_3_sigma,
                )

                kick_ratio = max(plus_kick_ratio, minus_kick_ratio)

                fhat = f_mnp/f_010

                # input(f"{fhat = }")

                if study == "SHINE_3HC":
                    print("\nSHINE_3HC:")
                    print(f"{kick_vert_plus_3_sigma = }")
                    print(f"{kick_vert_minus_3_sigma = }")
                    print(f"{kick_vert_E1_3_sigma = }")
                    print(f"{kick_vert_E2_3_sigma = }")
                    print(f"{plus_kick_ratio = }")
                    print(f"{minus_kick_ratio = }")

                if str(E1_mode) == "TM022" and str(E2_mode) == "TM220":
                    print(f"{E1_mode = }\n")
                    print(f"{E2_mode = }\n")
                    print(f"{kick_vert_E1_3_sigma = }\n")
                    print(f"{kick_vert_E2_3_sigma = }\n")
                    print(f"{kick_vert_plus_3_sigma = }\n")
                    print(f"{kick_vert_minus_3_sigma = }\n")
                    print(f"{plus_kick_ratio = }\n")
                    print(f"{minus_kick_ratio = }\n")
                    input(f"{kick_ratio = }\n")

                all_E1.append(str(E1_mode))
                all_E2.append(str(E2_mode))
                length_factors.append(float(l_factor))
                kick_Fhat.append(float(fhat))
                E1_kicks.append(float(kick_vert_E1_3_sigma))
                E2_kicks.append(float(kick_vert_E2_3_sigma))
                Eplus_kicks.append(float(kick_vert_plus_3_sigma))
                Eminus_kicks.append(float(kick_vert_minus_3_sigma))
                E_pm_max_kick.append(float(E_pm_max_k))
                plus_kick_ratios.append(float(plus_kick_ratio))
                minus_kick_ratios.append(float(minus_kick_ratio))
                kick_ratios.append(float(kick_ratio))

            else:
                pass

            # on axis loss if appropriate (i.e. if E1 or E1 has m index of 0)
            if calc_on_axis_loss:
                E1_on_axis_Vacc, E1_on_axis_loss = on_axis_Vz_loss_from_field(datapath, field_saved_fname_E1, f_010, f_mnp, l_factor, trans_res)
                E2_on_axis_Vacc, E2_on_axis_loss = on_axis_Vz_loss_from_field(datapath, field_saved_fname_E2, f_010, f_mnp, l_factor, trans_res)
                Eplus_on_axis_Vacc, Eplus_on_axis_loss = on_axis_Vz_loss_from_field(datapath, field_saved_fname_plus, f_010, f_mnp, l_factor, trans_res)
                Eminus_on_axis_Vacc, Eminus_on_axis_loss = on_axis_Vz_loss_from_field(datapath, field_saved_fname_minus, f_010, f_mnp, l_factor, trans_res)
                E_pm_max_l = float(max(Eplus_on_axis_loss, Eminus_on_axis_loss))
                plus_loss_ratio = min(Eplus_on_axis_loss / E1_on_axis_loss, Eplus_on_axis_loss / E2_on_axis_loss)
                minus_loss_ratio = min(Eminus_on_axis_loss / E1_on_axis_loss, Eminus_on_axis_loss / E2_on_axis_loss)

                print(f"{E1_on_axis_Vacc = }")
                print(f"{E2_on_axis_Vacc = }")
                print(f"{Eplus_on_axis_Vacc = }")
                print(f"{Eminus_on_axis_Vacc = }")

                print(f"{E1_on_axis_loss = }")
                print(f"{E2_on_axis_loss = }")
                print(f"{Eplus_on_axis_loss = }")
                print(f"{Eminus_on_axis_loss= }")

                # input()

                on_axis_mode_i.append(E1_mode)
                on_axis_mode_j.append(E2_mode)
                on_axis_length_factors.append(l_factor)
                on_axis_Fhat.append(f_mnp/f_010)
                E1_on_axis_Vaccs.append(E1_on_axis_Vacc)
                E2_on_axis_Vaccs.append(E2_on_axis_Vacc)
                Eplus_on_axis_Vaccs.append(Eplus_on_axis_Vacc)
                Eminus_on_axis_Vaccs.append(Eminus_on_axis_Vacc)
                E_pm_max_loss.append(E_pm_max_l)
                E1_on_axis_losses.append(E1_on_axis_loss)
                E2_on_axis_losses.append(E2_on_axis_loss)
                Eplus_on_axis_losses.append(Eplus_on_axis_loss)
                Eminus_on_axis_losses.append(Eminus_on_axis_loss)

                plus_loss_ratios.append(plus_loss_ratio)
                minus_loss_ratios.append(minus_loss_ratio)

                loss_ratio = max(plus_loss_ratio, minus_loss_ratio)

                loss_ratios.append(loss_ratio)

            else:
                pass



        # write out on-axis Vaccs
        with open(f"{base_dir}\\{study}_on_axis_voltages.txt", "w") as f:
            f.write("E1,E2,length_factor,Fhat,E1_Vacc,E1_Vacc_Eplus_Vacc,Eminus_Vacc\n")
            for idx in range(len(on_axis_mode_i)):
                f.write(f"{on_axis_mode_i[idx]},{on_axis_mode_j[idx]},{on_axis_length_factors[idx]},{on_axis_Fhat[idx]},{E1_on_axis_Vaccs[idx]},{E2_on_axis_Vaccs[idx]},{Eplus_on_axis_Vaccs[idx]},{Eminus_on_axis_Vaccs[idx]}\n")
            f.close()

        # Write out on-axis losses
        with open(f"{base_dir}\\{study}_on_axis_losses.txt", "w") as f:
            f.write("E1,E2,length_factor,Fhat,E1_loss,E2_loss,Eplus_loss,Eminus_loss,E_pm_max_loss,Eplus_loss_ratios,Eminus_loss_ratios,loss_ratio\n")
            for idx in range(len(on_axis_mode_i)):
                f.write(f"{on_axis_mode_i[idx]},{on_axis_mode_j[idx]},{on_axis_length_factors[idx]},{on_axis_Fhat[idx]},{E1_on_axis_losses[idx]},{E2_on_axis_losses[idx]},{Eplus_on_axis_losses[idx]},{Eminus_on_axis_losses[idx]},{E_pm_max_loss[idx]},{plus_loss_ratios[idx]},{minus_loss_ratios[idx]},{loss_ratios[idx]}\n")
            f.close()

        # Write out kicks
        with open(f"{base_dir}\\{study}_kick_ratios.txt", "w") as f:
            f.write("E1,E2,length_factors,Fhat,E1_kick,E2_kick,Eplus_kick,Eminus_kick,E_pm_max_kick,plus_kick_ratio,minus_kick_ratio,kick_ratio\n")
            for idx in range(len(all_E1)):
                f.write(f"{all_E1[idx]},{all_E2[idx]},{length_factors[idx]},{kick_Fhat[idx]},{E1_kicks[idx]},{E2_kicks[idx]},{Eplus_kicks[idx]},{Eminus_kicks[idx]},{E_pm_max_kick[idx]},{plus_kick_ratios[idx]},{minus_kick_ratios[idx]},{kick_ratios[idx]}\n")
            f.close()

        # plt.title(f"{study}")
        # plt.scatter(range(len(plus_kick_ratios)), plus_kick_ratios, label='plus')
        # plt.scatter(range(len(minus_kick_ratios)), minus_kick_ratios, label='minus')
        # plt.legend()
        # plt.show()










