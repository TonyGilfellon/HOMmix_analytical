import pickle as pkl
import numpy as np
import HOMmix_Master_Module as hmm
import matplotlib.pyplot as plt

import numpy as np



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
    loss = Vz ** 2. / 4.

    return loss



def Vz_loss_from_field(datapath, field_saved_fname, f_010, f_mnp, l_factor, Req_m, Rir_div_Req, plot=False):
    """
    for 150x150x150 cavity data
    N.B mid-pixel is 75
    """

    # load field
    Ez = np.load(f"{datapath}\\{field_saved_fname}")

    Ez = np.load(f"{datapath}\\{field_saved_fname}")



    # manipulate variables
    w_mnp = f_mnp * 2. * np.pi  # angular_freq
    lambda_010 = c_sol / f_010  # wavelength
    l_1_m = lambda_010 / 2.  # design cell length

    d_m = l_factor * l_1_m  # degenerate cell length

    # calculate longitudinal pixel length in metres
    n_pixels_longit = len(Ez[75, 75, :])
    len_pixel_longit = d_m / float((n_pixels_longit) - 1.)

    # generate z-vector in mm
    z_mm = [i * len_pixel_longit * 1.e3 for i in range(n_pixels_longit)]

    # calculate transverse pixel length in metres
    cav_diameter_m = 2. * Req_m
    n_pixel_trans = len(Ez[75, :, 75])
    len_pixel_trans = cav_diameter_m / float((n_pixel_trans) - 1.)

    # we want to analyse transversely up to quarter of the iris radius
    iris_radius_m = Req_m * Rir_div_Req
    iris_radius_pix = iris_radius_m / len_pixel_trans
    quarter_iris_radius_pix = int(np.floor(iris_radius_pix / 4.))
    print(f"{d_m = }")
    print(f"{n_pixel_trans = }")
    print(f"{len_pixel_trans = }")
    print(f"{iris_radius_m = }")
    print(f"{iris_radius_pix = }")
    print(f"{quarter_iris_radius_pix = }")

    # construct the radial coordinate vector in metres
    r_m = [i * len_pixel_trans for i in range(1, quarter_iris_radius_pix)]
    print(f"{r_m = }")

    # calculate the accelerating voltage for each radius up to Rir/2
    # ASSUME - high field is vertical - should be so from previous rotations
    Ez_vperm = Ez[75, 75, :]
    Vz_axis = Vacc_from_real_Ez(z_mm, Ez_vperm, w_mnp)

    # calculate the loss factors for each radius up to Rir/2
    loss_axis = loss_from_Vz(Vz_axis)

    return Vz_axis, loss_axis




if __name__ == "__main__":

    TESLA_TM022_030_dict = {
        "savepath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode043_mode046",
        "datapath": r"D:\PhD\HOMmix\TESLA\analysis\TM_modes\mode043_mode046\study_data",
        "f_010": 1.3e9,
        "f_mnp":3.90312e9,
        "f_E1": 3.755925e9,
        "f_E2": 3.920719e9,
        "l_factor": 0.93,
        "Req_m": 103.3 * 1.e-3,
        "Rir_div_Req": 35 / 103.3,
    }

    dict_list = [
        TESLA_TM022_030_dict,
    ]

    field_saved_fname_E1 = "E1_Ez.npy"
    field_saved_fname_E2 = "E2_Ez.npy"
    field_saved_fname_plus = "Ez_plus.npy"
    field_saved_fname_minus = "Ez_minus.npy"
    c_sol = 299792458.

    for didx, d in enumerate(dict_list):
        savepath = d["savepath"]
        datapath = d["datapath"]
        f_010 = d["f_010"]
        f_mnp = d["f_mnp"]
        f_E1 = d["f_E1"]
        f_E2 = d["f_E2"]
        l_factor = d["l_factor"]
        Req_m = d["Req_m"]
        Rir_div_Req = d["Rir_div_Req"]

        w_mnp = f_mnp * 2. * np.pi
        lambda_010 = c_sol / f_010
        ell_1_m = lambda_010 / 2.

        Vz_axis_E1, loss_axis_E1  = Vz_loss_from_field(
            datapath,
            field_saved_fname_E1,
            f_E1,
            f_mnp=f_mnp,
            l_factor=1.,
            Req_m=Req_m,
            Rir_div_Req=Rir_div_Req,

        )

        Vz_axis_E2, loss_axis_E2 = Vz_loss_from_field(
            datapath,
            field_saved_fname_E2,
            f_E2,
            f_mnp=f_mnp,
            l_factor=1.,
            Req_m=Req_m,
            Rir_div_Req=Rir_div_Req,
        )

        Vz_axis_plus, loss_axis_plus = Vz_loss_from_field(
            datapath,
            field_saved_fname_plus,
            f_010,
            f_mnp=f_mnp,
            l_factor=l_factor,
            Req_m=Req_m,
            Rir_div_Req=Rir_div_Req,
        )

        Vz_axis_minus, loss_axis_minus = Vz_loss_from_field(
            datapath,
            field_saved_fname_minus,
            f_010,
            f_mnp=f_mnp,
            l_factor=l_factor,
            Req_m=Req_m,
            Rir_div_Req=Rir_div_Req,
        )


        print(f"\n{Vz_axis_E1 = }")
        print(f"{loss_axis_E1 = }\n")

        print(f"{Vz_axis_E2 = }")
        print(f"{loss_axis_E2 = }\n")

        print(f"{Vz_axis_plus = }")
        print(f"{loss_axis_plus = }\n")

        print(f"{Vz_axis_minus = }")
        print(f"{loss_axis_minus = }\n")









