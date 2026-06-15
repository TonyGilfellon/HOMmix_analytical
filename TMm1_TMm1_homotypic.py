import HOMmix_analytical_master_module as hamm
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.patches import Rectangle
import numpy as np
import pickle as pkl
from pathlib import Path

""" Define Functions """

def assemble_all_dipole_data_dict(m_max, n_max, p_max,
                           frequency_010 = 1.3e9,
                           LF_start = 0.9,
                           LF_stop = 1.1,
                           param_sweep_resolution = 1000,
                           voxel_res = 21,
):

    csol = 299_792_458.0  # speed of light (m/s)
    lambda_010 = csol / frequency_010
    R = hamm.pillbox_radius_from_freq(frequency_010)

    m_list = np.array([1.])
    mint = [int(m) for m in m_list]
    n_list = np.linspace(1, n_max, n_max+1, endpoint=True)
    nint = [int(n) for n in n_list]
    p_list = np.linspace(0, p_max, p_max + 1, endpoint=True)
    pint = [int(p) for p in p_list]
    length_factor_vector = np.linspace(LF_start, LF_stop, param_sweep_resolution, endpoint=True)
    length_factor_vector_floats = [float(i) for i in length_factor_vector]

    all_data = {}
    all_data['TM'] = {}
    all_data['length_factor_vector'] = length_factor_vector_floats

    for pidx, p in enumerate(pint):
        for nidx, n in enumerate(nint):
            for midx, m in enumerate(mint):

                """ TM fields """

                """
                data = {"Ex": Exm, "Ey": Eym, "Ez": Ezm, "Eperp": Eperpm, "|E|": Emagm}
                """

                print(f"{m}{n}{p}")
                all_data['TM'][f"{m}{n}{p}"] = {}


                all_data['TM'][f"{m}{n}{p}"]['3D_Efield'] = hamm.pillbox_field_voxel_grid_xyz(
                    R=R,
                    L=lambda_010 / 2.,
                    m=m,
                    n=n,
                    p=p,
                    x_res=voxel_res,
                    y_res=voxel_res,
                    z_res=voxel_res,
                    E0=1.0,
                    mode="TM",
                    z_range=(0.0, 1.0),
                    dtype=np.float32,
                )

                TM_mode_data = []

                TM_normalised_mode_data = []

                for lidx, length_factor in enumerate(length_factor_vector_floats):
                    L = length_factor * lambda_010 / 2.

                    """ TM frequencies """

                    f_tm_val = hamm.f_tm(m, n, p, R, L)

                    # input(f"TM{m}{n}{p} at {length_factor} = {f_tm_val}")
                    TM_mode_data.append(f_tm_val)

                    TM_normalised_mode_data.append(f_tm_val / frequency_010)

                # param sweep data
                all_data['TM'][f"{m}{n}{p}"]['frequency_Hz'] = TM_mode_data
                all_data['TM'][f"{m}{n}{p}"]['frequency_normalised'] = TM_normalised_mode_data

                # design_freq_data
                design_freq_Hz = hamm.f_tm(m, n, p, R, lambda_010 / 2.)
                norm_design_freq = design_freq_Hz / f_010
                all_data['TM'][f"{m}{n}{p}"]['design_frequency_Hz'] = design_freq_Hz
                all_data['TM'][f"{m}{n}{p}"]['design_frequency_normalised'] = norm_design_freq

    return all_data

if __name__ == "__main__":
    datapath = r"D:\PhD\HOMmix\HOMmix_analytical\data"
    savepath = r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles"
    m_max, n_max, p_max = 1, 3, 3
    voxel_res = 151
    f_010 = 1.3e9
    c_sol = 299792458.0
    create_data = True
    create_fields = True
    if create_data:
        data_dict = assemble_all_dipole_data_dict(m_max, n_max, p_max,
                                                    frequency_010=f_010,
                                                    LF_start=0.7,
                                                    LF_stop=1.3,
                                                    param_sweep_resolution=1000,
                                                    voxel_res=voxel_res,
                                                    )

        hamm.pickle_save(data_dict, f"{datapath}\\TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl")

    else:
        data_dict = hamm.pickle_load(f"{datapath}\\TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl")

    print(data_dict['TM']['010'].keys())

    TM_crossing_results = hamm.find_mode_crossings_from_all_data(data_dict, mode_type="TM")

    # for key, val in TM_crossing_results["TM"].items():
    #     print(f"{key}: {val}")
    #
    # input()

    for key, val in TM_crossing_results['TM']['crossings'].items():
        print(f"{key}: {val}")
        mode_i = TM_crossing_results['TM']['crossings'][key]['mode_i'][-3:]
        mode_j = TM_crossing_results['TM']['crossings'][key]['mode_j'][-3:]
        mode_i_Ez = data_dict['TM'][f"{mode_i}"]['3D_Efield']['Ez']
        mode_j_Ez = data_dict['TM'][f"{mode_j}"]['3D_Efield']['Ez']

        # plt.imshow(mode_i_Ez[75, :, :])
        # plt.show()
        mode_i_Ex = data_dict['TM'][f"{mode_i}"]['3D_Efield']['Ex']
        mode_i_Ey = data_dict['TM'][f"{mode_i}"]['3D_Efield']['Ey']
        mode_i_Ez = data_dict['TM'][f"{mode_i}"]['3D_Efield']['Ez']
        mode_j_Ex = data_dict['TM'][f"{mode_j}"]['3D_Efield']['Ex']
        mode_j_Ey = data_dict['TM'][f"{mode_j}"]['3D_Efield']['Ey']
        mode_j_Ez = data_dict['TM'][f"{mode_j}"]['3D_Efield']['Ez']



        E1 = {
            "Ex": mode_i_Ex,
            "Ey": mode_i_Ey,
            "Ez": mode_i_Ez,
        }

        E2= {
            "Ex": mode_j_Ex,
            "Ey": mode_j_Ey,
            "Ez": mode_j_Ez,
        }

        array_path = f"{savepath}\\TM{mode_i}_TM{mode_j}"
        field_data = hamm.get_3D_data_monopole(
        E1,
        E2,
        array_path,
        plot = False,
        create_fields = create_fields,
        coord_unit = "mm",
        )

        # abs_add_vert_sec = field_data['abs_add'][75, :, :]
        # plt.imshow(abs_add_vert_sec)
        # plt.show()

        # savepath = array_path
        # datapath = d["datapath"]
        # f_010 = d["f_010"]
        f_mnp = TM_crossing_results['TM']['crossings'][key]['frequency_Hz']
        f_E1 = data_dict['TM'][f"{mode_i}"]['design_frequency_Hz']
        f_E2 = data_dict['TM'][f"{mode_j}"]['design_frequency_Hz']
        l_factor = TM_crossing_results['TM']['crossings'][key]['length_factor']
        Req_m = hamm.pillbox_radius_from_freq(f_010)


        w_mnp = f_mnp * 2. * np.pi
        lambda_010 = c_sol / f_010
        ell_1_m = lambda_010 / 2.

        Vz_axis_E1, loss_axis_E1 = hamm.Vz_loss_from_field(
            array_path,
            "E1_Ez.npy",
            f_E1,
            f_mnp=f_mnp,
            l_factor=1.,
            Req_m=Req_m,
            Rir_div_Req=0.25,
        )

        Vz_axis_E2, loss_axis_E2 = hamm.Vz_loss_from_field(
            array_path,
            "E2_Ez.npy",
            f_E2,
            f_mnp=f_mnp,
            l_factor=1.,
            Req_m=Req_m,
            Rir_div_Req=0.25,
        )

        Vz_axis_plus, loss_axis_plus = hamm.Vz_loss_from_field(
            array_path,
            "Ez_plus.npy",
            f_010,
            f_mnp=f_mnp,
            l_factor=l_factor,
            Req_m=Req_m,
            Rir_div_Req=0.25,
        )

        Vz_axis_minus, loss_axis_minus = hamm.Vz_loss_from_field(
            array_path,
            "Ez_minus.npy",
            f_010,
            f_mnp=f_mnp,
            l_factor=l_factor,
            Req_m=Req_m,
            Rir_div_Req=0.25,
        )

        print(f"\n{Vz_axis_E1 = }")
        print(f"{loss_axis_E1 = }\n")

        print(f"{Vz_axis_E2 = }")
        print(f"{loss_axis_E2 = }\n")

        print(f"{Vz_axis_plus = }")
        print(f"{loss_axis_plus = }\n")

        print(f"{Vz_axis_minus = }")
        print(f"{loss_axis_minus = }\n")

        field_keys = [
            "abs_E1",
            "abs_E2",
            "abs_add",
            "abs_sub",
            "E1_Ex",
            "E1_Ey",
            "E1_Ez",
            "E2_Ex",
            "E2_Ey",
            "E2_Ez",
            "trans_E1",
            "trans_E2",
            "Ex_plus",
            "Ey_plus",
            "Ez_plus",
            "Ex_minus",
            "Ey_minus",
            "Ez_minus",
        ]

        slice_dict = hamm.extract_3d_field_slices(field_data, field_keys)

        max_trans_E1 = np.max(field_data["trans_E1"])
        max_longit_E1 = np.max(np.abs(field_data["E1_Ez"]))

        max_trans_E2 = np.max(field_data["trans_E2"])
        max_longit_E2 = np.max(np.abs(field_data["E2_Ez"]))

        # print(
        #     f"{max_trans_E1 = }\n"
        #     f"{max_longit_E1 = }\n"
        #     f"{max_trans_E2 = }\n"
        #     f"{max_longit_E2 = }\n"
        # )

        with open(
                f"{array_path}\\TM_slice_dict.pkl",
                "wb",
        ) as handle:
            pkl.dump(slice_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # for key in slice_dict.keys():
        #     print(f"{key}")

        figs_plus, axes_out_plus, max_dict_plus = hamm.plot_all_plus(
            slice_dict,
            save_directory_fname=f"{array_path}\\TM_plus",
        )
        figs_minus, axes_out_minus, max_dict_minus = hamm.plot_all_minus(
            slice_dict,
            save_directory_fname=f"{array_path}\\TM_plus",
        )







