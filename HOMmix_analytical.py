import HOMmix_analytical_master_module as hamm
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.patches import Rectangle
import numpy as np
import pickle as pkl

""" Define Functions """


if __name__ == "__main__":
    datapath = r"D:\PhD\HOMmix\HOMmix_analytical\data"
    savepath = r"D:\PhD\HOMmix\HOMmix_analytical\analysis"

    process_data = False

    m_max, n_max, p_max = 2, 3, 3
    voxel_res = 151
    if process_data:
        all_data_dict = hamm.assemble_all_data_dict(m_max, n_max, p_max,
                                               frequency_010=1.3e9,
                                               LF_start=0.9,
                                               LF_stop=1.0,
                                               param_sweep_resolution=1000,
                                               voxel_res=voxel_res,
                                               )

        hamm.pickle_save(all_data_dict, f"{datapath}\\all_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl")
    else:
        all_data_dict = hamm.pickle_load(f"{datapath}\\all_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl")

        # for key, data in all_data_dict.items():
        #
        #     if type(data) == dict:
        #         print(data.keys())
        #         for key2, data2 in all_data_dict[key].items():
        #             print(f"{all_data_dict[key][key2] = }")
        #     else:
        #         print(f"{key}: {data}")
        #
        # input()


    # plt.imshow(all_data_dict['TM']['022']['3D_Efield']['Ez'][voxel_res//2, :, :])
    # plt.show()
    
    
    """ TM MODES """

    TM_crossing_results = hamm.find_mode_crossings_from_all_data(all_data_dict, mode_type="TM")
    
    TM_savenames = [
        'TM_monopoles', 
        'TM_dipoles', 
        'TM_quadrupoles', 
    ]
    
    m_filter_vals = [0, 1, 2]
    
    for idx, sn in enumerate(TM_savenames):
        hamm.plot_modes_from_all_data(
            all_data_dict,
            TM_crossing_results,
            savepath,
            sn,
            mode_type="TM",
            normalised=True,
            m_filter=m_filter_vals[idx],
        )

    counts, categories, all_crossings = hamm.plot_crossing_population_heatmap_TM(
        TM_crossing_results,
        savepath=savepath,
        savename="TM_crossing_heatmap_m012",
        inspect=False,
    )

    exit()

    """ TE MODES """

    TE_crossing_results = hamm.find_mode_crossings_from_all_data(all_data_dict, mode_type="TE")

    TE_savenames = [
        'TE_monopoles',
        'TE_dipoles',
        'TE_quadrupoles',
    ]

    m_filter_vals = [0, 1, 2]

    for idx, sn in enumerate(TE_savenames):
        hamm.plot_modes_from_all_data(
            all_data_dict,
            TE_crossing_results,
            savepath,
            sn,
            mode_type="TE",
            normalised=True,
            m_filter=m_filter_vals[idx],
        )
        
        

    """ BOTH MODES """

    BOTH_crossing_results = hamm.find_mode_crossings_from_all_data(all_data_dict, mode_type="BOTH")

    BOTH_savenames = [
        'BOTH_monopoles',
        'BOTH_dipoles',
        'BOTH_quadrupoles',
    ]

    m_filter_vals = [0, 1, 2]

    for idx, sn in enumerate(BOTH_savenames):
        hamm.plot_modes_from_all_data(
            all_data_dict,
            BOTH_crossing_results,
            savepath,
            sn,
            normalised=True,
            mode_type="BOTH",
            m_filter=m_filter_vals[idx],
        )

    hamm.plot_modes_from_all_data(
        all_data_dict,
        BOTH_crossing_results,
        savepath,
        'BOTH_m_indices_012',
        mode_type="BOTH",
        normalised=True,
        acceptance_fraction=0.01
    )

    counts, categories, all_crossings = hamm.plot_crossing_population_heatmap(
        BOTH_crossing_results,
        savepath=savepath,
        savename="crossing_heatmap_m012",
        inspect=False,
    )

    print(f"{counts = }")
    print(f"{categories = }")
    for idx, cross in enumerate(all_crossings):
        if np.abs(1.0 - float(cross['length_factor'])) <= 0.01:
            print(f"{cross}")




    # 3) generate the per-crossing figures into category folders
    made = hamm.generate_crossing_fieldmap_figures(
        all_data_dict,
        BOTH_crossing_results,
        counts,
        categories,
        out_root_dir=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\crossing_categories_and_analysis",
        field_container_key="3D_Efield",  # adjust if your key differs
        inspect=False,
    )
    print("Saved figures:", made)








