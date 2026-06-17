"""Run analytical homotypic TM m=1 dipole crossing analysis.

Use with HOMmix_analytical_master_module_dipole_stripped.py in the same folder.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import HOMmix_analytical_master_module_dipole as hamm


def assemble_all_dipole_data_dict(
    *,
    n_max: int = 3,
    p_max: int = 3,
    frequency_010: float = 1.3e9,
    LF_start: float = 0.7,
    LF_stop: float = 1.3,
    param_sweep_resolution: int = 1000,
    voxel_res: int = 151,
):
    """Build TM_1np frequency sweeps and design-length field maps."""
    lambda_010 = hamm.C0 / frequency_010
    design_L = lambda_010 / 2.0
    R = hamm.pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, param_sweep_resolution)

    all_data = {"TM": {}, "length_factor_vector": length_factors.tolist(), "metadata": {"frequency_010_Hz": frequency_010, "R_m": R, "design_L_m": design_L}}

    for p in range(p_max + 1):
        for n in range(1, n_max + 1):
            m = 1
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = hamm.pillbox_field_voxel_grid_xyz(R, design_L, m, n, p, voxel_res, voxel_res, voxel_res, E0=1.0, mode="TM")
            freqs = [hamm.f_tm(m, n, p, R, lf*design_L) for lf in length_factors]
            design_freq = hamm.f_tm(m, n, p, R, design_L)
            all_data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": freqs,
                "frequency_normalised": (np.asarray(freqs)/frequency_010).tolist(),
                "design_frequency_Hz": design_freq,
                "design_frequency_normalised": design_freq/frequency_010,
            }
    return all_data


def component_summary(field: dict) -> dict:
    s = {}
    for k in ["Ex", "Ey", "Ez", "Eperp", "|E|"]:
        arr = np.asarray(field[k])
        s[f"max_abs_{k}"] = float(np.nanmax(np.abs(arr)))
    ez = s["max_abs_Ez"]
    s["max_abs_Ex_over_Ez"] = float(s["max_abs_Ex"] / ez) if ez > 0 else np.inf
    s["max_abs_Ey_over_Ez"] = float(s["max_abs_Ey"] / ez) if ez > 0 else np.inf
    return s


def analyse_crossing(key: str, crossing: dict, data_dict: dict, save_root: Path, f_010: float, voxel_res: int, create_fields: bool = True) -> dict:
    mode_i = crossing["mode_i"].split("_", 1)[1]
    mode_j = crossing["mode_j"].split("_", 1)[1]
    out_dir = save_root / f"TM{mode_i}_TM{mode_j}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_i = data_dict["TM"][mode_i]["3D_Efield"]
    raw_j = data_dict["TM"][mode_j]["3D_Efield"]

    rot_i = hamm.align_field_to_vertical_plane(raw_i, out_plot=str(out_dir / f"TM{mode_i}_theta_r_rotation.png"), label=f"TM{mode_i}")
    rot_j = hamm.align_field_to_vertical_plane(raw_j, out_plot=str(out_dir / f"TM{mode_j}_theta_r_rotation.png"), label=f"TM{mode_j}")

    E1 = {"Ex": rot_i["Ex"], "Ey": rot_i["Ey"], "Ez": rot_i["Ez"]}
    E2 = {"Ex": rot_j["Ex"], "Ey": rot_j["Ey"], "Ez": rot_j["Ez"]}
    field_data = hamm.combine_fields(E1, E2)
    hamm.save_field_data_npz(field_data, str(out_dir / "field_data.npz"))
    hamm.pickle_save(hamm.extract_slices(field_data), out_dir / "slice_dict.pkl")
    hamm.plot_field_slices(field_data, str(out_dir / "plots"), title=f"TM{mode_i} / TM{mode_j}")

    f_cross = float(crossing["frequency_Hz"])
    lf_cross = float(crossing["length_factor"])
    Req_m = hamm.pillbox_radius_from_freq(f_010)

    kick_jobs = {
        "E1": (field_data["E1_Ez"], float(data_dict["TM"][mode_i]["design_frequency_Hz"]), 1.0),
        "E2": (field_data["E2_Ez"], float(data_dict["TM"][mode_j]["design_frequency_Hz"]), 1.0),
        "plus": (field_data["Ez_plus"], f_cross, lf_cross),
        "minus": (field_data["Ez_minus"], f_cross, lf_cross),
    }
    kicks = {}
    for name, (Ez, freq, lf) in kick_jobs.items():
        kicks[name] = hamm.kick_from_Ez_field(
            Ez,
            f_010=f_010,
            f_mnp=freq,
            l_factor=lf,
            Req_m=Req_m,
            axis="y",
            fit_pixels=8,
        )

    analysis = {
        "crossing_key": key,
        "crossing": crossing,
        "mode_i": mode_i,
        "mode_j": mode_j,
        "component_summary_i": component_summary(raw_i),
        "component_summary_j": component_summary(raw_j),
        "rotation_i": {"angle_deg": rot_i["rotation_angle_deg"], "peak_before": rot_i["peak_before"], "peak_after": rot_i["peak_after"]},
        "rotation_j": {"angle_deg": rot_j["rotation_angle_deg"], "peak_before": rot_j["peak_before"], "peak_after": rot_j["peak_after"]},
        "kick_units": "V/C/m/m",
        "kicks": kicks,
        "files": {"field_data_npz": str(out_dir / "field_data.npz"), "plots": str(out_dir / "plots")},
    }
    hamm.pickle_save(analysis, out_dir / "crossing_analysis.pkl")

    print(f"\n{key}")
    print(f"  TM{mode_i} rotation: {analysis['rotation_i']}")
    print(f"  TM{mode_j} rotation: {analysis['rotation_j']}")
    for name, k in kicks.items():
        print(f"  {name:5s} kick = {k['kick_V_per_C_per_m_per_m']:.6e} V/C/m/m")
    return analysis


def main():
    # Edit these two paths for your machine.
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    savepath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles")


    n_max, p_max = 3, 3
    voxel_res = 151
    f_010 = 1.3e9
    create_data = False
    create_fields = True

    datapath.mkdir(parents=True, exist_ok=True)
    savepath.mkdir(parents=True, exist_ok=True)
    data_file = datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl"

    if create_data:
        data_dict = assemble_all_dipole_data_dict(
            n_max=n_max,
            p_max=p_max,
            frequency_010=f_010,
            LF_start=0.7,
            LF_stop=1.3,
            param_sweep_resolution=1000,
            voxel_res=voxel_res,
        )
        hamm.pickle_save(data_dict, data_file)
    else:
        data_dict = hamm.pickle_load(data_file)

    # Diagnostic: Ex can be > Ez for p>0, but should not be enormous due to a missing 1/kc scaling.
    print("\nComponent max summaries at design length:")
    for mnp, d in data_dict["TM"].items():
        s = component_summary(d["3D_Efield"])
        print(f"  TM{mnp}: max|Ex|/max|Ez|={s['max_abs_Ex_over_Ez']:.4g}, max|Ey|/max|Ez|={s['max_abs_Ey_over_Ez']:.4g}")

    crossing_results = hamm.find_mode_crossings_from_all_data(data_dict, mode_type="TM")
    hamm.pickle_save(crossing_results, savepath / "crossing_results.pkl")

    analyses = {}
    for key, crossing in crossing_results["TM"]["crossings"].items():
        analyses[key] = analyse_crossing(key, crossing, data_dict, savepath, f_010, voxel_res, create_fields=create_fields)
    hamm.pickle_save(analyses, savepath / "all_crossing_analyses.pkl")


if __name__ == "__main__":
    main()
