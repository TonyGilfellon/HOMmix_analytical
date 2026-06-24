"""Run analytical homotypic TM m=2 quadrupole crossing analysis.

Use with HOMmix_analytical_master_module_quadrupole_stripped.py in the same folder.

Array convention: field[x_index, y_index, z_index].
Plot convention is handled by the helper module:
  - iris/transverse slices: x horizontal, y vertical
  - longitudinal slices: z horizontal, y vertical

The quadrupole focusing study fits the near-axis complex longitudinal voltage
Vz(x,y) to a quadratic polynomial and applies the Panofsky-Wenzel relation to
estimate the transverse-voltage gradient matrix in V/C/m/m.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import HOMmix_analytical_master_module_quadrupole as hamm
import matplotlib.pyplot as plt


def assemble_all_quadrupole_data_dict(
    *,
    n_max: int = 3,
    p_max: int = 3,
    frequency_010: float = 1.3e9,
    LF_start: float = 0.7,
    LF_stop: float = 1.3,
    param_sweep_resolution: int = 1000,
    voxel_res: int = 151,
):
    """Build TM_2np frequency sweeps and design-length field maps."""
    lambda_010 = hamm.C0 / frequency_010
    design_L = lambda_010 / 2.0
    R = hamm.pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, param_sweep_resolution)

    all_data = {
        "TM": {},
        "length_factor_vector": length_factors.tolist(),
        "metadata": {
            "frequency_010_Hz": frequency_010,
            "R_m": R,
            "design_L_m": design_L,
            "m": 2,
            "n_range": [1, n_max],
            "p_range": [0, p_max],
        },
    }

    for p in range(p_max + 1):
        for n in range(1, n_max + 1):
            m = 2
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = hamm.pillbox_field_voxel_grid_xyz(
                R,
                design_L,
                m,
                n,
                p,
                voxel_res,
                voxel_res,
                voxel_res,
                E0=1.0,
                mode="TM",
            )
            freqs = [hamm.f_tm(m, n, p, R, lf * design_L) for lf in length_factors]
            design_freq = hamm.f_tm(m, n, p, R, design_L)
            all_data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": freqs,
                "frequency_normalised": (np.asarray(freqs) / frequency_010).tolist(),
                "design_frequency_Hz": design_freq,
                "design_frequency_normalised": design_freq / frequency_010,
            }
    return all_data


def component_summary(field: dict) -> dict:
    """Max component diagnostics to catch bad field construction."""
    s = {}
    for k in ["Ex", "Ey", "Ez", "Eperp", "|E|"]:
        arr = np.asarray(field[k])
        s[f"max_abs_{k}"] = float(np.nanmax(np.abs(arr)))
    ez = s["max_abs_Ez"]
    s["max_abs_Ex_over_Ez"] = float(s["max_abs_Ex"] / ez) if ez > 0 else np.inf
    s["max_abs_Ey_over_Ez"] = float(s["max_abs_Ey"] / ez) if ez > 0 else np.inf
    return s


def focusing_summary_line(name: str, result: dict) -> str:
    """Compact one-line focusing/defocusing summary."""
    ecls = result["electron_force_classification"]
    return (
        f"  {name:5s}: Kxx={result['Kxx_V_per_C_per_m_per_m']:.6e}, "
        f"Kxy={result['Kxy_V_per_C_per_m_per_m']:.6e}, "
        f"Kyy={result['Kyy_V_per_C_per_m_per_m']:.6e} V/C/m/m; "
        f"electron x={ecls['x']}, y={ecls['y']}"
    )



def plot_field_slices_combined(field_data: dict, out_dir: str | Path, title: str = "") -> None:
    """Save 4x4 combined plots with columns [E1, E2, E+, E-].

    Saved files:
        iris_1.png
        iris_2.png
        transverse_mid.png
        longitudinal_mid.png

    Rows:
        Ex, Ey, Ez, |E|

    Colour meaning:
        - Ex, Ey and Ez rows are divided by max(abs(E1_Ez), abs(E2_Ez)).
          Therefore colourbar value 1 means the parent Ez reference amplitude.
        - |E| row is divided by max(|E1|, |E2|).
          Therefore colourbar value 1 means the parent |E| reference amplitude.
        - Colourbar limits extend beyond +/-1 or 1 if E+ or E- exceed the
          parent reference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_specs = {
        "iris_1": lambda F: np.asarray(F)[:, :, 0].T,
        "iris_2": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] - 1].T,
        "transverse_mid": lambda F: np.asarray(F)[:, :, np.asarray(F).shape[2] // 2].T,
        "longitudinal_mid": lambda F: np.asarray(F)[np.asarray(F).shape[0] // 2, :, :],
    }

    rows = [
        ("E1_Ex", "E2_Ex", "Ex_plus", "Ex_minus"),
        ("E1_Ey", "E2_Ey", "Ey_plus", "Ey_minus"),
        ("E1_Ez", "E2_Ez", "Ez_plus", "Ez_minus"),
        ("abs_E1", "abs_E2", "abs_plus", "abs_minus"),
    ]

    column_titles = ["E₁", "E₂", "E₊", "E₋"]
    row_titles = [
        "Eₓ / Ez ref",
        "Eᵧ / Ez ref",
        "Ez / Ez ref",
        "|E| / |E| ref",
    ]

    def _real_image(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        return np.real(a) if np.iscomplexobj(a) else a

    def _safe_ref(arrays: list[np.ndarray]) -> float:
        ref = max(float(np.nanmax(np.abs(a))) for a in arrays)
        if not np.isfinite(ref) or ref <= 0.0:
            ref = 1.0
        return ref

    def _safe_vmax(arrays: list[np.ndarray], ref: float) -> float:
        scaled_max = max(float(np.nanmax(np.abs(a / ref))) for a in arrays)
        if not np.isfinite(scaled_max) or scaled_max <= 0.0:
            scaled_max = 1.0
        return max(1.0, scaled_max)

    for stype, slicer in slice_specs.items():
        fig, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"{title} : plus/minus comparison : {stype}")

        parent_ez_ref = _safe_ref([
            slicer(_real_image(field_data["E1_Ez"])),
            slicer(_real_image(field_data["E2_Ez"])),
        ])

        parent_abs_ref = _safe_ref([
            slicer(_real_image(field_data["abs_E1"])),
            slicer(_real_image(field_data["abs_E2"])),
        ])

        for r, row_keys in enumerate(rows):
            raw_row_data = [slicer(_real_image(field_data[k])) for k in row_keys]

            is_abs_row = r == 3
            if is_abs_row:
                ref = parent_abs_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmin = 0.0
                vmax = _safe_vmax(raw_row_data, ref)
                cmap = "viridis"
            else:
                ref = parent_ez_ref
                row_data = [arr / ref for arr in raw_row_data]
                vmax = _safe_vmax(raw_row_data, ref)
                vmin = -vmax
                cmap = "RdBu_r"

            for c, (key, arr_raw, arr_scaled) in enumerate(zip(row_keys, raw_row_data, row_data)):
                ax = axes[r, c]
                im = ax.imshow(
                    arr_scaled,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )

                if r == 0:
                    ax.text(
                        0.5,
                        1.02,
                        column_titles[c],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontstyle="normal",
                        fontweight="bold",
                        zorder=100,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=2.0),
                        clip_on=False,
                    )

                if c == 0:
                    ax.text(
                        -0.12,
                        0.5,
                        row_titles[r],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=11,
                        zorder=100,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=2.0),
                        clip_on=False,
                    )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.02,
                    0.98,
                    f"max={np.nanmax(np.abs(arr_raw)):.2e}\n"
                    f"norm={np.nanmax(np.abs(arr_scaled)):.2g}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
                )

            fig.colorbar(im, ax=axes[r, :], fraction=0.02, pad=0.01)

        fig.savefig(out_dir / f"{stype}.png", dpi=300)
        plt.close(fig)

def analyse_crossing(
    key: str,
    crossing: dict,
    data_dict: dict,
    save_root: Path,
    f_010: float,
    voxel_res: int,
    *,
    create_fields: bool = True,
    fit_pixels: int = 8,
) -> dict:
    """Analyse one TM2np/TM2np crossing."""
    mode_i = crossing["mode_i"].split("_", 1)[1]
    mode_j = crossing["mode_j"].split("_", 1)[1]
    out_dir = save_root / f"TM{mode_i}_TM{mode_j}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_i = data_dict["TM"][mode_i]["3D_Efield"]
    raw_j = data_dict["TM"][mode_j]["3D_Efield"]

    # Rotate each parent field before combination. The helper verifies that the
    # maximum |E| voxel is in the vertical longitudinal plane |E|[mid,:,:].
    rot_i = hamm.align_field_to_vertical_plane(
        raw_i,
        out_plot=str(out_dir / f"TM{mode_i}_theta_r_rotation.png"),
        label=f"TM{mode_i}",
    )
    rot_j = hamm.align_field_to_vertical_plane(
        raw_j,
        out_plot=str(out_dir / f"TM{mode_j}_theta_r_rotation.png"),
        label=f"TM{mode_j}",
    )

    E1 = {"Ex": rot_i["Ex"], "Ey": rot_i["Ey"], "Ez": rot_i["Ez"]}
    E2 = {"Ex": rot_j["Ex"], "Ey": rot_j["Ey"], "Ez": rot_j["Ez"]}
    field_data = hamm.combine_fields(E1, E2)
    hamm.save_field_data_npz(field_data, str(out_dir / "field_data.npz"))
    hamm.pickle_save(hamm.extract_slices(field_data), out_dir / "slice_dict.pkl")
    hamm.plot_field_slices(field_data, str(out_dir / "plots"), title=f"TM{mode_i} / TM{mode_j}")
    plot_field_slices_combined(field_data, out_dir / "combined_plots", title=f"TM{mode_i} / TM{mode_j}")

    f_cross = float(crossing["frequency_Hz"])
    lf_cross = float(crossing["length_factor"])
    Req_m = hamm.pillbox_radius_from_freq(f_010)

    focus_jobs = {
        "E1": (field_data["E1_Ez"], float(data_dict["TM"][mode_i]["design_frequency_Hz"]), 1.0),
        "E2": (field_data["E2_Ez"], float(data_dict["TM"][mode_j]["design_frequency_Hz"]), 1.0),
        "plus": (field_data["Ez_plus"], f_cross, lf_cross),
        "minus": (field_data["Ez_minus"], f_cross, lf_cross),
    }

    focusing = {}
    for name, (Ez, freq, lf) in focus_jobs.items():
        focusing[name] = hamm.quadrupole_focusing_from_Ez_field(
            Ez,
            f_010=f_010,
            f_mnp=freq,
            l_factor=lf,
            Req_m=Req_m,
            fit_pixels=fit_pixels,
        )
        hamm.plot_quadrupole_voltage_fit(
            focusing[name],
            str(out_dir / f"quadrupole_fit_{name}.png"),
            title=f"TM{mode_i}/TM{mode_j} {name}",
        )

    analysis = {
        "crossing_key": key,
        "crossing": crossing,
        "mode_i": mode_i,
        "mode_j": mode_j,
        "component_summary_i": component_summary(raw_i),
        "component_summary_j": component_summary(raw_j),
        "rotation_i": {
            "angle_deg": rot_i["rotation_angle_deg"],
            "peak_before": rot_i["peak_before"],
            "peak_after": rot_i["peak_after"],
        },
        "rotation_j": {
            "angle_deg": rot_j["rotation_angle_deg"],
            "peak_before": rot_j["peak_before"],
            "peak_after": rot_j["peak_after"],
        },
        "focusing_units": "V/C/m/m",
        "focusing_sign_convention": (
            "K is the phase-aligned transverse-voltage gradient matrix. "
            "Positive Kxx deflects a positive test charge further +x; "
            "electron focusing/defocusing labels reverse that sign."
        ),
        "focusing": focusing,
        "files": {"field_data_npz": str(out_dir / "field_data.npz"), "plots": str(out_dir / "plots")},
    }
    hamm.pickle_save(analysis, out_dir / "crossing_analysis.pkl")

    print(f"\n{key}")
    print(f"  TM{mode_i} rotation: {analysis['rotation_i']}")
    print(f"  TM{mode_j} rotation: {analysis['rotation_j']}")
    for name, result in focusing.items():
        print(focusing_summary_line(name, result))
    return analysis


def main():
    # Edit these two paths for your machine.
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    savepath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles")

    n_max, p_max = 3, 3
    voxel_res = 151
    f_010 = 1.3e9
    create_data = True
    create_fields = True
    fit_pixels = 8

    datapath.mkdir(parents=True, exist_ok=True)
    savepath.mkdir(parents=True, exist_ok=True)
    data_file = datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl"

    if create_data:
        data_dict = assemble_all_quadrupole_data_dict(
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

    print("\nComponent max summaries at design length:")
    for mnp, d in data_dict["TM"].items():
        s = component_summary(d["3D_Efield"])
        print(
            f"  TM{mnp}: max|Ex|/max|Ez|={s['max_abs_Ex_over_Ez']:.4g}, "
            f"max|Ey|/max|Ez|={s['max_abs_Ey_over_Ez']:.4g}"
        )

    crossing_results = hamm.find_mode_crossings_from_all_data(data_dict, mode_type="TM")
    hamm.pickle_save(crossing_results, savepath / "crossing_results.pkl")

    analyses = {}
    for key, crossing in crossing_results["TM"]["crossings"].items():
        analyses[key] = analyse_crossing(
            key,
            crossing,
            data_dict,
            savepath,
            f_010,
            voxel_res,
            create_fields=create_fields,
            fit_pixels=fit_pixels,
        )
    hamm.pickle_save(analyses, savepath / "all_crossing_analyses.pkl")


if __name__ == "__main__":
    main()
