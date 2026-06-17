"""Heterotypic TM pillbox mode-crossing and field-mixing study.

This script is intended to be run after, or instead of, the separate monopole,
dipole and quadrupole analyses.  It assembles the length-factor parameter sweep
for TM_0np, TM_1np and TM_2np modes, finds heterotypic crossings of three types,
then saves parent/mixed field data and plots.

Crossing types
--------------
    monopole-dipole      : TM_0np with TM_1np
    dipole-quadrupole    : TM_1np with TM_2np
    monopole-quadrupole  : TM_0np with TM_2np

The code deliberately stops after generating field data and plots.  It does not
calculate monopole loss, dipole kick or quadrupole focusing figures of merit for
heterotypic mixtures, because the mixed fields do not have a single pure
multipole order.  Those quantities can be recovered later by fitting a modal or
Taylor expansion to the near-axis voltage map.

Requires
--------
Place this file in the same folder as
    HOMmix_analytical_master_module_quadrupole_stripped.py

That helper module is reused because it contains a corrected analytical TM field
builder, vector-field rotation, field combination and monopole-style plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import HOMmix_analytical_master_module_quadrupole as hamm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    # Edit for your machine.
    datapath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    savepath: Path = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings")

    # Mode range: m is fixed by family; n and p match the previous studies.
    n_max: int = 3
    p_max: int = 3

    # Pillbox / sweep settings.
    f_010: float = 1.3e9
    LF_start: float = 0.7
    LF_stop: float = 1.3
    param_sweep_resolution: int = 1000
    voxel_res: int = 151

    # Save/load behaviour.
    create_data: bool = False     # False = load previous family pkl if present.
    create_fields: bool = False   # False = load previous field_data.npz if present.
    make_plots: bool = True

    # Families to analyse.  Do not change unless you know why.
    families: tuple[int, ...] = (0, 1, 2)


FAMILY_LABEL = {
    0: "monopole",
    1: "dipole",
    2: "quadrupole",
}

PAIR_TYPES = {
    "monopole_dipole": (0, 1),
    "dipole_quadrupole": (1, 2),
    "monopole_quadrupole": (0, 2),
}


# -----------------------------------------------------------------------------
# Basic save/load helpers
# -----------------------------------------------------------------------------

def pickle_save(obj, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename: str | Path):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_npz_dict(filename: str | Path, data: dict[str, np.ndarray]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, **data)


def load_npz_dict(filename: str | Path) -> dict[str, np.ndarray]:
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# Family data: load previous sweeps if available, otherwise create again
# -----------------------------------------------------------------------------

def family_data_filename(datapath: Path, m: int, voxel_res: int) -> Path:
    return datapath / f"TMm{m}_TMm{m}_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl"


def assemble_family_data(
    *,
    m: int,
    n_max: int,
    p_max: int,
    frequency_010: float,
    LF_start: float,
    LF_stop: float,
    param_sweep_resolution: int,
    voxel_res: int,
) -> dict:
    """Build one TM_mnp family: frequency sweeps and design-length field maps."""
    lambda_010 = hamm.C0 / float(frequency_010)
    design_L = lambda_010 / 2.0
    R = hamm.pillbox_radius_from_freq(frequency_010)
    length_factors = np.linspace(LF_start, LF_stop, param_sweep_resolution)

    all_data = {
        "TM": {},
        "length_factor_vector": length_factors.tolist(),
        "metadata": {
            "family_m": int(m),
            "family_label": FAMILY_LABEL.get(int(m), f"m{m}"),
            "frequency_010_Hz": float(frequency_010),
            "R_m": float(R),
            "design_L_m": float(design_L),
            "voxel_res": int(voxel_res),
        },
    }

    for p in range(p_max + 1):
        for n in range(1, n_max + 1):
            mnp = f"{m}{n}{p}"
            print(f"Building TM{mnp}")
            field = hamm.pillbox_field_voxel_grid_xyz(
                R, design_L, m, n, p,
                voxel_res, voxel_res, voxel_res,
                E0=1.0,
                mode="TM",
            )
            freqs = [hamm.f_tm(m, n, p, R, lf * design_L) for lf in length_factors]
            design_freq = hamm.f_tm(m, n, p, R, design_L)
            all_data["TM"][mnp] = {
                "3D_Efield": field,
                "frequency_Hz": list(map(float, freqs)),
                "frequency_normalised": (np.asarray(freqs) / float(frequency_010)).tolist(),
                "design_frequency_Hz": float(design_freq),
                "design_frequency_normalised": float(design_freq / frequency_010),
            }

    return all_data


def load_or_create_family_data(config: RunConfig, m: int) -> dict:
    config.datapath.mkdir(parents=True, exist_ok=True)
    fname = family_data_filename(config.datapath, m, config.voxel_res)

    if (not config.create_data) and fname.exists():
        print(f"Loading TM m={m} family data from {fname}")
        return pickle_load(fname)

    print(f"Creating TM m={m} family data")
    data = assemble_family_data(
        m=m,
        n_max=config.n_max,
        p_max=config.p_max,
        frequency_010=config.f_010,
        LF_start=config.LF_start,
        LF_stop=config.LF_stop,
        param_sweep_resolution=config.param_sweep_resolution,
        voxel_res=config.voxel_res,
    )
    pickle_save(data, fname)
    return data


# -----------------------------------------------------------------------------
# Sweep assembly and crossing detection
# -----------------------------------------------------------------------------

def assemble_sweep_table(family_data: dict[int, dict], f_010: float) -> dict[str, dict]:
    """Flatten all length-factor sweeps into one mode-indexed dictionary."""
    table: dict[str, dict] = {}

    for m, data in family_data.items():
        L = np.asarray(data["length_factor_vector"], dtype=float)
        for mnp, entry in data["TM"].items():
            f_Hz = np.asarray(entry["frequency_Hz"], dtype=float)
            table[f"TM_{mnp}"] = {
                "family_m": int(m),
                "family_label": FAMILY_LABEL.get(int(m), f"m{m}"),
                "mnp": mnp,
                "length_factor": L,
                "frequency_Hz": f_Hz,
                "frequency_normalised": f_Hz / float(f_010),
                "design_frequency_Hz": float(entry["design_frequency_Hz"]),
            }

    return table


def detect_pair_crossings(
    sweep_table: dict[str, dict],
    *,
    m_a: int,
    m_b: int,
    pair_name: str,
) -> dict[str, dict]:
    """Find crossings between every TM_m_a mode and every TM_m_b mode."""
    modes_a = [name for name, d in sweep_table.items() if d["family_m"] == m_a]
    modes_b = [name for name, d in sweep_table.items() if d["family_m"] == m_b]

    crossings: dict[str, dict] = {}

    for name_a in modes_a:
        L = np.asarray(sweep_table[name_a]["length_factor"], dtype=float)
        fa = np.asarray(sweep_table[name_a]["frequency_Hz"], dtype=float)
        for name_b in modes_b:
            fb = np.asarray(sweep_table[name_b]["frequency_Hz"], dtype=float)
            if fa.shape != fb.shape:
                raise ValueError(f"Sweep shape mismatch for {name_a} and {name_b}")

            g = fa - fb
            candidate_indices = np.where(g[:-1] * g[1:] <= 0.0)[0]

            for idx in candidate_indices:
                # Avoid counting an exactly-zero plateau more than once.
                if np.isclose(g[idx], 0.0) and idx > 0 and np.isclose(g[idx - 1], 0.0):
                    continue

                if np.isclose(g[idx], 0.0):
                    Lc = float(L[idx])
                elif np.isclose(g[idx + 1], 0.0):
                    Lc = float(L[idx + 1])
                else:
                    Lc = float(brentq(
                        lambda xx: np.interp(xx, L, fa) - np.interp(xx, L, fb),
                        float(L[idx]),
                        float(L[idx + 1]),
                    ))

                fc = float(np.interp(Lc, L, fa))
                key = f"{pair_name}:{name_a}--{name_b}@{Lc:.8g}"
                crossings[key] = {
                    "pair_type": pair_name,
                    "mode_i": name_a,
                    "mode_j": name_b,
                    "m_i": int(m_a),
                    "m_j": int(m_b),
                    "length_factor": Lc,
                    "frequency_Hz": fc,
                    "frequency_normalised": fc / float(sweep_table[name_a]["frequency_Hz"][0] * 0 + 1.0),  # overwritten below
                }
                # Use the same f010 used for the sweeps.  It is not stored here,
                # so infer it from normalised/current for mode_a at the crossing.
                fhat_a = float(np.interp(Lc, L, sweep_table[name_a]["frequency_normalised"]))
                crossings[key]["frequency_normalised"] = fhat_a

    return crossings


def find_heterotypic_crossings(sweep_table: dict[str, dict]) -> dict[str, dict]:
    all_crossings: dict[str, dict] = {}
    for pair_name, (m_a, m_b) in PAIR_TYPES.items():
        pair_crossings = detect_pair_crossings(
            sweep_table,
            m_a=m_a,
            m_b=m_b,
            pair_name=pair_name,
        )
        all_crossings[pair_name] = pair_crossings
        print(f"{pair_name}: found {len(pair_crossings)} crossings")
    return all_crossings


# -----------------------------------------------------------------------------
# Plot sweeps and crossings
# -----------------------------------------------------------------------------

def plot_pair_sweeps(
    sweep_table: dict[str, dict],
    crossings_by_pair: dict[str, dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, (m_a, m_b) in PAIR_TYPES.items():
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for name, d in sweep_table.items():
            if d["family_m"] not in (m_a, m_b):
                continue
            L = d["length_factor"]
            fhat = d["frequency_normalised"]
            ls = "-" if d["family_m"] == m_a else "--"
            ax.plot(L, fhat, ls=ls, lw=1.0, alpha=0.8, label=name)

        for c in crossings_by_pair[pair_name].values():
            ax.scatter(c["length_factor"], c["frequency_normalised"], s=50, facecolors="none", edgecolors="k")

        ax.set_xlabel(r"Length factor, $\ell=L/L_0$")
        ax.set_ylabel(r"Normalised frequency, $\hat{f}=f/f_{010}$")
        ax.set_title(pair_name.replace("_", "-"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
        fig.savefig(out_dir / f"{pair_name}_sweeps.png", dpi=300)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Field preparation, rotation, mixing, saving and plotting
# -----------------------------------------------------------------------------

def parse_mode_name(mode_name: str) -> tuple[int, str]:
    # mode name format: "TM_012"
    _, mnp = mode_name.split("_", 1)
    return int(mnp[0]), mnp


def rotation_label(rot: dict | None) -> dict | None:
    if rot is None:
        return None
    keys = ["rotation_angle_deg", "peak_before", "peak_after", "global_max_after", "vertical_midplane_max_after"]
    return {k: rot.get(k) for k in keys if k in rot}


def prepare_field_for_mixing(field: dict, *, m: int, label: str, out_dir: Path) -> tuple[dict, dict | None]:
    """Return {'Ex','Ey','Ez'} for mixing, rotating non-monopole families.

    The monopole field is axisymmetric and is left unrotated.  Dipole and
    quadrupole fields are rotated with the same routine used in the previous
    analyses so that the dominant |E| structure is aligned with the vertical
    x=0 plane as well as possible.
    """
    if m == 0:
        return {"Ex": field["Ex"], "Ey": field["Ey"], "Ez": field["Ez"]}, None

    rot = hamm.align_field_to_vertical_plane(
        field,
        out_plot=str(out_dir / f"{label}_theta_r_rotation.png"),
        label=label,
    )
    return {"Ex": rot["Ex"], "Ey": rot["Ey"], "Ez": rot["Ez"]}, rot


def get_or_create_heterotypic_field_data(
    crossing_key: str,
    crossing: dict,
    family_data: dict[int, dict],
    out_dir: Path,
    *,
    create_fields: bool,
) -> dict:
    field_file = out_dir / "field_data.npz"
    analysis_file = out_dir / "heterotypic_crossing_analysis.pkl"

    if (not create_fields) and field_file.exists() and analysis_file.exists():
        return pickle_load(analysis_file)

    m_i, mnp_i = parse_mode_name(crossing["mode_i"])
    m_j, mnp_j = parse_mode_name(crossing["mode_j"])

    raw_i = family_data[m_i]["TM"][mnp_i]["3D_Efield"]
    raw_j = family_data[m_j]["TM"][mnp_j]["3D_Efield"]

    E1, rot_i = prepare_field_for_mixing(raw_i, m=m_i, label=f"TM{mnp_i}", out_dir=out_dir)
    E2, rot_j = prepare_field_for_mixing(raw_j, m=m_j, label=f"TM{mnp_j}", out_dir=out_dir)

    field_data = hamm.combine_fields(E1, E2)
    save_npz_dict(field_file, field_data)
    pickle_save(hamm.extract_slices(field_data), out_dir / "slice_dict.pkl")

    if hasattr(hamm, "plot_field_slices"):
        hamm.plot_field_slices(field_data, str(out_dir / "plots"), title=f"{crossing['mode_i']} / {crossing['mode_j']}")

    analysis = {
        "crossing_key": crossing_key,
        "crossing": crossing,
        "mode_i": crossing["mode_i"],
        "mode_j": crossing["mode_j"],
        "m_i": m_i,
        "m_j": m_j,
        "mnp_i": mnp_i,
        "mnp_j": mnp_j,
        "rotation_i": rotation_label(rot_i),
        "rotation_j": rotation_label(rot_j),
        "files": {
            "field_data_npz": str(field_file),
            "slice_dict_pkl": str(out_dir / "slice_dict.pkl"),
            "plots_dir": str(out_dir / "plots"),
        },
        "note": (
            "Heterotypic field mixing only.  No scalar loss/kick/focusing figure "
            "is assigned here because the mixed field is not a pure multipole."
        ),
    }
    pickle_save(analysis, analysis_file)
    return analysis


def safe_folder_name(crossing_key: str) -> str:
    s = crossing_key.replace(":", "__").replace("@", "__ell_")
    s = s.replace("--", "__")
    s = s.replace(".", "p")
    return s


def process_heterotypic_crossings(
    crossings_by_pair: dict[str, dict],
    family_data: dict[int, dict],
    savepath: Path,
    *,
    create_fields: bool,
) -> dict[str, dict]:
    analyses: dict[str, dict] = {}

    for pair_name, pair_crossings in crossings_by_pair.items():
        pair_dir = savepath / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        analyses[pair_name] = {}

        for key, crossing in pair_crossings.items():
            print(f"Processing {key}")
            out_dir = pair_dir / safe_folder_name(key)
            out_dir.mkdir(parents=True, exist_ok=True)
            analyses[pair_name][key] = get_or_create_heterotypic_field_data(
                key,
                crossing,
                family_data,
                out_dir,
                create_fields=create_fields,
            )

    pickle_save(analyses, savepath / "all_heterotypic_crossing_analyses.pkl")
    return analyses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    config = RunConfig()
    config.datapath.mkdir(parents=True, exist_ok=True)
    config.savepath.mkdir(parents=True, exist_ok=True)

    family_data = {m: load_or_create_family_data(config, m) for m in config.families}

    sweep_table = assemble_sweep_table(family_data, config.f_010)
    pickle_save(sweep_table, config.savepath / "heterotypic_sweep_table.pkl")

    crossings_by_pair = find_heterotypic_crossings(sweep_table)
    pickle_save(crossings_by_pair, config.savepath / "heterotypic_crossing_results.pkl")

    if config.make_plots:
        plot_pair_sweeps(sweep_table, crossings_by_pair, config.savepath / "sweep_plots")

    process_heterotypic_crossings(
        crossings_by_pair,
        family_data,
        config.savepath,
        create_fields=config.create_fields,
    )

    print("\nDone.  Heterotypic crossing fields and plots have been saved.")
    print("This script intentionally stops before calculating loss/kick/focusing metrics.")


if __name__ == "__main__":
    main()
