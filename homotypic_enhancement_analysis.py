"""Post-process homotypic pillbox mode-mixing analyses.

This script analyses outputs already saved by the monopole, dipole and quadrupole
workflows.  It does not re-run the field calculations.

It provides two main capabilities:

1. Enhancement tables for parent and mixed fields using

       R_max = max(M_plus, M_minus) / max(M_1, M_2)

   where M is the selected figure of merit.  For example, for dipole kicks,
   M is |kick| in V/C/m/m.

2. Reconstruction of the parameter-sweep plane, plotting length factor ell
   against normalised frequency f_hat = f / f_010, with detected crossings
   overlaid.

The functions are deliberately defensive about slightly different saved-output
structures from the mono-, di- and quadrupole scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import csv
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_save(obj: Any, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _finite_or_nan(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def safe_ratio(num: float, den: float) -> float:
    num = _finite_or_nan(num)
    den = _finite_or_nan(den)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(num / den)


def rmax_metric(parent_1: float, parent_2: float, plus: float, minus: float) -> dict[str, float | str]:
    """Return enhancement metrics for one scalar figure of merit.

    R_max follows the requested definition:

        R_max = max(K_+, K_-) / max(K_1, K_2)

    but the function is generic and can be applied to loss, kick, focusing, etc.
    """
    p1 = abs(_finite_or_nan(parent_1))
    p2 = abs(_finite_or_nan(parent_2))
    pp = abs(_finite_or_nan(plus))
    pm = abs(_finite_or_nan(minus))
    parent_max = max(p1, p2)
    mixed_max = max(pp, pm)
    winner = "plus" if pp >= pm else "minus"
    parent_winner = "E1" if p1 >= p2 else "E2"
    return {
        "E1": p1,
        "E2": p2,
        "plus": pp,
        "minus": pm,
        "parent_max": parent_max,
        "mixed_max": mixed_max,
        "R_max": safe_ratio(mixed_max, parent_max),
        "dominant_mixed": winner,
        "dominant_parent": parent_winner,
        "plus_over_parent_max": safe_ratio(pp, parent_max),
        "minus_over_parent_max": safe_ratio(pm, parent_max),
        "plus_over_parent_sum": safe_ratio(pp, p1 + p2),
        "minus_over_parent_sum": safe_ratio(pm, p1 + p2),
    }


# -----------------------------------------------------------------------------
# Normalise different saved analysis structures
# -----------------------------------------------------------------------------

def _analysis_items(saved: Any) -> list[tuple[str, dict[str, Any]]]:
    """Convert all_crossing_analyses.pkl content to [(key, analysis), ...]."""
    if isinstance(saved, dict):
        return [(str(k), v) for k, v in saved.items() if isinstance(v, dict)]
    if isinstance(saved, list):
        out = []
        for i, v in enumerate(saved):
            if isinstance(v, dict):
                key = v.get("crossing_key") or v.get("crossing", {}).get("key") or f"crossing_{i:04d}"
                out.append((str(key), v))
        return out
    raise TypeError(f"Unsupported saved analysis type: {type(saved)!r}")


def _crossing_label(analysis_key: str, analysis: dict[str, Any]) -> str:
    c = analysis.get("crossing", {})
    mi = analysis.get("mode_i") or analysis.get("modes", {}).get("E1") or c.get("mode_i", "E1")
    mj = analysis.get("mode_j") or analysis.get("modes", {}).get("E2") or c.get("mode_j", "E2")
    ell = c.get("length_factor")
    if ell is None:
        return str(analysis_key)
    return f"{mi}--{mj}@{float(ell):.8g}"


def _crossing_common_fields(analysis_key: str, analysis: dict[str, Any]) -> dict[str, Any]:
    c = analysis.get("crossing", {})
    return {
        "analysis_key": analysis_key,
        "crossing_label": _crossing_label(analysis_key, analysis),
        "mode_i": analysis.get("mode_i") or analysis.get("modes", {}).get("E1") or c.get("mode_i"),
        "mode_j": analysis.get("mode_j") or analysis.get("modes", {}).get("E2") or c.get("mode_j"),
        "length_factor": _finite_or_nan(c.get("length_factor", float("nan"))),
        "frequency_Hz": _finite_or_nan(c.get("frequency_Hz", float("nan"))),
    }


# -----------------------------------------------------------------------------
# Family-specific figure extraction
# -----------------------------------------------------------------------------

def _monopole_metrics(analysis: dict[str, Any]) -> dict[str, dict[str, float]]:
    a = analysis["analysis"]
    return {
        "loss": {k: abs(_finite_or_nan(a[k].get("loss", float("nan")))) for k in ("E1", "E2", "plus", "minus")},
        "Vz_abs": {k: abs(_finite_or_nan(a[k].get("Vz_V", float("nan")))) for k in ("E1", "E2", "plus", "minus")},
    }


def _dipole_metrics(analysis: dict[str, Any]) -> dict[str, dict[str, float]]:
    k = analysis["kicks"]
    return {
        "kick": {name: abs(_finite_or_nan(k[name].get("kick_V_per_C_per_m_per_m", float("nan")))) for name in ("E1", "E2", "plus", "minus")},
    }


def _quad_matrix_norm(q: dict[str, Any]) -> float:
    # Prefer explicit scalar if present; otherwise calculate Frobenius norm of the
    # symmetric 2x2 phase-aligned matrix [[Kxx,Kxy],[Kxy,Kyy]].
    for key in (
        "quadrupole_matrix_norm_V_per_C_per_m_per_m",
        "quad_norm_V_per_C_per_m_per_m",
        "K_norm_V_per_C_per_m_per_m",
    ):
        if key in q:
            return abs(_finite_or_nan(q[key]))
    Kxx = _finite_or_nan(q.get("Kxx_V_per_C_per_m_per_m", float("nan")))
    Kxy = _finite_or_nan(q.get("Kxy_V_per_C_per_m_per_m", float("nan")))
    Kyy = _finite_or_nan(q.get("Kyy_V_per_C_per_m_per_m", float("nan")))
    return float(np.sqrt(Kxx*Kxx + 2.0*Kxy*Kxy + Kyy*Kyy))


def _quad_max_axis(q: dict[str, Any]) -> float:
    return max(
        abs(_finite_or_nan(q.get("Kxx_V_per_C_per_m_per_m", float("nan")))),
        abs(_finite_or_nan(q.get("Kyy_V_per_C_per_m_per_m", float("nan")))),
        abs(_finite_or_nan(q.get("Kxy_V_per_C_per_m_per_m", float("nan")))),
    )


def _quadrupole_metrics(analysis: dict[str, Any]) -> dict[str, dict[str, float]]:
    f = analysis["focusing"]
    names = ("E1", "E2", "plus", "minus")
    return {
        "K_matrix_norm": {name: _quad_matrix_norm(f[name]) for name in names},
        "K_max_component": {name: _quad_max_axis(f[name]) for name in names},
        "Kxx": {name: abs(_finite_or_nan(f[name].get("Kxx_V_per_C_per_m_per_m", float("nan")))) for name in names},
        "Kxy": {name: abs(_finite_or_nan(f[name].get("Kxy_V_per_C_per_m_per_m", float("nan")))) for name in names},
        "Kyy": {name: abs(_finite_or_nan(f[name].get("Kyy_V_per_C_per_m_per_m", float("nan")))) for name in names},
    }


FAMILY_EXTRACTORS: dict[str, Callable[[dict[str, Any]], dict[str, dict[str, float]]]] = {
    "monopole": _monopole_metrics,
    "m0": _monopole_metrics,
    "dipole": _dipole_metrics,
    "m1": _dipole_metrics,
    "quadrupole": _quadrupole_metrics,
    "quad": _quadrupole_metrics,
    "m2": _quadrupole_metrics,
}


def analyse_homotypic_enhancements(saved_analysis: Any, *, family: str) -> list[dict[str, Any]]:
    """Return one row per crossing per metric, including R_max."""
    fam = family.lower()
    if fam not in FAMILY_EXTRACTORS:
        raise ValueError(f"family must be one of {sorted(FAMILY_EXTRACTORS)}")
    extractor = FAMILY_EXTRACTORS[fam]

    rows: list[dict[str, Any]] = []
    for key, analysis in _analysis_items(saved_analysis):
        common = _crossing_common_fields(key, analysis)
        metrics = extractor(analysis)
        for metric_name, vals in metrics.items():
            r = rmax_metric(vals["E1"], vals["E2"], vals["plus"], vals["minus"])
            rows.append({**common, "family": fam, "metric": metric_name, **r})
    return rows


def analyse_homotypic_enhancements_from_file(
    analysis_pkl: str | Path,
    *,
    family: str,
    out_csv: str | Path | None = None,
    out_pkl: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = analyse_homotypic_enhancements(pickle_load(analysis_pkl), family=family)
    if out_csv is not None:
        write_rows_csv(rows, out_csv)
    if out_pkl is not None:
        pickle_save(rows, out_pkl)
    return rows


def write_rows_csv(rows: list[dict[str, Any]], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        filename.write_text("")
        return
    # Stable human-friendly column order.
    preferred = [
        "family", "metric", "crossing_label", "mode_i", "mode_j", "length_factor", "frequency_Hz",
        "E1", "E2", "plus", "minus", "parent_max", "mixed_max", "R_max",
        "dominant_parent", "dominant_mixed", "plus_over_parent_max", "minus_over_parent_max",
        "plus_over_parent_sum", "minus_over_parent_sum", "analysis_key",
    ]
    keys = preferred + [k for k in rows[0].keys() if k not in preferred]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_enhancements(rows: list[dict[str, Any]], *, metric: str | None = None, n: int = 20) -> list[dict[str, Any]]:
    subset = [r for r in rows if metric is None or r["metric"] == metric]
    return sorted(subset, key=lambda r: (-np.nan_to_num(r["R_max"], nan=-np.inf), str(r["crossing_label"])))[:n]


# -----------------------------------------------------------------------------
# ell-fhat parameter plane reconstruction
# -----------------------------------------------------------------------------

def _infer_f010_from_data(data: dict[str, Any], default: float = 1.3e9) -> float:
    for meta_key in ("meta", "metadata"):
        meta = data.get(meta_key, {})
        for key in ("frequency_010_Hz", "f_010", "f010_Hz"):
            if key in meta:
                return float(meta[key])
    # Fall back to TM010 at design if available.
    try:
        return float(data["TM"]["010"]["design_frequency_Hz"])
    except Exception:
        return float(default)


def reconstruct_l_fhat_data(data_dict: dict[str, Any], crossing_results: dict[str, Any] | None = None, *, f_010: float | None = None) -> dict[str, Any]:
    """Return curves and crossing points for ell vs f_hat."""
    f0 = float(f_010) if f_010 is not None else _infer_f010_from_data(data_dict)
    ell = np.asarray(data_dict["length_factor_vector"], dtype=float)
    curves = []
    for fam, modes in data_dict.items():
        if fam not in ("TM", "TE") or not isinstance(modes, dict):
            continue
        for mnp, d in modes.items():
            if "frequency_normalised" in d:
                fhat = np.asarray(d["frequency_normalised"], dtype=float)
            else:
                fhat = np.asarray(d["frequency_Hz"], dtype=float) / f0
            curves.append({"family": fam, "mnp": str(mnp), "label": f"{fam}_{mnp}", "ell": ell, "fhat": fhat})

    crossing_points = []
    if crossing_results:
        for group, payload in crossing_results.items():
            if not isinstance(payload, dict) or "crossings" not in payload:
                continue
            for key, c in payload["crossings"].items():
                crossing_points.append({
                    "group": group,
                    "key": key,
                    "mode_i": c.get("mode_i"),
                    "mode_j": c.get("mode_j"),
                    "ell": float(c.get("length_factor")),
                    "fhat": float(c.get("frequency_Hz")) / f0,
                    "frequency_Hz": float(c.get("frequency_Hz")),
                })
    return {"f_010_Hz": f0, "ell": ell, "curves": curves, "crossings": crossing_points}


def plot_l_fhat_plane(
    plane: dict[str, Any],
    out_png: str | Path,
    *,
    title: str = r"Mode crossings in $(\ell,\hat{f})$",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    annotate_crossings: bool = True,
    annotate_modes: bool = False,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    curves = plane["curves"]
    for c in curves:
        ax.plot(c["ell"], c["fhat"], lw=1.0, alpha=0.8)
        if annotate_modes:
            ax.text(c["ell"][-1], c["fhat"][-1], c["label"], fontsize=7, ha="left", va="center")

    crossings = plane.get("crossings", [])
    if crossings:
        ax.scatter([c["ell"] for c in crossings], [c["fhat"] for c in crossings], s=70, facecolors="none", edgecolors="black", linewidths=1.5, zorder=5)
        if annotate_crossings:
            for c in crossings:
                label = f"{c['mode_i']}--{c['mode_j']}"
                ax.annotate(label, (c["ell"], c["fhat"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.axvline(1.0, ls="--", color="black", alpha=0.4, lw=1.0)
    ax.set_xlabel(r"Length factor $\ell=L/L_0$")
    ax.set_ylabel(r"Normalised frequency $\hat{f}=f/f_{010}$")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def write_crossing_points_csv(plane: dict[str, Any], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    rows = plane.get("crossings", [])
    if not rows:
        filename.write_text("")
        return
    keys = ["group", "key", "mode_i", "mode_j", "ell", "fhat", "frequency_Hz"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_and_plot_l_fhat_plane_from_files(
    data_pkl: str | Path,
    crossing_results_pkl: str | Path,
    out_png: str | Path,
    *,
    f_010: float | None = None,
    title: str | None = None,
    out_crossings_csv: str | Path | None = None,
    annotate_crossings: bool = True,
) -> dict[str, Any]:
    data = pickle_load(data_pkl)
    crossings = pickle_load(crossing_results_pkl)
    plane = reconstruct_l_fhat_data(data, crossings, f_010=f_010)
    if title is None:
        title = r"Mode crossings in $(\ell,\hat{f})$"
    plot_l_fhat_plane(plane, out_png, title=title, annotate_crossings=annotate_crossings)
    if out_crossings_csv is not None:
        write_crossing_points_csv(plane, out_crossings_csv)
    return plane


# -----------------------------------------------------------------------------
# One-call family helper
# -----------------------------------------------------------------------------

@dataclass
class FamilyPostprocessConfig:
    family: str
    data_pkl: Path
    crossing_results_pkl: Path
    all_crossing_analyses_pkl: Path
    output_dir: Path
    f_010: float | None = 1.3e9


def postprocess_family(cfg: FamilyPostprocessConfig) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    enhancement_rows = analyse_homotypic_enhancements_from_file(
        cfg.all_crossing_analyses_pkl,
        family=cfg.family,
        out_csv=cfg.output_dir / f"{cfg.family}_enhancement_summary.csv",
        out_pkl=cfg.output_dir / f"{cfg.family}_enhancement_summary.pkl",
    )
    plane = reconstruct_and_plot_l_fhat_plane_from_files(
        cfg.data_pkl,
        cfg.crossing_results_pkl,
        cfg.output_dir / f"{cfg.family}_ell_vs_fhat_crossings.png",
        f_010=cfg.f_010,
        title=f"{cfg.family}: homotypic crossings in $(\\ell,\\hat{{f}})$",
        out_crossings_csv=cfg.output_dir / f"{cfg.family}_crossing_points.csv",
    )
    return {"enhancements": enhancement_rows, "plane": plane}


if __name__ == "__main__":
    # Edit these paths for your machine.  This block is an example batch driver.
    datapath = Path(r"D:\PhD\HOMmix\HOMmix_analytical\data")
    root = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis")
    voxel_res = 151
    f_010 = 1.3e9

    configs = [
        FamilyPostprocessConfig(
            family="monopole",
            data_pkl=datapath / f"TMm0_TMm0_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
            crossing_results_pkl=root / "homotypic_monopoles" / "crossing_results.pkl",
            all_crossing_analyses_pkl=root / "homotypic_monopoles" / "all_crossing_analyses.pkl",
            output_dir=root / "homotypic_monopoles" / "postprocess",
            f_010=f_010,
        ),
        FamilyPostprocessConfig(
            family="dipole",
            data_pkl=datapath / f"TMm1_TMm1_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
            crossing_results_pkl=root / "homotypic_dipoles" / "crossing_results.pkl",
            all_crossing_analyses_pkl=root / "homotypic_dipoles" / "all_crossing_analyses.pkl",
            output_dir=root / "homotypic_dipoles" / "postprocess",
            f_010=f_010,
        ),
        FamilyPostprocessConfig(
            family="quadrupole",
            data_pkl=datapath / f"TMm2_TMm2_data_dict_{voxel_res}x{voxel_res}x{voxel_res}.pkl",
            crossing_results_pkl=root / "homotypic_quadrupoles" / "crossing_results.pkl",
            all_crossing_analyses_pkl=root / "homotypic_quadrupoles" / "all_crossing_analyses.pkl",
            output_dir=root / "homotypic_quadrupoles" / "postprocess",
            f_010=f_010,
        ),
    ]

    for c in configs:
        if c.data_pkl.exists() and c.crossing_results_pkl.exists() and c.all_crossing_analyses_pkl.exists():
            print(f"Post-processing {c.family}")
            postprocess_family(c)
        else:
            print(f"Skipping {c.family}: one or more input files not found")
