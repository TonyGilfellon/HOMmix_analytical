from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable


F_010_HZ = 1.3e9

DEFAULT_ANALYSIS_ROOT = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis")
DEFAULT_HOMOTYPIC_ROOT = DEFAULT_ANALYSIS_ROOT
DEFAULT_HETEROTYPIC_ROOT = DEFAULT_ANALYSIS_ROOT / "heterotypic_crossings"
DEFAULT_HETEROTYPIC_MULTIPOLE_PKL = DEFAULT_HETEROTYPIC_ROOT / "all_heterotypic_rf_multipole_analyses.pkl"
DEFAULT_HETEROTYPIC_HESSIAN_PKL = DEFAULT_HETEROTYPIC_ROOT / "all_heterotypic_multipole_analyses.pkl"
DEFAULT_PRAB_ROOT = Path(r"D:\PhD\PRAB")
DEFAULT_OUT_TEX = DEFAULT_PRAB_ROOT / "appendix_III_method_comparison.tex"


# -----------------------------------------------------------------------------
# Basic IO and formatting
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_load_if_exists(filename: str | Path) -> Any | None:
    filename = Path(filename)
    if not filename.exists():
        return None
    return pickle_load(filename)


def finite_or_nan(x: object) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def abs_finite_or_nan(x: object) -> float:
    try:
        y = abs(complex(x))
    except Exception:
        return float("nan")
    return float(y) if math.isfinite(float(y)) else float("nan")


def per_C_to_per_pC(x: object) -> float:
    return finite_or_nan(x) / 1.0e12


def fmt_sci(x: object, sig: int = 3) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def fmt_pct(x: object, ndp: int = 2) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    return rf"${x:.{ndp}f}$"


def normalise_mode_name(mode: object, default_family: str = "TM") -> str:
    """Return names as TM_012, TE_111, etc."""
    s = str(mode).strip()
    if not s or s.lower() == "none":
        return ""
    m = re.search(r"(TM|TE)[_\s-]*([0-9]{3,})", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}_{m.group(2).zfill(3)}"
    m = re.search(r"\b([0-9]{3,})\b", s)
    if m:
        return f"{default_family.upper()}_{m.group(1).zfill(3)}"
    return s.replace(" ", "_")


def latex_mode(mode: object) -> str:
    mode = normalise_mode_name(mode)
    if "_" in mode:
        fam, idx = mode.split("_", 1)
        return rf"$\mathrm{{{fam}_{{{idx}}}}}$"
    return str(mode).replace("_", r"\_")


def mode_azimuthal_index(mode: object) -> int | None:
    """Return m from a normalised mode name like TM_210, or None."""
    mode = normalise_mode_name(mode)
    m = re.search(r"_(\d)", mode)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(TM|TE)?[_\s-]*(\d)", mode, flags=re.IGNORECASE)
    if m:
        return int(m.group(2))
    return None


def filter_rows_by_m(rows: list[dict[str, Any]], required_m: int) -> list[dict[str, Any]]:
    """Keep only rows whose mode has the requested azimuthal index m."""
    return [row for row in rows if mode_azimuthal_index(row.get("mode")) == required_m]


def pct_difference(first: float, second: float) -> float:
    """100*(first-second)/second."""
    first = finite_or_nan(first)
    second = finite_or_nan(second)
    if not math.isfinite(first) or not math.isfinite(second) or second == 0.0:
        return float("nan")
    return 100.0 * (first - second) / second


def first_present(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return float("nan")


def representative(values: list[float]) -> float:
    vals = [finite_or_nan(v) for v in values if math.isfinite(finite_or_nan(v))]
    if not vals:
        return float("nan")
    return float(median(vals))


def relative_spread(values: list[float]) -> float:
    vals = [abs(finite_or_nan(v)) for v in values if math.isfinite(finite_or_nan(v))]
    if len(vals) < 2:
        return 0.0
    ref = max(median(vals), 1e-300)
    return (max(vals) - min(vals)) / ref


# -----------------------------------------------------------------------------
# Homotypic extraction: direct/specialised method values
# -----------------------------------------------------------------------------

def _crossing_metadata(item: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    crossing = item.get("crossing", {})
    if "modes" in item:
        mode_i = item["modes"].get("E1", crossing.get("mode_i"))
        mode_j = item["modes"].get("E2", crossing.get("mode_j"))
    else:
        mode_i = item.get("mode_i", crossing.get("mode_i"))
        mode_j = item.get("mode_j", crossing.get("mode_j"))
    return crossing, normalise_mode_name(mode_i), normalise_mode_name(mode_j)


def first_finite_from_dicts(dicts: Iterable[dict[str, Any]], keys: Iterable[str]) -> float:
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for key in keys:
            val = finite_or_nan(d.get(key, float("nan")))
            if math.isfinite(val) and val > 0.0:
                return val
    return float("nan")


def homotypic_crossing_length_m(item: dict[str, Any], crossing: dict[str, Any]) -> float:
    analysis = item.get("analysis", {})
    fields = item.get("fields", {})
    candidates = [
        item,
        crossing,
        analysis.get("E1", {}) if isinstance(analysis, dict) else {},
        fields.get("E1", {}) if isinstance(fields, dict) else {},
    ]
    return first_finite_from_dicts(
        candidates,
        ("length_m", "L_m", "cavity_length_m", "physical_length_m", "analysis_length_m"),
    )


def homotypic_loss_to_v_per_pc_per_m(loss_value: object, item: dict[str, Any], crossing: dict[str, Any]) -> float:
    loss = finite_or_nan(loss_value)
    if not math.isfinite(loss):
        return float("nan")
    L = homotypic_crossing_length_m(item, crossing)
    if math.isfinite(L) and L > 0.0:
        return abs(loss) / L
    return abs(loss)


def field_KQ_value(focusing: dict[str, Any], name: str) -> float:
    r = focusing.get(name, {}) if isinstance(focusing, dict) else {}
    for key in (
        "K_Q_V_per_pC_per_m3",
        "K_quad_strength_V_per_pC_per_m3",
        "KQ_V_per_pC_per_m3",
        "K_Q",
        "K_quad_strength",
    ):
        if isinstance(r, dict) and key in r:
            return abs(finite_or_nan(r[key]))
    return float("nan")


def load_homotypic_direct_values(homotypic_root: str | Path) -> dict[str, dict[str, list[float]]]:
    """Collect specialised homotypic values by parent mode.

    k_parallel: from monopole Vz^2/(4U)-style loss results.
    k_perp:     from dipole c k_parallel/(omega r^2)-style kick results.
    K_Q:        from quadrupole Brett azimuthal extraction results.
    """
    root = Path(homotypic_root)
    out: dict[str, dict[str, list[float]]] = {
        "k_parallel": {},
        "k_perp": {},
        "K_Q": {},
    }

    # Monopole k_parallel, stored as analysis[field]["loss"].
    mono_pkl = root / "homotypic_monopoles" / "all_crossing_analyses.pkl"
    if mono_pkl.exists():
        data = pickle_load(mono_pkl)
        items = data.values() if isinstance(data, dict) else data
        for item in items:
            crossing, mode_i, mode_j = _crossing_metadata(item)
            analysis = item.get("analysis", {})
            for field_name, mode in (("E1", mode_i), ("E2", mode_j)):
                try:
                    val = homotypic_loss_to_v_per_pc_per_m(analysis[field_name]["loss"], item, crossing)
                except Exception:
                    val = float("nan")
                if mode and math.isfinite(val):
                    out["k_parallel"].setdefault(mode, []).append(abs(val))

    # Dipole k_perp, stored as V/C/m^2; convert to V/pC/m^2.
    dip_pkl = root / "homotypic_dipoles" / "all_crossing_analyses.pkl"
    if dip_pkl.exists():
        data = pickle_load(dip_pkl)
        items = data.values() if isinstance(data, dict) else data
        for item in items:
            _, mode_i, mode_j = _crossing_metadata(item)
            kicks = item.get("kicks", {})
            for field_name, mode in (("E1", mode_i), ("E2", mode_j)):
                r = kicks.get(field_name, {}) if isinstance(kicks, dict) else {}
                val = finite_or_nan(first_present(r, ("kick_V_per_C_per_m_per_m", "kick_V_per_C_per_m2")))
                if mode and math.isfinite(val):
                    out["k_perp"].setdefault(mode, []).append(abs(per_C_to_per_pC(val)))

    # Quadrupole K_Q, current workflow: Brett azimuthal RF-multipole extraction.
    quad_pkl = root / "homotypic_quadrupoles" / "all_crossing_analyses.pkl"
    if quad_pkl.exists():
        data = pickle_load(quad_pkl)
        items = data.values() if isinstance(data, dict) else data
        for item in items:
            _, mode_i, mode_j = _crossing_metadata(item)
            focusing = item.get("focusing", {})
            for field_name, mode in (("E1", mode_i), ("E2", mode_j)):
                val = field_KQ_value(focusing, field_name)
                if mode and math.isfinite(val):
                    out["K_Q"].setdefault(mode, []).append(abs(val))

    return out


# -----------------------------------------------------------------------------
# Heterotypic extraction: Fourier/RF-multipole and Hessian/Taylor values
# -----------------------------------------------------------------------------

def figures(field_result: dict[str, Any]) -> dict[str, Any]:
    return field_result.get("figures_of_merit", {}) if isinstance(field_result, dict) else {}


def field_length_m(field_result: dict[str, Any]) -> float:
    return finite_or_nan(field_result.get("length_m", float("nan")))


def _kdiag_value(field_result: dict[str, Any], diag_key: str, value_key: str) -> float:
    try:
        return finite_or_nan(field_result.get("kparallel_diagnostics", {}).get(diag_key, {}).get(value_key))
    except Exception:
        return float("nan")


def heterotypic_loss_v_per_pc_per_m(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    direct = finite_or_nan(f.get("loss_like_V_per_pC_per_m"))
    if math.isfinite(direct):
        return abs(direct)

    direct = _kdiag_value(field_result, "fit_V0_U_CST", "k_V_per_pC_per_m")
    if math.isfinite(direct):
        return abs(direct)

    loss_v_per_pc = finite_or_nan(f.get("loss_like_V_per_pC"))
    L = field_length_m(field_result)
    if math.isfinite(loss_v_per_pc) and math.isfinite(L) and L > 0.0:
        return abs(loss_v_per_pc) / L

    return float("nan")


def heterotypic_kick_v_per_pc_per_m2(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return abs_finite_or_nan(first_present(f, ("kick_magnitude_V_per_pC_per_m2", "kick_mag_V_per_pC_per_m2")))


def heterotypic_Kxx(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return finite_or_nan(first_present(f, ("Kxx_V_per_pC_per_m3", "Kxx_U_CST_V_per_pC_per_m3")))


def heterotypic_Kyy(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return finite_or_nan(first_present(f, ("Kyy_V_per_pC_per_m3", "Kyy_U_CST_V_per_pC_per_m3")))


def heterotypic_Kxy(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return finite_or_nan(first_present(f, ("Kxy_V_per_pC_per_m3", "Kxy_U_CST_V_per_pC_per_m3")))


def heterotypic_KQ_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    direct = abs_finite_or_nan(first_present(f, ("K_Q_V_per_pC_per_m3", "KQ_V_per_pC_per_m3")))
    if math.isfinite(direct):
        return direct
    Kxx = heterotypic_Kxx(field_result)
    Kyy = heterotypic_Kyy(field_result)
    Kxy = heterotypic_Kxy(field_result)
    if not all(math.isfinite(v) for v in (Kxx, Kyy, Kxy)):
        return float("nan")
    return math.sqrt((Kxx - Kyy) ** 2 + 4.0 * Kxy ** 2)


def load_heterotypic_hessian_values(heterotypic_pkl: str | Path) -> dict[str, dict[str, list[float]]]:
    results = pickle_load(heterotypic_pkl)
    items = results.values() if isinstance(results, dict) else results
    out: dict[str, dict[str, list[float]]] = {
        "k_parallel": {},
        "k_perp": {},
        "K_Q": {},
    }

    for result in items:
        if not isinstance(result, dict):
            continue
        c = result.get("crossing", {})
        mode_i = normalise_mode_name(result.get("mode_i", c.get("mode_i", "")))
        mode_j = normalise_mode_name(result.get("mode_j", c.get("mode_j", "")))
        fields = result.get("fields", {})
        for field_name, mode in (("E1", mode_i), ("E2", mode_j)):
            field_result = fields.get(field_name, {}) if isinstance(fields, dict) else {}
            if not mode or not isinstance(field_result, dict):
                continue

            vals = {
                "k_parallel": heterotypic_loss_v_per_pc_per_m(field_result),
                "k_perp": heterotypic_kick_v_per_pc_per_m2(field_result),
                "K_Q": heterotypic_KQ_v_per_pc_per_m3(field_result),
            }
            for key, val in vals.items():
                if math.isfinite(val):
                    out[key].setdefault(mode, []).append(abs(val))

    return out


def _load_heterotypic_result_items_from_pkl_or_tree(
    path_or_root: str | Path,
    *,
    aggregate_filename: str,
    per_folder_filename: str,
) -> list[dict[str, Any]]:
    """Load heterotypic result dictionaries from either an aggregate pickle or folder tree.

    The latest RF/Fourier workflow saves per-crossing files named
    ``heterotypic_rf_multipole_analysis.pkl``.  Some runs also save an aggregate
    ``all_heterotypic_rf_multipole_analyses.pkl``.  This helper accepts either.
    """
    p = Path(path_or_root)

    if p.is_file():
        data = pickle_load(p)
        return list(data.values()) if isinstance(data, dict) else list(data)

    # Prefer the aggregate file if it exists.
    agg = p / aggregate_filename
    if agg.exists():
        data = pickle_load(agg)
        return list(data.values()) if isinstance(data, dict) else list(data)

    # Otherwise walk the crossing folders.
    files = sorted(p.rglob(per_folder_filename))
    return [pickle_load(f) for f in files]


def load_heterotypic_values_from_results(
    path_or_root: str | Path,
    *,
    aggregate_filename: str,
    per_folder_filename: str,
) -> dict[str, dict[str, list[float]]]:
    """Collect k_parallel, k_perp and K_Q for parent modes from heterotypic results."""
    items = _load_heterotypic_result_items_from_pkl_or_tree(
        path_or_root,
        aggregate_filename=aggregate_filename,
        per_folder_filename=per_folder_filename,
    )
    out: dict[str, dict[str, list[float]]] = {
        "k_parallel": {},
        "k_perp": {},
        "K_Q": {},
    }

    for result in items:
        if not isinstance(result, dict):
            continue
        c = result.get("crossing", {})
        mode_i = normalise_mode_name(result.get("mode_i", c.get("mode_i", "")))
        mode_j = normalise_mode_name(result.get("mode_j", c.get("mode_j", "")))
        fields = result.get("fields", {})
        for field_name, mode in (("E1", mode_i), ("E2", mode_j)):
            field_result = fields.get(field_name, {}) if isinstance(fields, dict) else {}
            if not mode or not isinstance(field_result, dict):
                continue
            vals = {
                "k_parallel": heterotypic_loss_v_per_pc_per_m(field_result),
                "k_perp": heterotypic_kick_v_per_pc_per_m2(field_result),
                "K_Q": heterotypic_KQ_v_per_pc_per_m3(field_result),
            }
            for key, val in vals.items():
                if math.isfinite(val):
                    out[key].setdefault(mode, []).append(abs(val))
    return out


def load_heterotypic_rf_multipole_values(heterotypic_root_or_pkl: str | Path) -> dict[str, dict[str, list[float]]]:
    return load_heterotypic_values_from_results(
        heterotypic_root_or_pkl,
        aggregate_filename="all_heterotypic_rf_multipole_analyses.pkl",
        per_folder_filename="heterotypic_rf_multipole_analysis.pkl",
    )


def load_heterotypic_hessian_values_tree(heterotypic_root_or_pkl: str | Path) -> dict[str, dict[str, list[float]]]:
    return load_heterotypic_values_from_results(
        heterotypic_root_or_pkl,
        aggregate_filename="all_heterotypic_multipole_analyses.pkl",
        per_folder_filename="heterotypic_multipole_analysis.pkl",
    )


# -----------------------------------------------------------------------------
# Comparison table construction
# -----------------------------------------------------------------------------

def comparison_rows(
    direct_values: dict[str, list[float]],
    hessian_values: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, dict[str, float]]]:
    direct_modes = set(direct_values)
    hessian_modes = set(hessian_values)
    common_modes = sorted(direct_modes & hessian_modes)

    rows: list[dict[str, Any]] = []
    spreads: dict[str, dict[str, float]] = {}
    for mode in common_modes:
        a = representative(direct_values[mode])
        b = representative(hessian_values[mode])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        rows.append({
            "mode": mode,
            "direct": a,
            "hessian": b,
            "pct_difference": pct_difference(a, b),
            "n_direct": len(direct_values[mode]),
            "n_hessian": len(hessian_values[mode]),
        })
        spreads[mode] = {
            "direct_relative_spread": relative_spread(direct_values[mode]),
            "hessian_relative_spread": relative_spread(hessian_values[mode]),
        }

    omitted = {
        "only_in_homotypic_direct": sorted(direct_modes - hessian_modes),
        "only_in_heterotypic_hessian": sorted(hessian_modes - direct_modes),
    }
    return rows, omitted, spreads


def latex_comparison_table(
    *,
    label: str,
    caption: str,
    rows: list[dict[str, Any]],
    direct_col: str,
    hessian_col: str,
) -> str:
    row_end = r" \\"
    body_lines: list[str] = []
    for row in rows:
        body_lines.append(
            " & ".join([
                latex_mode(row["mode"]),
                fmt_sci(row["direct"]),
                fmt_sci(row["hessian"]),
                fmt_pct(row["pct_difference"]),
            ]) + row_end
        )

    if not body_lines:
        body_lines.append("-- & -- & -- & --" + row_end)

    body = "\n".join(body_lines)

    header_end = r" \\"
    return rf"""
\begin{{table}}[htbp]
\caption{{{caption}}}
\label{{{label}}}
\begin{{ruledtabular}}
\begin{{tabular}}{{cccc}}
Mode & {direct_col} & {hessian_col} & $\Delta$ [\%]{header_end}
\hline
{body}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table}}
""".strip()
def appendix_start() -> str:
    return r"""
\clearpage
\section{Comparison of specialised homotypic and RF-multipole heterotypic beam-dynamics metrics}
\label{app:direct_rf_multipole_metric_comparison}

This appendix compares values obtained with the specialised homotypic methods against values obtained from the heterotypic Fourier/RF-multipole workflow for modes that appear as parent modes in both analyses.  The percentage difference in the first three tables is defined as $100(X_{\mathrm{homotypic}}-X_{\mathrm{RF}})/X_{\mathrm{RF}}$.  The final three tables compare the heterotypic Fourier/RF-multipole results directly with the heterotypic Hessian/Taylor-fit results using $100(X_{\mathrm{RF}}-X_{\mathrm{Hessian}})/X_{\mathrm{Hessian}}$.
""".strip()


def appendix_end() -> str:
    return r"% End of Appendix III: homotypic/RF-multipole/Hessian metric comparison"


def omitted_comment_block(all_omitted: dict[str, dict[str, list[str]]]) -> str:
    lines = ["", "% Modes omitted from comparison because they were not present in both datasets:"]
    for metric, omitted in all_omitted.items():
        lines.append(f"% {metric}:")
        for category, modes in omitted.items():
            pretty = ", ".join(modes) if modes else "none"
            lines.append(f"%   {category}: {pretty}")
    return "\n".join(lines)


def print_omitted_and_spreads(
    all_omitted: dict[str, dict[str, list[str]]],
    all_spreads: dict[str, dict[str, dict[str, float]]],
    *,
    spread_warn: float = 0.02,
) -> None:
    print("\nModes omitted from each comparison because they were not present in both datasets:")
    for metric, omitted in all_omitted.items():
        print(f"\n{metric}:")
        for category, modes in omitted.items():
            if modes:
                print(f"  {category}: {', '.join(modes)}")
            else:
                print(f"  {category}: none")

    print("\nDuplicate-value spread diagnostics:")
    any_warn = False
    for metric, spreads in all_spreads.items():
        for mode, s in spreads.items():
            if s["direct_relative_spread"] > spread_warn or s["hessian_relative_spread"] > spread_warn:
                any_warn = True
                print(
                    f"  {metric} {mode}: "
                    f"direct spread={100*s['direct_relative_spread']:.2f}%, "
                    f"hessian spread={100*s['hessian_relative_spread']:.2f}%"
                )
    if not any_warn:
        print("  none above warning threshold")


def latex_multipole_hessian_table(
    *,
    label: str,
    caption: str,
    rows_by_metric: dict[str, list[dict[str, Any]]],
) -> str:
    row_end = ' \\\\'
    metric_labels = {
        "k_parallel": r"k_{\parallel}",
        "k_perp": r"k_{\perp}",
        "K_Q": r"K_Q",
    }
    body_lines: list[str] = []
    for metric in ("k_parallel", "k_perp", "K_Q"):
        for row in rows_by_metric.get(metric, []):
            body_lines.append(
                " & ".join([
                    metric_labels.get(metric, metric.replace("_", r"\_")),
                    latex_mode(row["mode"]),
                    fmt_sci(row["direct"]),
                    fmt_sci(row["hessian"]),
                    fmt_pct(row["pct_difference"]),
                ]) + row_end
            )
    if not body_lines:
        body_lines.append("-- & -- & -- & -- & --" + row_end)
    body = "\n".join(body_lines)
    header_end = row_end
    return rf"""
\begin{{table}}[htbp]
\caption{{{caption}}}
\label{{{label}}}
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccc}}
Metric & Mode & RF multipole & Hessian/Taylor & $\Delta$ [\%]{header_end}
\hline
{body}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table}}
""".strip()



def latex_single_multipole_hessian_table(
    *,
    label: str,
    caption: str,
    metric: str,
    rows: list[dict[str, Any]],
) -> str:
    row_end = r" \\" 
    metric_labels = {
        "k_parallel": r"k_{\parallel}",
        "k_perp": r"k_{\perp}",
        "K_Q": r"K_Q",
    }
    body_lines: list[str] = []
    for row in rows:
        body_lines.append(
            " & ".join([
                latex_mode(row["mode"]),
                fmt_sci(row["direct"]),
                fmt_sci(row["hessian"]),
                fmt_pct(row["pct_difference"]),
            ]) + row_end
        )
    if not body_lines:
        body_lines.append("-- & -- & -- & --" + row_end)
    body = "\n".join(body_lines)
    header_end = row_end
    metric_label = metric_labels.get(metric, metric.replace("_", r"\_"))
    return rf"""
\begin{{table}}[htbp]
\caption{{{caption}}}
\label{{{label}}}
\begin{{ruledtabular}}
\begin{{tabular}}{{cccc}}
Mode & ${metric_label}^{{\mathrm{{RF}}}}$ & ${metric_label}^{{\mathrm{{Hessian}}}}$ & $\Delta$ [\%]{header_end}
\hline
{body}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table}}
""".strip()

def write_method_comparison_appendix(
    *,
    homotypic_root: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_multipole_root_or_pkl: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    heterotypic_hessian_root_or_pkl: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    out_tex: str | Path = DEFAULT_OUT_TEX,
) -> Path:
    """Write Appendix III.

    First three tables:
        specialised homotypic method values vs heterotypic Fourier/RF-multipole values.

    Final three tables:
        heterotypic Fourier/RF-multipole values vs heterotypic Hessian/Taylor values, split by monopole, dipole and quadrupole family.

    The heterotypic inputs may be either the heterotypic_crossings root folder or
    an aggregate pickle.  Per-crossing files are also supported:
        heterotypic_rf_multipole_analysis.pkl
        heterotypic_multipole_analysis.pkl
    """
    direct = load_homotypic_direct_values(homotypic_root)
    rf_multipole = load_heterotypic_rf_multipole_values(heterotypic_multipole_root_or_pkl)
    hessian = load_heterotypic_hessian_values_tree(heterotypic_hessian_root_or_pkl)

    kpar_rows, kpar_omitted, kpar_spreads = comparison_rows(direct["k_parallel"], rf_multipole["k_parallel"])
    kperp_rows, kperp_omitted, kperp_spreads = comparison_rows(direct["k_perp"], rf_multipole["k_perp"])
    KQ_rows, KQ_omitted, KQ_spreads = comparison_rows(direct["K_Q"], rf_multipole["K_Q"])

    # RF multipole vs Hessian/Taylor benchmark rows.
    mh_kpar_rows, mh_kpar_omitted, mh_kpar_spreads = comparison_rows(rf_multipole["k_parallel"], hessian["k_parallel"])
    mh_kperp_rows, mh_kperp_omitted, mh_kperp_spreads = comparison_rows(rf_multipole["k_perp"], hessian["k_perp"])
    mh_KQ_rows, mh_KQ_omitted, mh_KQ_spreads = comparison_rows(rf_multipole["K_Q"], hessian["K_Q"])

    # In the combined benchmark table, only compare the metric that is physically
    # relevant to each m-family: m=0 monopoles -> k_parallel,
    # m=1 dipoles -> k_perp, and m=2 quadrupoles -> K_Q.
    mh_kpar_rows_for_table = filter_rows_by_m(mh_kpar_rows, 0)
    mh_kperp_rows_for_table = filter_rows_by_m(mh_kperp_rows, 1)
    mh_KQ_rows_for_table = filter_rows_by_m(mh_KQ_rows, 2)

    tables = [
        latex_comparison_table(
            label="tab:kparallel_homotypic_rf_multipole_comparison",
            caption=(
                r"Comparison of $k_{\parallel}$ obtained from the direct "
                r"$|V_z|^2/(4U)$ homotypic calculation and from the "
                r"heterotypic Fourier/RF-multipole workflow."
            ),
            rows=kpar_rows,
            direct_col=r"$k_{\parallel}^{|V_z|^2/(4U)}$",
            hessian_col=r"$k_{\parallel}^{\mathrm{RF}}$",
        ),
        latex_comparison_table(
            label="tab:kperp_homotypic_rf_multipole_comparison",
            caption=(
                r"Comparison of $k_{\perp}$ obtained from the homotypic "
                r"$c k_{\parallel}/(\omega r^2)$ calculation and from the "
                r"heterotypic Fourier/RF-multipole workflow."
            ),
            rows=kperp_rows,
            direct_col=r"$k_{\perp}^{c k_{\parallel}/(\omega r^2)}$",
            hessian_col=r"$k_{\perp}^{\mathrm{RF}}$",
        ),
        latex_comparison_table(
            label="tab:KQ_brett_rf_multipole_comparison",
            caption=(
                r"Comparison of $K_Q$ obtained from the homotypic Brett-style "
                r"azimuthal RF-multipole extraction and from the heterotypic "
                r"Fourier/RF-multipole workflow."
            ),
            rows=KQ_rows,
            direct_col=r"$K_Q^{\mathrm{Brett}}$",
            hessian_col=r"$K_Q^{\mathrm{RF}}$",
        ),
        latex_single_multipole_hessian_table(
            label="tab:rf_multipole_hessian_monopole_benchmark",
            caption=(
                r"Benchmark comparison of heterotypic Fourier/RF-multipole and "
                r"Hessian/Taylor-fit $k_{\parallel}$ values for monopole modes."
            ),
            metric="k_parallel",
            rows=mh_kpar_rows_for_table,
        ),
        latex_single_multipole_hessian_table(
            label="tab:rf_multipole_hessian_dipole_benchmark",
            caption=(
                r"Benchmark comparison of heterotypic Fourier/RF-multipole and "
                r"Hessian/Taylor-fit $k_{\perp}$ values for dipole modes."
            ),
            metric="k_perp",
            rows=mh_kperp_rows_for_table,
        ),
        latex_single_multipole_hessian_table(
            label="tab:rf_multipole_hessian_quadrupole_benchmark",
            caption=(
                r"Benchmark comparison of heterotypic Fourier/RF-multipole and "
                r"Hessian/Taylor-fit $K_Q$ values for quadrupole modes."
            ),
            metric="K_Q",
            rows=mh_KQ_rows_for_table,
        ),
    ]

    all_omitted = {
        "k_parallel_homotypic_vs_RF": kpar_omitted,
        "k_perp_homotypic_vs_RF": kperp_omitted,
        "K_Q_homotypic_vs_RF": KQ_omitted,
        "k_parallel_RF_vs_Hessian": mh_kpar_omitted,
        "k_perp_RF_vs_Hessian": mh_kperp_omitted,
        "K_Q_RF_vs_Hessian": mh_KQ_omitted,
    }
    all_spreads = {
        "k_parallel_homotypic_vs_RF": kpar_spreads,
        "k_perp_homotypic_vs_RF": kperp_spreads,
        "K_Q_homotypic_vs_RF": KQ_spreads,
        "k_parallel_RF_vs_Hessian": mh_kpar_spreads,
        "k_perp_RF_vs_Hessian": mh_kperp_spreads,
        "K_Q_RF_vs_Hessian": mh_KQ_spreads,
    }

    tex = "\n\n".join([
        appendix_start(),
        *tables,
        omitted_comment_block(all_omitted),
        appendix_end(),
    ])

    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")

    print(f"Wrote {out_tex}")
    print(f"  k_parallel homotypic/RF rows: {len(kpar_rows)}")
    print(f"  k_perp homotypic/RF rows:     {len(kperp_rows)}")
    print(f"  K_Q homotypic/RF rows:        {len(KQ_rows)}")
    print(f"  k_parallel RF/Hessian rows:   {len(mh_kpar_rows)}")
    print(f"  k_perp RF/Hessian rows:       {len(mh_kperp_rows)}")
    print(f"  K_Q RF/Hessian rows:          {len(mh_KQ_rows)}")
    print_omitted_and_spreads(all_omitted, all_spreads)
    return out_tex


if __name__ == "__main__":
    write_method_comparison_appendix()
