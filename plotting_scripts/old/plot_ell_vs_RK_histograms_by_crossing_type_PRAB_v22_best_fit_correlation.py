from __future__ import annotations

"""
PRAB plotting utility for the unified homotypic RF/Fourier and heterotypic
Taylor/Hessian crossing analyses.

This R_total-only script writes:

    1. full ell-versus-R_total plot with logarithmic y-axis;
    2. configurable ell-versus-R_total zoom with linear y-axis;
    3. the same configurable zoom with logarithmic y-axis;
    4. overlaid R_total histograms; and
    5. overlaid ell histograms.

For every beam-dynamics metric K, the figure of merit is

    R_total,K = (K_plus + K_minus) / (K_1 + K_2).

Homotypic crossings contribute their family-specific metric. Heterotypic
crossings contribute one value for each of their two applicable metrics.
The ell histogram contains exactly one entry per unique physical crossing.
"""

import math
import pickle
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------

F_010_HZ = 1.3e9

DEFAULT_ANALYSIS_ROOT = Path(
    r"D:\PhD\HOMmix\HOMmix_analytical\analysis"
)
DEFAULT_HOMOTYPIC_ROOT = (
    DEFAULT_ANALYSIS_ROOT / "homotypic_rf_multipole"
)
DEFAULT_HETEROTYPIC_ROOT = (
    DEFAULT_ANALYSIS_ROOT / "heterotypic_crossings"
)

DEFAULT_PRAB_ROOT = Path(r"D:\PhD\PRAB")
DEFAULT_RTOTAL_PLOT_ROOT = (
    DEFAULT_PRAB_ROOT
    / "figures"
    / "ell_vs_Rtotal"
)

# Mixed-field metric cut-offs. A point is omitted when
# max(abs(E_minus), abs(E_plus)) is below the corresponding cut-off.
# Set a cut-off to 0.0 to retain every finite result for that metric.
DEFAULT_METRIC_CUTOFFS: dict[str, float] = {
    "K_parallel": 0.0,
    "K_perp": 0.0,
    "K_Q": 0.0,
}

# R_total zoom configuration. Both zoom plots use these same limits.
RTOTAL_ZOOM_ELL_MIN = 0.9
RTOTAL_ZOOM_ELL_MAX = 1.1
RTOTAL_ZOOM_MIN = 0.97
RTOTAL_ZOOM_MAX = 6.0

# Histogram configuration.
RTOTAL_HISTOGRAM_SCALE = "log"   # "linear" or "log"
RTOTAL_HISTOGRAM_N_BINS = 20
ELL_HISTOGRAM_N_BINS = 10
RK_STATISTICS_HISTOGRAM_N_BINS = 6
HISTOGRAM_ALPHA = 0.25

# PRAB one-column publication layout. All plots intentionally use the same
# physical dimensions so they can be inserted without inconsistent scaling.
PRAB_FIGSIZE = (3.35, 2.65)
AXIS_LABEL_FONTSIZE = 9.0
TICK_LABEL_FONTSIZE = 8.0
LEGEND_FONTSIZE = 4.0
LEGEND_MARKERSCALE = 0.6
ZOOM_LABEL_FONTSIZE = 7.0

# Publication colour and line-weight scheme. The red reference lines and
# black zoom box are retained, while the data colours use teal and purple.
# HOMOTYPIC_COLOUR = "red"  # "#00796B"      # deep teal
# HETEROTYPIC_COLOUR = "blue"  # "#6A3D9A"    # deep purple
# REFERENCE_COLOUR = "green"
HOMOTYPIC_COLOUR = "#0057B8"      # strong blue
HETEROTYPIC_COLOUR = "#F28E00"    # vivid orange
REFERENCE_COLOUR = "#D62728"      # red (unchanged)
ZOOM_BOX_COLOUR = "black"



# Line weights chosen to remain clear at PRAB one-column size.
DEFAULT_MARKER_SIZE = 5.5
DEFAULT_MARKER_EDGE_WIDTH = 0.9
ZOOM_MARKER_SIZE = 6.5
ZOOM_MARKER_EDGE_WIDTH = 1.0
REFERENCE_LINE_WIDTH = 1.1
ZOOM_BOX_LINE_WIDTH = 0.9
HISTOGRAM_EDGE_WIDTH = 1.4
LEGEND_FRAME_WIDTH = 0.7
SPINE_LINE_WIDTH = 0.7

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _apply_fine_plot_styling(ax: Any) -> None:
    """Apply consistent fine line weights to axes and tick marks."""
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)
    ax.tick_params(width=SPINE_LINE_WIDTH, labelsize=TICK_LABEL_FONTSIZE)


def _style_legend_frame(legend: Any) -> None:
    """Use a fine black border around a Matplotlib legend."""
    if legend is not None:
        legend.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)


def pickle_load(filename: str | Path) -> Any:
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def finite_or_nan(value: object) -> float:
    try:
        converted = float(value)
    except Exception:
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def abs_finite_or_nan(value: object) -> float:
    try:
        converted = float(abs(complex(value)))
    except Exception:
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def first_present(
    mapping: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    if not isinstance(mapping, dict):
        return float("nan")
    for key in keys:
        if key in mapping:
            return mapping[key]
    return float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    numerator = finite_or_nan(numerator)
    denominator = finite_or_nan(denominator)
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return float("nan")
    return numerator / denominator


def fmt_sci(
    value: object,
    significant_figures: int = 3,
) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"

    mantissa, exponent = (
        f"{value:.{significant_figures - 1}e}"
        .split("e")
    )
    return (
        rf"${mantissa}\times10^{{{int(exponent)}}}$"
    )


def fmt_fixed(
    value: object,
    decimal_places: int = 4,
) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{decimal_places}f}"


def normalise_mode_name(
    mode: object,
    default_family: str = "TM",
) -> str:
    text = str(mode).strip()
    if not text or text.lower() == "none":
        return ""

    match = re.search(
        r"(TM|TE)[_\s-]*([0-9]{3,})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return (
            f"{match.group(1).upper()}_"
            f"{match.group(2).zfill(3)}"
        )

    match = re.search(r"\b([0-9]{3,})\b", text)
    if match:
        return (
            f"{default_family.upper()}_"
            f"{match.group(1).zfill(3)}"
        )

    return text.replace(" ", "_")


def latex_mode(mode: object) -> str:
    mode = normalise_mode_name(mode)
    if "_" not in mode:
        return str(mode).replace("_", r"\_")
    family, indices = mode.split("_", 1)
    return rf"$\mathrm{{{family}_{{{indices}}}}}$"


def mode_azimuthal_index(mode: object) -> int | None:
    mode = normalise_mode_name(mode)
    match = re.search(r"_(\d)", mode)
    return int(match.group(1)) if match else None


# -----------------------------------------------------------------------------
# Result loading
# -----------------------------------------------------------------------------

def flatten_result_container(
    data: Any,
) -> list[dict[str, Any]]:
    """Flatten aggregate dictionaries/lists into crossing-result dictionaries."""
    results: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "fields" in value and "crossing" in value:
                results.append(value)
                return
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(data)
    return results


def load_result_items(
    path_or_root: str | Path,
    *,
    aggregate_filename: str,
    per_folder_filename: str,
) -> list[dict[str, Any]]:
    path = Path(path_or_root)

    if path.is_file():
        return flatten_result_container(
            pickle_load(path)
        )

    aggregate = path / aggregate_filename
    if aggregate.exists():
        return flatten_result_container(
            pickle_load(aggregate)
        )

    files = sorted(path.rglob(per_folder_filename))
    if not files:
        raise FileNotFoundError(
            f"No {aggregate_filename!r} or "
            f"{per_folder_filename!r} found below {path}."
        )

    results: list[dict[str, Any]] = []
    for filename in files:
        results.extend(
            flatten_result_container(
                pickle_load(filename)
            )
        )
    return results


def _result_identity(result: dict[str, Any]) -> tuple[Any, ...]:
    """Stable identity used to de-duplicate aggregate and per-folder results."""
    crossing = result.get("crossing", {})
    mode_i, mode_j = result_modes(result)
    family_m = result.get("family_m")
    if family_m is None:
        m_i = mode_azimuthal_index(mode_i)
        m_j = mode_azimuthal_index(mode_j)
        family_m = m_i if m_i == m_j else None

    return (
        family_m,
        mode_i,
        mode_j,
        finite_or_nan(crossing.get("length_factor", float("nan"))),
        finite_or_nan(crossing.get("frequency_Hz", float("nan"))),
    )


def load_homotypic_results(
    root_or_pickle: str | Path,
) -> list[dict[str, Any]]:
    """Load all homotypic families even when the aggregate pickle is partial.

    Earlier runs could leave ``all_homotypic_rf_multipole_analyses.pkl`` with
    only one family.  When a directory is supplied, read both the aggregate
    file and every per-crossing ``homotypic_rf_multipole_analysis.pkl`` below
    the root, then de-duplicate the combined results.
    """
    path = Path(root_or_pickle)

    if path.is_file():
        return flatten_result_container(pickle_load(path))

    collected: list[dict[str, Any]] = []

    aggregate = path / "all_homotypic_rf_multipole_analyses.pkl"
    if aggregate.exists():
        collected.extend(
            flatten_result_container(
                pickle_load(aggregate)
            )
        )

    per_crossing_files = sorted(
        path.rglob("homotypic_rf_multipole_analysis.pkl")
    )
    for filename in per_crossing_files:
        collected.extend(
            flatten_result_container(
                pickle_load(filename)
            )
        )

    if not collected:
        raise FileNotFoundError(
            "No homotypic aggregate or per-crossing RF-multipole "
            f"results were found below {path}."
        )

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in collected:
        unique[_result_identity(result)] = result

    results = list(unique.values())
    print(
        "Homotypic loader sources: "
        f"aggregate={'yes' if aggregate.exists() else 'no'}, "
        f"per-crossing files={len(per_crossing_files)}, "
        f"unique crossings={len(results)}"
    )
    return results


def load_heterotypic_results(
    root_or_pickle: str | Path,
) -> list[dict[str, Any]]:
    return load_result_items(
        root_or_pickle,
        aggregate_filename=(
            "all_heterotypic_multipole_analyses.pkl"
        ),
        per_folder_filename=(
            "heterotypic_multipole_analysis.pkl"
        ),
    )


# -----------------------------------------------------------------------------
# Result metadata and metric extraction
# -----------------------------------------------------------------------------

METRIC_INFO: dict[str, dict[str, Any]] = {
    "K_parallel": {
        "latex": r"$K_{\parallel}$",
        "units": r"$\mathrm{V/pC/m_z}$",
        "explicit_key": (
            "K_parallel_V_per_pC_per_m"
        ),
        "legacy_keys": (
            "loss_like_V_per_pC_per_m",
        ),
    },
    "K_perp": {
        "latex": r"$K_{\perp}$",
        "units": (
            r"$\mathrm{V/pC/m_{\perp}/m_z}$"
        ),
        "explicit_key": (
            "K_perp_V_per_pC_per_m2"
        ),
        "legacy_keys": (
            "kick_magnitude_V_per_pC_per_m2",
            "kick_mag_V_per_pC_per_m2",
        ),
    },
    "K_Q": {
        "latex": r"$K_Q$",
        "units": (
            r"$\mathrm{V/pC/m_{\perp}^{2}/m_z}$"
        ),
        "explicit_key": (
            "K_Q_V_per_pC_per_m3"
        ),
        "legacy_keys": (
            "KQ_V_per_pC_per_m3",
        ),
    },
}


def result_modes(
    result: dict[str, Any],
) -> tuple[str, str]:
    crossing = result.get("crossing", {})
    mode_i = normalise_mode_name(
        result.get(
            "mode_i",
            crossing.get("mode_i", ""),
        )
    )
    mode_j = normalise_mode_name(
        result.get(
            "mode_j",
            crossing.get("mode_j", ""),
        )
    )
    return mode_i, mode_j


def crossing_parameters(
    result: dict[str, Any],
) -> tuple[float, float]:
    crossing = result.get("crossing", {})
    length_factor = finite_or_nan(
        crossing.get(
            "length_factor",
            float("nan"),
        )
    )
    frequency_Hz = finite_or_nan(
        crossing.get(
            "frequency_Hz",
            float("nan"),
        )
    )
    frequency_normalised = (
        frequency_Hz / F_010_HZ
        if math.isfinite(frequency_Hz)
        else float("nan")
    )
    return length_factor, frequency_normalised


def pair_type_key(
    result: dict[str, Any],
) -> str:
    crossing = result.get("crossing", {})
    pair_type = (
        result.get("pair_type")
        or crossing.get("pair_type")
    )

    if pair_type is None and result.get(
        "crossing_folder"
    ):
        pair_type = Path(
            result["crossing_folder"]
        ).parent.name

    if pair_type is not None:
        return (
            str(pair_type)
            .lower()
            .replace("-", "_")
        )

    mode_i, mode_j = result_modes(result)
    m_pair = sorted([
        mode_azimuthal_index(mode_i),
        mode_azimuthal_index(mode_j),
    ])
    return {
        (0, 1): "monopole_dipole",
        (0, 2): "monopole_quadrupole",
        (1, 2): "dipole_quadrupole",
    }.get(tuple(m_pair), "heterotypic")


def homotypic_family_m(
    result: dict[str, Any],
) -> int:
    family_m = result.get("family_m")
    if family_m is not None:
        return int(family_m)

    mode_i, mode_j = result_modes(result)
    m_i = mode_azimuthal_index(mode_i)
    m_j = mode_azimuthal_index(mode_j)

    if m_i is None or m_j is None or m_i != m_j:
        raise ValueError(
            "Could not identify homotypic family for "
            f"{mode_i}--{mode_j}."
        )
    return int(m_i)


def figures_of_merit(
    field_result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(field_result, dict):
        return {}
    return field_result.get(
        "figures_of_merit",
        {},
    )


def metric_value(
    field_result: dict[str, Any],
    metric_key: str,
) -> float:
    info = METRIC_INFO[metric_key]
    figures = figures_of_merit(field_result)

    value = abs_finite_or_nan(
        first_present(
            figures,
            (
                info["explicit_key"],
                *info["legacy_keys"],
            ),
        )
    )
    if math.isfinite(value):
        return value

    # Older Taylor/Hessian longitudinal output.
    if metric_key == "K_parallel":
        value = abs_finite_or_nan(
            field_result.get(
                "kparallel_diagnostics",
                {},
            ).get(
                "fit_V0_U_CST",
                {},
            ).get(
                "k_V_per_pC_per_m",
                float("nan"),
            )
        )
        if math.isfinite(value):
            return value

    # Reconstruct K_Q from the Hessian components if no scalar is stored.
    if metric_key == "K_Q":
        Kxx = finite_or_nan(
            first_present(
                figures,
                (
                    "K_xx_V_per_pC_per_m3",
                    "Kxx_V_per_pC_per_m3",
                ),
            )
        )
        Kxy = finite_or_nan(
            first_present(
                figures,
                (
                    "K_xy_V_per_pC_per_m3",
                    "Kxy_V_per_pC_per_m3",
                ),
            )
        )
        Kyy = finite_or_nan(
            first_present(
                figures,
                (
                    "K_yy_V_per_pC_per_m3",
                    "Kyy_V_per_pC_per_m3",
                ),
            )
        )
        if all(
            math.isfinite(value)
            for value in (Kxx, Kxy, Kyy)
        ):
            return math.sqrt(
                (Kxx - Kyy) ** 2
                + 4.0 * Kxy ** 2
            )

    return float("nan")


def field_metric_values(
    result: dict[str, Any],
    metric_key: str,
) -> dict[str, float]:
    fields = result.get("fields", {})
    values = {
        field_name: metric_value(
            fields.get(field_name, {}),
            metric_key,
        )
        for field_name in (
            "E1",
            "E2",
            "minus",
            "plus",
        )
    }

    return values


# -----------------------------------------------------------------------------
# Aggregate table row construction
# -----------------------------------------------------------------------------

def homotypic_metric_key(
    result: dict[str, Any],
) -> str:
    return {
        0: "K_parallel",
        1: "K_perp",
        2: "K_Q",
    }[homotypic_family_m(result)]


def heterotypic_metric_keys(
    pair_type: str,
) -> tuple[str, str]:
    mapping = {
        "monopole_dipole": (
            "K_parallel",
            "K_perp",
        ),
        "monopole_quadrupole": (
            "K_parallel",
            "K_Q",
        ),
        "dipole_quadrupole": (
            "K_perp",
            "K_Q",
        ),
    }
    if pair_type not in mapping:
        raise KeyError(
            f"Unsupported heterotypic pair type: "
            f"{pair_type}"
        )
    return mapping[pair_type]


def sorted_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda result: (
            result_modes(result)[0],
            result_modes(result)[1],
            crossing_parameters(result)[0],
        ),
    )


# -----------------------------------------------------------------------------
# ell versus R_total plots
# -----------------------------------------------------------------------------

HETEROTYPIC_PLOT_INFO: dict[str, dict[str, str]] = {
    "monopole_dipole": {
        "label": "M-D",
        "marker_1": "p",
        "marker_2": "H",
    },
    "monopole_quadrupole": {
        "label": "M-Q",
        "marker_1": "s",
        "marker_2": "D",
    },
    "dipole_quadrupole": {
        "label": "D-Q",
        "marker_1": "<",
        "marker_2": ">",
    },
}

HOMOTYPIC_PLOT_INFO: dict[int, dict[str, str]] = {
    0: {"label": "M-M", "marker": "*"},
    1: {"label": "D-D", "marker": "p"},
    2: {"label": "Q-Q", "marker": "v"},
}

METRIC_FILENAME = {
    "K_parallel": "K_parallel",
    "K_perp": "K_perp",
    "K_Q": "K_Q",
}


def _normalised_metric_cutoffs(
    metric_cutoffs: dict[str, float] | None,
) -> dict[str, float]:
    cutoffs = dict(DEFAULT_METRIC_CUTOFFS)
    if metric_cutoffs is not None:
        unknown = set(metric_cutoffs) - set(METRIC_INFO)
        if unknown:
            raise KeyError(
                "Unknown metric cut-off key(s): "
                + ", ".join(sorted(unknown))
            )
        cutoffs.update(metric_cutoffs)

    for metric_key, cutoff in cutoffs.items():
        cutoff = finite_or_nan(cutoff)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError(
                f"Cut-off for {metric_key} must be finite and non-negative."
            )
        cutoffs[metric_key] = cutoff
    return cutoffs


def _is_heterotypic_result(result: dict[str, Any]) -> bool:
    mode_i, mode_j = result_modes(result)
    m_i = mode_azimuthal_index(mode_i)
    m_j = mode_azimuthal_index(mode_j)
    return m_i is not None and m_j is not None and m_i != m_j



def rtotal_plot_point(
    result: dict[str, Any],
    metric_key: str,
    *,
    metric_cutoff: float,
) -> tuple[float, float] | None:
    """Return ``(ell, R_total)`` for the selected metric.

    For both homotypic and heterotypic crossings, both parent-mode
    values are retained in the denominator:

        R_total = (K_plus + K_minus) / (K_1 + K_2)

    Heterotypic crossings are still evaluated separately for each of
    their two relevant metrics.
    """
    ell, _ = crossing_parameters(result)
    values = field_metric_values(result, metric_key)

    k1 = abs_finite_or_nan(values["E1"])
    k2 = abs_finite_or_nan(values["E2"])
    k_minus = abs_finite_or_nan(values["minus"])
    k_plus = abs_finite_or_nan(values["plus"])
    mixed_maximum = max(k_minus, k_plus)

    if (
        not math.isfinite(ell)
        or not all(math.isfinite(v) for v in (k1, k2, k_minus, k_plus))
        or mixed_maximum < metric_cutoff
    ):
        return None

    parent_total = k1 + k2
    r_total = safe_ratio(k_plus + k_minus, parent_total)
    if not math.isfinite(r_total):
        return None
    return ell, r_total


def _metric_legend_symbol(metric_key: str) -> str:
    return {
        "K_parallel": r"\parallel",
        "K_perp": r"\perp",
        "K_Q": "Q",
    }[metric_key]


def _fom_legend_label(base_label: str, metric_key: str) -> str:
    symbol = _metric_legend_symbol(metric_key)
    return rf"{base_label} $R_{{\mathrm{{total}},{symbol}}}$"

def _plot_hollow_markers(
    ax: Any,
    x_values: Iterable[float],
    y_values: Iterable[float],
    *,
    marker: str,
    label: str,
    color: str,
    markersize: float = 7.5,
    markeredgewidth: float = 1.35,
) -> None:
    """Plot markers with no connecting line and no filled marker face."""
    line_only_markers = {"+", "x", "1", "2", "3", "4", "|", "_"}
    kwargs: dict[str, Any] = {
        "linestyle": "none",
        "marker": marker,
        "label": label,
        "markersize": markersize,
        "markeredgewidth": markeredgewidth,
        "markeredgecolor": color,
        "alpha": 0.95,
        "zorder": 3,
    }
    if marker not in line_only_markers:
        kwargs["markerfacecolor"] = "none"
        kwargs["fillstyle"] = "none"
    ax.plot(list(x_values), list(y_values), **kwargs)





def _combined_groups(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], str, str, str]]:
    """Build plot groups for all homotypic and heterotypic metrics."""
    family_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}
    groups: list[tuple[list[dict[str, Any]], str, str, str]] = []

    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        family_results = [
            result for result in homotypic_results
            if homotypic_family_m(result) == family_m
        ]
        groups.append((
            family_results,
            family_metric[family_m],
            info["marker"],
            f"{info['label']}|||{HOMOTYPIC_COLOUR}",
        ))

    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        family_results = [
            result for result in heterotypic_results
            if pair_type_key(result) == pair_type
        ]
        for metric_index, metric_key in enumerate(
            heterotypic_metric_keys(pair_type), start=1
        ):
            groups.append((
                family_results,
                metric_key,
                info[f"marker_{metric_index}"],
                f"{info['label']}|||{HETEROTYPIC_COLOUR}",
            ))

    return groups


def _write_combined_rtotal_log_plot(
    *,
    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]],
    cutoffs: dict[str, float],
    output_root: Path,
    ell_min: float,
    ell_max: float,
    rtotal_min: float,
    rtotal_max: float,
    dpi: int,
) -> Path:
    """Write the combined ell--R_total plot on a logarithmic y-axis.

    A dashed rectangle identifies the configurable bounds used by the
    companion linear zoom plot.
    """
    if ell_min >= ell_max:
        raise ValueError("RTOTAL_ZOOM_ELL_MIN must be below RTOTAL_ZOOM_ELL_MAX.")
    if rtotal_min <= 0.0 or rtotal_min >= rtotal_max:
        raise ValueError(
            "For the log plot, RTOTAL_ZOOM_MIN must be positive and below "
            "RTOTAL_ZOOM_MAX."
        )

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)
    count = 0

    for group_results, group_metric, marker, label_color in combined_groups:
        label, color = label_color.split("|||", 1)
        points = [
            point
            for result in group_results
            if (point := rtotal_plot_point(
                result,
                group_metric,
                metric_cutoff=cutoffs[group_metric],
            )) is not None
        ]
        points.sort(key=lambda point: point[0])
        if not points:
            continue

        x_values, y_values = zip(*points)
        _plot_hollow_markers(
            ax,
            x_values,
            y_values,
            marker=marker,
            label=_fom_legend_label(label, group_metric),
            color=color,
            markersize=DEFAULT_MARKER_SIZE,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
        )
        count += len(points)

    # Mark the exact bounds used by the separate linear zoom figure.
    zoom_box = Rectangle(
        (ell_min, rtotal_min),
        ell_max - ell_min,
        rtotal_max - rtotal_min,
        fill=False,
        edgecolor=ZOOM_BOX_COLOUR,
        linestyle="--",
        linewidth=ZOOM_BOX_LINE_WIDTH,
        zorder=2,
        label="Zoom region",
    )
    ax.add_patch(zoom_box)
    ax.text(
        ell_min+0.05,
        rtotal_max * 1.05,
        "Zoom region",
        fontsize=ZOOM_LABEL_FONTSIZE,
        color=ZOOM_BOX_COLOUR,
        ha="left",
        va="bottom",
        zorder=4,
    )

    ax.set_xlabel(r"$\ell$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$R_{\mathrm{total}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_yscale("log")
    ax.axhline(
        1.0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label=r"$R_{\mathrm{total}}=1$",
        zorder=1,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        markerscale=LEGEND_MARKERSCALE,
    )
    _apply_fine_plot_styling(ax)
    legend = ax.get_legend()
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = (
        output_root
        / "homotypic_and_heterotypic_ell_vs_Rtotal_log.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  plotted points: {count}; "
        "cut-off=family-specific; y-scale=log; FoM=R_total"
    )
    return output_file



def _collect_rtotal_and_ell_populations(
    *,
    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]],
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Collect R_total entries and unique physical-crossing ell entries.

    The R_total histogram retains one entry per applicable beam-dynamics
    metric. Therefore, each heterotypic crossing may contribute two R_total
    values, as intended.

    The ell histogram is constructed independently from the R_total metric
    groups. Each physical homotypic or heterotypic crossing contributes at
    most one ell value, regardless of how many applicable metrics it has.
    """
    homotypic_rtotal: list[float] = []
    heterotypic_rtotal: list[float] = []

    # Preserve the existing metric-entry convention for the R_total histogram.
    for group_results, metric_key, _marker, label_color in combined_groups:
        is_homotypic = label_color.endswith(f"|||{HOMOTYPIC_COLOUR}")
        target_rtotal = (
            homotypic_rtotal if is_homotypic else heterotypic_rtotal
        )

        for result in group_results:
            point = rtotal_plot_point(
                result,
                metric_key,
                metric_cutoff=cutoffs[metric_key],
            )
            if point is not None:
                target_rtotal.append(point[1])

    # Build ell populations directly from physical crossings, not metric rows.
    def unique_ell_values(
        results: list[dict[str, Any]],
    ) -> list[float]:
        ell_by_crossing: dict[tuple[Any, ...], float] = {}
        for result in results:
            ell, _ = crossing_parameters(result)
            if math.isfinite(ell):
                ell_by_crossing.setdefault(
                    _result_identity(result),
                    ell,
                )
        return list(ell_by_crossing.values())

    return (
        homotypic_rtotal,
        heterotypic_rtotal,
        unique_ell_values(homotypic_results),
        unique_ell_values(heterotypic_results),
    )

def _write_combined_rtotal_zoom_plot(
    *,
    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]],
    cutoffs: dict[str, float],
    output_root: Path,
    ell_min: float,
    ell_max: float,
    rtotal_min: float,
    rtotal_max: float,
    y_scale: str,
    dpi: int,
) -> Path:
    """Write a configurable ell-versus-R_total zoom on linear or log scale."""
    if ell_min >= ell_max:
        raise ValueError("RTOTAL_ZOOM_ELL_MIN must be below RTOTAL_ZOOM_ELL_MAX.")
    if rtotal_min >= rtotal_max:
        raise ValueError("RTOTAL_ZOOM_MIN must be below RTOTAL_ZOOM_MAX.")
    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'.")
    if y_scale == "log" and rtotal_min <= 0.0:
        raise ValueError("The lower R_total zoom limit must be positive for log scale.")

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)
    count = 0

    for group_results, metric_key, marker, label_color in combined_groups:
        label, color = label_color.split("|||", 1)
        points: list[tuple[float, float]] = []
        for result in group_results:
            point = rtotal_plot_point(
                result,
                metric_key,
                metric_cutoff=cutoffs[metric_key],
            )
            if point is None:
                continue
            ell, r_total = point
            if ell_min <= ell <= ell_max and rtotal_min <= r_total <= rtotal_max:
                points.append((ell, r_total))

        if not points:
            continue
        points.sort(key=lambda item: item[0])
        x_values, y_values = zip(*points)
        _plot_hollow_markers(
            ax,
            x_values,
            y_values,
            marker=marker,
            label=_fom_legend_label(label, metric_key),
            color=color,
            markersize=ZOOM_MARKER_SIZE,
            markeredgewidth=ZOOM_MARKER_EDGE_WIDTH,
        )
        count += len(points)

    ax.set_xlim(ell_min, ell_max)
    ax.set_ylim(rtotal_min, rtotal_max)
    ax.set_yscale(y_scale)
    ax.set_xlabel(r"$\ell$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$R_{\mathrm{total}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.axhline(
        1.0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label=r"$R_{\mathrm{total}}=1$",
        zorder=1,
    )
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(
        loc="upper center",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        markerscale=LEGEND_MARKERSCALE,
    )
    _apply_fine_plot_styling(ax)
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = (
        output_root
        / f"homotypic_and_heterotypic_ell_vs_Rtotal_zoom_{y_scale}.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  plotted points: {count}; cut-off=family-specific; "
        f"ell=[{ell_min}, {ell_max}]; "
        f"R_total=[{rtotal_min}, {rtotal_max}]; y-scale={y_scale}"
    )
    return output_file


def _write_rtotal_histogram(
    *,
    homotypic_rtotal: list[float],
    heterotypic_rtotal: list[float],
    output_root: Path,
    rtotal_scale: str,
    n_bins: int,
    dpi: int,
) -> Path:
    """Write shared-bin overlaid histograms of homotypic/heterotypic R_total."""
    bins = _shared_histogram_bins(
        homotypic_rtotal + heterotypic_rtotal,
        scale=rtotal_scale,
        n_bins=n_bins,
    )

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)

    ax.hist(
        heterotypic_rtotal,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Heterotypic (n={len(heterotypic_rtotal)})",
        edgecolor=HETEROTYPIC_COLOUR,
        facecolor=HETEROTYPIC_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )

    ax.hist(
        homotypic_rtotal,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Homotypic (n={len(homotypic_rtotal)})",
        edgecolor=HOMOTYPIC_COLOUR,
        facecolor=HOMOTYPIC_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )

    ax.set_xscale(rtotal_scale)
    ax.set_xlabel(r"$R_{\mathrm{total}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Crossing count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.axvline(1.0, color=REFERENCE_COLOUR, linestyle="--", linewidth=REFERENCE_LINE_WIDTH,
               label=r"$R_{\mathrm{total}}=1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black", fontsize=LEGEND_FONTSIZE,
              markerscale=LEGEND_MARKERSCALE)
    _apply_fine_plot_styling(ax)
    legend = ax.get_legend()
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = output_root / f"homotypic_and_heterotypic_Rtotal_histogram_{rtotal_scale}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  histogram entries: homotypic={len(homotypic_rtotal)}, "
        f"heterotypic={len(heterotypic_rtotal)}; shared bins={n_bins}; "
        f"R_total scale={rtotal_scale}"
    )
    return output_file


def _write_ell_histogram(
    *,
    homotypic_ell: list[float],
    heterotypic_ell: list[float],
    output_root: Path,
    n_bins: int,
    dpi: int,
) -> Path:
    """Write shared, linearly spaced histograms of crossing length factor ell."""
    all_values = [
        value for value in homotypic_ell + heterotypic_ell
        if math.isfinite(value)
    ]
    if not all_values:
        raise ValueError("No finite ell values are available for histogramming.")
    if n_bins < 1:
        raise ValueError("ELL_The histogram bin count must be at least 1.")

    minimum = min(all_values)
    maximum = max(all_values)
    if minimum == maximum:
        minimum -= 0.05
        maximum += 0.05
    bins = np.linspace(minimum, maximum, n_bins + 1)

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)

    ax.hist(
        heterotypic_ell,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Heterotypic (n={len(heterotypic_ell)})",
        edgecolor=HETEROTYPIC_COLOUR,
        facecolor=HETEROTYPIC_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )

    ax.hist(
        homotypic_ell,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Homotypic (n={len(homotypic_ell)})",
        edgecolor=HOMOTYPIC_COLOUR,
        facecolor=HOMOTYPIC_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )

    ax.axvline(
        1.0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label=r"$\ell=1$",
        zorder=1,
    )
    ax.set_xlabel(r"$\ell$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Crossing count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black", fontsize=LEGEND_FONTSIZE,
              markerscale=LEGEND_MARKERSCALE)
    _apply_fine_plot_styling(ax)
    legend = ax.get_legend()
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = output_root / "homotypic_and_heterotypic_ell_histogram_linear.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  histogram entries: homotypic={len(homotypic_ell)}, "
        f"heterotypic={len(heterotypic_ell)}; shared bins={n_bins}"
    )
    return output_file



def _shared_histogram_bins(
    values: list[float],
    *,
    scale: str,
    n_bins: int,
) -> np.ndarray:
    finite_positive = np.asarray([
        value for value in values
        if math.isfinite(value) and value > 0.0
    ], dtype=float)
    if finite_positive.size == 0:
        raise ValueError("No positive finite R_total values are available.")
    if n_bins < 1:
        raise ValueError("The histogram bin count must be at least 1.")

    minimum = float(np.min(finite_positive))
    maximum = float(np.max(finite_positive))
    if minimum == maximum:
        minimum *= 0.9
        maximum *= 1.1

    if scale == "log":
        return np.logspace(
            math.log10(minimum),
            math.log10(maximum),
            n_bins + 1,
        )
    if scale == "linear":
        return np.linspace(minimum, maximum, n_bins + 1)
    raise ValueError(
        "RTOTAL_HISTOGRAM_SCALE must be either 'linear' or 'log'."
    )




# -----------------------------------------------------------------------------
# Crossing-type R_K histograms
# -----------------------------------------------------------------------------

RK_HISTOGRAM_SUBDIRECTORY = "crossing_type_RK_histograms"
DEFAULT_RK_STATISTICS_TEX = (
    DEFAULT_PRAB_ROOT / "crossing_type_RK_summary_statistics_PRAB_table.tex"
)
DEFAULT_RK_FIGURES_TEX = (
    DEFAULT_PRAB_ROOT / "crossing_type_RK_histogram_figures_PRAB.tex"
)
DEFAULT_RK_FIGURES_AND_TABLE_TEX = (
    DEFAULT_PRAB_ROOT / "crossing_type_RK_histograms_and_statistics_PRAB.tex"
)
RK_LATEX_FIGURE_DIRECTORY = "figs"
RK_LATEX_FIGURE_SCALE = 1.3

# Distinct line styles/colours for the two heterotypic metrics and their
# combined distribution. These are used only in the per-crossing-type
# histogram figures.
HETEROTYPIC_METRIC_COLOURS: dict[str, str] = {
    "K_parallel": "#0057B8",
    "K_perp": "#F28E00",
    "K_Q": "#6A3D9A",
}
HETEROTYPIC_COMBINED_COLOUR = "black"


def _collect_crossing_type_rk_values(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
) -> tuple[
    dict[int, list[float]],
    dict[str, dict[str, list[float]]],
]:
    """Collect R_K values by physical crossing type.

    Here

        R_K = (K_+ + K_-) / (K_1 + K_2),

    which is the quantity previously labelled R_total.

    Homotypic crossings contribute one family-specific R_K value:
        M-M -> K_parallel
        D-D -> K_perp
        Q-Q -> K_Q

    Heterotypic crossings contribute one R_K value for each of their two
    applicable metrics. A third, combined population is formed by concatenating
    those two metric-specific populations; it does not average the two values.
    """
    homotypic: dict[int, list[float]] = {
        family_m: [] for family_m in HOMOTYPIC_PLOT_INFO
    }
    heterotypic: dict[str, dict[str, list[float]]] = {
        pair_type: {
            metric_key: []
            for metric_key in heterotypic_metric_keys(pair_type)
        }
        for pair_type in HETEROTYPIC_PLOT_INFO
    }

    for result in homotypic_results:
        family_m = homotypic_family_m(result)
        metric_key = homotypic_metric_key(result)
        point = rtotal_plot_point(
            result,
            metric_key,
            metric_cutoff=cutoffs[metric_key],
        )
        if point is not None:
            homotypic[family_m].append(point[1])

    for result in heterotypic_results:
        pair_type = pair_type_key(result)
        if pair_type not in heterotypic:
            continue
        for metric_key in heterotypic_metric_keys(pair_type):
            point = rtotal_plot_point(
                result,
                metric_key,
                metric_cutoff=cutoffs[metric_key],
            )
            if point is not None:
                heterotypic[pair_type][metric_key].append(point[1])

    return homotypic, heterotypic


def _metric_histogram_label(metric_key: str) -> str:
    return {
        "K_parallel": r"$R_{K_{\parallel}}$",
        "K_perp": r"$R_{K_{\perp}}$",
        "K_Q": r"$R_{K_Q}$",
    }[metric_key]


def _write_homotypic_crossing_type_rk_histogram(
    *,
    family_m: int,
    rk_values: list[float],
    bins: np.ndarray,
    output_root: Path,
    rtotal_scale: str,
    dpi: int,
) -> Path:
    """Write one R_K histogram for a homotypic crossing type."""
    info = HOMOTYPIC_PLOT_INFO[family_m]
    metric_key = {
        0: "K_parallel",
        1: "K_perp",
        2: "K_Q",
    }[family_m]

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)
    ax.hist(
        rk_values,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        edgecolor=HOMOTYPIC_COLOUR,
        facecolor=HOMOTYPIC_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
        label=(
            f"{info['label']} "
            f"{_metric_histogram_label(metric_key)} "
            f"(n={len(rk_values)})"
        ),
    )

    ax.set_xscale(rtotal_scale)
    ax.set_xlabel(r"$R_K$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Crossing count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.axvline(
        1.0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label=r"$R_K=1$",
    )
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        markerscale=LEGEND_MARKERSCALE,
    )
    _apply_fine_plot_styling(ax)
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = (
        output_root
        / RK_HISTOGRAM_SUBDIRECTORY
        / f"{info['label'].replace('-', '_')}_RK_histogram_{rtotal_scale}.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)

    print(f"Wrote: {output_file}")
    print(
        f"  crossing type={info['label']}; metric={metric_key}; "
        f"entries={len(rk_values)}; shared bins={len(bins) - 1}"
    )
    return output_file


def _write_heterotypic_crossing_type_rk_histogram(
    *,
    pair_type: str,
    metric_values: dict[str, list[float]],
    bins: np.ndarray,
    output_root: Path,
    rtotal_scale: str,
    dpi: int,
) -> Path:
    """Write a three-population R_K histogram for one heterotypic type.

    The figure overlays:
        1. the first applicable metric;
        2. the second applicable metric; and
        3. the combined distribution containing both metric populations.

    All three use exactly the same bin edges.
    """
    info = HETEROTYPIC_PLOT_INFO[pair_type]
    metric_keys = heterotypic_metric_keys(pair_type)
    combined_values = [
        value
        for metric_key in metric_keys
        for value in metric_values[metric_key]
    ]

    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)

    for metric_key in metric_keys:
        values = metric_values[metric_key]
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=HISTOGRAM_EDGE_WIDTH,
            color=HETEROTYPIC_METRIC_COLOURS[metric_key],
            label=(
                f"{_metric_histogram_label(metric_key)} "
                f"(n={len(values)})"
            ),
        )

    ax.hist(
        combined_values,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        edgecolor=HETEROTYPIC_COMBINED_COLOUR,
        facecolor=HETEROTYPIC_COMBINED_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
        label=f"Combined (n={len(combined_values)})",
    )

    ax.set_xscale(rtotal_scale)
    ax.set_xlabel(r"$R_K$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Metric-entry count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.axvline(
        1.0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label=r"$R_K=1$",
    )
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        markerscale=LEGEND_MARKERSCALE,
        title=info["label"],
        title_fontsize=LEGEND_FONTSIZE,
    )
    _apply_fine_plot_styling(ax)
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = (
        output_root
        / RK_HISTOGRAM_SUBDIRECTORY
        / f"{info['label'].replace('-', '_')}_RK_histogram_{rtotal_scale}.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)

    print(f"Wrote: {output_file}")
    print(
        f"  crossing type={info['label']}; "
        + ", ".join(
            f"{metric_key}={len(metric_values[metric_key])}"
            for metric_key in metric_keys
        )
        + f", combined={len(combined_values)}; "
        f"shared bins={len(bins) - 1}"
    )
    return output_file


def _write_crossing_type_rk_histograms(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    output_root: Path,
    rtotal_scale: str,
    n_bins: int,
    dpi: int,
) -> list[Path]:
    """Write six crossing-type R_K histogram figures using global bin edges.

    A single set of bin edges is calculated from every retained R_K value
    across M-M, D-D, Q-Q, M-D, M-Q and D-Q. Consequently, all six figures
    have identical bin sizes and directly comparable x-axis binning.
    """
    homotypic, heterotypic = _collect_crossing_type_rk_values(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )

    all_values: list[float] = []
    for values in homotypic.values():
        all_values.extend(values)
    for metric_populations in heterotypic.values():
        for values in metric_populations.values():
            all_values.extend(values)

    bins = _shared_histogram_bins(
        all_values,
        scale=rtotal_scale,
        n_bins=n_bins,
    )

    written: list[Path] = []

    for family_m in (0, 1, 2):
        written.append(
            _write_homotypic_crossing_type_rk_histogram(
                family_m=family_m,
                rk_values=homotypic[family_m],
                bins=bins,
                output_root=output_root,
                rtotal_scale=rtotal_scale,
                dpi=dpi,
            )
        )

    for pair_type in (
        "monopole_dipole",
        "monopole_quadrupole",
        "dipole_quadrupole",
    ):
        written.append(
            _write_heterotypic_crossing_type_rk_histogram(
                pair_type=pair_type,
                metric_values=heterotypic[pair_type],
                bins=bins,
                output_root=output_root,
                rtotal_scale=rtotal_scale,
                dpi=dpi,
            )
        )

    return written



def _sample_standard_deviation(values: list[float]) -> float:
    """Return the sample standard deviation, or NaN for fewer than two values."""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if len(finite_values) < 2:
        return float("nan")
    return statistics.stdev(finite_values)


def _mean_or_nan(values: list[float]) -> float:
    """Return the arithmetic mean of finite values, or NaN if none exist."""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if not finite_values:
        return float("nan")
    return statistics.fmean(finite_values)


def _rk_statistics_rows(
    *,
    homotypic: dict[int, list[float]],
    heterotypic: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """Build PRAB statistics rows for each crossing-type distribution."""
    rows: list[dict[str, Any]] = []

    homotypic_metric = {
        0: "K_parallel",
        1: "K_perp",
        2: "K_Q",
    }
    for family_m in (0, 1, 2):
        values = homotypic[family_m]
        rows.append({
            "crossing_type": HOMOTYPIC_PLOT_INFO[family_m]["label"],
            "distribution": _metric_histogram_label(
                homotypic_metric[family_m]
            ),
            "values": values,
        })

    for pair_type in (
        "monopole_dipole",
        "monopole_quadrupole",
        "dipole_quadrupole",
    ):
        metric_keys = heterotypic_metric_keys(pair_type)
        for metric_key in metric_keys:
            rows.append({
                "crossing_type": HETEROTYPIC_PLOT_INFO[pair_type]["label"],
                "distribution": _metric_histogram_label(metric_key),
                "values": heterotypic[pair_type][metric_key],
            })

        combined_values = [
            value
            for metric_key in metric_keys
            for value in heterotypic[pair_type][metric_key]
        ]
        rows.append({
            "crossing_type": HETEROTYPIC_PLOT_INFO[pair_type]["label"],
            "distribution": "Combined",
            "values": combined_values,
        })

    return rows


def _write_crossing_type_rk_statistics_table(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    output_tex: str | Path = DEFAULT_RK_STATISTICS_TEX,
) -> Path:
    """Write a PRAB-format LaTeX table of mean and sample SD of R_K.

    The table is saved by default as

        D:\PhD\PRAB\crossing_type_RK_summary_statistics_PRAB_table.tex

    Heterotypic combined rows concatenate the two metric-specific
    populations; they do not average paired values crossing-by-crossing.
    """
    homotypic, heterotypic = _collect_crossing_type_rk_values(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )
    rows = _rk_statistics_rows(
        homotypic=homotypic,
        heterotypic=heterotypic,
    )

    body_lines: list[str] = []
    for row in rows:
        values = row["values"]
        mean_value = _mean_or_nan(values)
        sd_value = _sample_standard_deviation(values)

        body_lines.append(
            " & ".join([
                row["crossing_type"],
                row["distribution"],
                str(len(values)),
                fmt_fixed(mean_value, 4),
                fmt_fixed(sd_value, 4),
            ]) + r" \\"
        )

    table_text = rf"""
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\caption{{Mean and sample standard deviation of $R_K$ for each crossing-type
distribution. Here, $N$ is the number of retained metric entries,
$\bar{{R}}_K$ is the arithmetic mean, and $s_{{R_K}}$ is the sample standard
deviation. For heterotypic crossings, the combined distribution contains
both applicable metric populations.}}
\label{{tab_all_crossing_type_RK_statistics}}
\begin{{tabular}}{{@{{}}llccc@{{}}}}
\hline
Crossing type
& Distribution
& $N$
& $\bar{{R}}_K$
& $s_{{R_K}}$ \\
\hline
{chr(10).join(body_lines)}
\hline
\end{{tabular}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{table}}
""".strip()

    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(table_text + "\n", encoding="utf-8")

    print(f"Wrote: {output_tex}")
    print(f"  statistics rows: {len(rows)}")
    return output_tex



def _rk_symbol(metric_key: str) -> str:
    return {
        "K_parallel": r"\parallel",
        "K_perp": r"\perp",
        "K_Q": "Q",
    }[metric_key]


def _rk_mean_symbol(metric_key: str) -> str:
    symbol = _rk_symbol(metric_key)
    return rf"$\overline{{R}}_{{{symbol}}}$"


def _rk_sd_symbol(metric_key: str) -> str:
    symbol = _rk_symbol(metric_key)
    return rf"$s_{{R_{{{symbol}}}}}$"


def _format_caption_stat(value: float) -> str:
    return fmt_fixed(value, 3)


def _latex_figure_filename(crossing_label: str, rtotal_scale: str) -> str:
    return f"{crossing_label.replace('-', '_')}_RK_histogram_{rtotal_scale}.png"


def _write_crossing_type_rk_figures_tex(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    rtotal_scale: str,
    output_tex: str | Path = DEFAULT_RK_FIGURES_TEX,
    latex_figure_directory: str = RK_LATEX_FIGURE_DIRECTORY,
    figure_scale: float = RK_LATEX_FIGURE_SCALE,
) -> Path:
    """Write six LaTeX figure environments with computed statistics."""
    homotypic, heterotypic = _collect_crossing_type_rk_values(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )

    blocks: list[str] = []
    homotypic_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}

    for family_m in (0, 1, 2):
        crossing_label = HOMOTYPIC_PLOT_INFO[family_m]["label"]
        metric_key = homotypic_metric[family_m]
        values = homotypic[family_m]
        mean_value = _mean_or_nan(values)
        sd_value = _sample_standard_deviation(values)
        filename = _latex_figure_filename(crossing_label, rtotal_scale)
        label_suffix = crossing_label.replace("-", "_")
        metric_tex = _metric_histogram_label(metric_key).strip("$")

        blocks.append(rf"""\begin{{figure}}[!htb]
\centering
\includegraphics[scale={figure_scale:g}]{{{latex_figure_directory}/{filename}}}
\caption{{Histogram of the homotypic {crossing_label} crossing values of
${metric_tex}$. The sample mean {_rk_mean_symbol(metric_key)}={_format_caption_stat(mean_value)}
and the sample standard deviation is {_rk_sd_symbol(metric_key)}={_format_caption_stat(sd_value)}.}}
\label{{fig_hist_{label_suffix}}}
\end{{figure}}""")

    for pair_type in (
        "monopole_dipole",
        "monopole_quadrupole",
        "dipole_quadrupole",
    ):
        crossing_label = HETEROTYPIC_PLOT_INFO[pair_type]["label"]
        metric_1, metric_2 = heterotypic_metric_keys(pair_type)
        values_1 = heterotypic[pair_type][metric_1]
        values_2 = heterotypic[pair_type][metric_2]
        combined_values = values_1 + values_2

        mean_1 = _mean_or_nan(values_1)
        sd_1 = _sample_standard_deviation(values_1)
        mean_2 = _mean_or_nan(values_2)
        sd_2 = _sample_standard_deviation(values_2)
        combined_mean = _mean_or_nan(combined_values)
        combined_sd = _sample_standard_deviation(combined_values)

        filename = _latex_figure_filename(crossing_label, rtotal_scale)
        label_suffix = crossing_label.replace("-", "_")
        metric_1_tex = _metric_histogram_label(metric_1).strip("$")
        metric_2_tex = _metric_histogram_label(metric_2).strip("$")

        blocks.append(rf"""\begin{{figure}}[!htb]
\centering
\includegraphics[scale={figure_scale:g}]{{{latex_figure_directory}/{filename}}}
\caption{{Histogram of the heterotypic {crossing_label} crossing values of
${metric_1_tex}$ and ${metric_2_tex}$. The sample mean {_rk_mean_symbol(metric_1)}={_format_caption_stat(mean_1)}
and the sample standard deviation is {_rk_sd_symbol(metric_1)}={_format_caption_stat(sd_1)}.
The sample mean {_rk_mean_symbol(metric_2)}={_format_caption_stat(mean_2)} and the sample
standard deviation is {_rk_sd_symbol(metric_2)}={_format_caption_stat(sd_2)}.
The combined $\overline{{R}}_K$={_format_caption_stat(combined_mean)} and the sample
standard deviation is $s_{{R_K}}$={_format_caption_stat(combined_sd)}.}}
\label{{fig_hist_{label_suffix}}}
\end{{figure}}""")

    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote: {output_tex}")
    print(f"  figure environments: {len(blocks)}")
    return output_tex


def _write_crossing_type_rk_figures_and_table_tex(
    *,
    figures_tex: str | Path,
    table_tex: str | Path,
    output_tex: str | Path = DEFAULT_RK_FIGURES_AND_TABLE_TEX,
) -> Path:
    figures_text = Path(figures_tex).read_text(encoding="utf-8").rstrip()
    table_text = Path(table_tex).read_text(encoding="utf-8").rstrip()
    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(figures_text + "\n\n" + table_text + "\n", encoding="utf-8")
    print(f"Wrote: {output_tex}")
    return output_tex



# -----------------------------------------------------------------------------
# Histograms of crossing-type summary statistics
# -----------------------------------------------------------------------------

def _rk_statistics_plot_rows(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
) -> list[dict[str, Any]]:
    """Return statistics rows with calculated mean and sample SD values."""
    homotypic, heterotypic = _collect_crossing_type_rk_values(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )
    rows = _rk_statistics_rows(homotypic=homotypic, heterotypic=heterotypic)
    calculated_rows: list[dict[str, Any]] = []
    for row in rows:
        values = [float(v) for v in row["values"] if math.isfinite(float(v))]
        calculated_rows.append({
            "crossing_type": row["crossing_type"],
            "distribution": row["distribution"],
            "N": len(values),
            "mean": _mean_or_nan(values),
            "sample_sd": _sample_standard_deviation(values),
        })
    return calculated_rows


def _statistics_row_plot_label(row: dict[str, Any]) -> str:
    """Return a compact LaTeX label for a summary-statistics point.

    Examples:
        M-M $\parallel$
        M-Q $Q$
        M-Q $(\parallel,Q)$   [combined population]
    """
    crossing_type = str(row["crossing_type"])
    distribution = str(row["distribution"])

    metric_symbol = {
        r"$R_{K_{\parallel}}$": r"$\parallel$",
        r"$R_{K_{\perp}}$": r"$\perp$",
        r"$R_{K_Q}$": r"$Q$",
    }

    if distribution in metric_symbol:
        return f"{crossing_type} {metric_symbol[distribution]}"

    if distribution == "Combined":
        combined_symbols = {
            "M-D": r"$(\parallel,\perp)$",
            "M-Q": r"$(\parallel,Q)$",
            "D-Q": r"$(\perp,Q)$",
        }
        return f"{crossing_type} {combined_symbols[crossing_type]}"

    return f"{crossing_type} {distribution}"


def _statistics_label_offset(row: dict[str, Any]) -> tuple[float, float, str, str]:
    """Return hand-tuned annotation offsets for the crowded lower-left points."""
    key = (str(row["crossing_type"]), str(row["distribution"]))

    offsets: dict[tuple[str, str], tuple[float, float, str, str]] = {
        ("M-M", r"$R_{K_{\parallel}}$"): (8.0, -13.0, "left", "top"),
        ("D-D", r"$R_{K_{\perp}}$"): (8.0, 8.0, "left", "bottom"),
        ("Q-Q", r"$R_{K_Q}$"): (8.0, 20.0, "left", "bottom"),
        ("M-D", r"$R_{K_{\perp}}$"): (-8.0, -18.0, "right", "top"),
        ("M-D", "Combined"): (8.0, 18.0, "left", "bottom"),
        ("M-Q", r"$R_{K_Q}$"): (-8.0, 18.0, "right", "bottom"),
        ("D-Q", r"$R_{K_{\perp}}$"): (8.0, 8.0, "left", "bottom"),
    }

    return offsets.get(key, (4.0, 4.0, "left", "bottom"))


def _write_rk_summary_statistic_histogram(
    *,
    values: list[float],
    statistic_name: str,
    x_label: str,
    output_filename: str,
    output_root: Path,
    n_bins: int,
    dpi: int,
) -> Path:
    """Write a linear histogram of one set of crossing-type statistics."""
    finite_values = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        raise ValueError(f"No finite {statistic_name} values are available.")
    if n_bins < 1:
        raise ValueError("RK_STATISTICS_HISTOGRAM_N_BINS must be at least 1.")
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if minimum == maximum:
        padding = 0.05 * abs(minimum) if minimum != 0.0 else 0.05
        minimum -= padding
        maximum += padding
    bins = np.linspace(minimum, maximum, n_bins + 1)
    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)
    ax.hist(
        finite_values,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        edgecolor=HETEROTYPIC_COMBINED_COLOUR,
        facecolor=HETEROTYPIC_COMBINED_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
        label=f"Distributions (n={finite_values.size})",
    )
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Distribution count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(loc="best", frameon=True, framealpha=1.0,
                       facecolor="white", edgecolor="black",
                       fontsize=LEGEND_FONTSIZE)
    _apply_fine_plot_styling(ax)
    _style_legend_frame(legend)
    fig.tight_layout()
    output_file = output_root / RK_HISTOGRAM_SUBDIRECTORY / output_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(f"  statistic={statistic_name}; entries={finite_values.size}; bins={n_bins}")
    return output_file


def _write_rk_mean_vs_standard_deviation_plot(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    dpi: int,
) -> Path:
    """Plot mean R_K against sample standard deviation with a linear fit.

    The Pearson correlation coefficient and coefficient of determination are
    calculated across every finite crossing-type distribution, including the
    three heterotypic combined populations.
    """
    valid_rows = [
        row for row in rows
        if math.isfinite(row["mean"])
        and math.isfinite(row["sample_sd"])
    ]
    if len(valid_rows) < 2:
        raise ValueError(
            "At least two finite mean/sample-standard-deviation pairs "
            "are required."
        )

    means = np.asarray(
        [row["mean"] for row in valid_rows],
        dtype=float,
    )
    sample_sds = np.asarray(
        [row["sample_sd"] for row in valid_rows],
        dtype=float,
    )

    # Ordinary least-squares straight-line fit: s_RK = slope * mean_RK + intercept.
    slope, intercept = np.polyfit(means, sample_sds, deg=1)
    pearson_r = float(np.corrcoef(means, sample_sds)[0, 1])
    r_squared = pearson_r ** 2

    # Retain the PRAB one-column aspect ratio, but provide slightly more room
    # for annotations around the axes.
    fig, ax = plt.subplots(figsize=PRAB_FIGSIZE)

    for row in valid_rows:
        is_homotypic = row["crossing_type"] in {"M-M", "D-D", "Q-Q"}
        colour = (
            HOMOTYPIC_COLOUR
            if is_homotypic
            else HETEROTYPIC_COLOUR
        )
        marker = "o" if is_homotypic else "s"

        ax.plot(
            row["mean"],
            row["sample_sd"],
            linestyle="none",
            marker=marker,
            markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
            zorder=4,
        )

        dx, dy, horizontal_alignment, vertical_alignment = (
            _statistics_label_offset(row)
        )

        # Use leader lines for the crowded lower-left group, matching the
        # clearer layout of avg_R_vs_std_R_improved.png.
        crowded_lower_left = (
            row["mean"] < 2.4
            and row["sample_sd"] < 3.2
        )
        arrowprops = (
            {
                "arrowstyle": "->",
                "color": colour,
                "linewidth": 0.65,
                "shrinkA": 1.0,
                "shrinkB": 2.0,
            }
            if crowded_lower_left
            else None
        )

        ax.annotate(
            _statistics_row_plot_label(row),
            (row["mean"], row["sample_sd"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.5,
            ha=horizontal_alignment,
            va=vertical_alignment,
            color="black",
            arrowprops=arrowprops,
            annotation_clip=False,
            zorder=5,
        )

    # Draw the fitted line over the full displayed data range.
    x_padding = 0.04 * (float(np.max(means)) - float(np.min(means)))
    if x_padding <= 0.0:
        x_padding = 0.1
    fit_x = np.linspace(
        max(0.0, float(np.min(means)) - x_padding),
        float(np.max(means)) + x_padding,
        200,
    )
    fit_y = slope * fit_x + intercept

    ax.plot(
        fit_x,
        fit_y,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label="Linear best fit",
        zorder=2,
    )

    ax.set_xlabel(
        r"Sample mean, $\bar{R}_K$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        r"Sample standard deviation, $s_{R_K}$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    # Add modest data margins so edge labels are not clipped.
    y_range = float(np.max(sample_sds) - np.min(sample_sds))
    y_padding = 0.07 * y_range if y_range > 0.0 else 0.1
    ax.set_xlim(
        max(0.0, float(np.min(means)) - 2.0 * x_padding),
        float(np.max(means)) + 2.0 * x_padding,
    )
    ax.set_ylim(
        max(0.0, float(np.min(sample_sds)) - y_padding),
        float(np.max(sample_sds)) + y_padding,
    )

    ax.grid(True, which="both", alpha=0.3)

    # Dummy artists provide the two population entries in the legend.
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=HOMOTYPIC_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
        label="Homotypic",
    )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=HETEROTYPIC_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
        label="Heterotypic",
    )
    ax.plot(
        [],
        [],
        linestyle="none",
        label=rf"Pearson $r={pearson_r:.3f}$",
    )
    ax.plot(
        [],
        [],
        linestyle="none",
        label=rf"$R^2={r_squared:.3f}$",
    )

    # Reorder legend entries: populations, fit, then statistics.
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "Homotypic",
        "Heterotypic",
        "Linear best fit",
        rf"Pearson $r={pearson_r:.3f}$",
        rf"$R^2={r_squared:.3f}$",
    ]
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        for handle, label in zip(handles, labels):
            if label == desired_label:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break

    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.2,
        borderpad=0.45,
        labelspacing=0.35,
    )

    _apply_fine_plot_styling(ax)
    _style_legend_frame(legend)
    fig.tight_layout()

    output_file = (
        output_root
        / RK_HISTOGRAM_SUBDIRECTORY
        / "crossing_type_RK_mean_vs_sample_standard_deviation.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_file,
        bbox_inches="tight",
        dpi=dpi,
        format="png",
    )
    plt.close(fig)

    print(f"Wrote: {output_file}")
    print(
        f"  plotted distributions: {len(valid_rows)}; "
        f"best fit: s_RK = {slope:.6g} * mean_RK + {intercept:.6g}; "
        f"Pearson r={pearson_r:.6f}; R^2={r_squared:.6f}"
    )
    return output_file


def _write_rk_summary_statistics_plots(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    output_root: Path,
    n_bins: int,
    dpi: int,
) -> list[Path]:
    """Write histograms of means and SDs, plus mean-versus-SD scatter."""
    rows = _rk_statistics_plot_rows(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )
    means = [row["mean"] for row in rows]
    sample_sds = [row["sample_sd"] for row in rows]
    return [
        _write_rk_summary_statistic_histogram(
            values=means,
            statistic_name="sample means",
            x_label=r"Sample mean, $\bar{R}_K$",
            output_filename="crossing_type_RK_mean_histogram_linear.png",
            output_root=output_root,
            n_bins=n_bins,
            dpi=dpi,
        ),
        _write_rk_summary_statistic_histogram(
            values=sample_sds,
            statistic_name="sample standard deviations",
            x_label=r"Sample standard deviation, $s_{R_K}$",
            output_filename="crossing_type_RK_sample_standard_deviation_histogram_linear.png",
            output_root=output_root,
            n_bins=n_bins,
            dpi=dpi,
        ),
        _write_rk_mean_vs_standard_deviation_plot(
            rows=rows,
            output_root=output_root,
            dpi=dpi,
        ),
    ]

def write_rtotal_plots(
    *,
    homotypic_root_or_pickle: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_root_or_pickle: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    output_root: str | Path = DEFAULT_RTOTAL_PLOT_ROOT,
    metric_cutoffs: dict[str, float] | None = None,
    rtotal_zoom_ell_min: float = RTOTAL_ZOOM_ELL_MIN,
    rtotal_zoom_ell_max: float = RTOTAL_ZOOM_ELL_MAX,
    rtotal_zoom_min: float = RTOTAL_ZOOM_MIN,
    rtotal_zoom_max: float = RTOTAL_ZOOM_MAX,
    rtotal_histogram_scale: str = RTOTAL_HISTOGRAM_SCALE,
    rtotal_histogram_n_bins: int = RTOTAL_HISTOGRAM_N_BINS,
    ell_histogram_n_bins: int = ELL_HISTOGRAM_N_BINS,
    rk_statistics_histogram_n_bins: int = RK_STATISTICS_HISTOGRAM_N_BINS,
    rk_statistics_output_tex: str | Path = DEFAULT_RK_STATISTICS_TEX,
    rk_figures_output_tex: str | Path = DEFAULT_RK_FIGURES_TEX,
    rk_figures_and_table_output_tex: str | Path = DEFAULT_RK_FIGURES_AND_TABLE_TEX,
    rk_latex_figure_directory: str = RK_LATEX_FIGURE_DIRECTORY,
    rk_latex_figure_scale: float = RK_LATEX_FIGURE_SCALE,
    dpi: int = 300,
) -> list[Path]:
    """Write R_total plots and the R_total/ell population histograms."""
    cutoffs = _normalised_metric_cutoffs(metric_cutoffs)
    homotypic_results = load_homotypic_results(homotypic_root_or_pickle)
    heterotypic_results = load_heterotypic_results(heterotypic_root_or_pickle)
    output_root = Path(output_root)

    combined_groups = _combined_groups(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
    )
    (
        homotypic_rtotal,
        heterotypic_rtotal,
        homotypic_ell,
        heterotypic_ell,
    ) = _collect_rtotal_and_ell_populations(
        combined_groups=combined_groups,
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )

    written = [
        _write_combined_rtotal_log_plot(
            combined_groups=combined_groups,
            cutoffs=cutoffs,
            output_root=output_root,
            ell_min=rtotal_zoom_ell_min,
            ell_max=rtotal_zoom_ell_max,
            rtotal_min=rtotal_zoom_min,
            rtotal_max=rtotal_zoom_max,
            dpi=dpi,
        ),
        _write_combined_rtotal_zoom_plot(
            combined_groups=combined_groups,
            cutoffs=cutoffs,
            output_root=output_root,
            ell_min=rtotal_zoom_ell_min,
            ell_max=rtotal_zoom_ell_max,
            rtotal_min=rtotal_zoom_min,
            rtotal_max=rtotal_zoom_max,
            y_scale="linear",
            dpi=dpi,
        ),
        _write_combined_rtotal_zoom_plot(
            combined_groups=combined_groups,
            cutoffs=cutoffs,
            output_root=output_root,
            ell_min=rtotal_zoom_ell_min,
            ell_max=rtotal_zoom_ell_max,
            rtotal_min=rtotal_zoom_min,
            rtotal_max=rtotal_zoom_max,
            y_scale="log",
            dpi=dpi,
        ),
        _write_rtotal_histogram(
            homotypic_rtotal=homotypic_rtotal,
            heterotypic_rtotal=heterotypic_rtotal,
            output_root=output_root,
            rtotal_scale=rtotal_histogram_scale.lower(),
            n_bins=rtotal_histogram_n_bins,
            dpi=dpi,
        ),
        _write_ell_histogram(
            homotypic_ell=homotypic_ell,
            heterotypic_ell=heterotypic_ell,
            output_root=output_root,
            n_bins=ell_histogram_n_bins,
            dpi=dpi,
        ),
    ]

    written.extend(
        _write_crossing_type_rk_histograms(
            homotypic_results=homotypic_results,
            heterotypic_results=heterotypic_results,
            cutoffs=cutoffs,
            output_root=output_root,
            rtotal_scale=rtotal_histogram_scale.lower(),
            n_bins=rtotal_histogram_n_bins,
            dpi=dpi,
        )
    )

    written.extend(
        _write_rk_summary_statistics_plots(
            homotypic_results=homotypic_results,
            heterotypic_results=heterotypic_results,
            cutoffs=cutoffs,
            output_root=output_root,
            n_bins=rk_statistics_histogram_n_bins,
            dpi=dpi,
        )
    )

    statistics_tex = _write_crossing_type_rk_statistics_table(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
        output_tex=rk_statistics_output_tex,
    )
    written.append(statistics_tex)

    figures_tex = _write_crossing_type_rk_figures_tex(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
        rtotal_scale=rtotal_histogram_scale.lower(),
        output_tex=rk_figures_output_tex,
        latex_figure_directory=rk_latex_figure_directory,
        figure_scale=rk_latex_figure_scale,
    )
    written.append(figures_tex)

    combined_tex = _write_crossing_type_rk_figures_and_table_tex(
        figures_tex=figures_tex,
        table_tex=statistics_tex,
        output_tex=rk_figures_and_table_output_tex,
    )
    written.append(combined_tex)
    return written


if __name__ == "__main__":
    write_rtotal_plots(
        homotypic_root_or_pickle=DEFAULT_HOMOTYPIC_ROOT,
        heterotypic_root_or_pickle=DEFAULT_HETEROTYPIC_ROOT,
        output_root=DEFAULT_RTOTAL_PLOT_ROOT,
        metric_cutoffs=DEFAULT_METRIC_CUTOFFS,
        rtotal_zoom_ell_min=RTOTAL_ZOOM_ELL_MIN,
        rtotal_zoom_ell_max=RTOTAL_ZOOM_ELL_MAX,
        rtotal_zoom_min=RTOTAL_ZOOM_MIN,
        rtotal_zoom_max=RTOTAL_ZOOM_MAX,
        rtotal_histogram_scale=RTOTAL_HISTOGRAM_SCALE,
        rtotal_histogram_n_bins=RTOTAL_HISTOGRAM_N_BINS,
        ell_histogram_n_bins=ELL_HISTOGRAM_N_BINS,
        rk_statistics_histogram_n_bins=RK_STATISTICS_HISTOGRAM_N_BINS,
        rk_statistics_output_tex=DEFAULT_RK_STATISTICS_TEX,
        rk_figures_output_tex=DEFAULT_RK_FIGURES_TEX,
        rk_figures_and_table_output_tex=DEFAULT_RK_FIGURES_AND_TABLE_TEX,
        rk_latex_figure_directory=RK_LATEX_FIGURE_DIRECTORY,
        rk_latex_figure_scale=RK_LATEX_FIGURE_SCALE,
        dpi=300,
    )
