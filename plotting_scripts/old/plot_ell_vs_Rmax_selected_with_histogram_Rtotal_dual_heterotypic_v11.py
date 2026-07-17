from __future__ import annotations

"""
PRAB aggregate-table compiler for the unified homotypic RF/Fourier and
heterotypic Taylor/Hessian analyses.

The compiler writes six aggregate tables:

Homotypic RF/Fourier
    1. monopole--monopole: K_parallel
    2. dipole--dipole:     K_perp
    3. quadrupole--quadrupole: K_Q

Heterotypic Taylor/Hessian
    4. monopole--dipole:       K_parallel and K_perp
    5. monopole--quadrupole:   K_parallel and K_Q
    6. dipole--quadrupole:     K_perp and K_Q

For every crossing, fields are ordered as

    E1, E2, E-, E+, R_max

to match the appendix field-summary PDFs.

The compiler expects the agreed uppercase, length-normalised metric keys:

    K_parallel_V_per_pC_per_m
    K_perp_V_per_pC_per_m2
    K_Q_V_per_pC_per_m3

Legacy aliases are accepted as fallbacks.

LaTeX requirement: include \\usepackage{longtable} in the manuscript preamble. The generated output uses \\clearpage between tables and \\* between paired heterotypic rows.
"""

import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
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
DEFAULT_RMAX_PLOT_ROOT = (
    DEFAULT_PRAB_ROOT
    / "figures"
    / "ell_vs_Rmax"
)

# Mixed-field metric cut-offs. A point is omitted when
# max(abs(E_minus), abs(E_plus)) is below the corresponding cut-off.
# Set a cut-off to 0.0 to retain every finite result for that metric.
DEFAULT_METRIC_CUTOFFS: dict[str, float] = {
    "K_parallel": 0.0,
    "K_perp": 0.0,
    "K_Q": 0.0,
}

# Select one or both y-axis scales. Supported values: "linear" and "log".
# Each scale is written to a separately named PNG file.
DEFAULT_Y_SCALES: tuple[str, ...] = ("linear", "log")

# Histogram configuration. HISTOGRAM_RMAX_SCALE controls the R_max axis
# and may be "linear" or "log". Both homotypic and heterotypic data use
# exactly the same bin edges.
HISTOGRAM_RMAX_SCALE = "log"
HISTOGRAM_N_BINS = 20
HISTOGRAM_ALPHA = 0.45

# R_total zoom configuration.
RTOTAL_ZOOM_ELL_MIN = 0.85
RTOTAL_ZOOM_ELL_MAX = 1.15
RTOTAL_ZOOM_MIN = 1.0
RTOTAL_ZOOM_MAX = 5.0

# Histogram axis/bin configuration.
RTOTAL_HISTOGRAM_SCALE = "log"   # "linear" or "log"
RTOTAL_HISTOGRAM_N_BINS = 20
ELL_HISTOGRAM_N_BINS = 20


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

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

    parent_maximum = max(
        abs(values["E1"]),
        abs(values["E2"]),
    )
    mixed_maximum = max(
        abs(values["minus"]),
        abs(values["plus"]),
    )
    values["R_max"] = safe_ratio(
        mixed_maximum,
        parent_maximum,
    )
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
# ell versus R_max plots
# -----------------------------------------------------------------------------

HETEROTYPIC_PLOT_INFO: dict[str, dict[str, str]] = {
    "monopole_dipole": {
        "label": "M-D",
        "marker_1": "o",
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


def _heterotypic_parent_value(
    result: dict[str, Any],
    metric_key: str,
    values: dict[str, float],
) -> float:
    """Return the relevant parent value for a heterotypic metric.

    K_parallel belongs to the monopole parent (m=0), K_perp to the
    dipole parent (m=1), and K_Q to the quadrupole parent (m=2).
    """
    target_m = {"K_parallel": 0, "K_perp": 1, "K_Q": 2}[metric_key]
    mode_i, mode_j = result_modes(result)
    m_i = mode_azimuthal_index(mode_i)
    m_j = mode_azimuthal_index(mode_j)

    if m_i == target_m:
        return abs_finite_or_nan(values["E1"])
    if m_j == target_m:
        return abs_finite_or_nan(values["E2"])
    return float("nan")


def _is_heterotypic_result(result: dict[str, Any]) -> bool:
    mode_i, mode_j = result_modes(result)
    m_i = mode_azimuthal_index(mode_i)
    m_j = mode_azimuthal_index(mode_j)
    return m_i is not None and m_j is not None and m_i != m_j


def rmax_plot_point(
    result: dict[str, Any],
    metric_key: str,
    *,
    metric_cutoff: float,
) -> tuple[float, float] | None:
    """Return ``(ell, R_max)`` using homo/heterotypic conventions.

    Homotypic:
        max(K_plus, K_minus) / max(K_1, K_2)

    Heterotypic:
        max(K_plus, K_minus) / K_parent,
    where K_parent is the single parent belonging to the selected metric.
    """
    ell, _ = crossing_parameters(result)
    values = field_metric_values(result, metric_key)
    k_minus = abs_finite_or_nan(values["minus"])
    k_plus = abs_finite_or_nan(values["plus"])
    mixed_maximum = max(k_minus, k_plus)

    if (
        not math.isfinite(ell)
        or not math.isfinite(mixed_maximum)
        or mixed_maximum < metric_cutoff
    ):
        return None

    if _is_heterotypic_result(result):
        denominator = _heterotypic_parent_value(result, metric_key, values)
    else:
        denominator = max(
            abs_finite_or_nan(values["E1"]),
            abs_finite_or_nan(values["E2"]),
        )

    r_max = safe_ratio(mixed_maximum, denominator)
    if not math.isfinite(r_max):
        return None
    return ell, r_max


def rtotal_plot_point(
    result: dict[str, Any],
    metric_key: str,
    *,
    metric_cutoff: float,
) -> tuple[float, float] | None:
    """Return ``(ell, R_total)`` using homo/heterotypic conventions.

    Homotypic:
        (K_plus + K_minus) / (K_1 + K_2)

    Heterotypic:
        (K_plus + K_minus) / K_parent,
    where K_parent is the single parent belonging to the selected metric.
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

    if _is_heterotypic_result(result):
        denominator = _heterotypic_parent_value(result, metric_key, values)
    else:
        denominator = k1 + k2

    r_total = safe_ratio(k_plus + k_minus, denominator)
    if not math.isfinite(r_total):
        return None
    return ell, r_total


def _metric_legend_symbol(metric_key: str) -> str:
    return {
        "K_parallel": r"\parallel",
        "K_perp": r"\perp",
        "K_Q": "Q",
    }[metric_key]


def _fom_legend_label(base_label: str, metric_key: str, fom: str) -> str:
    symbol = _metric_legend_symbol(metric_key)
    if fom == "Rmax":
        return rf"{base_label} $R_{{\mathrm{{max}},{symbol}}}$"
    if fom == "Rtotal":
        return rf"{base_label} $R_{{\mathrm{{total}},{symbol}}}$"
    raise ValueError(f"Unknown FoM label type: {fom}")

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


def _scatter_result_group(
    ax: Any,
    results: Iterable[dict[str, Any]],
    metric_key: str,
    *,
    metric_cutoff: float,
    marker: str,
    label: str,
    color: str | None = None,
) -> int:
    points = [
        point
        for result in results
        if (point := rmax_plot_point(
            result,
            metric_key,
            metric_cutoff=metric_cutoff,
        )) is not None
    ]
    points.sort(key=lambda point: point[0])
    if not points:
        return 0

    x_values, y_values = zip(*points)
    resolved_color = color or ax._get_lines.get_next_color()
    _plot_hollow_markers(
        ax,
        x_values,
        y_values,
        marker=marker,
        label=label,
        color=resolved_color,
        markersize=7.5,
        markeredgewidth=1.35,
    )
    return len(points)


def _finish_rmax_plot(
    fig: Any,
    ax: Any,
    *,
    title: str,
    output_file: Path,
    show_legend: bool,
    dpi: int,
    y_scale: str,
) -> Path:
    if y_scale not in {"linear", "log"}:
        raise ValueError(f"Unsupported y-axis scale: {y_scale!r}")

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$R_{\max}$")
    ax.set_title(title)
    ax.set_yscale(y_scale)
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{\max}=1$",
        zorder=1,
    )
    ax.grid(True, which="both", alpha=0.3)
    if show_legend:
        if y_scale == "log":
            ax.legend(
                loc="lower right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
        else:
            ax.legend()
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    return output_file



def _write_annotated_combined_zoom_plot(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    output_root: Path,
    dpi: int,
) -> Path:
    """Write the annotated linear R_max zoom with both heterotypic metrics."""
    family_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}
    homotypic_colour = "#2A6FBB"
    heterotypic_colour = "#E07A1F"
    groups: list[tuple[list[dict[str, Any]], str, str, str, str]] = []

    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        groups.append((
            [r for r in homotypic_results if homotypic_family_m(r) == family_m],
            family_metric[family_m], info["marker"], info["label"],
            homotypic_colour,
        ))
    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        family_results = [r for r in heterotypic_results if pair_type_key(r) == pair_type]
        for metric_index, metric_key in enumerate(heterotypic_metric_keys(pair_type), start=1):
            groups.append((
                family_results,
                metric_key,
                info[f"marker_{metric_index}"],
                info["label"],
                heterotypic_colour,
            ))

    xmin, xmax = 0.85, 1.15
    ymin, ymax = 0.5, 5.0
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    total_plotted = 0

    for results, metric_key, marker, base_label, color in groups:
        selected: list[tuple[dict[str, Any], float, float]] = []
        for result in results:
            point = rmax_plot_point(
                result, metric_key, metric_cutoff=cutoffs[metric_key]
            )
            if point is None:
                continue
            ell, r_max = point
            if xmin <= ell <= xmax and ymin <= r_max <= ymax:
                selected.append((result, ell, r_max))
        if not selected:
            continue

        _plot_hollow_markers(
            ax, [v[1] for v in selected], [v[2] for v in selected],
            marker=marker,
            label=_fom_legend_label(base_label, metric_key, "Rmax"),
            color=color, markersize=12.0, markeredgewidth=1.4,
        )
        for result, ell, r_max in selected:
            mode_i, mode_j = result_modes(result)
            ax.annotate(
                f"{latex_mode(mode_i)}\n{latex_mode(mode_j)}",
                xy=(ell, r_max), xytext=(4, 4),
                textcoords="offset points", fontsize=12,
                ha="left", va="bottom", annotation_clip=True, zorder=4,
            )
        total_plotted += len(selected)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\ell$", fontsize=20)
    ax.set_ylabel(r"$R_{\max}$", fontsize=20)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label=r"$R_{\max}=1$", zorder=1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black", fontsize=10)
    fig.tight_layout()

    output_file = output_root / "homotypic_and_heterotypic_ell_vs_Rmax_annotated_zoom_linear.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  plotted annotated zoom points: {total_plotted}; "
        "heterotypic metrics=two per crossing"
    )
    return output_file

def _combined_groups_and_values(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
) -> tuple[
    list[tuple[list[dict[str, Any]], str, str, str]],
    list[float],
    list[float],
]:
    """Return plot groups and R_max populations.

    Each homotypic crossing contributes one metric-specific value. Each
    heterotypic crossing contributes two values, one for each relevant parent
    family metric.
    """
    family_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}
    homotypic_colour = "#2A6FBB"
    heterotypic_colour = "#E07A1F"

    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]] = []
    homotypic_rmax: list[float] = []
    heterotypic_rmax: list[float] = []

    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        metric_key = family_metric[family_m]
        family_results = [
            result for result in homotypic_results
            if homotypic_family_m(result) == family_m
        ]
        combined_groups.append((
            family_results, metric_key, info["marker"],
            f"{info['label']}|||{homotypic_colour}",
        ))
        for result in family_results:
            point = rmax_plot_point(
                result, metric_key, metric_cutoff=cutoffs[metric_key]
            )
            if point is not None:
                homotypic_rmax.append(point[1])

    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        family_results = [
            result for result in heterotypic_results
            if pair_type_key(result) == pair_type
        ]
        for metric_index, metric_key in enumerate(heterotypic_metric_keys(pair_type), start=1):
            combined_groups.append((
                family_results,
                metric_key,
                info[f"marker_{metric_index}"],
                f"{info['label']}|||{heterotypic_colour}",
            ))
            for result in family_results:
                point = rmax_plot_point(
                    result, metric_key, metric_cutoff=cutoffs[metric_key]
                )
                if point is not None:
                    heterotypic_rmax.append(point[1])

    return combined_groups, homotypic_rmax, heterotypic_rmax

def _write_combined_rtotal_log_plot(
    *,
    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]],
    cutoffs: dict[str, float],
    output_root: Path,
    dpi: int,
) -> Path:
    """Write the combined ell--R_total plot on a logarithmic y-axis."""
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
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
            label=_fom_legend_label(label, group_metric, "Rtotal"),
            color=color,
            markersize=7.5,
            markeredgewidth=1.35,
        )
        count += len(points)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$R_{\mathrm{total}}$")
    ax.set_yscale("log")
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
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
    )
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
    cutoffs: dict[str, float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Collect FoM populations while keeping one ell entry per crossing.

    Heterotypic R_total contains two values per crossing. The ell histogram is
    intentionally unchanged and contains one ell value per physical crossing.
    """
    homotypic_rtotal: list[float] = []
    heterotypic_rtotal: list[float] = []
    homotypic_ell_by_id: dict[tuple[Any, ...], float] = {}
    heterotypic_ell_by_id: dict[tuple[Any, ...], float] = {}

    for group_results, metric_key, _marker, label_color in combined_groups:
        is_homotypic = label_color.endswith("|||#2A6FBB")
        target_rtotal = homotypic_rtotal if is_homotypic else heterotypic_rtotal
        target_ell = homotypic_ell_by_id if is_homotypic else heterotypic_ell_by_id

        for result in group_results:
            point = rtotal_plot_point(
                result, metric_key, metric_cutoff=cutoffs[metric_key]
            )
            if point is None:
                continue
            ell, r_total = point
            target_rtotal.append(r_total)
            target_ell.setdefault(_result_identity(result), ell)

    return (
        homotypic_rtotal,
        heterotypic_rtotal,
        list(homotypic_ell_by_id.values()),
        list(heterotypic_ell_by_id.values()),
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
    dpi: int,
) -> Path:
    """Write a configurable linear-scale zoom of ell versus R_total."""
    if ell_min >= ell_max:
        raise ValueError("RTOTAL_ZOOM_ELL_MIN must be below RTOTAL_ZOOM_ELL_MAX.")
    if rtotal_min >= rtotal_max:
        raise ValueError("RTOTAL_ZOOM_MIN must be below RTOTAL_ZOOM_MAX.")

    fig, ax = plt.subplots(figsize=(10.0, 7.2))
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
            label=_fom_legend_label(label, metric_key, "Rtotal"),
            color=color,
            markersize=12.0,
            markeredgewidth=1.4,
        )
        count += len(points)

    ax.set_xlim(ell_min, ell_max)
    ax.set_ylim(rtotal_min, rtotal_max)
    ax.set_yscale("linear")
    ax.set_xlabel(r"$\ell$", fontsize=20)
    ax.set_ylabel(r"$R_{\mathrm{total}}$", fontsize=20)
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{\mathrm{total}}=1$",
        zorder=1,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    fig.tight_layout()

    output_file = output_root / "homotypic_and_heterotypic_ell_vs_Rtotal_zoom_linear.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        f"  plotted points: {count}; cut-off=family-specific; "
        f"ell=[{ell_min}, {ell_max}]; R_total=[{rtotal_min}, {rtotal_max}]"
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

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.hist(
        homotypic_rtotal,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Homotypic (n={len(homotypic_rtotal)})",
        edgecolor="#2A6FBB",
        facecolor="#2A6FBB",
        linewidth=1.2,
    )
    ax.hist(
        heterotypic_rtotal,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Heterotypic (n={len(heterotypic_rtotal)})",
        edgecolor="#E07A1F",
        facecolor="#E07A1F",
        linewidth=1.2,
    )
    ax.set_xscale(rtotal_scale)
    ax.set_xlabel(r"$R_{\mathrm{total}}$")
    ax.set_ylabel("Crossing count")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2,
               label=r"$R_{\mathrm{total}}=1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black")
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
        raise ValueError("ELL_HISTOGRAM_N_BINS must be at least 1.")

    minimum = min(all_values)
    maximum = max(all_values)
    if minimum == maximum:
        minimum -= 0.05
        maximum += 0.05
    bins = np.linspace(minimum, maximum, n_bins + 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.hist(
        homotypic_ell,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Homotypic (n={len(homotypic_ell)})",
        edgecolor="#2A6FBB",
        facecolor="#2A6FBB",
        linewidth=1.2,
    )
    ax.hist(
        heterotypic_ell,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Heterotypic (n={len(heterotypic_ell)})",
        edgecolor="#E07A1F",
        facecolor="#E07A1F",
        linewidth=1.2,
    )
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("Crossing count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black")
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


def _write_combined_log_plot(
    *,
    combined_groups: list[tuple[list[dict[str, Any]], str, str, str]],
    cutoffs: dict[str, float],
    output_root: Path,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    count = 0
    for group_results, group_metric, marker, label_color in combined_groups:
        base_label, color = label_color.split("|||", 1)
        points = [
            point for result in group_results
            if (point := rmax_plot_point(
                result, group_metric, metric_cutoff=cutoffs[group_metric]
            )) is not None
        ]
        points.sort(key=lambda item: item[0])
        if not points:
            continue
        x_values, y_values = zip(*points)
        _plot_hollow_markers(
            ax, x_values, y_values, marker=marker,
            label=_fom_legend_label(base_label, group_metric, "Rmax"),
            color=color, markersize=7.5, markeredgewidth=1.35,
        )
        count += len(points)

    output_file = output_root / "homotypic_and_heterotypic_ell_vs_Rmax_log.png"
    _finish_rmax_plot(
        fig, ax, title="", output_file=output_file, show_legend=True,
        dpi=dpi, y_scale="log",
    )
    print(
        f"  plotted points: {count}; cut-off=family-specific; "
        "y-scale=log; heterotypic metrics=two per crossing"
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
        raise ValueError("No positive finite R_max values are available.")
    if n_bins < 1:
        raise ValueError("HISTOGRAM_N_BINS must be at least 1.")

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
        "HISTOGRAM_RMAX_SCALE must be either 'linear' or 'log'."
    )


def _write_rmax_histogram(
    *,
    homotypic_rmax: list[float],
    heterotypic_rmax: list[float],
    output_root: Path,
    rmax_scale: str,
    n_bins: int,
    dpi: int,
) -> Path:
    all_values = homotypic_rmax + heterotypic_rmax
    bins = _shared_histogram_bins(
        all_values,
        scale=rmax_scale,
        n_bins=n_bins,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.hist(
        homotypic_rmax,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Homotypic (n={len(homotypic_rmax)})",
        edgecolor="#2A6FBB",
        facecolor="#2A6FBB",
        linewidth=1.2,
    )
    ax.hist(
        heterotypic_rmax,
        bins=bins,
        histtype="stepfilled",
        alpha=HISTOGRAM_ALPHA,
        label=f"Heterotypic (n={len(heterotypic_rmax)})",
        edgecolor="#E07A1F",
        facecolor="#E07A1F",
        linewidth=1.2,
    )

    ax.set_xscale(rmax_scale)
    ax.set_xlabel(r"$R_{\max}$")
    ax.set_ylabel("Crossing count")
    ax.axvline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{\max}=1$",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    fig.tight_layout()

    output_file = (
        output_root
        / f"homotypic_and_heterotypic_Rmax_histogram_{rmax_scale}.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        "  histogram entries: "
        f"homotypic={len(homotypic_rmax)}, "
        f"heterotypic={len(heterotypic_rmax)}; "
        f"shared bins={n_bins}; R_max scale={rmax_scale}"
    )
    return output_file


def write_selected_ell_vs_rmax_plots(
    *,
    homotypic_root_or_pickle: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_root_or_pickle: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    output_root: str | Path = DEFAULT_RMAX_PLOT_ROOT,
    metric_cutoffs: dict[str, float] | None = None,
    histogram_rmax_scale: str = HISTOGRAM_RMAX_SCALE,
    histogram_n_bins: int = HISTOGRAM_N_BINS,
    rtotal_zoom_ell_min: float = RTOTAL_ZOOM_ELL_MIN,
    rtotal_zoom_ell_max: float = RTOTAL_ZOOM_ELL_MAX,
    rtotal_zoom_min: float = RTOTAL_ZOOM_MIN,
    rtotal_zoom_max: float = RTOTAL_ZOOM_MAX,
    rtotal_histogram_scale: str = RTOTAL_HISTOGRAM_SCALE,
    rtotal_histogram_n_bins: int = RTOTAL_HISTOGRAM_N_BINS,
    ell_histogram_n_bins: int = ELL_HISTOGRAM_N_BINS,
    dpi: int = 300,
) -> list[Path]:
    """Write selected R_max/R_total plots and population histograms."""
    cutoffs = _normalised_metric_cutoffs(metric_cutoffs)
    homotypic_results = load_homotypic_results(homotypic_root_or_pickle)
    heterotypic_results = load_heterotypic_results(heterotypic_root_or_pickle)
    output_root = Path(output_root)

    combined_groups, homotypic_rmax, heterotypic_rmax = (
        _combined_groups_and_values(
            homotypic_results=homotypic_results,
            heterotypic_results=heterotypic_results,
            cutoffs=cutoffs,
        )
    )
    (
        homotypic_rtotal,
        heterotypic_rtotal,
        homotypic_ell,
        heterotypic_ell,
    ) = _collect_rtotal_and_ell_populations(
        combined_groups=combined_groups,
        cutoffs=cutoffs,
    )

    written = [
        _write_combined_log_plot(
            combined_groups=combined_groups,
            cutoffs=cutoffs,
            output_root=output_root,
            dpi=dpi,
        ),
        _write_combined_rtotal_log_plot(
            combined_groups=combined_groups,
            cutoffs=cutoffs,
            output_root=output_root,
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
        _write_annotated_combined_zoom_plot(
            homotypic_results=homotypic_results,
            heterotypic_results=heterotypic_results,
            cutoffs=cutoffs,
            output_root=output_root,
            dpi=dpi,
        ),
        _write_rmax_histogram(
            homotypic_rmax=homotypic_rmax,
            heterotypic_rmax=heterotypic_rmax,
            output_root=output_root,
            rmax_scale=histogram_rmax_scale.lower(),
            n_bins=histogram_n_bins,
            dpi=dpi,
        ),
    ]
    return written


if __name__ == "__main__":
    write_selected_ell_vs_rmax_plots(
        homotypic_root_or_pickle=DEFAULT_HOMOTYPIC_ROOT,
        heterotypic_root_or_pickle=DEFAULT_HETEROTYPIC_ROOT,
        output_root=DEFAULT_RMAX_PLOT_ROOT,
        metric_cutoffs=DEFAULT_METRIC_CUTOFFS,
        histogram_rmax_scale=HISTOGRAM_RMAX_SCALE,
        histogram_n_bins=HISTOGRAM_N_BINS,
        rtotal_zoom_ell_min=RTOTAL_ZOOM_ELL_MIN,
        rtotal_zoom_ell_max=RTOTAL_ZOOM_ELL_MAX,
        rtotal_zoom_min=RTOTAL_ZOOM_MIN,
        rtotal_zoom_max=RTOTAL_ZOOM_MAX,
        rtotal_histogram_scale=RTOTAL_HISTOGRAM_SCALE,
        rtotal_histogram_n_bins=RTOTAL_HISTOGRAM_N_BINS,
        ell_histogram_n_bins=ELL_HISTOGRAM_N_BINS,
        dpi=300,
    )
