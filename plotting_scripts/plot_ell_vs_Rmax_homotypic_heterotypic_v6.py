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
        "label": "monopole--dipole",
        "marker": "o",
    },
    "monopole_quadrupole": {
        "label": "monopole--quadrupole",
        "marker": "s",
    },
    "dipole_quadrupole": {
        "label": "dipole--quadrupole",
        "marker": "^",
    },
}

HOMOTYPIC_PLOT_INFO: dict[int, dict[str, str]] = {
    0: {"label": "monopole--monopole", "marker": "*"},
    1: {"label": "dipole--dipole", "marker": "+"},
    2: {"label": "quadrupole--quadrupole", "marker": "v"},
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


def rmax_plot_point(
    result: dict[str, Any],
    metric_key: str,
    *,
    metric_cutoff: float,
) -> tuple[float, float] | None:
    """Return (ell, R_max), applying the mixed-field metric cut-off."""
    ell, _ = crossing_parameters(result)
    values = field_metric_values(result, metric_key)
    mixed_maximum = max(
        abs_finite_or_nan(values["minus"]),
        abs_finite_or_nan(values["plus"]),
    )
    r_max = finite_or_nan(values["R_max"])

    if (
        not math.isfinite(ell)
        or not math.isfinite(r_max)
        or not math.isfinite(mixed_maximum)
        or mixed_maximum < metric_cutoff
    ):
        return None
    return ell, r_max


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
    """Write the annotated linear-scale zoom of the combined data set."""
    family_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}
    homotypic_colour = "#2A6FBB"
    heterotypic_colour = "#E07A1F"

    groups: list[tuple[list[dict[str, Any]], str, str, str, str]] = []
    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        groups.append((
            [r for r in homotypic_results if homotypic_family_m(r) == family_m],
            family_metric[family_m],
            info["marker"],
            f"homotypic {info['label']}",
            homotypic_colour,
        ))
    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        # Match the metric selection used by the full combined figure.
        groups.append((
            [r for r in heterotypic_results if pair_type_key(r) == pair_type],
            heterotypic_metric_keys(pair_type)[0],
            info["marker"],
            f"heterotypic {info['label']}",
            heterotypic_colour,
        ))

    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    total_plotted = 0

    for results, metric_key, marker, label, color in groups:
        selected: list[tuple[dict[str, Any], float, float]] = []
        for result in results:
            point = rmax_plot_point(
                result,
                metric_key,
                metric_cutoff=cutoffs[metric_key],
            )
            if point is None:
                continue
            ell, r_max = point
            if 0.85 <= ell <= 1.15 and 1.0 <= r_max <= 10.0:
                selected.append((result, ell, r_max))

        if not selected:
            continue

        x_values = [item[1] for item in selected]
        y_values = [item[2] for item in selected]
        _plot_hollow_markers(
            ax,
            x_values,
            y_values,
            marker=marker,
            label=label,
            color=color,
            markersize=8.0,
            markeredgewidth=1.4,
        )

        for result, ell, r_max in selected:
            mode_i, mode_j = result_modes(result)
            annotation = f"{latex_mode(mode_i)}\n{latex_mode(mode_j)}"
            ax.annotate(
                annotation,
                xy=(ell, r_max),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=12,
                ha="left",
                va="bottom",
                annotation_clip=True,
                zorder=4,
            )
        total_plotted += len(selected)

    xmin=0.85
    xmax=1.15
    ymin=0.5
    ymax=5.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale("linear")
    ax.set_xlabel(r"$\ell$", fontsize=20)
    ax.set_ylabel(r"$R_{\max}$", fontsize=20)
    # ax.set_title(
    #     r"Homotypic and heterotypic crossings: annotated zoom "
    #     f"({xmin}"r"$\leq\ell\leq$"f"{xmax}, {ymin}"r"$\leq R_{\max}\leq$"f"{ymax})"
    # )
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{\max}=1$",
        zorder=1,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()

    output_file = (
        output_root
        / "homotypic_and_heterotypic_ell_vs_Rmax_annotated_zoom_linear.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=dpi, format="png")
    plt.close(fig)
    print(f"Wrote: {output_file}")
    print(
        "  plotted annotated zoom points: "
        f"{total_plotted}; cut-off=family-specific; y-scale=linear"
    )
    return output_file

def write_ell_vs_rmax_plots(
    *,
    homotypic_root_or_pickle: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_root_or_pickle: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    output_root: str | Path = DEFAULT_RMAX_PLOT_ROOT,
    metric_cutoffs: dict[str, float] | None = None,
    y_scales: Iterable[str] = DEFAULT_Y_SCALES,
    dpi: int = 300,
) -> list[Path]:
    """Write linear and/or logarithmic ell-versus-R_max PNG figures."""
    cutoffs = _normalised_metric_cutoffs(metric_cutoffs)
    homotypic_results = load_homotypic_results(homotypic_root_or_pickle)
    heterotypic_results = load_heterotypic_results(heterotypic_root_or_pickle)
    output_root = Path(output_root)

    scales = tuple(dict.fromkeys(str(scale).lower() for scale in y_scales))
    if not scales:
        raise ValueError("At least one y-axis scale must be selected.")
    invalid = set(scales) - {"linear", "log"}
    if invalid:
        raise ValueError("Unsupported y-axis scale(s): " + ", ".join(sorted(invalid)))

    written: list[Path] = []

    def save_group(
        *,
        groups: list[tuple[list[dict[str, Any]], str, str, str | None]],
        metric_key: str | None,
        title: str,
        output_stem: Path,
        show_legend: bool,
    ) -> None:
        for y_scale in scales:
            fig, ax = plt.subplots(figsize=(6.8, 5.0))
            count = 0
            for group_results, group_metric, marker, label_color in groups:
                label, color = label_color.split("|||", 1) if label_color and "|||" in label_color else (label_color or "", None)
                count += _scatter_result_group(
                    ax,
                    group_results,
                    group_metric,
                    metric_cutoff=cutoffs[group_metric],
                    marker=marker,
                    label=label,
                    color=color,
                )
            output_file = output_stem.with_name(f"{output_stem.name}_{y_scale}.png")
            written.append(_finish_rmax_plot(
                fig,
                ax,
                title=title,
                output_file=output_file,
                show_legend=show_legend,
                dpi=dpi,
                y_scale=y_scale,
            ))
            cutoff_text = cutoffs[metric_key] if metric_key else "family-specific"
            print(f"  plotted points: {count}; cut-off={cutoff_text}; y-scale={y_scale}")

    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        family_results = [r for r in heterotypic_results if pair_type_key(r) == pair_type]
        for metric_key in heterotypic_metric_keys(pair_type):
            save_group(
                groups=[(family_results, metric_key, info["marker"], info["label"])],
                metric_key=metric_key,
                title=(rf"Heterotypic {info['label']}: $\ell$ versus $R_{{\max}}$ "
                       rf"({METRIC_INFO[metric_key]['latex'].strip('$')})"),
                output_stem=output_root / "heterotypic" /
                    f"heterotypic_{pair_type}_ell_vs_Rmax_{METRIC_FILENAME[metric_key]}",
                show_legend=False,
            )

    for metric_key in METRIC_INFO:
        groups = []
        for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
            if metric_key in heterotypic_metric_keys(pair_type):
                family_results = [r for r in heterotypic_results if pair_type_key(r) == pair_type]
                groups.append((family_results, metric_key, info["marker"], info["label"]))
        save_group(
            groups=groups,
            metric_key=metric_key,
            title=(rf"All heterotypic crossings: $\ell$ versus $R_{{\max}}$ "
                   rf"({METRIC_INFO[metric_key]['latex'].strip('$')})"),
            output_stem=output_root / "heterotypic" /
                f"all_heterotypic_ell_vs_Rmax_{METRIC_FILENAME[metric_key]}",
            show_legend=True,
        )

    family_metric = {0: "K_parallel", 1: "K_perp", 2: "K_Q"}
    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        metric_key = family_metric[family_m]
        family_results = [r for r in homotypic_results if homotypic_family_m(r) == family_m]
        save_group(
            groups=[(family_results, metric_key, info["marker"], info["label"])],
            metric_key=metric_key,
            title=(rf"Homotypic {info['label']}: $\ell$ versus $R_{{\max}}$ "
                   rf"({METRIC_INFO[metric_key]['latex'].strip('$')})"),
            output_stem=output_root / "homotypic" /
                f"homotypic_{info['label'].replace('--', '_')}_ell_vs_Rmax_{METRIC_FILENAME[metric_key]}",
            show_legend=False,
        )

    homo_groups = []
    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        metric_key = family_metric[family_m]
        homo_groups.append((
            [r for r in homotypic_results if homotypic_family_m(r) == family_m],
            metric_key,
            info["marker"],
            info["label"],
        ))
    save_group(
        groups=homo_groups,
        metric_key=None,
        title=r"All homotypic crossings: $\ell$ versus $R_{\max}$",
        output_stem=output_root / "homotypic" / "all_homotypic_ell_vs_Rmax",
        show_legend=True,
    )

    homotypic_colour = "#2A6FBB"
    heterotypic_colour = "#E07A1F"
    combined_groups = []
    for family_m, info in HOMOTYPIC_PLOT_INFO.items():
        metric_key = family_metric[family_m]
        combined_groups.append((
            [r for r in homotypic_results if homotypic_family_m(r) == family_m],
            metric_key,
            info["marker"],
            f"homotypic {info['label']}|||{homotypic_colour}",
        ))
    for pair_type, info in HETEROTYPIC_PLOT_INFO.items():
        metric_key = heterotypic_metric_keys(pair_type)[0]
        combined_groups.append((
            [r for r in heterotypic_results if pair_type_key(r) == pair_type],
            metric_key,
            info["marker"],
            f"heterotypic {info['label']}|||{heterotypic_colour}",
        ))
    save_group(
        groups=combined_groups,
        metric_key=None,
        title=r"Homotypic and heterotypic crossings: $\ell$ versus $R_{\max}$",
        output_stem=output_root / "homotypic_and_heterotypic_ell_vs_Rmax",
        show_legend=True,
    )

    written.append(_write_annotated_combined_zoom_plot(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
        output_root=output_root,
        dpi=dpi,
    ))
    return written


if __name__ == "__main__":
    write_ell_vs_rmax_plots(
        homotypic_root_or_pickle=DEFAULT_HOMOTYPIC_ROOT,
        heterotypic_root_or_pickle=DEFAULT_HETEROTYPIC_ROOT,
        output_root=DEFAULT_RMAX_PLOT_ROOT,
        metric_cutoffs=DEFAULT_METRIC_CUTOFFS,
        y_scales=DEFAULT_Y_SCALES,
        dpi=300,
    )
