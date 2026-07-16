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
# Zoom-region PRAB table configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_TEX = DEFAULT_PRAB_ROOT / "zoom_crossings_PRAB_table.tex"

# These limits must match the annotated ell-versus-R_max zoom figure.
ZOOM_XMIN = 0.85
ZOOM_XMAX = 1.15
ZOOM_YMIN = 0.5
ZOOM_YMAX = 5.0

# A crossing is retained only when max(abs(E_minus), abs(E_plus)) is at least
# the cut-off for the metric used to represent that crossing in the combined
# homotypic-and-heterotypic plot.
DEFAULT_METRIC_CUTOFFS: dict[str, float] = {
    "K_parallel": 0.0,
    "K_perp": 0.0,
    "K_Q": 0.0,
}

HOMOTYPIC_TYPE_LABELS = {
    0: "homotypic monopole--monopole",
    1: "homotypic dipole--dipole",
    2: "homotypic quadrupole--quadrupole",
}

HETEROTYPIC_TYPE_LABELS = {
    "monopole_dipole": "heterotypic monopole--dipole",
    "monopole_quadrupole": "heterotypic monopole--quadrupole",
    "dipole_quadrupole": "heterotypic dipole--quadrupole",
}


def normalised_metric_cutoffs(
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


def selected_metric_for_combined_plot(
    result: dict[str, Any],
    *,
    population: str,
) -> str:
    """Return the metric used for this crossing in the combined plot."""
    if population == "homotypic":
        return homotypic_metric_key(result)
    if population == "heterotypic":
        # This intentionally matches the combined plot: the first applicable
        # metric is used for each heterotypic pair type.
        return heterotypic_metric_keys(pair_type_key(result))[0]
    raise ValueError(f"Unsupported population: {population!r}")


def zoom_table_record(
    result: dict[str, Any],
    *,
    population: str,
    cutoffs: dict[str, float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> dict[str, Any] | None:
    metric_key = selected_metric_for_combined_plot(
        result,
        population=population,
    )
    values = field_metric_values(result, metric_key)
    ell, f_hat = crossing_parameters(result)
    r_max = finite_or_nan(values["R_max"])
    mixed_maximum = max(
        abs_finite_or_nan(values["minus"]),
        abs_finite_or_nan(values["plus"]),
    )

    if (
        not math.isfinite(ell)
        or not math.isfinite(f_hat)
        or not math.isfinite(r_max)
        or not math.isfinite(mixed_maximum)
        or mixed_maximum < cutoffs[metric_key]
        or not (xmin <= ell <= xmax)
        or not (ymin <= r_max <= ymax)
    ):
        return None

    mode_i, mode_j = result_modes(result)
    if population == "homotypic":
        crossing_type = HOMOTYPIC_TYPE_LABELS[homotypic_family_m(result)]
    else:
        crossing_type = HETEROTYPIC_TYPE_LABELS[pair_type_key(result)]

    return {
        "population": population,
        "crossing_type": crossing_type,
        "mode_i": mode_i,
        "mode_j": mode_j,
        "ell": ell,
        "f_hat": f_hat,
        "metric_key": metric_key,
        "E1": values["E1"],
        "E2": values["E2"],
        "minus": values["minus"],
        "plus": values["plus"],
        "R_max": r_max,
        "mixed_maximum": mixed_maximum,
    }


def collect_zoom_table_records(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for result in homotypic_results:
        record = zoom_table_record(
            result,
            population="homotypic",
            cutoffs=cutoffs,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        if record is not None:
            records.append(record)

    for result in heterotypic_results:
        record = zoom_table_record(
            result,
            population="heterotypic",
            cutoffs=cutoffs,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        if record is not None:
            records.append(record)

    return sorted(
        records,
        key=lambda row: (
            row["ell"],
            row["R_max"],
            row["population"],
            row["mode_i"],
            row["mode_j"],
        ),
    )


def latex_crossing_type(text: str) -> str:
    return text.replace("--", r"--")


def build_zoom_prab_longtable(
    records: list[dict[str, Any]],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    cutoffs: dict[str, float],
) -> str:
    body: list[str] = []

    for row in records:
        metric = METRIC_INFO[row["metric_key"]]
        body.append(
            " & ".join([
                latex_crossing_type(row["crossing_type"]),
                latex_mode(row["mode_i"]),
                latex_mode(row["mode_j"]),
                fmt_fixed(row["ell"], 4),
                fmt_fixed(row["f_hat"], 4),
                metric["latex"],
                fmt_sci(row["E1"]),
                fmt_sci(row["E2"]),
                fmt_sci(row["minus"]),
                fmt_sci(row["plus"]),
                fmt_fixed(row["R_max"], 3),
            ]) + r" \\" 
        )

    if not body:
        body.append(
            "-- & -- & -- & -- & -- & -- & -- & -- & -- & -- & --"
            + r" \\" 
        )

    cutoff_text = (
        rf"$K_{{\parallel}}\geq {fmt_sci(cutoffs['K_parallel']).strip('$')}$, "
        rf"$K_{{\perp}}\geq {fmt_sci(cutoffs['K_perp']).strip('$')}$, and "
        rf"$K_Q\geq {fmt_sci(cutoffs['K_Q']).strip('$')}$"
    )

    caption = (
        "Homotypic and heterotypic crossings appearing in the annotated "
        rf"$\ell$--$R_{{\max}}$ zoom, with "
        rf"${xmin:.2f}\leq\ell\leq{xmax:.2f}$ and "
        rf"${ymin:.2f}\leq R_{{\max}}\leq{ymax:.2f}$. "
        "Each crossing is evaluated using the same family-specific metric "
        "used in the combined plot. The mixed-field values must satisfy the "
        rf"configured cut-offs: {cutoff_text}."
    )

    return rf"""
\begingroup
\small
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\setlength{{\tabcolsep}}{{2.7pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{@{{}}lcccccccccc@{{}}}}
\caption{{{caption}}}
\label{{tab:zoom_crossings_ell_Rmax}}\\
\hline
Crossing type
& Mode 1
& Mode 2
& $\ell$
& $\hat{{f}}$
& Metric
& $E_1$
& $E_2$
& $E_-$
& $E_+$
& $R_{{\max}}$ \\
\hline
\endfirsthead

\multicolumn{{11}}{{c}}{{\tablename\ \thetable{{}} continued}}\\
\hline
Crossing type
& Mode 1
& Mode 2
& $\ell$
& $\hat{{f}}$
& Metric
& $E_1$
& $E_2$
& $E_-$
& $E_+$
& $R_{{\max}}$ \\
\hline
\endhead

\hline
\multicolumn{{11}}{{r}}{{Continued on next page}}\\
\endfoot

\hline
\endlastfoot

{chr(10).join(body)}
\end{{longtable}}
\renewcommand{{\arraystretch}}{{1.0}}
\endgroup
""".strip()


def write_zoom_crossings_prab_table(
    *,
    homotypic_root_or_pickle: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_root_or_pickle: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    output_tex: str | Path = DEFAULT_OUTPUT_TEX,
    metric_cutoffs: dict[str, float] | None = None,
    xmin: float = ZOOM_XMIN,
    xmax: float = ZOOM_XMAX,
    ymin: float = ZOOM_YMIN,
    ymax: float = ZOOM_YMAX,
) -> Path:
    if xmin > xmax:
        raise ValueError("xmin must not exceed xmax.")
    if ymin > ymax:
        raise ValueError("ymin must not exceed ymax.")

    cutoffs = normalised_metric_cutoffs(metric_cutoffs)
    homotypic_results = load_homotypic_results(homotypic_root_or_pickle)
    heterotypic_results = load_heterotypic_results(heterotypic_root_or_pickle)

    records = collect_zoom_table_records(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    table_text = build_zoom_prab_longtable(
        records,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        cutoffs=cutoffs,
    )

    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(table_text + "\n", encoding="utf-8")

    homotypic_count = sum(r["population"] == "homotypic" for r in records)
    heterotypic_count = sum(r["population"] == "heterotypic" for r in records)
    print(f"Wrote: {output_tex}")
    print(
        "Zoom limits: "
        f"{xmin} <= ell <= {xmax}, "
        f"{ymin} <= R_max <= {ymax}"
    )
    print(f"Crossings written: {len(records)}")
    print(f"  homotypic: {homotypic_count}")
    print(f"  heterotypic: {heterotypic_count}")
    for metric_key in ("K_parallel", "K_perp", "K_Q"):
        count = sum(r["metric_key"] == metric_key for r in records)
        print(
            f"  {metric_key}: {count}; "
            f"cut-off={cutoffs[metric_key]}"
        )

    return output_tex


if __name__ == "__main__":
    write_zoom_crossings_prab_table(
        homotypic_root_or_pickle=DEFAULT_HOMOTYPIC_ROOT,
        heterotypic_root_or_pickle=DEFAULT_HETEROTYPIC_ROOT,
        output_tex=DEFAULT_OUTPUT_TEX,
        metric_cutoffs=DEFAULT_METRIC_CUTOFFS,
        xmin=ZOOM_XMIN,
        xmax=ZOOM_XMAX,
        ymin=ZOOM_YMIN,
        ymax=ZOOM_YMAX,
    )
