from __future__ import annotations

"""
PRAB aggregate-table compiler for the unified homotypic RF/Fourier and
heterotypic Taylor/Hessian analyses.

The compiler writes two PRAB LaTeX tables:

    1. A detailed table of metric entries inside configurable ell and
       R_total zoom limits.
    2. Sample means and sample standard deviations of ell and R_total for
       homotypic, heterotypic and combined data above the metric thresholds.

The underlying analyses comprise:

Homotypic RF/Fourier
    1. monopole--monopole: K_parallel
    2. dipole--dipole:     K_perp
    3. quadrupole--quadrupole: K_Q

Heterotypic Taylor/Hessian
    4. monopole--dipole:       K_parallel and K_perp
    5. monopole--quadrupole:   K_parallel and K_Q
    6. dipole--quadrupole:     K_perp and K_Q

For every crossing, fields are ordered as

    E1, E2, E-, E+, R_total

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
import statistics
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
DEFAULT_RTOT_PLOT_ROOT = (
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
    """Return absolute metric magnitudes and both figures of merit.

    The total-ratio definition used throughout is

        R_total,K = (K_+ + K_-) / (K_1 + K_2).

    Both parent values are therefore retained for homotypic and heterotypic
    crossings, even when one parent has only a small value of the metric.
    """
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

    parent_maximum = max(values["E1"], values["E2"])
    mixed_maximum = max(values["minus"], values["plus"])
    parent_total = values["E1"] + values["E2"]
    mixed_total = values["minus"] + values["plus"]

    values["parent_maximum"] = parent_maximum
    values["mixed_maximum"] = mixed_maximum
    values["parent_total"] = parent_total
    values["mixed_total"] = mixed_total
    values["R_max"] = safe_ratio(mixed_maximum, parent_maximum)
    values["R_total"] = safe_ratio(mixed_total, parent_total)
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

DEFAULT_OUTPUT_TEX = (
    DEFAULT_PRAB_ROOT
    / "zoom_crossings_and_Rtotal_summary_statistics_PRAB_tables.tex"
)

# These limits should match the ell-versus-R_total zoom figure.
ZOOM_ELL_MIN = 0.9
ZOOM_ELL_MAX = 1.1
ZOOM_RTOTAL_MIN = 1.0
ZOOM_RTOTAL_MAX = 5.0

# A row is retained only when max(K_minus, K_plus) is at least the cut-off
# for that row's metric. Set a cut-off to 0.0 to retain every finite result.
DEFAULT_METRIC_CUTOFFS: dict[str, float] = {
    "K_parallel": 0.0,
    "K_perp": 0.0,
    "K_Q": 0.0,
}

HOMOTYPIC_TYPE_LABELS = {
    0: "M-M",
    1: "D-D",
    2: "Q-Q",
}

HETEROTYPIC_TYPE_LABELS = {
    "monopole_dipole": "M-D",
    "monopole_quadrupole": "M-Q",
    "dipole_quadrupole": "D-Q",
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


def metric_keys_for_table(
    result: dict[str, Any],
    *,
    population: str,
) -> tuple[str, ...]:
    """Return one homotypic metric or both heterotypic metrics.

    Thus every homotypic crossing contributes at most one row, whereas every
    heterotypic crossing can contribute up to two rows to the R_total zoom
    table, matching the plotting script.
    """
    if population == "homotypic":
        return (homotypic_metric_key(result),)
    if population == "heterotypic":
        return heterotypic_metric_keys(pair_type_key(result))
    raise ValueError(f"Unsupported population: {population!r}")


def zoom_table_record(
    result: dict[str, Any],
    *,
    population: str,
    metric_key: str,
    cutoffs: dict[str, float],
    ell_min: float,
    ell_max: float,
    rtotal_min: float,
    rtotal_max: float,
) -> dict[str, Any] | None:
    """Build one metric-specific row when it lies inside the zoom bounds."""
    values = field_metric_values(result, metric_key)
    ell, f_hat = crossing_parameters(result)
    r_total = finite_or_nan(values["R_total"])
    mixed_maximum = finite_or_nan(values["mixed_maximum"])

    if (
        not math.isfinite(ell)
        or not math.isfinite(f_hat)
        or not math.isfinite(r_total)
        or not math.isfinite(mixed_maximum)
        or mixed_maximum < cutoffs[metric_key]
        or not (ell_min <= ell <= ell_max)
        or not (rtotal_min <= r_total <= rtotal_max)
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
        "K1": values["E1"],
        "K2": values["E2"],
        "K_minus": values["minus"],
        "K_plus": values["plus"],
        "parent_total": values["parent_total"],
        "mixed_total": values["mixed_total"],
        "R_total": r_total,
    }


def collect_zoom_table_records(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
    ell_min: float,
    ell_max: float,
    rtotal_min: float,
    rtotal_max: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for population, results in (
        ("homotypic", homotypic_results),
        ("heterotypic", heterotypic_results),
    ):
        for result in results:
            for metric_key in metric_keys_for_table(
                result,
                population=population,
            ):
                record = zoom_table_record(
                    result,
                    population=population,
                    metric_key=metric_key,
                    cutoffs=cutoffs,
                    ell_min=ell_min,
                    ell_max=ell_max,
                    rtotal_min=rtotal_min,
                    rtotal_max=rtotal_max,
                )
                if record is not None:
                    records.append(record)

    return sorted(
        records,
        key=lambda row: (
            row["ell"],
            row["R_total"],
            row["population"],
            row["crossing_type"],
            row["metric_key"],
            row["mode_i"],
            row["mode_j"],
        ),
    )


def latex_crossing_type(text: str) -> str:
    return text.replace("--", r"--")


def build_zoom_prab_longtable(
    records: list[dict[str, Any]],
    *,
    ell_min: float,
    ell_max: float,
    rtotal_min: float,
    rtotal_max: float,
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
                fmt_sci(row["parent_total"]),
                fmt_sci(row["mixed_total"]),
                fmt_fixed(row["R_total"], 3),
            ]) + r" \\" 
        )

    if not body:
        body.append(
            "-- & -- & -- & -- & -- & -- & -- & -- & --"
            + r" \\" 
        )

    cutoff_text = (
        rf"$K_{{\parallel}}\geq {fmt_sci(cutoffs['K_parallel']).strip('$')}$, "
        rf"$K_{{\perp}}\geq {fmt_sci(cutoffs['K_perp']).strip('$')}$, and "
        rf"$K_Q\geq {fmt_sci(cutoffs['K_Q']).strip('$')}$"
    )

    caption = (
        "Homotypic and heterotypic metric entries appearing in the annotated "
        rf"$\ell$--$R_{{\mathrm{{total}}}}$ zoom, with "
        rf"${ell_min:.2f}\leq\ell\leq{ell_max:.2f}$ and "
        rf"${rtotal_min:.2f}\leq R_{{\mathrm{{total}}}}\leq{rtotal_max:.2f}$. "
        "Each homotypic crossing contributes its family-specific metric. Each "
        "heterotypic crossing contributes a separate row for both applicable "
        "metrics. In every row, "
        rf"$R_{{\mathrm{{total}},K}}=(K_+ + K_-)/(K_1 + K_2)$, so both parent "
        "modes are retained. The mixed-field values must satisfy the configured "
        rf"cut-offs: {cutoff_text}."
    )

    return rf"""
\begingroup
\small
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\setlength{{\tabcolsep}}{{2.7pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{@{{}}lcccccccc@{{}}}}
\caption{{{caption}}}
\label{{tab:zoom_crossings_ell_Rtotal}}\\
\hline
Crossing type
& Mode 1
& Mode 2
& $\ell$
& $\hat{{f}}$
& Metric
& $K_1+K_2$
& $K_+ + K_-$
& $R_{{\mathrm{{total}}}}$ \\
\hline
\endfirsthead

\multicolumn{{9}}{{c}}{{\tablename\ \thetable{{}} continued}}\\
\hline
Crossing type
& Mode 1
& Mode 2
& $\ell$
& $\hat{{f}}$
& Metric
& $K_1+K_2$
& $K_+ + K_-$
& $R_{{\mathrm{{total}}}}$ \\
\hline
\endhead

\hline
\multicolumn{{9}}{{r}}{{Continued on next page}}\\
\endfoot

\hline
\endlastfoot

{chr(10).join(body)}
\end{{longtable}}
\renewcommand{{\arraystretch}}{{1.0}}
\endgroup
""".strip()



# -----------------------------------------------------------------------------
# All-data summary statistics above the configured metric thresholds
# -----------------------------------------------------------------------------

def threshold_table_record(
    result: dict[str, Any],
    *,
    population: str,
    metric_key: str,
    cutoffs: dict[str, float],
) -> dict[str, Any] | None:
    """Return one finite metric entry that passes its mixed-field threshold.

    No zoom limits are applied here. The same R_total definition and metric
    selection used by the zoom table are retained:

        R_total,K = (K_+ + K_-) / (K_1 + K_2).

    A homotypic crossing contributes one family-specific entry. A heterotypic
    crossing contributes one entry for each of its two applicable metrics.
    """
    values = field_metric_values(result, metric_key)
    ell, _ = crossing_parameters(result)
    r_total = finite_or_nan(values["R_total"])
    mixed_maximum = finite_or_nan(values["mixed_maximum"])

    if (
        not math.isfinite(ell)
        or not math.isfinite(r_total)
        or not math.isfinite(mixed_maximum)
        or mixed_maximum < cutoffs[metric_key]
    ):
        return None

    return {
        "population": population,
        "metric_key": metric_key,
        "ell": ell,
        "R_total": r_total,
    }


def collect_threshold_statistics_records(
    *,
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    cutoffs: dict[str, float],
) -> list[dict[str, Any]]:
    """Collect all finite metric entries above threshold, without zoom cuts."""
    records: list[dict[str, Any]] = []

    for population, results in (
        ("homotypic", homotypic_results),
        ("heterotypic", heterotypic_results),
    ):
        for result in results:
            for metric_key in metric_keys_for_table(
                result,
                population=population,
            ):
                record = threshold_table_record(
                    result,
                    population=population,
                    metric_key=metric_key,
                    cutoffs=cutoffs,
                )
                if record is not None:
                    records.append(record)

    return records


def sample_standard_deviation(values: list[float]) -> float:
    """Return the sample standard deviation, or NaN when fewer than 2 values."""
    finite_values = [value for value in values if math.isfinite(value)]
    if len(finite_values) < 2:
        return float("nan")
    return statistics.stdev(finite_values)


def population_summary_row(
    records: list[dict[str, Any]],
    *,
    population: str | None,
    label: str,
) -> dict[str, Any]:
    selected = (
        records
        if population is None
        else [row for row in records if row["population"] == population]
    )
    ell_values = [finite_or_nan(row["ell"]) for row in selected]
    rtotal_values = [finite_or_nan(row["R_total"]) for row in selected]
    ell_values = [value for value in ell_values if math.isfinite(value)]
    rtotal_values = [value for value in rtotal_values if math.isfinite(value)]

    # Each retained record has both finite ell and R_total, so these counts
    # should be equal. Use the record count explicitly for clarity.
    return {
        "label": label,
        "N": len(selected),
        "ell_mean": statistics.fmean(ell_values) if ell_values else float("nan"),
        "ell_sd": sample_standard_deviation(ell_values),
        "R_total_mean": (
            statistics.fmean(rtotal_values) if rtotal_values else float("nan")
        ),
        "R_total_sd": sample_standard_deviation(rtotal_values),
    }


def build_summary_statistics_table(
    records: list[dict[str, Any]],
    *,
    cutoffs: dict[str, float],
) -> str:
    """Build a PRAB-style table of all-data means and sample SDs."""
    summaries = [
        population_summary_row(
            records, population="homotypic", label="Homotypic"
        ),
        population_summary_row(
            records, population="heterotypic", label="Heterotypic"
        ),
        population_summary_row(
            records, population=None, label="Combined"
        ),
    ]

    body = "\n".join(
        " & ".join([
            row["label"],
            str(row["N"]),
            fmt_fixed(row["ell_mean"], 4),
            fmt_fixed(row["ell_sd"], 4),
            fmt_fixed(row["R_total_mean"], 3),
            fmt_fixed(row["R_total_sd"], 3),
        ]) + r" \\"
        for row in summaries
    )

    cutoff_text = (
        rf"$K_{{\parallel}}\geq {fmt_sci(cutoffs['K_parallel']).strip('$')}$, "
        rf"$K_{{\perp}}\geq {fmt_sci(cutoffs['K_perp']).strip('$')}$, and "
        rf"$K_Q\geq {fmt_sci(cutoffs['K_Q']).strip('$')}$"
    )

    caption = (
        r"Summary statistics for all analysed metric entries satisfying "
        r"the configured mixed-field thresholds. Here, $N$ is the number "
        r"of retained entries, $\bar{x}$ denotes the sample mean, and "
        r"$s_x$ denotes the sample standard deviation. Homotypic crossings "
        r"contribute one family-specific metric entry, whereas heterotypic "
        r"crossings contribute one entry for each applicable metric. In all "
        r"cases, $R_{\mathrm{total},K}=(K_+ + K_-)/(K_1 + K_2)$. "
        rf"The thresholds are {cutoff_text}."
    )

    return rf"""
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\caption{{{caption}}}
\label{{tab:Rtotal_ell_summary_statistics}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\hline
Population
& $N$
& $\bar{{\ell}}$
& $s_{{\ell}}$
& $\overline{{R}}_{{\mathrm{{total}}}}$
& $s_{{R_{{\mathrm{{total}}}}}}$ \\
\hline
{body}
\hline
\end{{tabular}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{table}}
""".strip()

def write_zoom_crossings_prab_table(
    *,
    homotypic_root_or_pickle: str | Path = DEFAULT_HOMOTYPIC_ROOT,
    heterotypic_root_or_pickle: str | Path = DEFAULT_HETEROTYPIC_ROOT,
    output_tex: str | Path = DEFAULT_OUTPUT_TEX,
    metric_cutoffs: dict[str, float] | None = None,
    ell_min: float = ZOOM_ELL_MIN,
    ell_max: float = ZOOM_ELL_MAX,
    rtotal_min: float = ZOOM_RTOTAL_MIN,
    rtotal_max: float = ZOOM_RTOTAL_MAX,
) -> Path:
    if ell_min > ell_max:
        raise ValueError("ell_min must not exceed ell_max.")
    if rtotal_min > rtotal_max:
        raise ValueError("rtotal_min must not exceed rtotal_max.")
    if rtotal_min < 0.0:
        raise ValueError("rtotal_min must be non-negative.")

    cutoffs = normalised_metric_cutoffs(metric_cutoffs)
    homotypic_results = load_homotypic_results(homotypic_root_or_pickle)
    heterotypic_results = load_heterotypic_results(heterotypic_root_or_pickle)

    records = collect_zoom_table_records(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
        ell_min=ell_min,
        ell_max=ell_max,
        rtotal_min=rtotal_min,
        rtotal_max=rtotal_max,
    )

    zoom_table_text = build_zoom_prab_longtable(
        records,
        ell_min=ell_min,
        ell_max=ell_max,
        rtotal_min=rtotal_min,
        rtotal_max=rtotal_max,
        cutoffs=cutoffs,
    )

    statistics_records = collect_threshold_statistics_records(
        homotypic_results=homotypic_results,
        heterotypic_results=heterotypic_results,
        cutoffs=cutoffs,
    )
    statistics_table_text = build_summary_statistics_table(
        statistics_records,
        cutoffs=cutoffs,
    )

    table_text = zoom_table_text + "\n\n\\clearpage\n\n" + statistics_table_text

    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(table_text + "\n", encoding="utf-8")

    homotypic_count = sum(r["population"] == "homotypic" for r in records)
    heterotypic_count = sum(r["population"] == "heterotypic" for r in records)
    unique_heterotypic = {
        (r["mode_i"], r["mode_j"], r["ell"], r["f_hat"])
        for r in records
        if r["population"] == "heterotypic"
    }

    print(f"Wrote: {output_tex}")
    print(
        "Zoom limits: "
        f"{ell_min} <= ell <= {ell_max}, "
        f"{rtotal_min} <= R_total <= {rtotal_max}"
    )
    print(f"Table rows written: {len(records)}")
    print(f"  homotypic metric rows: {homotypic_count}")
    print(f"  heterotypic metric rows: {heterotypic_count}")
    print(f"  unique heterotypic crossings represented: {len(unique_heterotypic)}")
    print("All-data statistics entries above thresholds:")
    print(
        "  homotypic: "
        f"{sum(r['population'] == 'homotypic' for r in statistics_records)}"
    )
    print(
        "  heterotypic: "
        f"{sum(r['population'] == 'heterotypic' for r in statistics_records)}"
    )
    print(f"  combined: {len(statistics_records)}")
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
        ell_min=ZOOM_ELL_MIN,
        ell_max=ZOOM_ELL_MAX,
        rtotal_min=ZOOM_RTOTAL_MIN,
        rtotal_max=ZOOM_RTOTAL_MAX,
    )
