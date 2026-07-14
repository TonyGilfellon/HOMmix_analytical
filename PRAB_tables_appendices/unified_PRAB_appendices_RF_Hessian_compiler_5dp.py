from __future__ import annotations

"""
Unified PRAB appendix LaTeX compiler.

Appendix I
    Comparison of homotypic RF/Fourier multipole results with heterotypic
    Taylor/Hessian results.  Only like physical families are compared:
        m=0 -> K_parallel
        m=1 -> K_perp
        m=2 -> K_Q
    and only modes present in both analyses are retained.

Appendix II
    Homotypic RF/Fourier mixing pages from the unified homotypic analysis.

Appendix III
    Heterotypic Taylor/Hessian mixing pages from the unified heterotypic
    Hessian analysis.

The table order is deliberately
    E1, E2, E-, E+, R_max
to match the field-summary PDF column order requested for the manuscript.

The compiler expects the agreed explicit figure-of-merit keys:
    k_parallel_V_per_pC
    K_parallel_V_per_pC_per_m
    k_perp_V_per_pC_per_m
    K_perp_V_per_pC_per_m2
    k_Q_V_per_pC_per_m2
    K_Q_V_per_pC_per_m3

Legacy aliases are accepted only as fallbacks.
"""

import math
import pickle
import re
import shutil
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


# -----------------------------------------------------------------------------
# Paths
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

DEFAULT_HOMOTYPIC_AGGREGATE = (
    DEFAULT_HOMOTYPIC_ROOT / "all_homotypic_rf_multipole_analyses.pkl"
)
DEFAULT_HETEROTYPIC_HESSIAN_AGGREGATE = (
    DEFAULT_HETEROTYPIC_ROOT / "all_heterotypic_multipole_analyses.pkl"
)

DEFAULT_PRAB_ROOT = Path(r"D:\PhD\PRAB")
DEFAULT_FIGS_DIR = DEFAULT_PRAB_ROOT / "figs"
DEFAULT_OUT_TEX = DEFAULT_PRAB_ROOT / "appendices_I_II_III_unified.tex"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def finite_or_nan(value: object) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def abs_finite_or_nan(value: object) -> float:
    try:
        x = float(abs(complex(value)))
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
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


def representative(values: list[float]) -> float:
    finite = [
        finite_or_nan(v)
        for v in values
        if math.isfinite(finite_or_nan(v))
    ]
    return float(median(finite)) if finite else float("nan")


def relative_spread(values: list[float]) -> float:
    finite = [
        abs(finite_or_nan(v))
        for v in values
        if math.isfinite(finite_or_nan(v))
    ]
    if len(finite) < 2:
        return 0.0
    reference = max(float(median(finite)), 1.0e-300)
    return float((max(finite) - min(finite)) / reference)


def percentage_difference(first: float, second: float) -> float:
    """Return 100*(first-second)/second."""
    first = finite_or_nan(first)
    second = finite_or_nan(second)
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or second == 0.0
    ):
        return float("nan")
    return 100.0 * (first - second) / second


def fmt_sci(value: object, significant_figures: int = 5) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    mantissa, exponent = (
        f"{value:.{significant_figures - 1}e}".split("e")
    )
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def fmt_float(value: object, decimal_places: int = 5) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{decimal_places}f}"


def fmt_percentage(value: object, decimal_places: int = 5) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "--"
    return rf"${value:.{decimal_places}f}$"


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


def latex_mode(mode: object, *, include_math: bool = True) -> str:
    normalised = normalise_mode_name(mode)
    if "_" in normalised:
        family, indices = normalised.split("_", 1)
        value = rf"\mathrm{{{family}_{{{indices}}}}}"
    else:
        value = normalised.replace("_", r"\_")
    return f"${value}$" if include_math else value


def mode_azimuthal_index(mode: object) -> int | None:
    normalised = normalise_mode_name(mode)
    match = re.search(r"_(\d)", normalised)
    return int(match.group(1)) if match else None


def latex_label_safe(value: object) -> str:
    text = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(value),
    ).strip("_")
    return text or "entry"


# -----------------------------------------------------------------------------
# Loading result trees
# -----------------------------------------------------------------------------

def _flatten_result_container(data: Any) -> list[dict[str, Any]]:
    """
    Flatten aggregate formats such as:
        {crossing_key: result}
        {0: {crossing_key: result}, 1: {...}, 2: {...}}
        [result, ...]
    """
    out: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "fields" in value and "crossing" in value:
                out.append(value)
                return
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(data)
    return out


def load_result_items(
    path_or_root: str | Path,
    *,
    aggregate_filename: str,
    per_folder_filename: str,
) -> list[dict[str, Any]]:
    path = Path(path_or_root)

    if path.is_file():
        return _flatten_result_container(pickle_load(path))

    aggregate = path / aggregate_filename
    if aggregate.exists():
        return _flatten_result_container(pickle_load(aggregate))

    per_folder = sorted(path.rglob(per_folder_filename))
    if not per_folder:
        raise FileNotFoundError(
            f"No {aggregate_filename!r} or "
            f"{per_folder_filename!r} files found below {path}."
        )

    items: list[dict[str, Any]] = []
    for filename in per_folder:
        items.extend(_flatten_result_container(pickle_load(filename)))
    return items


def load_homotypic_results(
    root_or_pkl: str | Path = DEFAULT_HOMOTYPIC_ROOT,
) -> list[dict[str, Any]]:
    return load_result_items(
        root_or_pkl,
        aggregate_filename=(
            "all_homotypic_rf_multipole_analyses.pkl"
        ),
        per_folder_filename=(
            "homotypic_rf_multipole_analysis.pkl"
        ),
    )


def load_heterotypic_hessian_results(
    root_or_pkl: str | Path = DEFAULT_HETEROTYPIC_ROOT,
) -> list[dict[str, Any]]:
    return load_result_items(
        root_or_pkl,
        aggregate_filename=(
            "all_heterotypic_multipole_analyses.pkl"
        ),
        per_folder_filename=(
            "heterotypic_multipole_analysis.pkl"
        ),
    )


# -----------------------------------------------------------------------------
# Agreed metric extraction
# -----------------------------------------------------------------------------

METRIC_INFO: dict[str, dict[str, Any]] = {
    "K_parallel": {
        "latex": r"$K_{\parallel}$",
        "units": r"$\mathrm{V/pC/m_z}$",
        "required_m": 0,
        "explicit_key": "K_parallel_V_per_pC_per_m",
        "legacy_keys": (
            "loss_like_V_per_pC_per_m",
        ),
    },
    "K_perp": {
        "latex": r"$K_{\perp}$",
        "units": r"$\mathrm{V/pC/m_{\perp}/m_z}$",
        "required_m": 1,
        "explicit_key": "K_perp_V_per_pC_per_m2",
        "legacy_keys": (
            "kick_magnitude_V_per_pC_per_m2",
            "kick_mag_V_per_pC_per_m2",
        ),
    },
    "K_Q": {
        "latex": r"$K_Q$",
        "units": r"$\mathrm{V/pC/m_{\perp}^{2}/m_z}$",
        "required_m": 2,
        "explicit_key": "K_Q_V_per_pC_per_m3",
        "legacy_keys": (
            "KQ_V_per_pC_per_m3",
        ),
    },
}


def figures(field_result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(field_result, dict):
        return {}
    return field_result.get("figures_of_merit", {})


def metric_value(
    field_result: dict[str, Any],
    metric_key: str,
) -> float:
    info = METRIC_INFO[metric_key]
    fom = figures(field_result)

    value = abs_finite_or_nan(
        first_present(
            fom,
            (
                info["explicit_key"],
                *info["legacy_keys"],
            ),
        )
    )
    if math.isfinite(value):
        return value

    # K_parallel diagnostic fallback for older Hessian outputs.
    if metric_key == "K_parallel":
        diagnostics = field_result.get(
            "kparallel_diagnostics",
            {},
        )
        value = abs_finite_or_nan(
            diagnostics.get(
                "fit_V0_U_CST",
                {},
            ).get(
                "k_V_per_pC_per_m",
                float("nan"),
            )
        )
        if math.isfinite(value):
            return value

    # Reconstruct K_Q from the reported Hessian matrix if needed.
    if metric_key == "K_Q":
        Kxx = finite_or_nan(
            first_present(
                fom,
                (
                    "K_xx_V_per_pC_per_m3",
                    "Kxx_V_per_pC_per_m3",
                ),
            )
        )
        Kxy = finite_or_nan(
            first_present(
                fom,
                (
                    "K_xy_V_per_pC_per_m3",
                    "Kxy_V_per_pC_per_m3",
                ),
            )
        )
        Kyy = finite_or_nan(
            first_present(
                fom,
                (
                    "K_yy_V_per_pC_per_m3",
                    "Kyy_V_per_pC_per_m3",
                ),
            )
        )
        if all(math.isfinite(v) for v in (Kxx, Kxy, Kyy)):
            return math.sqrt(
                (Kxx - Kyy) ** 2
                + 4.0 * Kxy ** 2
            )

    return float("nan")


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


def field_values(
    result: dict[str, Any],
    metric_key: str,
) -> dict[str, float]:
    fields = result.get("fields", {})
    values = {
        name: metric_value(
            fields.get(name, {}),
            metric_key,
        )
        for name in ("E1", "E2", "plus", "minus")
    }

    parent_max = max(
        abs(values["E1"]),
        abs(values["E2"]),
    )
    mixed_max = max(
        abs(values["plus"]),
        abs(values["minus"]),
    )

    values["R_max"] = safe_ratio(
        mixed_max,
        parent_max,
    )
    return values


def relevant_metric_for_homotypic_result(
    result: dict[str, Any],
) -> str:
    family_m = result.get("family_m")
    if family_m is None:
        mode_i, _ = result_modes(result)
        family_m = mode_azimuthal_index(mode_i)

    mapping = {
        0: "K_parallel",
        1: "K_perp",
        2: "K_Q",
    }
    if family_m not in mapping:
        raise ValueError(
            f"Could not determine m=0,1,2 family for "
            f"{result_modes(result)}."
        )
    return mapping[int(family_m)]


def pair_type_key(result: dict[str, Any]) -> str:
    crossing = result.get("crossing", {})
    pair_type = (
        result.get("pair_type")
        or crossing.get("pair_type")
    )
    if pair_type is None and result.get("crossing_folder"):
        pair_type = Path(
            result["crossing_folder"]
        ).parent.name
    return (
        str(pair_type or "heterotypic")
        .lower()
        .replace("-", "_")
    )


def relevant_metrics_for_heterotypic_result(
    result: dict[str, Any],
) -> list[str]:
    pair_type = pair_type_key(result)

    exact = {
        "monopole_dipole": [
            "K_parallel",
            "K_perp",
        ],
        "monopole_quadrupole": [
            "K_parallel",
            "K_Q",
        ],
        "dipole_quadrupole": [
            "K_perp",
            "K_Q",
        ],
    }
    if pair_type in exact:
        return exact[pair_type]

    mode_i, mode_j = result_modes(result)
    m_values = {
        mode_azimuthal_index(mode_i),
        mode_azimuthal_index(mode_j),
    }
    metrics: list[str] = []
    if 0 in m_values:
        metrics.append("K_parallel")
    if 1 in m_values:
        metrics.append("K_perp")
    if 2 in m_values:
        metrics.append("K_Q")
    return metrics or [
        "K_parallel",
        "K_perp",
        "K_Q",
    ]


# -----------------------------------------------------------------------------
# Appendix I: RF/Fourier versus Taylor/Hessian comparison
# -----------------------------------------------------------------------------

def collect_parent_mode_values(
    results: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, dict[str, list[float]]]:
    out = {
        "K_parallel": {},
        "K_perp": {},
        "K_Q": {},
    }

    for result in results:
        mode_i, mode_j = result_modes(result)
        fields = result.get("fields", {})

        for field_name, mode in (
            ("E1", mode_i),
            ("E2", mode_j),
        ):
            if not mode:
                continue

            mode_m = mode_azimuthal_index(mode)
            metric_key = {
                0: "K_parallel",
                1: "K_perp",
                2: "K_Q",
            }.get(mode_m)
            if metric_key is None:
                continue

            value = metric_value(
                fields.get(field_name, {}),
                metric_key,
            )
            if math.isfinite(value):
                out[metric_key].setdefault(
                    mode,
                    [],
                ).append(abs(value))

    print(
        f"Collected parent-mode values from {method}: "
        + ", ".join(
            f"{key}={len(values)} modes"
            for key, values in out.items()
        )
    )
    return out


def build_comparison_rows(
    rf_values: dict[str, list[float]],
    hessian_values: dict[str, list[float]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, dict[str, float]],
]:
    rf_modes = set(rf_values)
    hessian_modes = set(hessian_values)
    common_modes = sorted(rf_modes & hessian_modes)

    rows: list[dict[str, Any]] = []
    spreads: dict[str, dict[str, float]] = {}

    for mode in common_modes:
        rf_value = representative(rf_values[mode])
        hessian_value = representative(
            hessian_values[mode]
        )
        if not (
            math.isfinite(rf_value)
            and math.isfinite(hessian_value)
        ):
            continue

        rows.append({
            "mode": mode,
            "rf": rf_value,
            "hessian": hessian_value,
            "percentage_difference": (
                percentage_difference(
                    rf_value,
                    hessian_value,
                )
            ),
            "n_rf": len(rf_values[mode]),
            "n_hessian": len(
                hessian_values[mode]
            ),
        })
        spreads[mode] = {
            "rf_relative_spread": relative_spread(
                rf_values[mode]
            ),
            "hessian_relative_spread": relative_spread(
                hessian_values[mode]
            ),
        }

    omitted = {
        "only_in_homotypic_RF": sorted(
            rf_modes - hessian_modes
        ),
        "only_in_heterotypic_Hessian": sorted(
            hessian_modes - rf_modes
        ),
    }
    return rows, omitted, spreads


def comparison_table(
    metric_key: str,
    rows: list[dict[str, Any]],
) -> str:
    info = METRIC_INFO[metric_key]
    body: list[str] = []

    for row in rows:
        body.append(
            " & ".join([
                latex_mode(row["mode"]),
                fmt_sci(row["rf"]),
                fmt_sci(row["hessian"]),
                fmt_percentage(row["percentage_difference"], 5),
            ])
            + r" \\"
        )

    if not body:
        body.append(
            "-- & -- & -- & --" + r" \\"
        )

    captions = {
        "K_parallel": (
            r"Comparison of the length-normalised longitudinal "
            r"metric $K_{\parallel}$ obtained from the "
            r"homotypic RF/Fourier analysis and from the "
            r"heterotypic Taylor/Hessian analysis for "
            r"monopole modes present in both datasets."
        ),
        "K_perp": (
            r"Comparison of the length-normalised transverse "
            r"metric $K_{\perp}$ obtained from the "
            r"homotypic RF/Fourier analysis and from the "
            r"heterotypic Taylor/Hessian analysis for "
            r"dipole modes present in both datasets."
        ),
        "K_Q": (
            r"Comparison of the length-normalised quadrupole "
            r"metric $K_Q$ obtained from the homotypic "
            r"RF/Fourier analysis and from the heterotypic "
            r"Taylor/Hessian analysis for quadrupole modes "
            r"present in both datasets."
        ),
    }

    symbol = {
        "K_parallel": r"K_{\parallel}",
        "K_perp": r"K_{\perp}",
        "K_Q": r"K_Q",
    }[metric_key]

    return rf"""
\begin{{table}}[htbp]
\caption{{{captions[metric_key]}}}
\label{{tab:{metric_key.lower()}_rf_hessian_comparison}}
\begin{{ruledtabular}}
\begin{{tabular}}{{cccc}}
Mode
& ${symbol}^{{\mathrm{{RF}}}}$
& ${symbol}^{{\mathrm{{Hessian}}}}$
& $\Delta$ [\%] \\
\hline
{chr(10).join(body)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table}}
""".strip()


def appendix_i_comparison(
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, dict[str, float]]],
]:
    rf = collect_parent_mode_values(
        homotypic_results,
        method="homotypic RF/Fourier",
    )
    hessian = collect_parent_mode_values(
        heterotypic_results,
        method="heterotypic Taylor/Hessian",
    )

    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_omitted: dict[
        str,
        dict[str, list[str]],
    ] = {}
    all_spreads: dict[
        str,
        dict[str, dict[str, float]],
    ] = {}

    for metric_key in (
        "K_parallel",
        "K_perp",
        "K_Q",
    ):
        rows, omitted, spreads = build_comparison_rows(
            rf[metric_key],
            hessian[metric_key],
        )
        required_m = METRIC_INFO[
            metric_key
        ]["required_m"]
        rows = [
            row
            for row in rows
            if mode_azimuthal_index(
                row["mode"]
            ) == required_m
        ]
        all_rows[metric_key] = rows
        all_omitted[metric_key] = omitted
        all_spreads[metric_key] = spreads

    omitted_comments = [
        "",
        "% Modes omitted from Appendix I comparisons:",
    ]
    for metric_key, omitted in all_omitted.items():
        omitted_comments.append(f"% {metric_key}:")
        for category, modes in omitted.items():
            omitted_comments.append(
                f"%   {category}: "
                + (
                    ", ".join(modes)
                    if modes
                    else "none"
                )
            )

    tex = "\n\n".join([
        r"""
\clearpage
\appendix
\section{RF-multipole and Taylor--Hessian comparison}
\label{app:rf_hessian_comparison}

The tables compare the parent-mode beam-dynamics metrics obtained from the
homotypic RF/Fourier multipole analysis with those obtained independently
from the heterotypic Taylor/Hessian analysis.  Only modes represented in both
datasets are included.  Monopole modes are compared using
$K_{\parallel}$, dipole modes using $K_{\perp}$ and quadrupole modes using
$K_Q$.  The percentage difference is
$100(K_{\mathrm{RF}}-K_{\mathrm{Hessian}})/K_{\mathrm{Hessian}}$.
""".strip(),
        comparison_table(
            "K_parallel",
            all_rows["K_parallel"],
        ),
        comparison_table(
            "K_perp",
            all_rows["K_perp"],
        ),
        comparison_table(
            "K_Q",
            all_rows["K_Q"],
        ),
        "\n".join(omitted_comments),
        r"% End of Appendix I",
    ])
    return tex, all_omitted, all_spreads


# -----------------------------------------------------------------------------
# Figure path and copy helpers
# -----------------------------------------------------------------------------

def homotypic_class_key(
    result: dict[str, Any],
) -> str:
    family_m = result.get("family_m")
    if family_m is None:
        mode_i, _ = result_modes(result)
        family_m = mode_azimuthal_index(mode_i)
    return {
        0: "monopole",
        1: "dipole",
        2: "quadrupole",
    }[int(family_m)]


def result_crossing_folder(
    result: dict[str, Any],
) -> Path | None:
    if result.get("crossing_folder"):
        return Path(result["crossing_folder"])

    # The unified homotypic aggregate currently does not store crossing_folder.
    # Reconstruct it from the result's modes and family root when possible.
    return None


def homotypic_prab_pdf_name(
    result: dict[str, Any],
) -> str:
    mode_i, mode_j = result_modes(result)
    return (
        f"homotypic_{homotypic_class_key(result)}_"
        f"{mode_i.replace('_', '')}_"
        f"{mode_j.replace('_', '')}_"
        f"field_summary.pdf"
    )


def homotypic_latex_pdf_path(
    result: dict[str, Any],
    figs_dir_latex: str,
) -> str:
    return (
        f"{figs_dir_latex}/"
        f"{homotypic_prab_pdf_name(result)}"
    )


def _homotypic_possible_folders(
    result: dict[str, Any],
    homotypic_root: Path,
) -> list[Path]:
    mode_i, mode_j = result_modes(result)
    family_dir = {
        0: "monopole_monopole",
        1: "dipole_dipole",
        2: "quadrupole_quadrupole",
    }[int(
        result.get(
            "family_m",
            mode_azimuthal_index(mode_i),
        )
    )]

    names = [
        f"{mode_i}_{mode_j}".replace(
            "TM_",
            "TM",
        ),
        f"{mode_i.replace('_', '')}_"
        f"{mode_j.replace('_', '')}",
        f"{mode_i}_{mode_j}",
    ]
    return [
        homotypic_root / family_dir / name
        for name in names
    ]


def find_homotypic_summary_pdf(
    result: dict[str, Any],
    homotypic_root: str | Path,
) -> Path | None:
    root = Path(homotypic_root)
    expected_name = homotypic_prab_pdf_name(
        result
    )

    candidates: list[Path] = []
    for folder in _homotypic_possible_folders(
        result,
        root,
    ):
        candidates.extend([
            folder
            / "slice_summary_pdfs"
            / f"{folder.name}_field_summary.pdf",
            folder
            / "slice_summary_pdfs"
            / expected_name,
            folder / expected_name,
        ])
        candidates.extend(
            sorted(
                (
                    folder
                    / "slice_summary_pdfs"
                ).glob("*field_summary.pdf")
            )
            if (
                folder
                / "slice_summary_pdfs"
            ).exists()
            else []
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def heterotypic_prab_pdf_name(
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

    mode_i, mode_j = result_modes(result)
    return (
        f"heterotypic_{pair_type}_"
        f"{mode_i}__{mode_j}__"
        f"field_summary.pdf"
    )


def heterotypic_latex_pdf_path(
    result: dict[str, Any],
    figs_dir_latex: str,
) -> str:
    return (
        f"{figs_dir_latex}/"
        f"{heterotypic_prab_pdf_name(result)}"
    )


def find_heterotypic_summary_pdf(
    result: dict[str, Any],
) -> Path | None:
    if not result.get("crossing_folder"):
        return None
    folder = Path(result["crossing_folder"])
    candidate = (
        folder
        / "slice_summary_pdfs"
        / f"{folder.name}_field_summary.pdf"
    )
    if candidate.exists():
        return candidate

    alternatives = sorted(
        (
            folder / "slice_summary_pdfs"
        ).glob("*field_summary.pdf")
    ) if (
        folder / "slice_summary_pdfs"
    ).exists() else []
    return alternatives[0] if alternatives else None


def copy_summary_pdfs(
    homotypic_results: list[dict[str, Any]],
    heterotypic_results: list[dict[str, Any]],
    *,
    homotypic_root: str | Path,
    destination: str | Path,
    overwrite: bool = True,
) -> None:
    destination = Path(destination)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing: list[str] = []

    for result in homotypic_results:
        source = find_homotypic_summary_pdf(
            result,
            homotypic_root,
        )
        if source is None:
            missing.append(
                "homotypic: "
                + " -- ".join(result_modes(result))
            )
            continue
        target = (
            destination
            / homotypic_prab_pdf_name(result)
        )
        if overwrite or not target.exists():
            shutil.copy2(source, target)

    for result in heterotypic_results:
        source = find_heterotypic_summary_pdf(
            result
        )
        if source is None:
            missing.append(
                "heterotypic: "
                + " -- ".join(result_modes(result))
            )
            continue
        target = (
            destination
            / heterotypic_prab_pdf_name(result)
        )
        if overwrite or not target.exists():
            shutil.copy2(source, target)

    if missing:
        print(
            "\nSummary PDFs not found for:"
        )
        for entry in missing:
            print(f"  {entry}")


# -----------------------------------------------------------------------------
# Appendix II: homotypic mixing
# -----------------------------------------------------------------------------

def crossing_parameters(
    result: dict[str, Any],
) -> tuple[float, float]:
    crossing = result.get("crossing", {})
    ell = finite_or_nan(
        crossing.get(
            "length_factor",
            float("nan"),
        )
    )
    fhat = finite_or_nan(
        crossing.get(
            "frequency_Hz",
            float("nan"),
        )
    ) / F_010_HZ
    return ell, fhat


def homotypic_crossing_title(
    result: dict[str, Any],
) -> str:
    mode_i, mode_j = result_modes(result)
    ell, fhat = crossing_parameters(result)
    class_name = homotypic_class_key(result)
    return (
        rf"Homotypic {class_name} crossing for "
        rf"{latex_mode(mode_i)} and {latex_mode(mode_j)}, "
        rf"$\ell={ell:.4f}$, "
        rf"$\hat{{f}}={fhat:.4f}$"
    )


def homotypic_metric_table(
    result: dict[str, Any],
    *,
    include_title: bool = True,
) -> str:
    metric_key = (
        relevant_metric_for_homotypic_result(
            result
        )
    )
    info = METRIC_INFO[metric_key]
    values = field_values(
        result,
        metric_key,
    )

    title = ""
    if include_title:
        title = (
            rf"\textbf{{"
            rf"{homotypic_crossing_title(result)}"
            rf"}}\\[0.45em]"
        )

    # E- precedes E+ to match the field-summary PDF columns.
    return rf"""
\begin{{center}}
\small
\renewcommand{{\arraystretch}}{{1.25}}
\setlength{{\tabcolsep}}{{4.5pt}}
{title}
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccccc}}
Metric & Units & $E_1$ & $E_2$ & $E_-$ & $E_+$ & $R_{{\max}}$ \\
\hline
{info["latex"]}
& {info["units"]}
& {fmt_sci(values["E1"])}
& {fmt_sci(values["E2"])}
& {fmt_sci(values["minus"])}
& {fmt_sci(values["plus"])}
& {fmt_float(values["R_max"], 3)} \\
\end{{tabular}}
\end{{ruledtabular}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{center}}
""".strip()


def homotypic_summary_figure(
    result: dict[str, Any],
    *,
    figs_dir_latex: str,
) -> str:
    class_key = homotypic_class_key(result)
    if class_key == "quadrupole":
        width = r"\textwidth"
        height = r"0.62\textheight"
    else:
        width = r"0.65\textwidth"
        height = r"0.68\textheight"

    return rf"""
\begin{{center}}
\includegraphics[
    width={width},
    height={height},
    keepaspectratio
]{{{homotypic_latex_pdf_path(result, figs_dir_latex)}}}
\end{{center}}
""".strip()


def homotypic_page(
    result: dict[str, Any],
    *,
    figs_dir_latex: str,
) -> str:
    if homotypic_class_key(result) == "quadrupole":
        return rf"""
\clearpage
\Needspace{{0.92\textheight}}
\begin{{samepage}}
\noindent\textbf{{{homotypic_crossing_title(result)}}}

\vspace{{0.45em}}

{homotypic_metric_table(result, include_title=False)}

\vspace{{0.45em}}

{homotypic_summary_figure(result, figs_dir_latex=figs_dir_latex)}
\end{{samepage}}
\clearpage
""".strip()

    return "\n\n".join([
        r"\clearpage",
        homotypic_metric_table(result),
        homotypic_summary_figure(
            result,
            figs_dir_latex=figs_dir_latex,
        ),
        r"\clearpage",
    ])


def sort_homotypic_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda result: (
            int(
                result.get(
                    "family_m",
                    mode_azimuthal_index(
                        result_modes(result)[0]
                    ),
                )
            ),
            result_modes(result)[0],
            result_modes(result)[1],
            crossing_parameters(result)[0],
        ),
    )


def appendix_ii_homotypic(
    results: list[dict[str, Any]],
    *,
    figs_dir_latex: str,
) -> str:
    pages = [
        homotypic_page(
            result,
            figs_dir_latex=figs_dir_latex,
        )
        for result in sort_homotypic_results(
            results
        )
    ]
    return "\n\n".join([
        r"""
\clearpage
\section{TM homotypic mixing}
\label{app:tm_homotypic_mixing}

The homotypic mixed fields are characterised using the azimuthal
RF/Fourier multipole decomposition.  Monopole--monopole crossings report
$K_{\parallel}$, dipole--dipole crossings report $K_{\perp}$ and
quadrupole--quadrupole crossings report $K_Q$.  Each uppercase metric is the
corresponding integrated lowercase quantity divided by the analysed structure
length $d$.  The table columns are ordered as $E_1$, $E_2$, $E_-$ and $E_+$ to
match the accompanying field-summary figures.
""".strip(),
        *pages,
        r"% End of Appendix II",
    ])


# -----------------------------------------------------------------------------
# Appendix III: heterotypic mixing
# -----------------------------------------------------------------------------

def heterotypic_crossing_title(
    result: dict[str, Any],
) -> str:
    mode_i, mode_j = result_modes(result)
    ell, fhat = crossing_parameters(result)
    pair_type = (
        pair_type_key(result)
        .replace("_", "--")
    )
    return (
        rf"Heterotypic {pair_type} crossing "
        rf"{latex_mode(mode_i)}--{latex_mode(mode_j)}, "
        rf"$\ell={ell:.4f}$, "
        rf"$\hat{{f}}={fhat:.4f}$"
    )


def heterotypic_metric_table(
    result: dict[str, Any],
) -> str:
    rows: list[str] = []

    for metric_key in (
        relevant_metrics_for_heterotypic_result(
            result
        )
    ):
        info = METRIC_INFO[metric_key]
        values = field_values(
            result,
            metric_key,
        )
        rows.append(
            " & ".join([
                info["latex"],
                info["units"],
                fmt_sci(values["E1"]),
                fmt_sci(values["E2"]),
                fmt_sci(values["minus"]),
                fmt_sci(values["plus"]),
                fmt_float(
                    values["R_max"],
                    3,
                ),
            ])
            + r" \\"
        )

    return rf"""
\begin{{center}}
\footnotesize
\renewcommand{{\arraystretch}}{{1.08}}
\setlength{{\tabcolsep}}{{3.6pt}}
\textbf{{{heterotypic_crossing_title(result)}}}\\[0.15em]
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccccc}}
Beam-dynamics metric
& Units
& $E_1$
& $E_2$
& $E_-$
& $E_+$
& $R_{{\max}}$ \\
\hline
{chr(10).join(rows)}
\end{{tabular}}
\end{{ruledtabular}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{center}}
""".strip()


def heterotypic_figure_size(
    result: dict[str, Any],
) -> tuple[str, str]:
    pair_type = pair_type_key(result)
    if pair_type in {
        "monopole_quadrupole",
        "dipole_quadrupole",
    }:
        return (
            r"0.44\textwidth",
            r"0.62\textheight",
        )
    return (
        r"0.54\textwidth",
        r"0.70\textheight",
    )


def heterotypic_summary_figure(
    result: dict[str, Any],
    *,
    figs_dir_latex: str,
) -> str:
    width, height = heterotypic_figure_size(
        result
    )
    return rf"""
\vspace{{-0.8em}}
\begin{{center}}
\includegraphics[
    width={width},
    height={height},
    keepaspectratio
]{{{heterotypic_latex_pdf_path(result, figs_dir_latex)}}}
\end{{center}}
\vspace{{-0.6em}}
""".strip()


def heterotypic_page(
    result: dict[str, Any],
    *,
    figs_dir_latex: str,
) -> str:
    return "\n\n".join([
        r"\clearpage",
        heterotypic_metric_table(result),
        heterotypic_summary_figure(
            result,
            figs_dir_latex=figs_dir_latex,
        ),
        r"\clearpage",
    ])


def sort_heterotypic_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda result: (
            pair_type_key(result),
            result_modes(result)[0],
            result_modes(result)[1],
            crossing_parameters(result)[0],
        ),
    )


def appendix_iii_heterotypic(
    results: list[dict[str, Any]],
    *,
    figs_dir_latex: str,
) -> str:
    pages = [
        heterotypic_page(
            result,
            figs_dir_latex=figs_dir_latex,
        )
        for result in sort_heterotypic_results(
            results
        )
    ]
    return "\n\n".join([
        r"""
\clearpage
\section{TM heterotypic mixing}
\label{app:tm_heterotypic_mixing}

The heterotypic mixed fields are characterised using the near-axis
Taylor/Hessian analysis.  Each page reports only the uppercase
length-normalised metrics relevant to the two parent families:
$K_{\parallel}$ for monopole content, $K_{\perp}$ for dipole content and
$K_Q$ for quadrupole content.  The table columns are ordered as $E_1$, $E_2$,
$E_-$ and $E_+$ to match the accompanying field-summary figures.
""".strip(),
        *pages,
        r"% End of Appendix III",
    ])


# -----------------------------------------------------------------------------
# Diagnostics and unified writer
# -----------------------------------------------------------------------------

def print_comparison_diagnostics(
    omitted: dict[
        str,
        dict[str, list[str]],
    ],
    spreads: dict[
        str,
        dict[str, dict[str, float]],
    ],
    *,
    spread_warning: float = 0.02,
) -> None:
    print(
        "\nAppendix I modes omitted because they "
        "were not present in both datasets:"
    )
    for metric_key, categories in omitted.items():
        print(f"\n{metric_key}:")
        for category, modes in categories.items():
            print(
                f"  {category}: "
                + (
                    ", ".join(modes)
                    if modes
                    else "none"
                )
            )

    print(
        "\nRepeated parent-mode spread diagnostics:"
    )
    warnings = 0
    for metric_key, modes in spreads.items():
        for mode, spread in modes.items():
            if (
                spread["rf_relative_spread"]
                > spread_warning
                or spread[
                    "hessian_relative_spread"
                ] > spread_warning
            ):
                warnings += 1
                print(
                    f"  {metric_key} {mode}: "
                    f"RF={100.0 * spread['rf_relative_spread']:.2f}%, "
                    f"Hessian={100.0 * spread['hessian_relative_spread']:.2f}%"
                )
    if warnings == 0:
        print(
            "  none above the warning threshold"
        )


def write_unified_appendices(
    *,
    homotypic_root_or_pkl: str | Path = (
        DEFAULT_HOMOTYPIC_ROOT
    ),
    heterotypic_hessian_root_or_pkl: str | Path = (
        DEFAULT_HETEROTYPIC_ROOT
    ),
    out_tex: str | Path = DEFAULT_OUT_TEX,
    figures_destination: str | Path = (
        DEFAULT_FIGS_DIR
    ),
    figs_dir_latex: str = "figs",
    copy_figures: bool = True,
    overwrite_figures: bool = True,
) -> Path:
    homotypic_results = load_homotypic_results(
        homotypic_root_or_pkl
    )
    heterotypic_results = (
        load_heterotypic_hessian_results(
            heterotypic_hessian_root_or_pkl
        )
    )

    appendix_i, omitted, spreads = (
        appendix_i_comparison(
            homotypic_results,
            heterotypic_results,
        )
    )

    if copy_figures:
        homotypic_root = (
            Path(homotypic_root_or_pkl)
            if Path(
                homotypic_root_or_pkl
            ).is_dir()
            else Path(
                homotypic_root_or_pkl
            ).parent
        )
        copy_summary_pdfs(
            homotypic_results,
            heterotypic_results,
            homotypic_root=homotypic_root,
            destination=figures_destination,
            overwrite=overwrite_figures,
        )

    appendix_ii = appendix_ii_homotypic(
        homotypic_results,
        figs_dir_latex=figs_dir_latex,
    )
    appendix_iii = appendix_iii_heterotypic(
        heterotypic_results,
        figs_dir_latex=figs_dir_latex,
    )

    tex = "\n\n".join([
        appendix_i,
        appendix_ii,
        appendix_iii,
    ])

    out_tex = Path(out_tex)
    out_tex.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_tex.write_text(
        tex,
        encoding="utf-8",
    )

    print(f"\nWrote {out_tex}")
    print(
        f"  Appendix I: 3 comparison tables"
    )
    print(
        f"  Appendix II: "
        f"{len(homotypic_results)} homotypic pages"
    )
    print(
        f"  Appendix III: "
        f"{len(heterotypic_results)} heterotypic pages"
    )
    print_comparison_diagnostics(
        omitted,
        spreads,
    )
    return out_tex


if __name__ == "__main__":
    write_unified_appendices(
        homotypic_root_or_pkl=(
            DEFAULT_HOMOTYPIC_ROOT
        ),
        heterotypic_hessian_root_or_pkl=(
            DEFAULT_HETEROTYPIC_ROOT
        ),
        out_tex=DEFAULT_OUT_TEX,
        figures_destination=DEFAULT_FIGS_DIR,
        figs_dir_latex="figs",
        copy_figures=True,
        overwrite_figures=True,
    )
