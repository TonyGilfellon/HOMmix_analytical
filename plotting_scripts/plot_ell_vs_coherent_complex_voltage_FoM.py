from __future__ import annotations

"""
Plot ell versus log10 of a coherent complex-response figure of merit.

The script reads the same saved homotypic RF/Fourier and heterotypic
Taylor/Hessian result pickles used by the PRAB appendix/table compilers.

For each crossing it constructs a complex, energy- and length-normalised
response vector for the relevant multipole order:

    monopole:
        a = V0 / sqrt(4 U d)
        K_parallel = ||a||^2

    dipole:
        a = sqrt(c / (4 U omega d)) [Vx, Vy]
        K_perp = ||a||^2

    quadrupole, RF/Fourier:
        a = 4 c / (omega sqrt(4 U) d) [c2, s2]
        K_Q = ||a||

    quadrupole, Taylor/Hessian:
        a = 1 / (sqrt(4 U) d) [Kxx_raw - Kyy_raw, 2 Kxy_raw]
        K_Q = ||a||

For two modes A and B with an unknown relative RF phase theta, the combined
complex response is

    a_combined(theta) = a_A + exp(i theta) a_B.

The maximum coherent envelope over theta is

    ||a_combined||^2_max
        = ||a_A||^2 + ||a_B||^2 + 2 |a_A^H a_B|.

For monopole and dipole metrics the figure of merit is this squared norm.
For the quadrupole metric the figure of merit is the norm itself.

The plotted ratio is

    R_coherent =
        FoM_coherent(E_minus, E_plus)
        --------------------------------
        FoM_coherent(E1, E2),

and the vertical axis is log10(R_coherent).

Interpretation:
    log10(R_coherent) > 0  : larger phase-optimised coherent response after mixing
    log10(R_coherent) = 0  : unchanged coherent envelope
    log10(R_coherent) < 0  : smaller coherent envelope after mixing

This is an equal-weight, phase-optimised envelope.  It is not a prediction of
one specific bunch passage unless the two modes are excited with those relative
weights and phases.  For modes with different frequencies it is the upper
envelope accessible during their beat cycle.
"""

import csv
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


C0 = 299_792_458.0

DEFAULT_ANALYSIS_ROOT = Path(
    r"D:\PhD\HOMmix\HOMmix_analytical\analysis"
)
DEFAULT_HOMOTYPIC_ROOT = (
    DEFAULT_ANALYSIS_ROOT / "homotypic_rf_multipole"
)
DEFAULT_HETEROTYPIC_ROOT = (
    DEFAULT_ANALYSIS_ROOT / "heterotypic_crossings"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\PhD\PRAB\figures\ell_vs_coherent_complex_voltage"
)


@dataclass
class PlotConfig:
    homotypic_root_or_pickle: Path = DEFAULT_HOMOTYPIC_ROOT
    heterotypic_root_or_pickle: Path = DEFAULT_HETEROTYPIC_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR

    include_homotypic: bool = True
    include_heterotypic: bool = True

    # "max_envelope" optimises over the relative phase.
    # "fixed_phase" uses fixed_relative_phase_rad.
    phase_mode: str = "max_envelope"
    fixed_relative_phase_rad: float = 0.0

    annotate: bool = False
    marker_size: float = 44.0
    dpi: int = 300

    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None

    make_family_plots: bool = True
    make_combined_plot: bool = True


METRIC_LABELS = {
    "K_parallel": r"$K_{\parallel}$",
    "K_perp": r"$K_{\perp}$",
    "K_Q": r"$K_Q$",
}

CATEGORY_STYLE = {
    "homotypic_monopole": ("*", "hom. monopole"),
    "homotypic_dipole": ("+", "hom. dipole"),
    "homotypic_quadrupole": ("v", "hom. quadrupole"),
    "heterotypic_monopole_dipole_K_parallel": (
        "o",
        r"het. M--D, $K_{\parallel}$",
    ),
    "heterotypic_monopole_dipole_K_perp": (
        "o",
        r"het. M--D, $K_{\perp}$",
    ),
    "heterotypic_monopole_quadrupole_K_parallel": (
        "s",
        r"het. M--Q, $K_{\parallel}$",
    ),
    "heterotypic_monopole_quadrupole_K_Q": (
        "s",
        r"het. M--Q, $K_Q$",
    ),
    "heterotypic_dipole_quadrupole_K_perp": (
        "^",
        r"het. D--Q, $K_{\perp}$",
    ),
    "heterotypic_dipole_quadrupole_K_Q": (
        "^",
        r"het. D--Q, $K_Q$",
    ),
}


# -----------------------------------------------------------------------------
# Generic loading and metadata helpers
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


def mode_azimuthal_index(mode: object) -> int | None:
    match = re.search(
        r"_(\d)",
        normalise_mode_name(mode),
    )
    return int(match.group(1)) if match else None


def result_modes(
    result: dict[str, Any],
) -> tuple[str, str]:
    crossing = result.get("crossing", {})
    return (
        normalise_mode_name(
            result.get(
                "mode_i",
                crossing.get("mode_i", ""),
            )
        ),
        normalise_mode_name(
            result.get(
                "mode_j",
                crossing.get("mode_j", ""),
            )
        ),
    )


def crossing_ell(
    result: dict[str, Any],
) -> float:
    return finite_or_nan(
        result.get("crossing", {}).get(
            "length_factor",
            float("nan"),
        )
    )


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

    if pair_type:
        return (
            str(pair_type)
            .lower()
            .replace("-", "_")
        )

    mode_i, mode_j = result_modes(result)
    pair = tuple(
        sorted([
            mode_azimuthal_index(mode_i),
            mode_azimuthal_index(mode_j),
        ])
    )
    return {
        (0, 1): "monopole_dipole",
        (0, 2): "monopole_quadrupole",
        (1, 2): "dipole_quadrupole",
    }.get(pair, "heterotypic")


def flatten_result_container(
    data: Any,
) -> list[dict[str, Any]]:
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


def result_identity(
    result: dict[str, Any],
) -> tuple[Any, ...]:
    crossing = result.get("crossing", {})
    mode_i, mode_j = result_modes(result)
    return (
        result.get("family_m"),
        pair_type_key(result),
        mode_i,
        mode_j,
        finite_or_nan(
            crossing.get(
                "length_factor",
                float("nan"),
            )
        ),
        finite_or_nan(
            crossing.get(
                "frequency_Hz",
                float("nan"),
            )
        ),
    )


def load_results(
    root_or_pickle: str | Path,
    *,
    aggregate_filename: str,
    per_crossing_filename: str,
    merge_aggregate_and_per_crossing: bool,
) -> list[dict[str, Any]]:
    path = Path(root_or_pickle)

    if path.is_file():
        return flatten_result_container(
            pickle_load(path)
        )

    collected: list[dict[str, Any]] = []
    aggregate = path / aggregate_filename

    if aggregate.exists():
        collected.extend(
            flatten_result_container(
                pickle_load(aggregate)
            )
        )
        if not merge_aggregate_and_per_crossing:
            return collected

    files = sorted(
        path.rglob(per_crossing_filename)
    )
    for filename in files:
        collected.extend(
            flatten_result_container(
                pickle_load(filename)
            )
        )

    if not collected:
        raise FileNotFoundError(
            f"No results found below {path}. Expected "
            f"{aggregate_filename!r} or "
            f"{per_crossing_filename!r}."
        )

    unique: dict[
        tuple[Any, ...],
        dict[str, Any],
    ] = {}
    for result in collected:
        unique[result_identity(result)] = result

    return list(unique.values())


def load_homotypic_results(
    root_or_pickle: str | Path,
) -> list[dict[str, Any]]:
    return load_results(
        root_or_pickle,
        aggregate_filename=(
            "all_homotypic_rf_multipole_analyses.pkl"
        ),
        per_crossing_filename=(
            "homotypic_rf_multipole_analysis.pkl"
        ),
        merge_aggregate_and_per_crossing=True,
    )


def load_heterotypic_results(
    root_or_pickle: str | Path,
) -> list[dict[str, Any]]:
    return load_results(
        root_or_pickle,
        aggregate_filename=(
            "all_heterotypic_multipole_analyses.pkl"
        ),
        per_crossing_filename=(
            "heterotypic_multipole_analysis.pkl"
        ),
        merge_aggregate_and_per_crossing=False,
    )


# -----------------------------------------------------------------------------
# Complex normalized response vectors
# -----------------------------------------------------------------------------

def field_energy_J(
    field_result: dict[str, Any],
) -> float:
    energy = field_result.get(
        "energy_diagnostics",
        {},
    )
    for key in (
        "U_CST_equiv_J",
        "U_used_J",
        "U_Etotal_used_J",
    ):
        value = finite_or_nan(
            energy.get(key, float("nan"))
        )
        if math.isfinite(value) and value > 0.0:
            return value
    raise KeyError(
        "Could not find a positive stored-energy value "
        "in field_result['energy_diagnostics']."
    )


def field_length_m(
    field_result: dict[str, Any],
) -> float:
    value = finite_or_nan(
        field_result.get(
            "length_m",
            float("nan"),
        )
    )
    if not math.isfinite(value) or value <= 0.0:
        raise KeyError(
            "Missing or invalid field length_m."
        )
    return value


def field_omega(
    field_result: dict[str, Any],
) -> float:
    frequency = finite_or_nan(
        field_result.get(
            "frequency_Hz",
            field_result.get(
                "figures_of_merit",
                {},
            ).get(
                "frequency_Hz",
                float("nan"),
            ),
        )
    )
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise KeyError(
            "Missing or invalid field frequency_Hz."
        )
    return 2.0 * np.pi * frequency


def rf_response_vector(
    field_result: dict[str, Any],
    metric_key: str,
) -> np.ndarray:
    """Normalized complex response from saved RF/Fourier coefficients."""
    U = field_energy_J(field_result)
    d = field_length_m(field_result)
    omega = field_omega(field_result)
    multipoles = field_result.get(
        "multipole_coefficients",
        {},
    )

    if metric_key == "K_parallel":
        c0 = complex(
            multipoles["c0_axis_V_per_C"]
        )
        return np.asarray([
            c0 / np.sqrt(4.0 * U * d)
        ], dtype=complex)

    if metric_key == "K_perp":
        c1 = complex(
            multipoles[
                "c1_cos_phi_V_per_C_per_m"
            ]
        )
        s1 = complex(
            multipoles[
                "s1_sin_phi_V_per_C_per_m"
            ]
        )
        scale = np.sqrt(
            C0 / (4.0 * U * omega * d)
        )
        return scale * np.asarray(
            [c1, s1],
            dtype=complex,
        )

    if metric_key == "K_Q":
        c2 = complex(
            multipoles[
                "c2_cos_2phi_V_per_C_per_m2"
            ]
        )
        s2 = complex(
            multipoles[
                "s2_sin_2phi_V_per_C_per_m2"
            ]
        )

        # The saved unified homotypic workflow uses the factor-four convention
        # so that ||response|| reproduces its stored K_Q.
        scale = (
            4.0
            * C0
            / (
                omega
                * np.sqrt(4.0 * U)
                * d
            )
        )
        return scale * np.asarray(
            [c2, s2],
            dtype=complex,
        )

    raise KeyError(metric_key)


def hessian_response_vector(
    field_result: dict[str, Any],
    metric_key: str,
) -> np.ndarray:
    """Normalized complex response from saved Taylor/Hessian coefficients."""
    U = field_energy_J(field_result)
    d = field_length_m(field_result)
    omega = field_omega(field_result)
    fom = field_result.get(
        "figures_of_merit",
        {},
    )

    if metric_key == "K_parallel":
        V0 = complex(
            fom["V0_V_per_C"]
        )
        return np.asarray([
            V0 / np.sqrt(4.0 * U * d)
        ], dtype=complex)

    if metric_key == "K_perp":
        Vx = complex(
            fom[
                "dVz_dx_V_per_C_per_m"
            ]
        )
        Vy = complex(
            fom[
                "dVz_dy_V_per_C_per_m"
            ]
        )
        scale = np.sqrt(
            C0 / (4.0 * U * omega * d)
        )
        return scale * np.asarray(
            [Vx, Vy],
            dtype=complex,
        )

    if metric_key == "K_Q":
        Kxx_raw = complex(
            fom[
                "Kxx_V_per_C_per_m_per_m"
            ]
        )
        Kxy_raw = complex(
            fom[
                "Kxy_V_per_C_per_m_per_m"
            ]
        )
        Kyy_raw = complex(
            fom[
                "Kyy_V_per_C_per_m_per_m"
            ]
        )
        scale = 1.0 / (
            np.sqrt(4.0 * U) * d
        )
        return scale * np.asarray(
            [
                Kxx_raw - Kyy_raw,
                2.0 * Kxy_raw,
            ],
            dtype=complex,
        )

    raise KeyError(metric_key)


def response_vector(
    field_result: dict[str, Any],
    metric_key: str,
) -> np.ndarray:
    method = str(
        field_result.get(
            "analysis_method",
            "",
        )
    ).lower()

    if (
        "rf" in method
        or "fourier" in method
        or "multipole_coefficients"
        in field_result
    ):
        return rf_response_vector(
            field_result,
            metric_key,
        )

    return hessian_response_vector(
        field_result,
        metric_key,
    )


# -----------------------------------------------------------------------------
# Coherent combination
# -----------------------------------------------------------------------------

def combined_norm_squared(
    response_a: np.ndarray,
    response_b: np.ndarray,
    *,
    phase_mode: str,
    fixed_relative_phase_rad: float,
) -> float:
    a = np.asarray(
        response_a,
        dtype=complex,
    ).ravel()
    b = np.asarray(
        response_b,
        dtype=complex,
    ).ravel()

    if a.shape != b.shape:
        raise ValueError(
            "Response vectors must have matching dimensions: "
            f"{a.shape} versus {b.shape}."
        )

    if phase_mode == "max_envelope":
        value = (
            np.vdot(a, a).real
            + np.vdot(b, b).real
            + 2.0 * abs(np.vdot(a, b))
        )
        return float(max(value, 0.0))

    if phase_mode == "fixed_phase":
        combined = (
            a
            + np.exp(
                1j
                * float(
                    fixed_relative_phase_rad
                )
            )
            * b
        )
        return float(
            np.vdot(
                combined,
                combined,
            ).real
        )

    raise ValueError(
        "phase_mode must be 'max_envelope' "
        "or 'fixed_phase'."
    )


def coherent_metric(
    response_a: np.ndarray,
    response_b: np.ndarray,
    *,
    metric_key: str,
    phase_mode: str,
    fixed_relative_phase_rad: float,
) -> float:
    norm_squared = combined_norm_squared(
        response_a,
        response_b,
        phase_mode=phase_mode,
        fixed_relative_phase_rad=(
            fixed_relative_phase_rad
        ),
    )

    if metric_key in (
        "K_parallel",
        "K_perp",
    ):
        return norm_squared

    if metric_key == "K_Q":
        return float(
            np.sqrt(norm_squared)
        )

    raise KeyError(metric_key)


def coherent_ratio(
    result: dict[str, Any],
    metric_key: str,
    *,
    phase_mode: str,
    fixed_relative_phase_rad: float,
) -> tuple[float, float, float]:
    fields = result["fields"]

    parent = coherent_metric(
        response_vector(
            fields["E1"],
            metric_key,
        ),
        response_vector(
            fields["E2"],
            metric_key,
        ),
        metric_key=metric_key,
        phase_mode=phase_mode,
        fixed_relative_phase_rad=(
            fixed_relative_phase_rad
        ),
    )
    mixed = coherent_metric(
        response_vector(
            fields["minus"],
            metric_key,
        ),
        response_vector(
            fields["plus"],
            metric_key,
        ),
        metric_key=metric_key,
        phase_mode=phase_mode,
        fixed_relative_phase_rad=(
            fixed_relative_phase_rad
        ),
    )

    ratio = (
        mixed / parent
        if (
            math.isfinite(parent)
            and parent > 0.0
        )
        else float("nan")
    )
    return float(ratio), float(parent), float(mixed)


# -----------------------------------------------------------------------------
# Point collection
# -----------------------------------------------------------------------------

def homotypic_metric_key(
    result: dict[str, Any],
) -> str:
    family_m = result.get("family_m")
    if family_m is None:
        family_m = mode_azimuthal_index(
            result_modes(result)[0]
        )

    return {
        0: "K_parallel",
        1: "K_perp",
        2: "K_Q",
    }[int(family_m)]


def heterotypic_metric_keys(
    result: dict[str, Any],
) -> tuple[str, str]:
    return {
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
    }[pair_type_key(result)]


def collect_points(
    config: PlotConfig,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    if config.include_homotypic:
        for result in load_homotypic_results(
            config.homotypic_root_or_pickle
        ):
            metric_key = homotypic_metric_key(
                result
            )
            ratio, parent, mixed = (
                coherent_ratio(
                    result,
                    metric_key,
                    phase_mode=(
                        config.phase_mode
                    ),
                    fixed_relative_phase_rad=(
                        config.fixed_relative_phase_rad
                    ),
                )
            )
            ell = crossing_ell(result)
            if (
                not math.isfinite(ratio)
                or ratio <= 0.0
                or not math.isfinite(ell)
            ):
                continue

            mode_i, mode_j = result_modes(
                result
            )
            family_m = int(
                result.get(
                    "family_m",
                    mode_azimuthal_index(
                        mode_i
                    ),
                )
            )
            family_name = {
                0: "monopole",
                1: "dipole",
                2: "quadrupole",
            }[family_m]

            points.append({
                "analysis": "homotypic",
                "category": (
                    f"homotypic_{family_name}"
                ),
                "pair_type": (
                    f"{family_name}_{family_name}"
                ),
                "metric": metric_key,
                "mode_i": mode_i,
                "mode_j": mode_j,
                "ell": ell,
                "coherent_parent": parent,
                "coherent_mixed": mixed,
                "R_coherent": ratio,
                "log10_R_coherent": (
                    float(np.log10(ratio))
                ),
            })

    if config.include_heterotypic:
        for result in load_heterotypic_results(
            config.heterotypic_root_or_pickle
        ):
            pair_type = pair_type_key(
                result
            )
            mode_i, mode_j = result_modes(
                result
            )
            ell = crossing_ell(result)

            for metric_key in (
                heterotypic_metric_keys(result)
            ):
                ratio, parent, mixed = (
                    coherent_ratio(
                        result,
                        metric_key,
                        phase_mode=(
                            config.phase_mode
                        ),
                        fixed_relative_phase_rad=(
                            config.fixed_relative_phase_rad
                        ),
                    )
                )
                if (
                    not math.isfinite(ratio)
                    or ratio <= 0.0
                    or not math.isfinite(ell)
                ):
                    continue

                points.append({
                    "analysis": "heterotypic",
                    "category": (
                        f"heterotypic_"
                        f"{pair_type}_"
                        f"{metric_key}"
                    ),
                    "pair_type": pair_type,
                    "metric": metric_key,
                    "mode_i": mode_i,
                    "mode_j": mode_j,
                    "ell": ell,
                    "coherent_parent": parent,
                    "coherent_mixed": mixed,
                    "R_coherent": ratio,
                    "log10_R_coherent": (
                        float(np.log10(ratio))
                    ),
                })

    return points


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_points_csv(
    points: list[dict[str, Any]],
    filename: str | Path,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "analysis",
        "category",
        "pair_type",
        "metric",
        "mode_i",
        "mode_j",
        "ell",
        "coherent_parent",
        "coherent_mixed",
        "R_coherent",
        "log10_R_coherent",
    ]

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(points)


def plot_points(
    points: list[dict[str, Any]],
    filename: str | Path,
    *,
    title: str,
    config: PlotConfig,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8.4, 5.6),
    )

    categories = sorted({
        point["category"]
        for point in points
    })

    for category in categories:
        category_points = [
            point
            for point in points
            if point["category"] == category
        ]
        marker, label = CATEGORY_STYLE.get(
            category,
            ("o", category),
        )

        x = [
            point["ell"]
            for point in category_points
        ]
        y = [
            point["log10_R_coherent"]
            for point in category_points
        ]

        scatter_kwargs = {
            "marker": marker,
            "s": config.marker_size,
            "label": label,
        }

        # Use open markers where Matplotlib supports a face colour.
        if marker not in ("+", "x"):
            scatter_kwargs["facecolors"] = "none"

        axis.scatter(
            x,
            y,
            **scatter_kwargs,
        )

        if config.annotate:
            for point in category_points:
                axis.annotate(
                    (
                        point["mode_i"]
                        .replace("_", "")
                        + "\n"
                        + point["mode_j"]
                        .replace("_", "")
                    ),
                    (
                        point["ell"],
                        point[
                            "log10_R_coherent"
                        ],
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel(r"Length factor, $\ell$")
    axis.set_ylabel(
        r"$\log_{10}\!\left(R_{\mathrm{coh}}\right)$"
    )
    axis.set_title(title)
    axis.grid(
        True,
        which="both",
        alpha=0.25,
    )

    if config.x_limits is not None:
        axis.set_xlim(*config.x_limits)
    if config.y_limits is not None:
        axis.set_ylim(*config.y_limits)

    axis.legend(
        fontsize=8,
        frameon=False,
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(
        filename,
        dpi=config.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(
    config: PlotConfig = PlotConfig(),
) -> list[dict[str, Any]]:
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    points = collect_points(config)
    if not points:
        raise RuntimeError(
            "No valid coherent-response points were collected."
        )

    save_points_csv(
        points,
        config.output_dir
        / "ell_vs_coherent_complex_voltage_FoM.csv",
    )

    phase_description = (
        "phase-optimised envelope"
        if config.phase_mode == "max_envelope"
        else (
            "fixed relative phase "
            f"{config.fixed_relative_phase_rad:.3f} rad"
        )
    )

    if config.make_combined_plot:
        plot_points(
            points,
            config.output_dir
            / (
                "homotypic_and_heterotypic_"
                "ell_vs_log10_Rcoherent.png"
            ),
            title=(
                "Coherent complex-response ratio "
                f"({phase_description})"
            ),
            config=config,
        )

    if config.make_family_plots:
        for pair_type in sorted({
            point["pair_type"]
            for point in points
        }):
            selected = [
                point
                for point in points
                if point["pair_type"]
                == pair_type
            ]
            plot_points(
                selected,
                config.output_dir
                / (
                    f"{pair_type}_"
                    "ell_vs_log10_Rcoherent.png"
                ),
                title=(
                    pair_type.replace(
                        "_",
                        " ",
                    ).title()
                    + " coherent complex-response ratio"
                ),
                config=config,
            )

    print(
        f"Saved {len(points)} points below "
        f"{config.output_dir}"
    )
    return points


if __name__ == "__main__":
    config = PlotConfig(
        homotypic_root_or_pickle=(
            DEFAULT_HOMOTYPIC_ROOT
        ),
        heterotypic_root_or_pickle=(
            DEFAULT_HETEROTYPIC_ROOT
        ),
        output_dir=DEFAULT_OUTPUT_DIR,
        include_homotypic=True,
        include_heterotypic=True,
        phase_mode="max_envelope",
        fixed_relative_phase_rad=0.0,
        annotate=False,
        x_limits=None,
        y_limits=None,
        make_family_plots=True,
        make_combined_plot=True,
    )
    main(config)
