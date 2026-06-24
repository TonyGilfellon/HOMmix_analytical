"""Build PRAB/REVTeX tables of length-normalised heterotypic effects.

Reads:
    all_heterotypic_multipole_analyses.pkl

Writes one formatted table per heterotypic crossing.

Columns:
    Effect | Units | E1 | E2 | E+ | E- | R_max

where
    R_max = max(|E+|, |E-|) / max(|E1|, |E2|)

Reported length-normalised geometric quantities:
    loss       : k_parallel^(1)  [V/pC/m]
    kick       : k_perp^(2)      [V/pC/m^2]
    focusing   : K_xx^(3)        [V/pC/m^3]
    defocusing : K_yy^(3)        [V/pC/m^3]
    skew       : K_xy^(3)        [V/pC/m^3]

The loss-like quantity in the saved analysis is normally stored as V/pC;
this table divides by the analysed length, giving V/pC/m.

By default this script writes non-floating ruledtabular blocks to avoid
LaTeX's "too many unprocessed floats" error. Set USE_FLOAT_TABLES=True
if you want table* floats.
"""

from __future__ import annotations

from pathlib import Path
import math
import pickle
import re
from typing import Any

import numpy as np


F_010_HZ = 1.3e9

INPUT_PKL = Path(
    r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings"
    r"\all_heterotypic_multipole_analyses.pkl"
)

OUT_TEX = Path(
    r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings"
    r"\heterotypic_length_normalised_prab_tables.tex"
)

TOP_N: int | None = None
USE_FLOAT_TABLES = False


def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


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


def safe_ratio(num: float, den: float) -> float:
    num = finite_or_nan(num)
    den = finite_or_nan(den)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return float("nan")
    return num / den


def fmt_float(x: object, ndp: int = 4) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{ndp}f}"


def fmt_sci(x: object, sig: int = 3) -> str:
    """
    Return LaTeX scientific notation with sig significant figures.

    Example:
        1.799e-28 -> $1.80\\times10^{-28}$
        4.683e+03 -> $4.68\\times10^{3}$
    """
    x = finite_or_nan(x)

    if not math.isfinite(x):
        return "--"

    s = f"{x:.{sig-1}e}"

    mantissa, exponent = s.split("e")
    exponent = int(exponent)

    return rf"${mantissa}\times10^{{{exponent}}}$"


def mode_to_latex(mode: object) -> str:
    if mode is None:
        return "--"

    s = str(mode).strip()

    m = re.search(r"(TM|TE)[_\s-]*([0-9]{3,})", s, flags=re.IGNORECASE)
    if m:
        return rf"${m.group(1).upper()}_{{{m.group(2).zfill(3)}}}$"

    m = re.search(r"\b(\d{3,})\b", s)
    if m:
        return rf"$TM_{{{m.group(1)}}}$"

    return s.replace("_", r"\_")


def latex_label_safe(s: object) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
    return out or "crossing"


def pair_type_to_text(pair_type: object) -> str:
    if pair_type is None:
        return "heterotypic"
    return str(pair_type).replace("_", "--")


def first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return float("nan")


EFFECT_ROWS = [
    {
        "key": "loss",
        "label": r"$k_{\parallel}^{(1)}$",
        "units": r"$\mathrm{V/pC/m}$",
    },
    {
        "key": "kick",
        "label": r"$k_{\perp}^{(2)}$",
        "units": r"$\mathrm{V/pC/m^2}$",
    },
    {
        "key": "focusing",
        "label": r"$K_{xx}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
    },
    {
        "key": "defocusing",
        "label": r"$K_{yy}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
    },
    {
        "key": "skew",
        "label": r"$K_{xy}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
    },
]


def field_length_m(field_result: dict[str, Any]) -> float:
    return finite_or_nan(field_result.get("length_m", float("nan")))


def figures(field_result: dict[str, Any]) -> dict[str, Any]:
    return field_result.get("figures_of_merit", {})


def loss_v_per_pc_per_m(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    loss_v_per_pc = finite_or_nan(
        first_present(
            f,
            (
                "loss_like_V_per_pC",
                "loss_like_V2_per_C2",  # updated workflow alias; value is V/pC
            ),
        )
    )
    length_m = field_length_m(field_result)

    if not math.isfinite(loss_v_per_pc) or not math.isfinite(length_m) or length_m <= 0.0:
        return float("nan")

    return abs(loss_v_per_pc) / length_m


def kick_v_per_pc_per_m2(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return abs_finite_or_nan(
        first_present(
            f,
            (
                "kick_magnitude_V_per_pC_per_m2",
                "kick_mag_V_per_pC_per_m2",
                "kick_magnitude_V_per_C_per_m",  # updated workflow alias; value is V/pC/m^2
            ),
        )
    )


def Kxx_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return abs_finite_or_nan(
        first_present(
            f,
            (
                "Kxx_V_per_pC_per_m3",
                "Kxx_V_per_C_per_m_per_m",  # updated workflow alias; value is V/pC/m^3
            ),
        )
    )


def Kyy_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return abs_finite_or_nan(
        first_present(
            f,
            (
                "Kyy_V_per_pC_per_m3",
                "Kyy_V_per_C_per_m_per_m",  # updated workflow alias; value is V/pC/m^3
            ),
        )
    )


def Kxy_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return abs_finite_or_nan(
        first_present(
            f,
            (
                "Kxy_V_per_pC_per_m3",
                "Kxy_V_per_C_per_m_per_m",  # updated workflow alias; value is V/pC/m^3
            ),
        )
    )


def effect_value(field_result: dict[str, Any], effect_key: str) -> float:
    if effect_key == "loss":
        return loss_v_per_pc_per_m(field_result)
    if effect_key == "kick":
        return kick_v_per_pc_per_m2(field_result)
    if effect_key == "focusing":
        return Kxx_v_per_pc_per_m3(field_result)
    if effect_key == "defocusing":
        return Kyy_v_per_pc_per_m3(field_result)
    if effect_key == "skew":
        return Kxy_v_per_pc_per_m3(field_result)
    raise KeyError(effect_key)


def effect_row(result: dict[str, Any], effect_key: str) -> dict[str, float]:
    vals = {}
    for name in ("E1", "E2", "plus", "minus"):
        vals[name] = effect_value(result["fields"][name], effect_key)

    parent_max = max(vals["E1"], vals["E2"])
    mixed_max = max(vals["plus"], vals["minus"])

    vals["parent_max"] = parent_max
    vals["mixed_max"] = mixed_max
    vals["R_max"] = safe_ratio(mixed_max, parent_max)
    return vals


def crossing_short_title(result: dict[str, Any]) -> str:
    c = result["crossing"]
    pair_type = pair_type_to_text(result.get("pair_type") or c.get("pair_type"))
    mode_i = mode_to_latex(result.get("mode_i"))
    mode_j = mode_to_latex(result.get("mode_j"))
    ell = float(c["length_factor"])
    fhat = float(c["frequency_Hz"]) / F_010_HZ

    return (
        rf"{pair_type} crossing {mode_i}--{mode_j}, "
        rf"$\ell={ell:.4f}$, $\hat{{f}}={fhat:.4f}$"
    )


def crossing_caption(result: dict[str, Any]) -> str:
    c = result["crossing"]
    pair_type = pair_type_to_text(result.get("pair_type") or c.get("pair_type"))
    mode_i = mode_to_latex(result.get("mode_i"))
    mode_j = mode_to_latex(result.get("mode_j"))
    ell = float(c["length_factor"])
    fhat = float(c["frequency_Hz"]) / F_010_HZ

    return (
        rf"Length-normalised geometric figures of merit for the heterotypic "
        rf"{pair_type} crossing {mode_i}--{mode_j}. "
        rf"The crossing occurs at $\ell={ell:.4f}$ and $\hat{{f}}={fhat:.4f}$. "
        rf"For each row, $R_{{\max}}=\max(E_+,E_-)/\max(E_1,E_2)$."
    )


def latex_table_for_crossing(result: dict[str, Any], table_index: int) -> str:
    c = result["crossing"]
    pair = latex_label_safe(result.get("pair_type") or c.get("pair_type", "heterotypic"))
    mi = latex_label_safe(result.get("mode_i"))
    mj = latex_label_safe(result.get("mode_j"))
    ell_label = latex_label_safe(f"{float(c['length_factor']):.8g}")

    lines = []

    if USE_FLOAT_TABLES:
        lines.append(r"\begin{table*}")
        lines.append(
            rf"\caption{{\label{{tab:heterotypic_effects_{table_index:03d}_{pair}_{mi}_{mj}_{ell_label}}}"
            + crossing_caption(result)
            + r"}"
        )
    else:
        lines.append(r"\begin{center}")
        lines.append(rf"\textbf{{{crossing_short_title(result)}}}\\[0.5em]")

    lines.append(r"\begin{ruledtabular}")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(
        r"Effect & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{\max}$ \\"
    )
    lines.append(r"\hline")

    for info in EFFECT_ROWS:
        row = effect_row(result, info["key"])
        cells = [
            info["label"],
            info["units"],
            fmt_sci(row["E1"]),
            fmt_sci(row["E2"]),
            fmt_sci(row["plus"]),
            fmt_sci(row["minus"]),
            fmt_float(row["R_max"], 3),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{ruledtabular}")

    if USE_FLOAT_TABLES:
        lines.append(r"\end{table*}")
    else:
        lines.append(r"\end{center}")

    return "\n".join(lines)


def sorted_results(results: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def keyfunc(item: tuple[str, dict[str, Any]]):
        _, r = item
        c = r.get("crossing", {})
        return (
            str(r.get("pair_type") or c.get("pair_type", "")),
            str(r.get("mode_i", "")),
            str(r.get("mode_j", "")),
            float(c.get("length_factor", np.inf)),
        )

    return sorted(results.items(), key=keyfunc)


def build_heterotypic_effect_tables(
    *,
    input_pkl: str | Path = INPUT_PKL,
    out_tex: str | Path = OUT_TEX,
    top_n: int | None = TOP_N,
) -> None:
    results = pickle_load(input_pkl)
    items = sorted_results(results)

    if top_n is not None:
        items = items[:top_n]

    blocks = []
    for i, (_, result) in enumerate(items, start=1):
        blocks.append(latex_table_for_crossing(result, i))

    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    build_heterotypic_effect_tables()
