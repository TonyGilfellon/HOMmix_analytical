"""Build PRAB/REVTeX tables of fitted Taylor coefficients for heterotypic crossings.

This script reads the saved output from:

    heterotypic_multipole_analysis_length_normalised.py

specifically:

    all_heterotypic_multipole_analyses.pkl

and writes one LaTeX table per crossing.

Each table has columns:

    coefficient | E1 | E2 | E+ | E- | R_max

where

    R_max = max(|E+|, |E-|) / max(|E1|, |E2|)

for each fitted Taylor coefficient in the vector

    c = (V0, ax, ay, bxx, bxy, byy).

The table values are the magnitudes of the complex fitted coefficients:
    |V0|   [V]
    |ax|   [V/m]
    |ay|   [V/m]
    |bxx|  [V/m^2]
    |bxy|  [V/m^2]
    |byy|  [V/m^2]

The length-normalised physical figures of merit are still stored in the
heterotypic analysis output; this table is specifically for showing how the
near-axis fitted voltage expansion changes under mixing.
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
    r"\heterotypic_taylor_coefficients_prab_tables.tex"
)

TOP_N: int | None = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def finite_or_nan(x: object) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


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
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{sig}e}"


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


# -----------------------------------------------------------------------------
# Fitted coefficient extraction
# -----------------------------------------------------------------------------

COEFF_INFO = [
    ("V0",  r"$|V_0|$",       r"$\mathrm{V}$"),
    ("ax",  r"$|a_x|$",       r"$\mathrm{V/m}$"),
    ("ay",  r"$|a_y|$",       r"$\mathrm{V/m}$"),
    ("bxx", r"$|b_{xx}|$",    r"$\mathrm{V/m^2}$"),
    ("bxy", r"$|b_{xy}|$",    r"$\mathrm{V/m^2}$"),
    ("byy", r"$|b_{yy}|$",    r"$\mathrm{V/m^2}$"),
]


def get_coefficients(field_result: dict[str, Any]) -> dict[str, complex]:
    """
    Return the complex Taylor coefficients for one field result.

    Expected location:
        field_result["fit"]["coefficients"]

    with keys:
        V0, ax, ay, bxx, bxy, byy
    """
    coeffs = field_result.get("fit", {}).get("coefficients", {})
    if not coeffs:
        raise KeyError("Could not find field_result['fit']['coefficients']")

    out = {}
    for key, _, _ in COEFF_INFO:
        out[key] = complex(coeffs.get(key, np.nan + 0.0j))
    return out


def coefficient_row(result: dict[str, Any], coeff_key: str) -> dict[str, float]:
    vals = {}
    for name in ("E1", "E2", "plus", "minus"):
        coeffs = get_coefficients(result["fields"][name])
        vals[name] = abs(coeffs[coeff_key])

    parent_max = max(vals["E1"], vals["E2"])
    mixed_max = max(vals["plus"], vals["minus"])
    vals["R_max"] = safe_ratio(mixed_max, parent_max)
    vals["parent_max"] = parent_max
    vals["mixed_max"] = mixed_max
    return vals


# -----------------------------------------------------------------------------
# LaTeX table construction
# -----------------------------------------------------------------------------

def crossing_caption(result: dict[str, Any]) -> str:
    c = result["crossing"]
    pair_type = pair_type_to_text(result.get("pair_type") or c.get("pair_type"))
    mode_i = mode_to_latex(result.get("mode_i"))
    mode_j = mode_to_latex(result.get("mode_j"))
    ell = float(c["length_factor"])
    fhat = float(c["frequency_Hz"]) / F_010_HZ

    return (
        rf"Near-axis Taylor-coefficient comparison for the heterotypic "
        rf"{pair_type} crossing {mode_i}--{mode_j}. "
        rf"The crossing occurs at $\ell={ell:.4f}$ and $\hat{{f}}={fhat:.4f}$. "
        rf"Rows give magnitudes of the complex fitted coefficients in "
        rf"$V_z(x,y)=V_0+a_xx+a_yy+b_{{xx}}x^2+b_{{xy}}xy+b_{{yy}}y^2$. "
        rf"For each row, $R_{{\max}}=\max(E_+,E_-)/\max(E_1,E_2)$."
    )

def crossing_short_title(result: dict) -> str:
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

def latex_table_for_crossing(result: dict[str, Any], table_index: int) -> str:
    c = result["crossing"]
    pair = latex_label_safe(result.get("pair_type") or c.get("pair_type", "heterotypic"))
    mi = latex_label_safe(result.get("mode_i"))
    mj = latex_label_safe(result.get("mode_j"))
    ell_label = latex_label_safe(f"{float(c['length_factor']):.8g}")

    lines = []
    lines.append(r"\begin{center}")
    lines.append(rf"\textbf{{{crossing_short_title(result)}}}\\[0.5em]")
    lines.append(r"\begin{ruledtabular}")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(
        r"Coefficient & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{\max}$ \\"
    )
    lines.append(r"\hline")

    for coeff_key, coeff_latex, units in COEFF_INFO:
        row = coefficient_row(result, coeff_key)
        cells = [
            coeff_latex,
            units,
            fmt_sci(row["E1"]),
            fmt_sci(row["E2"]),
            fmt_sci(row["plus"]),
            fmt_sci(row["minus"]),
            fmt_float(row["R_max"], 3),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{ruledtabular}")
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


def build_heterotypic_coefficient_tables(
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
    build_heterotypic_coefficient_tables()
