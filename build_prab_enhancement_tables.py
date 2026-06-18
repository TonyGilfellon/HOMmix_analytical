"""
Build PRAB/REVTeX ruledtabular LaTeX tables from enhancement summary CSV files.

Updates in this version
-----------------------
1. The E1 and E2 table entries are rendered as mode labels, e.g. TM_{011}.
2. The final table column is read directly from the CSV column named mixed_max.
3. The frequency_Hz column is converted to f_hat = frequency_Hz / F_010_HZ
   unless a f_hat/frequency_normalised column already exists.

Edit INPUT_FILES and OUT_TEX for your machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd


F_010_HZ = 1.3e9

INPUT_FILES = {
    "monopole": Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_monopoles\postprocess\monopole_enhancement_summary.csv"),
    "dipole": Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles\postprocess\dipole_enhancement_summary.csv"),
    "quadrupole": Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles\postprocess\quadrupole_enhancement_summary.csv"),
}

OUT_TEX = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\enhancement_tables_prab.tex")
TOP_N_PER_FAMILY = None


@dataclass(frozen=True)
class FamilyTableSpec:
    family: str
    metric_heading: str
    caption: str
    label: str


TABLE_SPECS = {
    "monopole": FamilyTableSpec(
        family="monopole",
        metric_heading=r"$k_{\parallel,\max(\pm)}$",
        label="tab:monopole_enhancement",
        caption=(
            r"Enhancement summary for homotypic monopole crossings. "
            r"The normalized frequency is $\hat{f}=f/f_{010}$ and "
            r"$R_{\max}=\max(M_+,M_-)/\max(M_1,M_2)$. "
            r"The final column is the \texttt{mixed\_max} value from the CSV."
        ),
    ),
    "dipole": FamilyTableSpec(
        family="dipole",
        metric_heading=r"$k_{\perp,\max(\pm)}$",
        label="tab:dipole_enhancement",
        caption=(
            r"Enhancement summary for homotypic dipole crossings. "
            r"The final column is the \texttt{mixed\_max} value from the CSV."
        ),
    ),
    "quadrupole": FamilyTableSpec(
        family="quadrupole",
        metric_heading=r"$K_{\max(\pm)}$",
        label="tab:quadrupole_enhancement",
        caption=(
            r"Enhancement summary for homotypic quadrupole crossings. "
            r"The final column is the \texttt{mixed\_max} value from the CSV."
        ),
    ),
}


def find_col(df: pd.DataFrame, candidates: list[str], *, required: bool = False) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")
    return None


def fmt_float(x: object, ndp: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return "--"
    if not np.isfinite(xf):
        return "--"
    return f"{xf:.{ndp}f}"


def fmt_sci(x: object, sig: int = 3) -> str:
    try:
        xf = float(x)
    except Exception:
        return "--"
    if not np.isfinite(xf):
        return "--"
    return f"{xf:.{sig}e}"


def mode_to_latex(mode: object) -> str:
    """
    Convert mode labels such as TM011, TM_011, or 011 into $TM_{011}$.
    If the family prefix is missing, TM is assumed.
    """
    if mode is None or (isinstance(mode, float) and math.isnan(mode)):
        return "--"

    s = str(mode).strip()

    m = re.search(
        r"(TM|TE)[_\s-]*([0-9]{3,})",
        s,
        flags=re.IGNORECASE,
    )

    if m:
        fam = m.group(1).upper()
        idx = m.group(2).zfill(3)
        return rf"${fam}_{{{idx}}}$"

    m = re.search(r"\b(\d{3,})\b", s)
    if m:
        return rf"$TM_{{{m.group(1)}}}$"

    return s.replace("_", r"\_")

def get_metric(row: pd.Series, df: pd.DataFrame) -> str:
    col = find_col(df, ["metric"], required=True)
    return str(row[col])

def get_modes(row: pd.Series, df: pd.DataFrame) -> tuple[str, str]:
    """
    Read parent modes directly from the enhancement summary.

    Expected columns:
        mode_i
        mode_j

    Examples
    --------
    TM011  -> $TM_{011}$
    TM_011 -> $TM_{011}$
    011    -> $TM_{011}$
    """

    col_i = find_col(df, ["mode_i"], required=True)
    col_j = find_col(df, ["mode_j"], required=True)

    mode_i = str(row[col_i]).strip()
    mode_j = str(row[col_j]).strip()

    return mode_to_latex(mode_i), mode_to_latex(mode_j)


def get_length_factor(row: pd.Series, df: pd.DataFrame) -> float:
    col = find_col(df, ["length_factor", "ell", "l_factor", "LF"], required=True)
    return float(row[col])


def get_f_hat(row: pd.Series, df: pd.DataFrame, f010_hz: float) -> float:
    col = find_col(df, ["f_hat", "fhat", "frequency_normalised", "frequency_normalized"])
    if col:
        return float(row[col])

    col = find_col(df, ["frequency_Hz", "frequency_hz", "freq_Hz", "freq_hz"], required=True)
    return float(row[col]) / float(f010_hz)


def get_rmax(row: pd.Series, df: pd.DataFrame) -> float:
    col = find_col(df, ["R_max", "Rmax", "r_max", "enhancement", "enhancement_ratio"], required=True)
    return float(row[col])


def get_mixed_max(row: pd.Series, df: pd.DataFrame) -> float:
    col = find_col(df, ["mixed_max"], required=True)
    return float(row[col])


def load_rows_for_family(
    csv_path: Path,
    *,
    f010_hz: float,
    top_n: int | None = None,
) -> list[list[str]]:
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        E1, E2 = get_modes(row, df)
        ell = get_length_factor(row, df)
        fhat = get_f_hat(row, df, f010_hz)
        rmax = get_rmax(row, df)
        mixed_max = get_mixed_max(row, df)

        if "quadrupole" in str(csv_path).lower():

            metric = get_metric(row, df)

            cells = [
                E1,
                E2,
                fmt_float(ell, 4),
                fmt_float(fhat, 4),
                metric_to_latex(metric),
                fmt_float(rmax, 3),
                fmt_sci(mixed_max, 3),
            ]

        else:

            cells = [
                E1,
                E2,
                fmt_float(ell, 4),
                fmt_float(fhat, 4),
                fmt_float(rmax, 3),
                fmt_sci(mixed_max, 3),
            ]

        rows.append(
            {
                "sort_Rmax": rmax,
                "cells": cells,
            }
        )

    rows.sort(key=lambda d: -d["sort_Rmax"] if np.isfinite(d["sort_Rmax"]) else np.inf)

    if top_n is not None:
        rows = rows[:top_n]

    return [r["cells"] for r in rows]

def metric_to_latex(metric: str) -> str:

    mapping = {
        "kxx": r"$K_{xx}$",
        "kyy": r"$K_{yy}$",
        "kxy": r"$K_{xy}$",
        "K_matrix_norm": r"$K_Q$",
        "K_max_component": r"$K_{\max}$",
    }

    return mapping.get(metric, metric)

def latex_table(spec: FamilyTableSpec, rows: list[list[str]]) -> str:
    lines = []
    lines.append(r"\begin{table*}")
    lines.append(rf"\caption{{\label{{{spec.label}}}{spec.caption}}}")
    lines.append(r"\begin{ruledtabular}")
    if spec.family == "quadrupole":
        lines.append(r"\begin{tabular}{ccccccc}")
        lines.append(
            r"$E_1$ & $E_2$ & $\ell$ & $\hat{f}$ & metric & $R_{\max}$ & "
            + spec.metric_heading
            + r" \\"
        )
    else:
        lines.append(r"\begin{tabular}{cccccc}")
        lines.append(
            r"$E_1$ & $E_2$ & $\ell$ & $\hat{f}$ & $R_{\max}$ & "
            + spec.metric_heading
            + r" \\"
        )

    lines.append(r"\hline")
    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{ruledtabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_all_tables(
    input_files: dict[str, Path],
    out_tex: Path,
    *,
    f010_hz: float = F_010_HZ,
    top_n_per_family: int | None = TOP_N_PER_FAMILY,
):
    blocks = []

    for family in ("monopole", "dipole", "quadrupole"):
        path = input_files[family]
        if not path.exists():
            print(f"Skipping {family}: file not found: {path}")
            continue

        spec = TABLE_SPECS[family]
        rows = load_rows_for_family(path, f010_hz=f010_hz, top_n=top_n_per_family)
        blocks.append(latex_table(spec, rows))

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    build_all_tables(INPUT_FILES, OUT_TEX)
