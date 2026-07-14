from __future__ import annotations

import math
import pickle as pkl
from pathlib import Path
from typing import Any, Iterable


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------

def pickle_save(data_dict: Any, dir_fname: str | Path) -> None:
    dir_fname = Path(dir_fname)
    dir_fname.parent.mkdir(parents=True, exist_ok=True)
    with dir_fname.open("wb") as handle:
        pkl.dump(data_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(dir_fname: str | Path) -> Any:
    with Path(dir_fname).open("rb") as handle:
        return pkl.load(handle)


# -----------------------------------------------------------------------------
# LaTeX formatting helpers
# -----------------------------------------------------------------------------

def finite_or_nan(x: object) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def latex_num(x: float | None, sig: int = 3) -> str:
    """
    Return a LaTeX number for table cells, matching the heterotypic appendix
    style: $a\\times10^{b}$ for nonzero values, 0 for zero, -- for NaN.
    """
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"

    s = f"{x:.{sig - 1}e}"
    mantissa, exponent = s.split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def latex_ratio(x: object, ndp: int = 3) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{ndp}f}"


def per_C_to_per_pC(x: object) -> float:
    """
    Convert values normalised per coulomb to values normalised per pC.

    1 C = 10^12 pC, so V/C/... -> V/pC/... by dividing by 10^12.
    """
    return finite_or_nan(x) / 1.0e12


def latex_mode(mode: str) -> str:
    """
    Convert TM_012, TM012 or 012 -> \\mathrm{TM_{012}}.
    """
    s = str(mode)
    if "_" in s:
        family, indices = s.split("_", 1)
    elif s[:2].upper() in {"TM", "TE"}:
        family, indices = s[:2], s[2:]
    else:
        family, indices = "TM", s
    return rf"\mathrm{{{family.upper()}_{{{indices}}}}}"


def normalise_mode_name(mode: str, family: str = "TM") -> str:
    """
    Convert 012 or TM012 to TM_012, leaving TM_012 unchanged.
    """
    s = str(mode)
    if "_" in s:
        fam, mnp = s.split("_", 1)
        return f"{fam.upper()}_{mnp}"
    if s[:2].upper() in {"TM", "TE"}:
        return f"{s[:2].upper()}_{s[2:]}"
    return f"{family.upper()}_{s}"


def prab_summary_pdf_path(row: dict, class_key: str, figs_dir: str = "figs") -> str:
    """
    Return the copied PRAB figure path, e.g.
        figs/homotypic_monopole_TM012_TM020_field_summary.pdf
        figs/homotypic_dipole_TM112_TM120_field_summary.pdf
        figs/homotypic_quadrupole_TM213_TM222_field_summary.pdf

    Deliberately use a single underscore between the two mode names.  Do not use
    the double-underscore convention used by some heterotypic scripts.
    """
    mode_i = row["mode_i"].replace("_", "")
    mode_j = row["mode_j"].replace("_", "")
    return f"{figs_dir}/homotypic_{class_key}_{mode_i}_{mode_j}_field_summary.pdf"

def appendix_i_start() -> str:
    """
    Start Appendix I on its own page.

    Requires this in the main manuscript preamble:
        \\usepackage{lipsum}
        \\usepackage{needspace}
        \\usepackage{placeins}
    """
    return r"""
\clearpage
\appendix
\section{TM homotypic mixing}
\label{app:tm_homotypic_mixing}

\lipsum[1-2]

The tables report beam-dynamics metrics derived from the field distributions: longitudinal loss factor $k_{\parallel}$ for monopole--monopole crossings, dipole kick factor $k_{\perp}$ for dipole--dipole crossings, and scalar quadrupole strength $K_Q$ for quadrupole--quadrupole crossings.  For the homotypic quadrupole cases, $K_Q$ is obtained from the azimuthal RF-multipole extraction and reported in the same $\mathrm{V/pC/m^3}$ normalisation used for the mode-mixing figures of merit.

\clearpage
""".strip()


def appendix_i_end() -> str:
    return r"""
% End of Appendix I: TM homotypic mixing
""".strip()


# -----------------------------------------------------------------------------
# Row extraction from the three homotypic analysis pickle formats
# -----------------------------------------------------------------------------

def _ratio_max(mixed_a: float, mixed_b: float, parent_a: float, parent_b: float) -> float:
    parent_ref = max(abs(float(parent_a)), abs(float(parent_b)))
    if parent_ref <= 0.0:
        return float("nan")
    return max(abs(float(mixed_a)), abs(float(mixed_b))) / parent_ref


def _crossing_metadata(item: dict) -> tuple[dict, str, str]:
    """
    Return crossing dict, mode_i and mode_j as TM_XXX strings.

    Handles:
      - monopole entries with item["modes"]["E1"], item["modes"]["E2"]
      - dipole/quadrupole entries with item["mode_i"], item["mode_j"]
      - all entries with item["crossing"]
    """
    crossing = item.get("crossing", {})

    if "modes" in item:
        mode_i = item["modes"].get("E1", crossing.get("mode_i"))
        mode_j = item["modes"].get("E2", crossing.get("mode_j"))
    else:
        mode_i = item.get("mode_i", crossing.get("mode_i"))
        mode_j = item.get("mode_j", crossing.get("mode_j"))

    mode_i = normalise_mode_name(mode_i)
    mode_j = normalise_mode_name(mode_j)

    return crossing, mode_i, mode_j


def first_finite_from_dicts(dicts: Iterable[dict], keys: Iterable[str]) -> float:
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for key in keys:
            if key in d:
                val = finite_or_nan(d[key])
                if math.isfinite(val) and val > 0.0:
                    return val
    return float("nan")


def crossing_length_m(item: dict, crossing: dict) -> float:
    """
    Best-effort physical length used to convert V/pC loss to V/pC/m.

    If no length is present in older homotypic outputs, the loss is left
    unchanged rather than an arbitrary length being invented.
    """
    analysis = item.get("analysis", {})
    fields = item.get("fields", {})

    candidates = [
        item,
        crossing,
        analysis.get("E1", {}) if isinstance(analysis, dict) else {},
        fields.get("E1", {}) if isinstance(fields, dict) else {},
    ]

    return first_finite_from_dicts(
        candidates,
        (
            "length_m",
            "L_m",
            "cavity_length_m",
            "physical_length_m",
            "analysis_length_m",
        ),
    )


def loss_to_v_per_pc_per_m(loss_value: object, item: dict, crossing: dict) -> float:
    """
    Convert a stored monopole loss to V/pC/m when possible.

    Newer pipelines store V/pC and a physical length separately; in that case
    divide by length. If no length is available, preserve the stored value.
    """
    loss = finite_or_nan(loss_value)
    if not math.isfinite(loss):
        return float("nan")

    length_m = crossing_length_m(item, crossing)
    if math.isfinite(length_m) and length_m > 0.0:
        return abs(loss) / length_m

    return abs(loss)



def load_monopole_rows(mono_path: str | Path) -> dict[str, dict]:
    """
    Load homotypic monopole crossing entries from all_crossing_analyses.pkl.
    Expected metrics are item["analysis"][E1/E2/plus/minus]["loss"].
    """
    data = pickle_load(Path(mono_path) / "all_crossing_analyses.pkl")
    rows: dict[str, dict] = {}

    for item in data:
        crossing, mode_i, mode_j = _crossing_metadata(item)
        analysis = item["analysis"]

        loss_E1 = loss_to_v_per_pc_per_m(analysis["E1"]["loss"], item, crossing)
        loss_E2 = loss_to_v_per_pc_per_m(analysis["E2"]["loss"], item, crossing)
        loss_Eplus = loss_to_v_per_pc_per_m(analysis["plus"]["loss"], item, crossing)
        loss_Eminus = loss_to_v_per_pc_per_m(analysis["minus"]["loss"], item, crossing)

        row_key = f"{mode_i}-{mode_j}"

        rows[row_key] = {
            "class_key": "monopole",
            "class_label": "monopole",
            "mode_i": mode_i,
            "mode_j": mode_j,
            "length_factor": float(crossing["length_factor"]),
            "frequency_Hz": float(crossing["frequency_Hz"]),
            "frequency_normalised": float(crossing["frequency_Hz"])/1.3e9,
            "metrics": [
                {
                    "metric": r"$k_{\parallel}$",
                    "effect": r"$k_{\parallel}$",
                    "units": r"$\mathrm{V/pC/m}$",
                    "E1": loss_E1,
                    "E2": loss_E2,
                    "Eplus": loss_Eplus,
                    "Eminus": loss_Eminus,
                    "R_max": _ratio_max(loss_Eplus, loss_Eminus, loss_E1, loss_E2),
                }
            ],
            # Backwards-compatible names for the original one-line monopole table.
            "loss_E1": loss_E1,
            "loss_E2": loss_E2,
            "loss_Eplus": loss_Eplus,
            "loss_Eminus": loss_Eminus,
            "R_max": _ratio_max(loss_Eplus, loss_Eminus, loss_E1, loss_E2),
        }

    return rows


def load_dipole_rows(di_path: str | Path) -> dict[str, dict]:
    """
    Load homotypic dipole crossing entries from all_crossing_analyses.pkl.
    Expected metrics are item["kicks"][E1/E2/plus/minus]["kick_V_per_C_per_m_per_m"].
    """
    data = pickle_load(Path(di_path) / "all_crossing_analyses.pkl")
    items = data.values() if isinstance(data, dict) else data
    rows: dict[str, dict] = {}

    for item in items:
        crossing, mode_i, mode_j = _crossing_metadata(item)
        kicks = item["kicks"]

        k_E1 = per_C_to_per_pC(kicks["E1"]["kick_V_per_C_per_m_per_m"])
        k_E2 = per_C_to_per_pC(kicks["E2"]["kick_V_per_C_per_m_per_m"])
        k_Eplus = per_C_to_per_pC(kicks["plus"]["kick_V_per_C_per_m_per_m"])
        k_Eminus = per_C_to_per_pC(kicks["minus"]["kick_V_per_C_per_m_per_m"])

        row_key = f"{mode_i}-{mode_j}"

        rows[row_key] = {
            "class_key": "dipole",
            "class_label": "dipole",
            "mode_i": mode_i,
            "mode_j": mode_j,
            "length_factor": float(crossing["length_factor"]),
            "frequency_Hz": float(crossing["frequency_Hz"]),
            "frequency_normalised": float(crossing["frequency_Hz"])/1.3e9,
            "metrics": [
                {
                    "metric": r"$k_{\perp}$",
                    "effect": r"$k_{\perp}$",
                    "units": r"$\mathrm{V/pC/m^2}$",
                    "E1": k_E1,
                    "E2": k_E2,
                    "Eplus": k_Eplus,
                    "Eminus": k_Eminus,
                    "R_max": _ratio_max(k_Eplus, k_Eminus, k_E1, k_E2),
                }
            ],
        }

    return rows


def _focus_value(result: dict, key: str, *, signed: bool = True) -> float:
    """Return a quadrupole table value in V/pC/m^3.

    The current homotypic quadrupole workflow reports only the positive scalar
    strength K_Q from the Brett-style azimuthal RF-multipole extraction.  This
    loader remains tolerant of older Hessian/signed-K outputs by accepting the
    historical K_quad_strength aliases.
    """
    alias_keys = [key]
    if key == "K_Q":
        alias_keys += ["KQ", "K_quad_strength", "K_quad"]

    per_pc_candidates: list[str] = []
    per_c_candidates: list[str] = []
    for alias in alias_keys:
        per_pc_candidates.extend([
            f"{alias}_V_per_pC_per_m3",
            f"{alias}_V_per_pC_per_m_per_m_per_m",
            alias,
        ])
        per_c_candidates.extend([
            f"{alias}_V_per_C_per_m3",
            f"{alias}_V_per_C_per_m_per_m",
        ])

    for candidate in per_pc_candidates:
        if candidate in result:
            val = finite_or_nan(result[candidate])
            return val if signed else abs(val)

    for candidate in per_c_candidates:
        if candidate in result:
            val = per_C_to_per_pC(result[candidate])
            return val if signed else abs(val)

    raise KeyError(f"Could not find {key} in quadrupole focusing result.")

def _quadrupole_metric_values(focusing: dict, key: str, *, signed: bool) -> tuple[float, float, float, float]:
    return (
        _focus_value(focusing["E1"], key, signed=signed),
        _focus_value(focusing["E2"], key, signed=signed),
        _focus_value(focusing["plus"], key, signed=signed),
        _focus_value(focusing["minus"], key, signed=signed),
    )

def load_quadrupole_rows(quad_path: str | Path) -> dict[str, dict]:
    """
    Load homotypic quadrupole crossing entries from all_crossing_analyses.pkl.

    The homotypic quadrupole appendix now reports a single scalar row only:
    K_Q in V/pC/m^3 for E1, E2, E+ and E-.  This matches the Brett-style
    azimuthal RF-multipole quadrupole analysis while remaining compatible with
    older outputs that stored K_Q under K_quad_strength.
    """
    data = pickle_load(Path(quad_path) / "all_crossing_analyses.pkl")
    items = data.values() if isinstance(data, dict) else data
    rows: dict[str, dict] = {}

    for item in items:
        crossing, mode_i, mode_j = _crossing_metadata(item)
        focusing = item["focusing"]

        E1, E2, Eplus, Eminus = _quadrupole_metric_values(
            focusing,
            "K_Q",
            signed=False,
        )

        row_key = f"{mode_i}-{mode_j}"
        rows[row_key] = {
            "class_key": "quadrupole",
            "class_label": "quadrupole",
            "mode_i": mode_i,
            "mode_j": mode_j,
            "length_factor": float(crossing["length_factor"]),
            "frequency_Hz": float(crossing["frequency_Hz"]),
            "frequency_normalised": float(crossing["frequency_Hz"]) / 1.3e9,
            "metrics": [
                {
                    "metric": r"$K_Q$",
                    "effect": r"$K_Q$",
                    "units": r"$\mathrm{V/pC/m^3}$",
                    "E1": E1,
                    "E2": E2,
                    "Eplus": Eplus,
                    "Eminus": Eminus,
                    "R_max": _ratio_max(Eplus, Eminus, E1, E2),
                }
            ],
        }

    return rows

# -----------------------------------------------------------------------------
# LaTeX table and figure generation
# -----------------------------------------------------------------------------

def crossing_title(row: dict) -> str:
    mode_i = row["mode_i"]
    mode_j = row["mode_j"]
    class_label = row.get("class_label", row.get("class_key", "mode"))
    return (
        rf"Homotypic {class_label} crossing for ${latex_mode(mode_i)}$ and "
        rf"${latex_mode(mode_j)}$, $\ell = {row['length_factor']:.4f}$, "
        rf"$\hat{{f}} = {row['frequency_normalised']:.4f}$"
    )


def single_crossing_metric_table(row: dict, *, include_title: bool = True) -> str:
    """
    Generic table for mono-, di- and quadrupole entries.

    For quadrupole pages, include_title=False is used so the heading can be
    placed outside the centered table and wrapped with the table and figure.
    This avoids the heading/table overlap seen when a long title is placed in
    the center environment immediately above a ruledtabular.
    """
    metric_rows = []
    for metric in row["metrics"]:
        metric_rows.append(
            rf"{metric.get('metric', metric.get('effect'))} "
            rf"& {metric['units']} "
            rf"& {latex_num(metric['E1'])} "
            rf"& {latex_num(metric['E2'])} "
            rf"& {latex_num(metric['Eplus'])} "
            rf"& {latex_num(metric['Eminus'])} "
            rf"& {latex_ratio(metric['R_max'], 3)} \\"
        )

    title_line = ""
    if include_title:
        title_line = rf"\textbf{{{crossing_title(row)}}}\\[0.45em]"

    return rf"""
\begin{{center}}
\small
\renewcommand{{\arraystretch}}{{1.25}}
\setlength{{\tabcolsep}}{{4.5pt}}
{title_line}
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccccc}}
Metric & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{{\max}}$ \\
\hline
{chr(10).join(metric_rows)}
\end{{tabular}}
\end{{ruledtabular}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{center}}
""".strip()


def single_crossing_summary_figure(
    row: dict,
    image_width: str = "0.65\\textwidth",
    *,
    image_height: str | None = None,
) -> str:
    """
    Include the single-page summary PDF generated by save_four_slice_pdfs_and_merge().
    """
    pdf_path = prab_summary_pdf_path(row, class_key=row["class_key"])

    if image_height is None:
        include_opts = f"width={image_width}"
    else:
        include_opts = f"width={image_width},height={image_height},keepaspectratio"

    return rf"""
\begin{{center}}
\includegraphics[{include_opts}]{{{pdf_path}}}
\end{{center}}
""".strip()


def quadrupole_crossing_page(row: dict, clearpage: bool = True) -> str:
    """
    One homotypic quadrupole crossing per page.

    The heading, table and field-section PDF are kept in one samepage block.
    A clearpage before the block gives it a fresh page; Needspace protects
    against page breaks if the function is reused without the preceding clearpage.
    """

    title_tex = crossing_title(row)

    table_tex = single_crossing_metric_table(row, include_title=False)

    # figure_tex = single_crossing_summary_figure(
    #     row=row,
    #     image_width=r"0.98\textwidth",
    #     image_height=r"0.48\textheight",
    # )

    figure_tex = single_crossing_summary_figure(
        row=row,
        image_width=r"\textwidth",
        image_height=r"0.62\textheight",
    )

    page = rf"""
\clearpage
\Needspace{{0.92\textheight}}
\begin{{samepage}}
\noindent\textbf{{{title_tex}}}

\vspace{{0.45em}}

{table_tex}

\vspace{{0.45em}}

{figure_tex}
\end{{samepage}}
""".strip()

    if clearpage:
        page += "\n\n\\clearpage"

    return page

def single_crossing_page(row: dict, clearpage: bool = True) -> str:
    """
    One crossing per page: table first, then the summary PDF.
    Quadrupoles use a tighter, non-overlapping samepage layout.
    """
    if row["class_key"] == "quadrupole":
        return quadrupole_crossing_page(row=row, clearpage=clearpage)

    if row["class_key"] == "dipole":
        image_width = "0.65\\textwidth"
    else:  # monopole
        image_width = "0.65\\textwidth"

    page = (
        single_crossing_metric_table(row)
        + "\n\n"
        + single_crossing_summary_figure(row=row, image_width=image_width)
    )

    if clearpage:
        page += "\n\n\\clearpage"

    return page


def write_tm_homotypic_appendix(
    monopole_rows: dict[str, dict],
    dipole_rows: dict[str, dict],
    quadrupole_rows: dict[str, dict],
    out_dir: str | Path,
    filename: str = "appendix_I_TM_homotypic_mixing.tex",
    image_width: str = "0.65\\textwidth",
) -> Path:
    """
    Write Appendix I with a standalone appendix title/text page followed by
    one table + one summary PDF per crossing.

    Order:
        1. monopole crossings
        2. dipole crossings
        3. quadrupole crossings
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_rows: list[dict] = []
    ordered_rows.extend(row for _, row in sorted(monopole_rows.items()))
    ordered_rows.extend(row for _, row in sorted(dipole_rows.items()))
    ordered_rows.extend(row for _, row in sorted(quadrupole_rows.items()))

    pages = [
        single_crossing_page(row=row, clearpage=True)
        for row in ordered_rows
    ]

    tex = "\n\n".join([
        appendix_i_start(),
        *pages,
        appendix_i_end(),
    ])

    out_file = out_dir / filename
    out_file.write_text(tex, encoding="utf-8")

    print(f"Wrote {out_file}")
    print(f"  monopole rows:    {len(monopole_rows)}")
    print(f"  dipole rows:      {len(dipole_rows)}")
    print(f"  quadrupole rows:  {len(quadrupole_rows)}")
    print(f"  total entries:    {len(ordered_rows)}")

    return out_file


# -----------------------------------------------------------------------------
# Optional compact overview table for monopoles, retained from the original script
# -----------------------------------------------------------------------------

def monopole_loss_table(local_table_dict: dict) -> str:
    rows = []

    for _, row in local_table_dict.items():
        R_max = row["R_max"]

        rows.append(
            rf"{row['mode_i']}--{row['mode_j']} "
            rf"& {row['length_factor']:.4f} "
            rf"& {row['frequency_normalised']:.4f} "
            rf"& {latex_num(row['loss_E1'])} "
            rf"& {latex_num(row['loss_E2'])} "
            rf"& {latex_num(row['loss_Eplus'])} "
            rf"& {latex_num(row['loss_Eminus'])} "
            rf"& {latex_ratio(R_max, 3)} \\"
        )

    rows = "\n".join(rows)

    return rf"""
\begin{{center}}
\small
\textbf{{Homotypic monopole crossings}}\\[0.5em]
\begin{{ruledtabular}}
\begin{{tabular}}{{cccccccc}}
Modes & $\ell$ & $\hat{{f}}$ & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{{\max}}$ \\
\hline
{rows}
\end{{tabular}}
\end{{ruledtabular}}
\end{{center}}
""".strip()


def write_monopole_loss_table(
    local_table_dict: dict,
    out_dir: str | Path,
    filename: str = "homotypic_monopole_loss_table.tex",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tex = monopole_loss_table(local_table_dict)

    out_file = out_dir / filename
    out_file.write_text(tex, encoding="utf-8")

    print(f"Wrote {out_file}")
    return out_file


# -----------------------------------------------------------------------------
# Main script configuration
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis")
    OUT_PATH = Path(r"D:\PhD\PRAB")

    MONO_PATH = ROOT / "homotypic_monopoles"
    DI_PATH = ROOT / "homotypic_dipoles"
    QUAD_PATH = ROOT / "homotypic_quadrupoles"

    monopole_rows = load_monopole_rows(MONO_PATH)
    dipole_rows = load_dipole_rows(DI_PATH)
    quadrupole_rows = load_quadrupole_rows(QUAD_PATH)

    # Optional separate overview table retained from the original monopole-only script.
    write_monopole_loss_table(monopole_rows, OUT_PATH)

    write_tm_homotypic_appendix(
        monopole_rows=monopole_rows,
        dipole_rows=dipole_rows,
        quadrupole_rows=quadrupole_rows,
        out_dir=OUT_PATH,
        filename="appendix_I_TM_homotypic_mixing.tex",
        image_width="0.65\\textwidth",
    )
