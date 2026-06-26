from __future__ import annotations

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

def latex_num(x: float | None, precision: int = 4) -> str:
    if x is None:
        return "--"
    return f"{float(x):.{precision}g}"


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
        figs/homotypic_quadrupole_TM212_TM220_field_summary.pdf
    """
    mode_i = row["mode_i"].replace("_", "")
    mode_j = row["mode_j"].replace("_", "")
    return f"{figs_dir}/homotypic_{class_key}_{mode_i}_{mode_j}_field_summary.pdf"


def appendix_i_start() -> str:
    """
    Start Appendix I on its own page.

    Requires this in the main manuscript preamble:
        \\usepackage{lipsum}
    """
    return r"""
\clearpage
\appendix
\section{TM homotypic mixing}
\label{app:tm_homotypic_mixing}

\lipsum[1-2]

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

        loss_E1 = float(analysis["E1"]["loss"])
        loss_E2 = float(analysis["E2"]["loss"])
        loss_Eplus = float(analysis["plus"]["loss"])
        loss_Eminus = float(analysis["minus"]["loss"])

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
                    "effect": r"$k_{\parallel}$",
                    "units": r"V\,pC$^{-1}$\,m$^{-1}$",
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

        k_E1 = float(kicks["E1"]["kick_V_per_C_per_m_per_m"])
        k_E2 = float(kicks["E2"]["kick_V_per_C_per_m_per_m"])
        k_Eplus = float(kicks["plus"]["kick_V_per_C_per_m_per_m"])
        k_Eminus = float(kicks["minus"]["kick_V_per_C_per_m_per_m"])

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
                    "effect": r"$k_{\perp}$",
                    "units": r"V\,C$^{-1}$\,m$^{-2}$",
                    "E1": k_E1,
                    "E2": k_E2,
                    "Eplus": k_Eplus,
                    "Eminus": k_Eminus,
                    "R_max": _ratio_max(k_Eplus, k_Eminus, k_E1, k_E2),
                }
            ],
        }

    return rows


def _focus_value(result: dict, key: str) -> float:
    """
    Helper for quadrupole dictionaries, allowing for future minor naming changes.
    """
    candidates = [
        f"{key}_V_per_C_per_m_per_m",
        key,
    ]
    for candidate in candidates:
        if candidate in result:
            return float(result[candidate])
    raise KeyError(f"Could not find {key} in quadrupole focusing result.")


def load_quadrupole_rows(quad_path: str | Path) -> dict[str, dict]:
    """
    Load homotypic quadrupole crossing entries from all_crossing_analyses.pkl.
    Expected metrics are item["focusing"][E1/E2/plus/minus] with Kxx, Kxy and Kyy.
    """
    data = pickle_load(Path(quad_path) / "all_crossing_analyses.pkl")
    items = data.values() if isinstance(data, dict) else data
    rows: dict[str, dict] = {}

    for item in items:
        crossing, mode_i, mode_j = _crossing_metadata(item)
        focusing = item["focusing"]

        metrics = []
        for effect_key, effect_tex in [
            ("Kxx", r"$K_{xx}$"),
            ("Kyy", r"$K_{yy}$"),
            ("Kxy", r"$K_{xy}$"),
        ]:
            E1 = _focus_value(focusing["E1"], effect_key)
            E2 = _focus_value(focusing["E2"], effect_key)
            Eplus = _focus_value(focusing["plus"], effect_key)
            Eminus = _focus_value(focusing["minus"], effect_key)

            metrics.append(
                {
                    "effect": effect_tex,
                    "units": r"V\,C$^{-1}$\,m$^{-2}$",
                    "E1": E1,
                    "E2": E2,
                    "Eplus": Eplus,
                    "Eminus": Eminus,
                    "R_max": _ratio_max(Eplus, Eminus, E1, E2),
                }
            )

        row_key = f"{mode_i}-{mode_j}"

        rows[row_key] = {
            "class_key": "quadrupole",
            "class_label": "quadrupole",
            "mode_i": mode_i,
            "mode_j": mode_j,
            "length_factor": float(crossing["length_factor"]),
            "frequency_Hz": float(crossing["frequency_Hz"]),
            "frequency_normalised": float(crossing["frequency_Hz"])/1.3e9,
            "metrics": metrics,
        }

    return rows


# -----------------------------------------------------------------------------
# LaTeX table and figure generation
# -----------------------------------------------------------------------------

def single_crossing_metric_table(row: dict) -> str:
    """
    Generic table for mono-, di- and quadrupole entries.

    The title carries the mode names, length factor and normalised frequency.
    The rows carry the metric values.
    """
    mode_i = row["mode_i"]
    mode_j = row["mode_j"]
    class_label = row.get("class_label", row.get("class_key", "mode"))

    metric_rows = []
    for metric in row["metrics"]:
        metric_rows.append(
            rf"{metric['effect']} "
            rf"& {metric['units']} "
            rf"& {latex_num(metric['E1'])} "
            rf"& {latex_num(metric['E2'])} "
            rf"& {latex_num(metric['Eplus'])} "
            rf"& {latex_num(metric['Eminus'])} "
            rf"& {metric['R_max']:.2f} \\"
        )

    return rf"""
\begin{{center}}
\small
\renewcommand{{\arraystretch}}{{1.35}}
\setlength{{\tabcolsep}}{{6pt}}
\textbf{{Homotypic {class_label} crossing for ${latex_mode(mode_i)}$ and ${latex_mode(mode_j)}$, $\ell = {row['length_factor']:.4f}$, $\hat{{f}} = {row['frequency_normalised']:.4f}$}}\\[0.35em]
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccccc}}
Effect & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{{\max}}$ \\
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
) -> str:
    """
    Include the single-page tall summary PDF generated by
    save_four_slice_pdfs_and_merge().
    """
    pdf_path = prab_summary_pdf_path(row, class_key=row["class_key"])

    return rf"""
\vspace{{-1.0em}}
\begin{{center}}
\includegraphics[width={image_width}]{{{pdf_path}}}
\end{{center}}
\vspace{{-0.6em}}
""".strip()


def single_crossing_page(
    row: dict,
    clearpage: bool = True,

) -> str:
    """
    One crossing per page: table first, then the summary PDF.
    """
    if row["class_key"] == "quadrupole":
        image_width = "0.58\\textwidth"
    elif row["class_key"] == "dipole":
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
            rf"& {R_max:.2f} \\"
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
