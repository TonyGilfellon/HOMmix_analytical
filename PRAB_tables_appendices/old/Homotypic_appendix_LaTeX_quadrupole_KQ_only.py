from __future__ import annotations

import math
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

F_010_HZ = 1.3e9

DEFAULT_ANALYSIS_ROOT = Path(r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles")
DEFAULT_INPUT_PKL = DEFAULT_ANALYSIS_ROOT / "all_crossing_analyses.pkl"
DEFAULT_PRAB_ROOT = Path(r"D:\PhD\PRAB")
DEFAULT_OUT_TEX = DEFAULT_PRAB_ROOT / "appendix_I_TM_homotypic_quadrupole_mixing.tex"
DEFAULT_FIGS_DIR = DEFAULT_PRAB_ROOT / "figs"


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


def fmt_float(x: object, ndp: int = 3) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{ndp}f}"


def fmt_sci(x: object, sig: int = 3) -> str:
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def mode_to_latex(mode: object) -> str:
    if mode is None:
        return "--"
    s = str(mode).strip()
    m = re.search(r"(TM|TE)[_\s-]*([0-9]{3,})", s, flags=re.IGNORECASE)
    if m:
        return rf"$\mathrm{{{m.group(1).upper()}_{{{m.group(2).zfill(3)}}}}}$"
    m = re.search(r"\b(\d{3,})\b", s)
    if m:
        return rf"$\mathrm{{TM_{{{m.group(1)}}}}}$"
    return s.replace("_", r"\_")


def latex_label_safe(s: object) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
    return out or "crossing"


def field_KQ_value(focusing: dict[str, Any], name: str) -> float:
    r = focusing.get(name, {})
    for key in (
        "K_Q_V_per_pC_per_m3",
        "K_quad_strength_V_per_pC_per_m3",
        "KQ_V_per_pC_per_m3",
    ):
        if key in r:
            return abs(finite_or_nan(r[key]))
    return float("nan")


def KQ_row(analysis: dict[str, Any]) -> dict[str, float]:
    focusing = analysis["focusing"]
    vals = {name: field_KQ_value(focusing, name) for name in ("E1", "E2", "plus", "minus")}
    vals["R_max"] = safe_ratio(max(vals["plus"], vals["minus"]), max(vals["E1"], vals["E2"]))
    return vals


def crossing_short_title(analysis: dict[str, Any]) -> str:
    c = analysis["crossing"]
    mode_i = mode_to_latex(analysis.get("mode_i", c.get("mode_i")))
    mode_j = mode_to_latex(analysis.get("mode_j", c.get("mode_j")))
    ell = float(c["length_factor"])
    fhat = float(c["frequency_Hz"]) / F_010_HZ
    return rf"Homotypic quadrupole crossing {mode_i}--{mode_j}, $\ell={ell:.4f}$, $\hat{{f}}={fhat:.4f}$"


def crossing_folder(analysis: dict[str, Any], analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT) -> Path:
    files = analysis.get("files", {})
    if "field_data_npz" in files:
        return Path(files["field_data_npz"]).parent
    mode_i = str(analysis.get("mode_i", analysis.get("crossing", {}).get("mode_i", "E1"))).replace("_", "")
    mode_j = str(analysis.get("mode_j", analysis.get("crossing", {}).get("mode_j", "E2"))).replace("_", "")
    if not mode_i.startswith("TM"):
        mode_i = "TM" + mode_i
    if not mode_j.startswith("TM"):
        mode_j = "TM" + mode_j
    return Path(analysis_root) / f"{mode_i}_{mode_j}"


def source_summary_pdf(analysis: dict[str, Any]) -> Path:
    folder = crossing_folder(analysis)
    pdf = analysis.get("files", {}).get("merged_slice_pdf")
    if pdf:
        return Path(pdf)
    return folder / "slice_summary_pdfs" / f"{folder.name}_field_summary.pdf"


def prab_pdf_name(analysis: dict[str, Any]) -> str:
    c = analysis.get("crossing", {})
    mode_i = str(analysis.get("mode_i", c.get("mode_i", "E1"))).replace("_", "")
    mode_j = str(analysis.get("mode_j", c.get("mode_j", "E2"))).replace("_", "")
    if not mode_i.startswith("TM"):
        mode_i = "TM" + mode_i
    if not mode_j.startswith("TM"):
        mode_j = "TM" + mode_j
    return f"homotypic_quadrupole_{mode_i}__{mode_j}__field_summary.pdf"


def copy_summary_pdfs_to_prab_figs(
    analyses: dict[str, Any],
    dest_dir: str | Path = DEFAULT_FIGS_DIR,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    missing: list[Path] = []
    for key, analysis in sorted_analyses(analyses):
        src = source_summary_pdf(analysis)
        dst = dest_dir / prab_pdf_name(analysis)
        if not src.exists():
            missing.append(src)
            continue
        if dst.exists() and not overwrite:
            copied[key] = dst
            continue
        shutil.copy2(src, dst)
        copied[key] = dst
        print(f"Copied: {src} -> {dst}")
    if missing:
        print("\nMissing homotypic quadrupole summary PDFs:")
        for src in missing:
            print(f"  {src}")
    return copied


def latex_table_for_crossing(analysis: dict[str, Any]) -> str:
    row = KQ_row(analysis)
    cells = [
        r"$K_Q$",
        r"$\mathrm{V/pC/m^3}$",
        fmt_sci(row["E1"]),
        fmt_sci(row["E2"]),
        fmt_sci(row["plus"]),
        fmt_sci(row["minus"]),
        fmt_float(row["R_max"], 3),
    ]
    return "\n".join([
        r"\begin{center}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3.6pt}",
        rf"\textbf{{{crossing_short_title(analysis)}}}\\[0.15em]",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{ccccccc}",
        r"Metric & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{\max}$ \\",
        r"\hline",
        " & ".join(cells) + r" \\",
        r"\end{tabular}",
        r"\end{ruledtabular}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\end{center}",
    ])


def single_crossing_summary_figure(
    analysis: dict[str, Any],
    *,
    image_width: str = "0.54\\textwidth",
    image_height: str = "0.68\\textheight",
    figs_dir_latex: str = "figs",
) -> str:
    pdf_path = f"{figs_dir_latex}/{prab_pdf_name(analysis)}"
    return rf"""
\vspace{{-0.8em}}
\begin{{center}}
\includegraphics[width={image_width},height={image_height},keepaspectratio]{{{pdf_path}}}
\end{{center}}
\vspace{{-0.6em}}
""".strip()


def single_crossing_page(
    analysis: dict[str, Any],
    *,
    image_width: str = "0.54\\textwidth",
    image_height: str = "0.68\\textheight",
    figs_dir_latex: str = "figs",
    clearpage: bool = True,
) -> str:
    page = latex_table_for_crossing(analysis) + "\n\n" + single_crossing_summary_figure(
        analysis,
        image_width=image_width,
        image_height=image_height,
        figs_dir_latex=figs_dir_latex,
    )
    if clearpage:
        page += "\n\n\\clearpage"
    return page


def appendix_start() -> str:
    return r"""
\clearpage
\section{TM homotypic quadrupole mixing}
\label{app:tm_homotypic_quadrupole_mixing}

For the homotypic quadrupole--quadrupole crossings, the scalar quadrupole strength $K_Q$ is reported for each parent and mixed field. The value is obtained from the azimuthal RF-multipole extraction and reported in the same $\mathrm{V/pC/m^3}$ normalisation used for the mode-mixing figures of merit.

\clearpage
""".strip()


def appendix_end() -> str:
    return r"% End of TM homotypic quadrupole mixing appendix"


def sorted_analyses(analyses: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def keyfunc(item: tuple[str, dict[str, Any]]):
        _, a = item
        c = a.get("crossing", {})
        return (str(a.get("mode_i", "")), str(a.get("mode_j", "")), float(c.get("length_factor", np.inf)))
    return sorted(analyses.items(), key=keyfunc)


def write_tm_homotypic_quadrupole_appendix(
    *,
    input_pkl: str | Path = DEFAULT_INPUT_PKL,
    out_tex: str | Path = DEFAULT_OUT_TEX,
    figs_dest_dir: str | Path = DEFAULT_FIGS_DIR,
    figs_dir_latex: str = "figs",
    image_width: str = "0.54\\textwidth",
    image_height: str = "0.68\\textheight",
    copy_figures: bool = True,
    top_n: int | None = None,
) -> Path:
    analyses = pickle_load(input_pkl)
    if copy_figures:
        copy_summary_pdfs_to_prab_figs(analyses, figs_dest_dir)
    items = sorted_analyses(analyses)
    if top_n is not None:
        items = items[:top_n]
    pages = [
        single_crossing_page(
            analysis,
            image_width=image_width,
            image_height=image_height,
            figs_dir_latex=figs_dir_latex,
            clearpage=True,
        )
        for _, analysis in items
    ]
    tex = "\n\n".join([appendix_start(), *pages, appendix_end()])
    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")
    print(f"Wrote {out_tex}")
    print(f"  homotypic quadrupole entries: {len(items)}")
    return out_tex


if __name__ == "__main__":
    write_tm_homotypic_quadrupole_appendix()
