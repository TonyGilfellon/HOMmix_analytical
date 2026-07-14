from __future__ import annotations

import math
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

F_010_HZ = 1.3e9

DEFAULT_ANALYSIS_ROOT = Path(
    r"D:\PhD\HOMmix\HOMmix_analytical\analysis\heterotypic_crossings"
)
DEFAULT_INPUT_PKL = DEFAULT_ANALYSIS_ROOT / "all_heterotypic_multipole_analyses.pkl"
DEFAULT_PRAB_ROOT = Path(r"D:\PhD\PRAB")
DEFAULT_OUT_TEX = DEFAULT_PRAB_ROOT / "appendix_II_TM_heterotypic_mixing.tex"
DEFAULT_FIGS_DIR = DEFAULT_PRAB_ROOT / "figs"


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------

def pickle_load(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Numeric / LaTeX helpers
# -----------------------------------------------------------------------------

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



def signed_real_finite_or_nan(x: object) -> float:
    """Return the real part with sign preserved, or nan.

    The diagnostic script saves U_CST-normalised K entries as real signed
    values.  This helper also tolerates complex fallbacks by taking their real
    part rather than their magnitude.
    """
    try:
        y = complex(x).real
    except Exception:
        return float("nan")
    return float(y) if math.isfinite(float(y)) else float("nan")


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
    """Return LaTeX scientific notation with sig significant figures."""
    x = finite_or_nan(x)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    s = f"{x:.{sig - 1}e}"
    mantissa, exponent = s.split("e")
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


def pair_type_to_text(pair_type: object) -> str:
    if pair_type is None:
        return "heterotypic"
    return str(pair_type).replace("_", "--")


def first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return float("nan")


# -----------------------------------------------------------------------------
# Effect extraction, matching heterotypic_prab_loss_kick_quad_tables.py
# -----------------------------------------------------------------------------
EFFECT_ROWS = [
    {
        "key": "loss",
        "label": r"$k_{\parallel}$",
        "units": r"$\mathrm{V/pC/m}$",
        "signed": False,
    },
    {
        "key": "kick",
        "label": r"$k_{\perp}$",
        "units": r"$\mathrm{V/pC/m^2}$",
        "signed": False,
    },
    {
        "key": "K_iso",
        "label": r"$K_{\mathrm{iso}}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "signed": True,
    },
    {
        "key": "K_Q",
        "label": r"$K_Q$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "signed": False,
    },
    {
        "key": "focusing",
        "label": r"$K_{xx}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "signed": True,
    },
    {
        "key": "defocusing",
        "label": r"$K_{yy}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "signed": True,
    },
    {
        "key": "skew",
        "label": r"$K_{xy}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "signed": True,
    },
]


def field_length_m(field_result: dict[str, Any]) -> float:
    return finite_or_nan(field_result.get("length_m", float("nan")))


def figures(field_result: dict[str, Any]) -> dict[str, Any]:
    return field_result.get("figures_of_merit", {})


def _kdiag_value(field_result: dict[str, Any], diag_key: str, value_key: str) -> float:
    try:
        return finite_or_nan(field_result.get("kparallel_diagnostics", {}).get(diag_key, {}).get(value_key))
    except Exception:
        return float("nan")


def loss_v_per_pc_per_m(field_result: dict[str, Any]) -> float:
    # Preferred: explicit U_CST-normalised value written by the diagnostic script.
    f = figures(field_result)
    direct = finite_or_nan(f.get("loss_like_V_per_pC_per_m"))
    if math.isfinite(direct):
        return abs(direct)

    # Fallback: U_CST k_parallel diagnostic from the fitted V0.
    direct = _kdiag_value(field_result, "fit_V0_U_CST", "k_V_per_pC_per_m")
    if math.isfinite(direct):
        return abs(direct)

    # Last-resort fallback: U_CST-normalised V/pC divided by length.
    loss_v_per_pc = finite_or_nan(f.get("loss_like_V_per_pC"))
    length_m = field_length_m(field_result)
    if not math.isfinite(loss_v_per_pc) or not math.isfinite(length_m) or length_m <= 0.0:
        return float("nan")
    return abs(loss_v_per_pc) / length_m


def kick_v_per_pc_per_m2(field_result: dict[str, Any]) -> float:
    # Preferred: U_CST-normalised dipole kick factor |V_perp|^2/(4U_CST).
    f = figures(field_result)
    return abs_finite_or_nan(
        first_present(
            f,
            (
                "kick_magnitude_V_per_pC_per_m2",
                "kick_mag_V_per_pC_per_m2",
            ),
        )
    )


def Kxx_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return signed_real_finite_or_nan(
        first_present(
            f,
            (
                "Kxx_V_per_pC_per_m3",
                "Kxx_U_CST_V_per_pC_per_m3",
            ),
        )
    )


def Kyy_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return signed_real_finite_or_nan(
        first_present(
            f,
            (
                "Kyy_V_per_pC_per_m3",
                "Kyy_U_CST_V_per_pC_per_m3",
            ),
        )
    )


def Kxy_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    f = figures(field_result)
    return signed_real_finite_or_nan(
        first_present(
            f,
            (
                "Kxy_V_per_pC_per_m3",
                "Kxy_U_CST_V_per_pC_per_m3",
            ),
        )
    )


def Kiso_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    Kxx = Kxx_v_per_pc_per_m3(field_result)
    Kyy = Kyy_v_per_pc_per_m3(field_result)
    if not math.isfinite(Kxx) or not math.isfinite(Kyy):
        return float("nan")
    return 0.5 * (Kxx + Kyy)


def KQ_v_per_pc_per_m3(field_result: dict[str, Any]) -> float:
    Kxx = Kxx_v_per_pc_per_m3(field_result)
    Kyy = Kyy_v_per_pc_per_m3(field_result)
    Kxy = Kxy_v_per_pc_per_m3(field_result)
    if not all(math.isfinite(v) for v in (Kxx, Kyy, Kxy)):
        return float("nan")
    return math.sqrt((Kxx - Kyy) ** 2 + 4.0 * Kxy ** 2)


def effect_value(field_result: dict[str, Any], effect_key: str) -> float:
    if effect_key == "loss":
        return loss_v_per_pc_per_m(field_result)
    if effect_key == "kick":
        # k_perp is a magnitude by definition in the current diagnostic output.
        # A physically signed dipole row would need a chosen transverse axis,
        # e.g. k_x or k_y, not |k_perp|.
        return kick_v_per_pc_per_m2(field_result)
    if effect_key == "K_iso":
        return Kiso_v_per_pc_per_m3(field_result)
    if effect_key == "K_Q":
        return KQ_v_per_pc_per_m3(field_result)
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

    # R_max is a size/enhancement metric, so compare magnitudes even for
    # signed rows such as K_iso, Kxx, Kyy and Kxy.
    parent_max = max(abs(vals["E1"]), abs(vals["E2"]))
    mixed_max = max(abs(vals["plus"]), abs(vals["minus"]))

    vals["parent_max"] = parent_max
    vals["mixed_max"] = mixed_max
    vals["R_max"] = safe_ratio(mixed_max, parent_max)
    return vals


# -----------------------------------------------------------------------------
# Figure path handling
# -----------------------------------------------------------------------------

def crossing_folder_name(result: dict[str, Any]) -> str:
    folder = result.get("crossing_folder")
    if folder:
        return Path(folder).name

    c = result.get("crossing", {})
    pair_type = result.get("pair_type") or c.get("pair_type", "heterotypic")
    mode_i = str(result.get("mode_i", c.get("mode_i", "E1"))).replace("_", "")
    mode_j = str(result.get("mode_j", c.get("mode_j", "E2"))).replace("_", "")
    ell = latex_label_safe(f"{float(c.get('length_factor', 0.0)):.8g}")
    return f"{latex_label_safe(pair_type)}__{mode_i}__{mode_j}__ell_{ell}"


def heterotypic_source_summary_pdf(result: dict[str, Any]) -> Path:
    folder = Path(result["crossing_folder"])
    return folder / "slice_summary_pdfs" / f"{folder.name}_field_summary.pdf"


def heterotypic_prab_pdf_name(result: dict[str, Any]) -> str:
    """
    Return the copied PRAB figure filename.

    Example
    -------
    heterotypic_dipole_quadrupole_TM_111__TM_210__field_summary.pdf
    """

    c = result.get("crossing", {})

    pair_type = (
        result.get("pair_type")
        or c.get("pair_type")
        or Path(result["crossing_folder"]).parent.name
    )

    pair_type = str(pair_type)

    mode_i = str(result.get("mode_i", c.get("mode_i"))).replace("-", "_")
    mode_j = str(result.get("mode_j", c.get("mode_j"))).replace("-", "_")

    return (
        f"heterotypic_{pair_type}"
        f"_{mode_i}__{mode_j}__field_summary.pdf"
    )


def heterotypic_prab_summary_pdf_path(
    result: dict[str, Any],
    figs_dir_latex: str = "figs",
) -> str:
    return f"{figs_dir_latex}/{heterotypic_prab_pdf_name(result)}"


def copy_heterotypic_summary_pdfs_to_prab_figs(
    results: dict[str, Any],
    dest_dir: str | Path = DEFAULT_FIGS_DIR,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Copy crossing-specific summary PDFs into the PRAB figure directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    missing: list[Path] = []

    for key, result in sorted_results(results):
        src = heterotypic_source_summary_pdf(result)
        dst = dest_dir / heterotypic_prab_pdf_name(result)

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
        print("\nMissing heterotypic summary PDFs:")
        for src in missing:
            print(f"  {src}")

    print(f"\nCopied/found {len(copied)} heterotypic summary PDFs in {dest_dir}")
    return copied


# -----------------------------------------------------------------------------
# LaTeX block generation
# -----------------------------------------------------------------------------

def crossing_short_title(result: dict[str, Any]) -> str:
    c = result["crossing"]
    pair_type = pair_type_to_text(result.get("pair_type") or c.get("pair_type"))
    mode_i = mode_to_latex(result.get("mode_i"))
    mode_j = mode_to_latex(result.get("mode_j"))
    ell = float(c["length_factor"])
    fhat = float(c["frequency_Hz"]) / F_010_HZ
    return (
        rf"Heterotypic {pair_type} crossing {mode_i}--{mode_j}, "
        rf"$\ell={ell:.4f}$, $\hat{{f}}={fhat:.4f}$"
    )


def appendix_ii_start() -> str:
    """Start Appendix II. Assumes \appendix has already been issued by Appendix I."""
    return r"""
\clearpage
\section{TM heterotypic mixing}
\label{app:tm_heterotypic_mixing}

\lipsum[1-2]

In the tables below, only the figures of merit relevant to the crossing type are shown: monopole--dipole pages retain $k_{\parallel}$ and $k_{\perp}$ only; monopole--quadrupole pages omit $k_{\perp}$; and dipole--quadrupole pages omit $k_{\parallel}$. The quantities $K_{xx}$, $K_{yy}$, $K_{xy}$ and $K_{\mathrm{iso}}=(K_{xx}+K_{yy})/2$ retain their phase-aligned signs. The scalar quantities $k_{\parallel}$, $k_{\perp}$ and $K_Q=\sqrt{(K_{xx}-K_{yy})^2+4K_{xy}^2}$ are reported as magnitudes.

\clearpage
""".strip()


def appendix_ii_end() -> str:
    return r"""
% End of Appendix II: TM heterotypic mixing
""".strip()




def pair_type_key(result: dict[str, Any]) -> str:
    """Return normalised heterotypic pair type from result metadata/path."""
    c = result.get("crossing", {})
    pt = result.get("pair_type") or c.get("pair_type")
    if pt is None and result.get("crossing_folder"):
        pt = Path(result["crossing_folder"]).parent.name
    return str(pt or "heterotypic").lower().replace("-", "_")


def effect_rows_for_crossing(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Select only physically relevant rows for the heterotypic pair.

    Compact appendix policy:
      * monopole--dipole: keep k_parallel and k_perp only; discard all K rows.
      * monopole--quadrupole: discard k_perp; keep k_parallel and K rows.
      * dipole--quadrupole: discard k_parallel; keep k_perp and K rows.

    K rows include signed K_iso, unsigned K_Q, and signed Kxx/Kyy/Kxy so the
    isotropic monopole-like curvature and quadrupole-like anisotropy remain
    distinguishable.
    """
    pt = pair_type_key(result)
    rows = []
    for row in EFFECT_ROWS:
        key = row["key"]
        if pt == "monopole_dipole" and key in {"K_iso", "K_Q", "focusing", "defocusing", "skew"}:
            continue
        if pt == "monopole_quadrupole" and key == "kick":
            continue
        if pt == "dipole_quadrupole" and key == "loss":
            continue
        rows.append(row)
    return rows


def figure_size_for_crossing(result: dict[str, Any], default_width: str, default_height: str) -> tuple[str, str]:
    """Use a smaller figure on quadrupole-containing pages so table+figure fit."""
    pt = pair_type_key(result)
    if pt in {"monopole_quadrupole", "dipole_quadrupole"}:
        return "0.44\\textwidth", "0.62\\textheight"
    if pt == "monopole_dipole":
        return "0.54\\textwidth", "0.70\\textheight"
    return default_width, default_height


def latex_table_for_crossing(result: dict[str, Any]) -> str:
    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\footnotesize")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\setlength{\tabcolsep}{3.6pt}")
    lines.append(rf"\textbf{{{crossing_short_title(result)}}}\\[0.15em]")
    lines.append(r"\begin{ruledtabular}")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(
        r"Beam-dynamics metric & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{\max}$ \\"
    )
    lines.append(r"\hline")

    for info in effect_rows_for_crossing(result):
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
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\end{center}")
    return "\n".join(lines)


def single_crossing_summary_figure(
    result: dict[str, Any],
    *,
    image_width: str = "0.50\\textwidth",
    image_height: str = "0.66\\textheight",
    figs_dir_latex: str = "figs",
) -> str:
    pdf_path = heterotypic_prab_summary_pdf_path(result, figs_dir_latex=figs_dir_latex)
    width, height = figure_size_for_crossing(result, image_width, image_height)
    return rf"""
\vspace{{-0.8em}}
\begin{{center}}
\includegraphics[width={width},height={height},keepaspectratio]{{{pdf_path}}}
\end{{center}}
\vspace{{-0.6em}}
""".strip()


def single_crossing_page(
    result: dict[str, Any],
    *,
    image_width: str = "0.50\\textwidth",
    image_height: str = "0.66\\textheight",
    figs_dir_latex: str = "figs",
    clearpage: bool = True,
) -> str:
    page = (
        latex_table_for_crossing(result)
        + "\n\n"
        + single_crossing_summary_figure(
            result,
            image_width=image_width,
            image_height=image_height,
            figs_dir_latex=figs_dir_latex,
        )
    )
    if clearpage:
        page += "\n\n\\clearpage"
    return page


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


def write_tm_heterotypic_appendix(
    *,
    input_pkl: str | Path = DEFAULT_INPUT_PKL,
    out_tex: str | Path = DEFAULT_OUT_TEX,
    figs_dest_dir: str | Path = DEFAULT_FIGS_DIR,
    figs_dir_latex: str = "figs",
    image_width: str = "0.50\\textwidth",
    image_height: str = "0.66\\textheight",
    copy_figures: bool = True,
    top_n: int | None = None,
) -> Path:
    results = pickle_load(input_pkl)

    if copy_figures:
        copy_heterotypic_summary_pdfs_to_prab_figs(results, figs_dest_dir)

    items = sorted_results(results)
    if top_n is not None:
        items = items[:top_n]

    pages = [
        single_crossing_page(
            result=result,
            image_width=image_width,
            image_height=image_height,
            figs_dir_latex=figs_dir_latex,
            clearpage=True,
        )
        for _, result in items
    ]

    tex = "\n\n".join([
        appendix_ii_start(),
        *pages,
        appendix_ii_end(),
    ])

    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")

    print(f"Wrote {out_tex}")
    print(f"  heterotypic entries: {len(items)}")
    return out_tex


if __name__ == "__main__":
    write_tm_heterotypic_appendix(
        input_pkl=DEFAULT_INPUT_PKL,
        out_tex=DEFAULT_OUT_TEX,
        figs_dest_dir=DEFAULT_FIGS_DIR,
        figs_dir_latex="figs",
        image_width="0.50\\textwidth",
        image_height="0.66\\textheight",
        copy_figures=True,
        top_n=None,
    )
