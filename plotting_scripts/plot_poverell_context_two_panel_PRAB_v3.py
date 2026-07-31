import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm

import TM_analytical_fields as tmaf


def fit_powerlaw_A_p_r2(x, y, *, x_min_positive=1e-12):
    """
    Fit y = A * x^p using log-linear regression on positive x and y.

    Returns
    -------
    dict
        {"success": bool, "A": float, "p": float, "r2": float}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > x_min_positive) & (y > 0)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return {"success": False}

    lx = np.log(x)
    ly = np.log(y)

    design_matrix = np.vstack([np.ones_like(lx), lx]).T
    coeff, *_ = np.linalg.lstsq(design_matrix, ly, rcond=None)
    b0, b1 = coeff

    ly_hat = b0 + b1 * lx
    ss_res = float(np.sum((ly - ly_hat) ** 2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    A = float(np.exp(b0))
    exponent = float(b1)

    if not (
        np.isfinite(A)
        and np.isfinite(exponent)
        and np.isfinite(r2)
        and A > 0
    ):
        return {"success": False}

    return {
        "success": True,
        "A": A,
        "p": exponent,
        "r2": r2,
    }


def calculate_normalised_mode_frequencies(
    *,
    m_values,
    n_values,
    p_values,
    radius,
    length_factors,
    reference_frequency,
):
    """
    Calculate normalised TM mode frequencies over the length-factor sweep.
    """
    mode_frequencies = {}

    for p in p_values:
        for n in n_values:
            for m in m_values:
                frequencies = np.asarray(
                    [
                        tmaf.f_tm(m, n, p, radius, length_factor)
                        for length_factor in length_factors
                    ],
                    dtype=float,
                )
                mode_frequencies[(m, n, p)] = frequencies / reference_frequency

    return mode_frequencies


def write_two_panel_p_over_ell_plot(
    *,
    length_factors,
    mode_frequencies,
    p_values,
    highlighted_p,
    savepath,
    dpi=300,
):
    """
    Write a two-panel PRAB figure.

    Left panel: all TM_01p modes in the selected p range.
    Right panel: the highlighted TM_01p mode alone.
    """
    savepath = Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    cmap = cm["viridis"]
    colours = cmap(np.linspace(0.0, 1.0, len(p_values)))
    colour_by_p = {p: colours[index] for index, p in enumerate(p_values)}

    fig, (ax_context, ax_highlight) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        sharex=True,
        sharey=True,
    )

    for p in p_values:
        key = (0, 1, p)
        if key not in mode_frequencies:
            continue

        ax_context.plot(
            length_factors,
            mode_frequencies[key],
            color=colour_by_p[p],
            linewidth=1.25,
            label=rf"$\mathrm{{TM}}_{{01{p}}}$",
        )

    highlighted_key = (0, 1, highlighted_p)
    if highlighted_key not in mode_frequencies:
        raise KeyError(
            f"No calculated data were found for mode TM_01{highlighted_p}."
        )

    ax_highlight.plot(
        length_factors,
        mode_frequencies[highlighted_key],
        color=colour_by_p[highlighted_p],
        linewidth=1.6,
        label=rf"$\mathrm{{TM}}_{{01{highlighted_p}}}$",
    )

    for axis in (ax_context, ax_highlight):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$")
        axis.grid(True, which="both", linewidth=0.45, alpha=0.35)
        axis.tick_params(which="both", direction="in", top=True, right=True)

    ax_context.set_ylabel(r"$\hat{f}=f_{mnp}/f_{010}$")

    # Indicative boundaries separating the three characteristic regimes.
    region_boundaries = (0.2, 2.0)
    for boundary in region_boundaries:
        ax_highlight.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            color="red",
            zorder=2,
        )

    # Characteristic scaling relations.  Place these on the highlighted panel
    # to keep the full-family context panel legible.

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.18,
        top=0.975,
        wspace=0.08,
    )

    png_path = savepath / "poverell_context_two_panel.png"
    pdf_path = savepath / "poverell_context_two_panel.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")

    return png_path, pdf_path


if __name__ == "__main__":
    savepath = Path(r"D:\PhD\PRAB")

    csol = 299_792_458.0
    frequency_010 = 1.3e9
    lambda_010 = csol / frequency_010
    radius = tmaf.pillbox_radius_from_freq(frequency_010)

    pindex_max = 5
    highlighted_p = 5

    m_values = [0]
    n_values = [1]
    p_values = list(range(pindex_max + 1))

    length_factor_start_exponent = -3
    length_factor_stop_exponent = 2
    parameter_sweep_resolution = 100

    length_factors = np.logspace(
        length_factor_start_exponent,
        length_factor_stop_exponent,
        parameter_sweep_resolution,
        endpoint=True,
    )

    mode_frequencies = calculate_normalised_mode_frequencies(
        m_values=m_values,
        n_values=n_values,
        p_values=p_values,
        radius=radius,
        length_factors=length_factors,
        reference_frequency=frequency_010,
    )

    for p in p_values:
        mode_array = np.asarray(
            [
                mode_frequencies[(m, n, p)]
                for n in n_values
                for m in m_values
            ],
            dtype=float,
        )
        np.save(savepath / f"P{p}_modes.npy", mode_array)

    write_two_panel_p_over_ell_plot(
        length_factors=length_factors,
        mode_frequencies=mode_frequencies,
        p_values=p_values,
        highlighted_p=highlighted_p,
        savepath=savepath,
        dpi=300,
    )
