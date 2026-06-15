import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import os
from scipy.optimize import linear_sum_assignment
from scipy.optimize import brentq
from typing import Dict, Any, Optional, List, Tuple, Literal
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from scipy.special import jn_zeros
from scipy.interpolate import RegularGridInterpolator
from importlib.machinery import SourceFileLoader
import pickle as pkl



phdmm_path = r"C:\Users\zup98752\PycharmProjects\PhD"
sys.path.insert(0, phdmm_path)

import PhD_Master_Module as pmm

####################################################################################################################
#                                                                                                                  #
#                                            PARAMETER SWEEP FUNCTIONS                                             #
#                                                                                                                  #
####################################################################################################################



def parse_mode_sweep_directory(directory):
    """
    Parses all CST per-mode sweep files in a directory into:
    {
        "mode 1": {"length_factor": [...], "frequency_GHz": [...]},
        ...
    }

    Assumptions:
    - One mode per file (e.g. Mode_001.txt)
    - Header contains something like: "Mode 1/real"
    - Data lines contain exactly two numeric columns:
        length_factor, frequency_GHz
    """

    modes = {}

    # Regex to detect "Mode <number>" in header
    mode_re = re.compile(r"mode\s+(\d+)", re.IGNORECASE)

    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(directory, fname)

        current_mode = None

        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # Detect mode from header line (preferred, robust)
                if current_mode is None:
                    m = mode_re.search(line)
                    if m:
                        mode_num = int(m.group(1))
                        current_mode = f"mode {mode_num}"
                        modes[current_mode] = {
                            "length_factor": [],
                            "frequency_GHz": []
                        }
                    continue

                # Parse numeric data lines
                parts = line.split()
                if len(parts) == 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                    except ValueError:
                        continue

                    modes[current_mode]["length_factor"].append(x)
                    modes[current_mode]["frequency_GHz"].append(y)

        # Safety check: file had no recognizable mode
        if current_mode is None:
            raise ValueError(f"Could not determine mode number from file: {fname}")

    return modes

def parse_mode_sweep_file(filename):
    """
    Robustly parses CST-style multi-mode sweep file into:
    {
        "mode 1": {"length_factor": [...], "frequency_GHz": [...]},
        ...
    }
    """
    modes = {}
    current_mode = None

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Start of a new mode block
            if line.startswith("Curvelabel"):
                label = line.split("=", 1)[1].strip()  # "Mode 1"
                current_mode = label.lower()  # "mode 1"
                modes[current_mode] = {"length_factor": [], "frequency_GHz": []}
                continue

            # Ignore anything until we know what mode we're in
            if current_mode is None:
                continue

            # Collect numeric data lines: must be exactly two numeric columns
            parts = line.split()
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                modes[current_mode]["length_factor"].append(x)
                modes[current_mode]["frequency_GHz"].append(y)

    return modes



def redistribute_modes_for_continuity_smooth_log_log(
    mode_dict,
    *,
    freq_scale=1.3,
    target_x=1.0,
    max_mode=None,

    # preprocessing / space
    use_loglog=True,
    resample_to_common_grid=True,
    init_sort_by_freq=True,

    # smoothing objective weights (matching space)
    w_cont=0.10,
    w_pred=1.00,
    w_slope=0.85,
    w_curv=1.00,

    # NEW: leapfrog/lookahead behavior
    w_lookahead=0.90,          # strength of "project to t+1 and prefer continuity there"
    look_slope_weight=0.35,    # include slope consistency into lookahead min-cost
    look_curv_weight=0.35,     # include curvature proxy into lookahead min-cost

    robust_scale=True,
    eps=1e-12,

    # prioritize first N output modes (optional)
    prioritize_first_k=True,
    priority_buffer=0,
):
    """
    Log–log Hungarian mode tracker with one-step lookahead:
      - If a point at time t is a poor local fit (often near crossings),
        the lookahead term makes the assignment prefer the candidate at t
        that best continues at t+1 (i.e., "leapfrogs" the misleading point).
      - After redistribution, modes are renumbered by increasing freq at target_x,
        then strictly truncated to modes 1..max_mode.

    NOTE: Lookahead uses a min over candidates at t+1 (ignoring exclusivity),
          which is fast and very effective at crossings.
    """

    # ---------------- helpers ----------------
    def mode_num(k: str) -> int:
        return int(k.strip().lower().split()[-1])

    def mad_scale(A):
        med = np.median(A)
        mad = np.median(np.abs(A - med))
        return mad + eps

    def renumber_by_frequency_at_x(md, x_grid, tx):
        if not (x_grid.min() <= tx <= x_grid.max()):
            raise ValueError(f"target_x={tx} is outside length_factor range.")
        freq_at = []
        for name, data in md.items():
            f = np.asarray(data["frequency_GHz"], float)
            ftx = np.interp(tx, x_grid, f)
            freq_at.append((name, ftx))
        freq_at.sort(key=lambda t: t[1])
        out = {}
        for i, (old_name, _) in enumerate(freq_at, start=1):
            out[f"mode {i}"] = md[old_name]
        return out

    def truncate_modes(md, n):
        n = int(n)
        out = {}
        for i in range(1, n + 1):
            k = f"mode {i}"
            if k not in md:
                break
            out[k] = md[k]
        return out

    # ---------------- collect / validate ----------------
    keys = sorted(mode_dict.keys(), key=mode_num)
    if not keys:
        return {}

    xs, freqs = [], []
    for k in keys:
        xk = np.asarray(mode_dict[k]["length_factor"], dtype=float)
        fk = np.asarray(mode_dict[k]["frequency_GHz"], dtype=float) / float(freq_scale)

        if len(xk) != len(fk):
            raise ValueError(f"{k}: length_factor and frequency_GHz lengths differ.")
        if np.any(~np.isfinite(xk)) or np.any(xk <= 0):
            raise ValueError(f"{k}: length_factor must be finite and > 0.")
        if np.any(~np.isfinite(fk)) or np.any(fk <= 0):
            raise ValueError(f"{k}: frequency_GHz must be finite and > 0 after scaling.")

        xs.append(xk)
        freqs.append(fk)

    # ---------------- resample onto common grid ----------------
    if resample_to_common_grid:
        ref_idx = int(np.argmax([len(xk) for xk in xs]))
        x_ref = np.asarray(xs[ref_idx], float)
        x_ref = x_ref[np.argsort(x_ref)]
        if np.any(np.diff(x_ref) <= 0):
            raise ValueError("Reference length_factor grid must be strictly increasing.")

        F_list = []
        for xk, fk in zip(xs, freqs):
            o = np.argsort(xk)
            xk = xk[o]
            fk = fk[o]
            F_list.append(np.interp(x_ref, xk, fk))

        x = x_ref
        F = np.vstack(F_list).T  # T x M
    else:
        x0 = np.asarray(xs[0], float)
        for xk in xs[1:]:
            if len(xk) != len(x0) or np.max(np.abs(np.asarray(xk) - x0)) > 1e-12:
                raise ValueError("Enable resample_to_common_grid=True (grids differ).")
        x = x0
        F = np.vstack(freqs).T

    if np.any(np.diff(x) <= 0):
        raise ValueError("length_factor must be strictly increasing.")

    T, M = F.shape
    if T <= 0 or M <= 0:
        return {}

    # ---------------- matching space ----------------
    if use_loglog:
        X = np.log(np.clip(x, eps, None))
        Y = np.log(np.clip(F, eps, None))
    else:
        X = x.copy()
        Y = F.copy()

    # ---------------- initial anchor ----------------
    if init_sort_by_freq:
        Y = Y[:, np.argsort(Y[0, :])]

    tracked = np.empty_like(Y)
    tracked[0, :] = Y[0, :]

    if T == 1:
        tracked_F = (np.exp(tracked) if use_loglog else tracked) * float(freq_scale)
        redistributed = {
            f"mode {m+1}": {"length_factor": x.tolist(), "frequency_GHz": tracked_F[:, m].tolist()}
            for m in range(M)
        }
        ren = renumber_by_frequency_at_x(redistributed, x, target_x)
        return truncate_modes(ren, max_mode) if max_mode is not None else ren

    # step 1: continuity only
    cost01 = np.abs(tracked[0, :][:, None] - Y[1, :][None, :])
    _, col = linear_sum_assignment(cost01)
    tracked[1, :] = Y[1, col]

    # prioritize rows 0..K-1 if requested
    if max_mode is None:
        K_rows = M
    else:
        K_rows = min(int(max_mode), M)
        if prioritize_first_k and priority_buffer > 0:
            K_rows = min(M, K_rows + int(priority_buffer))

    # ---------------- main loop with lookahead ----------------
    for t in range(2, T):
        y_prev = tracked[t - 1, :]
        y_prevprev = tracked[t - 2, :]

        dx = X[t] - X[t - 1]
        dx_prev = X[t - 1] - X[t - 2]
        if abs(dx) < eps: dx = eps
        if abs(dx_prev) < eps: dx_prev = eps

        slope_prev = (y_prev - y_prevprev) / dx_prev
        y_pred = y_prev + slope_prev * dx

        curr = Y[t, :]  # candidates at t

        # --- base costs (M x M) ---
        cont_cost = np.abs(y_prev[:, None] - curr[None, :])
        pred_cost = np.abs(y_pred[:, None] - curr[None, :])

        slope_cand = (curr[None, :] - y_prev[:, None]) / dx
        slope_cost = np.abs(slope_prev[:, None] - slope_cand)
        curv_cost = np.abs(slope_cand - slope_prev[:, None])

        if robust_scale:
            cont_cost /= mad_scale(cont_cost)
            pred_cost /= mad_scale(pred_cost)
            slope_cost /= mad_scale(slope_cost)
            curv_cost /= mad_scale(curv_cost)

        base_cost = (w_cont * cont_cost + w_pred * pred_cost + w_slope * slope_cost + w_curv * curv_cost)

        # --- NEW: lookahead cost ---
        # For each (i,j): if we pick curr[j] for tracked i at time t,
        # project to t+1 and see if there's a good continuation there.
        if (w_lookahead > 0) and (t < T - 1):
            nxt = Y[t + 1, :]  # candidates at t+1
            dx_next = X[t + 1] - X[t]
            if abs(dx_next) < eps:
                dx_next = eps

            # if we pick curr_j at t, candidate slope becomes:
            # slope_ij = (curr_j - y_prev_i) / dx
            # then predicted y at t+1 is:
            # y_next_pred_ij = curr_j + slope_ij * dx_next
            # We'll score continuation by min_k of:
            #   |y_next_pred_ij - nxt_k| + slope/curv consistency terms
            # (min over k gives the "leapfrog" behavior)
            curr_j = curr[None, :, None]            # shape: (1, M, 1)
            y_prev_i = y_prev[:, None, None]        # shape: (M, 1, 1)

            slope_ij = (curr_j - y_prev_i) / dx     # (M, M, 1)
            y_next_pred_ij = curr_j + slope_ij * dx_next  # (M, M, 1)

            nxt_k = nxt[None, None, :]              # (1, 1, M)

            look_pred = np.abs(y_next_pred_ij - nxt_k)  # (M, M, M)

            # optional slope/curv terms in lookahead:
            # slope to reach nxt_k from curr_j:
            slope_next_ijk = (nxt_k - curr_j) / dx_next   # (M, M, M)
            look_slope = np.abs(slope_next_ijk - slope_ij)  # slope consistency
            look_curv = np.abs(slope_next_ijk - slope_prev[:, None, None])  # curvature-ish

            # combine and min over k (candidate at t+1)
            look_cost_ijk = look_pred + look_slope_weight * look_slope + look_curv_weight * look_curv
            look_cost = np.min(look_cost_ijk, axis=2)  # (M, M)

            if robust_scale:
                look_cost /= mad_scale(look_cost)

            cost = base_cost + w_lookahead * look_cost
        else:
            cost = base_cost

        # --- assignment (priority-first or full) ---
        if (max_mode is not None) and prioritize_first_k:
            # 1) assign priority rows
            cost_pri = cost[:K_rows, :]  # K x M
            pri_rows, pri_cols = linear_sum_assignment(cost_pri)

            next_y = np.empty(M, dtype=Y.dtype)
            next_y[:K_rows] = curr[pri_cols]

            # 2) remainder rows get remainder cols
            rem_rows = np.arange(K_rows, M)
            used = set(pri_cols.tolist())
            rem_cols = np.array([j for j in range(M) if j not in used], dtype=int)

            if len(rem_rows) > 0:
                cost_rem = cost[np.ix_(rem_rows, rem_cols)]
                r2, c2 = linear_sum_assignment(cost_rem)
                next_y[rem_rows] = curr[rem_cols[c2]]

            tracked[t, :] = next_y
        else:
            _, col = linear_sum_assignment(cost)
            tracked[t, :] = curr[col]

    # ---------------- back-transform ----------------
    tracked_F = (np.exp(tracked) if use_loglog else tracked) * float(freq_scale)

    redistributed = {
        f"mode {m + 1}": {
            "length_factor": x.tolist(),
            "frequency_GHz": tracked_F[:, m].tolist(),
        }
        for m in range(M)
    }

    # renumber after redistribution
    renumbered = renumber_by_frequency_at_x(redistributed, x, target_x)

    # STRICT: only modes 1..max_mode in output
    if max_mode is not None:
        renumbered = truncate_modes(renumbered, max_mode)

    return renumbered




def redistribute_modes_for_continuity_smooth_log_log_extrap(
    mode_dict,
    *,
    freq_scale=1.3,
    target_x=1.0,
    max_mode=None,

    # preprocessing / space
    use_loglog=True,
    resample_to_common_grid=True,
    init_sort_by_freq=True,

    # smoothing objective weights (matching space)
    w_cont=0.10,
    w_pred=1.00,
    w_slope=0.85,
    w_curv=1.00,

    # lookahead / leapfrog
    w_lookahead=0.90,
    look_slope_weight=0.35,
    look_curv_weight=0.35,

    # NEW: segment extrapolation (log–log)
    w_extrap=1.25,            # weight of proximity-to-extrapolated-model term
    extrap_window_max=6,      # max recent points to consider for the "consistent segment"
    extrap_window_min=3,      # minimum points to declare a segment
    extrap_mad_tol=2.5,       # consistency threshold vs local MAD(y)
    extrap_poly_order=1,      # 1=linear, 2=quadratic (in log–log space)
    extrap_alpha_max=1.0,     # how strongly extrap pred can blend into the 1-step predictor

    # scaling / robustness
    robust_scale=True,
    eps=1e-12,

    # prioritize first N output modes (optional)
    prioritize_first_k=True,
    priority_buffer=0,
):
    """
    Log–log Hungarian mode tracker with:
      - continuity + predictor + slope + curvature costs
      - one-step lookahead (min over t+1 candidates, ignoring exclusivity)
      - NEW: segment-based extrapolation in log–log space

    Segment extrapolation:
      For each tracked row i, find the longest recent contiguous window ending at t-1
      that fits a poly in (X=log(x), Y=log(f)) with low residual; extrapolate to X[t]
      and penalize candidates far from that extrapolated model.

    After tracking, modes are renumbered by increasing freq at target_x and optionally
    truncated to modes 1..max_mode.
    """

    # ---------------- helpers ----------------
    def mode_num(k: str) -> int:
        return int(k.strip().lower().split()[-1])

    def mad_scale(A):
        A = np.asarray(A, float)
        med = np.median(A)
        mad = np.median(np.abs(A - med))
        return mad + eps

    def renumber_by_frequency_at_x(md, x_grid, tx):
        if not (x_grid.min() <= tx <= x_grid.max()):
            raise ValueError(f"target_x={tx} is outside length_factor range.")
        freq_at = []
        for name, data in md.items():
            f = np.asarray(data["frequency_GHz"], float)
            ftx = np.interp(tx, x_grid, f)
            freq_at.append((name, ftx))
        freq_at.sort(key=lambda t: t[1])
        out = {}
        for i, (old_name, _) in enumerate(freq_at, start=1):
            out[f"mode {i}"] = md[old_name]
        return out

    def truncate_modes(md, n):
        n = int(n)
        out = {}
        for i in range(1, n + 1):
            k = f"mode {i}"
            if k not in md:
                break
            out[k] = md[k]
        return out

    def fit_extrap_segment(Xhist, yhist):
        """
        Select a "consistent" recent window ending at last sample, fit poly in X->y.

        Returns:
          coeffs (np.ndarray): poly coefficients for np.polyval
          slope_last (float): derivative at last X
          alpha (float): confidence weight in [0, extrap_alpha_max]
        """
        n = len(yhist)
        if n < 2:
            return np.array([yhist[-1]]), 0.0, 0.0

        y_scale = mad_scale(yhist)
        maxL = min(int(extrap_window_max), n)
        minL = max(2, int(extrap_window_min))
        ord_ = 2 if int(extrap_poly_order) >= 2 else 1

        best = None
        for L in range(maxL, minL - 1, -1):  # prefer longest consistent window
            if L < ord_ + 1:
                continue
            Xw = Xhist[-L:]
            yw = yhist[-L:]
            coeffs = np.polyfit(Xw, yw, ord_)
            yhat = np.polyval(coeffs, Xw)
            resid = yw - yhat
            rmse = float(np.sqrt(np.mean(resid**2)))
            if rmse <= float(extrap_mad_tol) * y_scale:
                best = (L, coeffs)
                break

        if best is None:
            # fallback: last 2 points linear
            L = 2
            coeffs = np.polyfit(Xhist[-2:], yhist[-2:], 1)
        else:
            L, coeffs = best

        # slope at last X
        if ord_ == 1:
            slope_last = float(coeffs[0])
        else:
            a, b, _c = coeffs
            x_last = float(Xhist[-1])
            slope_last = float(2 * a * x_last + b)

        # alpha grows with L (more evidence => more trust)
        denom = max(1, min(maxL, n) - 2)
        alpha = float(extrap_alpha_max) * (L - 2) / denom
        alpha = float(np.clip(alpha, 0.0, float(extrap_alpha_max)))

        return np.asarray(coeffs, float), slope_last, alpha

    # ---------------- collect / validate ----------------
    keys = sorted(mode_dict.keys(), key=mode_num)
    if not keys:
        return {}

    xs, freqs = [], []
    for k in keys:
        xk = np.asarray(mode_dict[k]["length_factor"], dtype=float)
        fk = np.asarray(mode_dict[k]["frequency_GHz"], dtype=float) / float(freq_scale)

        if len(xk) != len(fk):
            raise ValueError(f"{k}: length_factor and frequency_GHz lengths differ.")
        if np.any(~np.isfinite(xk)) or np.any(xk <= 0):
            raise ValueError(f"{k}: length_factor must be finite and > 0.")
        if np.any(~np.isfinite(fk)) or np.any(fk <= 0):
            raise ValueError(f"{k}: frequency_GHz must be finite and > 0 after scaling.")

        xs.append(xk)
        freqs.append(fk)

    # ---------------- resample onto common grid ----------------
    if resample_to_common_grid:
        ref_idx = int(np.argmax([len(xk) for xk in xs]))
        x_ref = np.asarray(xs[ref_idx], float)
        x_ref = x_ref[np.argsort(x_ref)]
        if np.any(np.diff(x_ref) <= 0):
            raise ValueError("Reference length_factor grid must be strictly increasing.")

        F_list = []
        for xk, fk in zip(xs, freqs):
            o = np.argsort(xk)
            xk = xk[o]
            fk = fk[o]
            F_list.append(np.interp(x_ref, xk, fk))

        x = x_ref
        F = np.vstack(F_list).T  # T x M
    else:
        x0 = np.asarray(xs[0], float)
        for xk in xs[1:]:
            if len(xk) != len(x0) or np.max(np.abs(np.asarray(xk) - x0)) > 1e-12:
                raise ValueError("Enable resample_to_common_grid=True (grids differ).")
        x = x0
        F = np.vstack(freqs).T

    if np.any(np.diff(x) <= 0):
        raise ValueError("length_factor must be strictly increasing.")

    T, M = F.shape
    if T <= 0 or M <= 0:
        return {}

    # ---------------- matching space ----------------
    if use_loglog:
        X = np.log(np.clip(x, eps, None))
        Y = np.log(np.clip(F, eps, None))
    else:
        X = x.copy()
        Y = F.copy()

    # ---------------- initial anchor ----------------
    if init_sort_by_freq:
        Y = Y[:, np.argsort(Y[0, :])]

    tracked = np.empty_like(Y)
    tracked[0, :] = Y[0, :]

    if T == 1:
        tracked_F = (np.exp(tracked) if use_loglog else tracked) * float(freq_scale)
        redistributed = {
            f"mode {m + 1}": {"length_factor": x.tolist(), "frequency_GHz": tracked_F[:, m].tolist()}
            for m in range(M)
        }
        ren = renumber_by_frequency_at_x(redistributed, x, target_x)
        return truncate_modes(ren, max_mode) if max_mode is not None else ren

    # step 1: continuity only
    cost01 = np.abs(tracked[0, :][:, None] - Y[1, :][None, :])
    _, col = linear_sum_assignment(cost01)
    tracked[1, :] = Y[1, col]

    # prioritize rows 0..K-1 if requested
    if max_mode is None:
        K_rows = M
    else:
        K_rows = min(int(max_mode), M)
        if prioritize_first_k and priority_buffer > 0:
            K_rows = min(M, K_rows + int(priority_buffer))

    # ---------------- main loop ----------------
    for t in range(2, T):
        y_prev = tracked[t - 1, :]
        y_prevprev = tracked[t - 2, :]

        dx = X[t] - X[t - 1]
        dx_prev = X[t - 1] - X[t - 2]
        if abs(dx) < eps:
            dx = eps
        if abs(dx_prev) < eps:
            dx_prev = eps

        slope_prev = (y_prev - y_prevprev) / dx_prev
        y_pred = y_prev + slope_prev * dx

        curr = Y[t, :]  # candidates at t

        # ---- NEW: segment extrapolation predictions (per tracked row i) ----
        if w_extrap > 0:
            y_extrap = np.empty(M, dtype=Y.dtype)
            slope_extrap = np.empty(M, dtype=Y.dtype)
            alpha = np.empty(M, dtype=float)

            Xhist = X[:t]  # same for all rows
            for i in range(M):
                coeffs, s_last, a = fit_extrap_segment(Xhist, tracked[:t, i])
                y_extrap[i] = np.polyval(coeffs, X[t])
                slope_extrap[i] = s_last
                alpha[i] = a

            # Blend extrapolated model into the usual one-step predictor when confident
            y_model = (1.0 - alpha) * y_pred + alpha * y_extrap
            slope_model = (1.0 - alpha) * slope_prev + alpha * slope_extrap
        else:
            y_extrap = None
            y_model = y_pred
            slope_model = slope_prev

        # ---- base costs (M x M) ----
        cont_cost = np.abs(y_prev[:, None] - curr[None, :])
        pred_cost = np.abs(y_model[:, None] - curr[None, :])

        slope_cand = (curr[None, :] - y_prev[:, None]) / dx
        slope_cost = np.abs(slope_model[:, None] - slope_cand)
        curv_cost = np.abs(slope_cand - slope_model[:, None])

        if w_extrap > 0:
            extrap_cost = np.abs(y_extrap[:, None] - curr[None, :])
        else:
            extrap_cost = 0.0

        if robust_scale:
            cont_cost /= mad_scale(cont_cost)
            pred_cost /= mad_scale(pred_cost)
            slope_cost /= mad_scale(slope_cost)
            curv_cost /= mad_scale(curv_cost)
            if w_extrap > 0:
                extrap_cost = extrap_cost / mad_scale(extrap_cost)

        base_cost = (
            w_cont * cont_cost
            + w_pred * pred_cost
            + w_slope * slope_cost
            + w_curv * curv_cost
            + (w_extrap * extrap_cost if w_extrap > 0 else 0.0)
        )

        # ---- lookahead ----
        if (w_lookahead > 0) and (t < T - 1):
            nxt = Y[t + 1, :]
            dx_next = X[t + 1] - X[t]
            if abs(dx_next) < eps:
                dx_next = eps

            curr_j = curr[None, :, None]
            y_prev_i = y_prev[:, None, None]

            slope_ij = (curr_j - y_prev_i) / dx
            y_next_pred_ij = curr_j + slope_ij * dx_next

            nxt_k = nxt[None, None, :]
            look_pred = np.abs(y_next_pred_ij - nxt_k)

            slope_next_ijk = (nxt_k - curr_j) / dx_next
            look_slope = np.abs(slope_next_ijk - slope_ij)
            look_curv = np.abs(slope_next_ijk - slope_model[:, None, None])

            look_cost_ijk = look_pred + look_slope_weight * look_slope + look_curv_weight * look_curv
            look_cost = np.min(look_cost_ijk, axis=2)

            if robust_scale:
                look_cost /= mad_scale(look_cost)

            cost = base_cost + w_lookahead * look_cost
        else:
            cost = base_cost

        # ---- assignment (priority-first or full) ----
        if (max_mode is not None) and prioritize_first_k:
            cost_pri = cost[:K_rows, :]
            _pri_rows, pri_cols = linear_sum_assignment(cost_pri)

            next_y = np.empty(M, dtype=Y.dtype)
            next_y[:K_rows] = curr[pri_cols]

            rem_rows = np.arange(K_rows, M)
            used = set(pri_cols.tolist())
            rem_cols = np.array([j for j in range(M) if j not in used], dtype=int)

            if len(rem_rows) > 0:
                cost_rem = cost[np.ix_(rem_rows, rem_cols)]
                _r2, c2 = linear_sum_assignment(cost_rem)
                next_y[rem_rows] = curr[rem_cols[c2]]

            tracked[t, :] = next_y
        else:
            _, col = linear_sum_assignment(cost)
            tracked[t, :] = curr[col]

    # ---------------- back-transform ----------------
    tracked_F = (np.exp(tracked) if use_loglog else tracked) * float(freq_scale)

    redistributed = {
        f"mode {m + 1}": {"length_factor": x.tolist(), "frequency_GHz": tracked_F[:, m].tolist()}
        for m in range(M)
    }

    renumbered = renumber_by_frequency_at_x(redistributed, x, target_x)
    if max_mode is not None:
        renumbered = truncate_modes(renumbered, max_mode)
    return renumbered

def redistribute_modes_for_continuity_smooth(mode_dict, alpha=0.3, beta=1.0, gamma=2.0):
    """
    Reorder modes to maximize continuity AND smoothness.

    Cost at step t (Hungarian assignment):
        cost = beta * |predicted - curr|
             + alpha * |prev - curr|
             + gamma * |(curr - 2*prev + prevprev)|   (curvature penalty)

    Notes:
      - Numeric sorting of 'mode N' keys (critical for N>=10)
      - Curvature penalty discourages zig-zag / swapping near crossings
      - Returns new_mode_dict with 'mode 1'..'mode M' keys
    """

    def mode_num(k: str) -> int:
        # tolerate "mode 1", "Mode 1", etc.
        parts = k.strip().lower().split()
        return int(parts[-1])

    # --- assemble matrix with numeric mode ordering ---
    keys = sorted(mode_dict.keys(), key=mode_num)
    lengths = np.asarray(mode_dict[keys[0]]["length_factor"], dtype=float)

    freq_matrix = np.vstack([np.asarray(mode_dict[k]["frequency_GHz"], dtype=float) for k in keys]).T
    T, M = freq_matrix.shape
    tracked = np.empty_like(freq_matrix)

    # Step 0: keep original ordering
    tracked[0, :] = freq_matrix[0, :]

    if T == 1:
        return {
            f"mode {m+1}": {"length_factor": lengths.tolist(), "frequency_GHz": tracked[:, m].tolist()}
            for m in range(M)
        }

    # Step 1: match to step 0 (simple continuity)
    cost01 = np.abs(tracked[0, :][:, None] - freq_matrix[1, :][None, :])
    _, col = linear_sum_assignment(cost01)
    tracked[1, :] = freq_matrix[1, col]

    # Steps 2..T-1: predictive + smoothness (curvature) assignment
    for t in range(2, T):
        prev = tracked[t - 1, :]
        prevprev = tracked[t - 2, :]

        # linear extrapolation prediction
        pred = prev + (prev - prevprev)

        curr = freq_matrix[t, :]  # unordered at time t

        # build cost matrix: (M tracked modes) x (M current candidates)
        cont_cost = alpha * np.abs(prev[:, None] - curr[None, :])
        pred_cost = beta  * np.abs(pred[:, None] - curr[None, :])

        # curvature uses (curr - 2*prev + prevprev)
        curv_cost = gamma * np.abs((curr[None, :] - 2.0 * prev[:, None] + prevprev[:, None]))

        cost = cont_cost + pred_cost + curv_cost

        _, col = linear_sum_assignment(cost)
        tracked[t, :] = curr[col]

    # --- rebuild dict ---
    new_mode_dict = {}
    for m in range(M):
        new_mode_dict[f"mode {m + 1}"] = {
            "length_factor": lengths.tolist(),
            "frequency_GHz": tracked[:, m].tolist(),
        }


    return new_mode_dict



def renumber_modes_by_frequency_at_1p0(mode_dict, target_x=1.0):
    """
    Reassigns mode names according to increasing frequency at x=1.0.

    Returns:
        new_mode_dict = {
            'mode 1': { ... },
            'mode 2': { ... },
            ...
        }
    """

    # -----------------------------------------------------------
    # 1. Measure interpolated frequency at x = 1.0 for each mode
    # -----------------------------------------------------------
    freq_at_x = []

    for old_mode, data in mode_dict.items():
        L = np.array(data["length_factor"])
        F = np.array(data["frequency_GHz"])

        # Check that target_x is within this mode's range
        if not (L.min() <= target_x <= L.max()):
            raise ValueError(f"Mode {old_mode} does not span x={target_x}")

        # Interpolated frequency at x = target_x
        f_interp = np.interp(target_x, L, F)

        freq_at_x.append((old_mode, f_interp))

    # -----------------------------------------------------------
    # 2. Sort modes by increasing frequency at x=1.0
    # -----------------------------------------------------------
    freq_at_x.sort(key=lambda t: t[1])  # sort by frequency

    # -----------------------------------------------------------
    # 3. Build new mode dictionary with new numbering
    # -----------------------------------------------------------
    new_mode_dict = {}

    for i, (old_name, _) in enumerate(freq_at_x, start=1):
        new_name = f"mode {i}"

        new_mode_dict[new_name] = {
            "length_factor": mode_dict[old_name]["length_factor"],
            "frequency_GHz": mode_dict[old_name]["frequency_GHz"],
        }

    return new_mode_dict

def make_mode_label(mode_name, technical_mode_dict, *, tech_fields=("TM_TE", "m", "n", "p")):
    """
    Create a readable legend label from technical metadata, falling back to mode_name.
    Example: "mode 2 — TE(m=1,n=1,p=0)"
    """
    import re

    def normalize_mode_key(name: str) -> str:
        m = re.search(r"(\d+)", name)
        if not m:
            return name.strip().lower()
        return f"mode_{int(m.group(1))}"

    k = normalize_mode_key(mode_name)
    info = technical_mode_dict.get(k)
    if not info:
        return mode_name

    try:
        tm_te = info.get("TM_TE", "?")
        m = info.get("m", "?")
        n = info.get("n", "?")
        p = info.get("p", "?")
        return f"{mode_name} — {tm_te}(m={m},n={n},p={p})"
    except Exception:
        return mode_name

def find_mode_crossings_non_consecutive(
    mode_dict,
    target_x=1.0,
    relabel_in_order=False,
):
    """
    Reorders modes according to increasing frequency at x = target_x
    and detects all mode crossings with high precision.

    Works with non-consecutive input keys like: "mode 1", "mode 4", "mode 11", ...

    Parameters
    ----------
    mode_dict : dict
        { "mode k": {"length_factor": [...], "frequency_GHz": [...]} , ... }
    target_x : float
        x-value at which to rank modes by frequency.
    relabel_in_order : bool
        If True, renames modes to consecutive "mode 1", "mode 2", ... in physical order.
        If False (default), preserves original names and returns them ordered physically.

    Returns
    -------
    crossings : dict
        {
            "mode a–mode b": {
                "mode_i": "mode a",
                "mode_j": "mode b",
                "length_factor": Lc,
                "frequency_GHz": Fc
            },
            ...
        }
    all_modes_that_cross : list[str]
        Unique list of mode names that participate in at least one crossing.
    ordered_modes : list[str]
        Mode names in physical order (increasing frequency at target_x).
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def mode_index(name: str):
        """
        Extracts the integer from strings like 'mode 11'.
        Falls back to the raw name if no integer is found.
        """
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else name

    # ----------------------------
    # 1) Compute f(target_x) for each mode; sort by that
    # ----------------------------
    sort_info = []

    for mode_name, data in mode_dict.items():
        L = np.asarray(data["length_factor"], dtype=float)
        F = np.asarray(data["frequency_GHz"], dtype=float)

        # Ensure increasing L for np.interp
        if np.any(np.diff(L) < 0):
            order = np.argsort(L)
            L = L[order]
            F = F[order]

        if not (L.min() <= target_x <= L.max()):
            raise ValueError(f"Mode {mode_name} does not span x={target_x}")

        f_interp = np.interp(target_x, L, F)
        sort_info.append((mode_name, f_interp))

    # Sort by increasing frequency at target_x; tie-break by numeric mode index
    sort_info.sort(key=lambda t: (t[1], mode_index(t[0])))
    ordered_modes = [name for name, _ in sort_info]




    # Optionally relabel to consecutive mode numbers in physical order
    if relabel_in_order:
        reordered = {
            f"mode {i+1}": mode_dict[old]
            for i, old in enumerate(ordered_modes)
        }
        ordered_modes = list(reordered.keys())
    else:
        # Preserve original names, just in physical order
        reordered = {name: mode_dict[name] for name in ordered_modes}

    # ----------------------------
    # 2) Build shared length axis and frequency matrix
    # ----------------------------
    modes = ordered_modes
    N = len(modes)

    L = np.asarray(reordered[modes[0]]["length_factor"], dtype=float)

    # Ensure increasing L
    if np.any(np.diff(L) < 0):
        L = np.sort(L)

    # Frequencies matrix (len(L), N)
    Fmat = np.zeros((len(L), N), dtype=float)

    for i, m in enumerate(modes):
        Li = np.asarray(reordered[m]["length_factor"], dtype=float)
        Fi = np.asarray(reordered[m]["frequency_GHz"], dtype=float)

        # sort each mode data by Li
        if np.any(np.diff(Li) < 0):
            order = np.argsort(Li)
            Li = Li[order]
            Fi = Fi[order]

        # Ensure coverage of shared axis
        if Li.min() > L.min() or Li.max() < L.max():
            raise ValueError(
                f"Mode {m} does not span the shared length axis range "
                f"[{L.min()}, {L.max()}]."
            )

        Fmat[:, i] = np.interp(L, Li, Fi)

    # ----------------------------
    # 3) Detect crossings between all mode pairs
    # ----------------------------
    crossings = {}
    all_modes_that_cross = set()

    for i in range(N):
        for j in range(i + 1, N):

            Mi, Mj = modes[i], modes[j]
            fi = Fmat[:, i]
            fj = Fmat[:, j]

            g = fi - fj
            idxs = np.where(np.diff(np.sign(g)) != 0)[0]

            if idxs.size == 0:
                continue

            # Root function
            def gfun(x):
                fi_x = np.interp(x, L, fi)
                fj_x = np.interp(x, L, fj)
                return fi_x - fj_x

            for idx in idxs:
                L1, L2 = L[idx], L[idx + 1]

                try:
                    Lc = brentq(gfun, L1, L2)
                except ValueError:
                    continue

                Fc = np.interp(Lc, L, fi)

                key = f"{Mi}–{Mj}"
                crossings[key] = {
                    "mode_i": Mi,
                    "mode_j": Mj,
                    "length_factor": float(f"{Lc:.12g}"),
                    "frequency_GHz": float(f"{Fc:.12g}"),
                }

                all_modes_that_cross.add(Mi)
                all_modes_that_cross.add(Mj)

    return crossings, sorted(all_modes_that_cross, key=mode_index), ordered_modes

def find_mode_crossings_excluding_same_technical(
    mode_dict,
    technical_mode_dict,
    target_x=1.0,
    relabel_in_order=False,
):
    """
    Like find_mode_crossings_non_consecutive, but excludes crossings between modes
    that share the same "technical name" in `technical_mode_dict`.

    "Technical name" is defined as the tuple:
        (TM_TE, m, n, p)

    Parameters
    ----------
    mode_dict : dict
        { "mode k": {"length_factor": [...], "frequency_GHz": [...]} , ... }
        Keys may be "mode 2" or "mode_2" etc.
    technical_mode_dict : dict
        e.g. the attached TESLA_ALL_modes_dict mapping "mode_2" -> {"TM_TE":"TE","m":1,"n":1,"p":0}
        :contentReference[oaicite:1]{index=1}
    target_x : float
        x-value at which to rank modes by frequency.
    relabel_in_order : bool
        If True, renames to consecutive "mode 1", "mode 2", ... in physical order.

    Returns
    -------
    crossings : dict
    all_modes_that_cross : list[str]
    ordered_modes : list[str]
    skipped_pairs_same_technical : list[tuple[str,str]]
        Pairs that were skipped due to identical technical tuple.
    """

    def mode_index(name: str):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else name

    def normalize_mode_key(name: str) -> str:
        """
        Convert 'mode 2' or 'mode_2' or 'Mode-2' -> 'mode_2' for lookup in technical_mode_dict.
        """
        m = re.search(r"(\d+)", name)
        if not m:
            return name.strip().lower()
        return f"mode_{int(m.group(1))}"

    def tech_tuple(mode_name: str):
        """
        Return (TM_TE, m, n, p) if available, else None.
        """
        k = normalize_mode_key(mode_name)
        info = technical_mode_dict.get(k)
        if not info:
            return None
        # Only treat as comparable if all fields exist
        try:
            return (info["TM_TE"], int(info["m"]), int(info["n"]), int(info["p"]))
        except Exception:
            return None

    # ----------------------------
    # 1) Compute f(target_x) for each mode; sort by that
    # ----------------------------
    sort_info = []
    for mode_name, data in mode_dict.items():
        L = np.asarray(data["length_factor"], dtype=float)
        F = np.asarray(data["frequency_GHz"], dtype=float)

        if np.any(np.diff(L) < 0):
            order = np.argsort(L)
            L = L[order]
            F = F[order]

        if not (L.min() <= target_x <= L.max()):
            raise ValueError(f"Mode {mode_name} does not span x={target_x}")

        f_interp = np.interp(target_x, L, F)
        sort_info.append((mode_name, f_interp))

    sort_info.sort(key=lambda t: (t[1], mode_index(t[0])))
    ordered_modes = [name for name, _ in sort_info]

    # Optionally relabel
    if relabel_in_order:
        reordered = {f"mode {i+1}": mode_dict[old] for i, old in enumerate(ordered_modes)}
        ordered_modes = list(reordered.keys())
    else:
        reordered = {name: mode_dict[name] for name in ordered_modes}

    # ----------------------------
    # 2) Shared length axis and interpolated frequency matrix
    # ----------------------------
    modes = ordered_modes
    N = len(modes)

    L = np.asarray(reordered[modes[0]]["length_factor"], dtype=float)
    if np.any(np.diff(L) < 0):
        L = np.sort(L)

    Fmat = np.zeros((len(L), N), dtype=float)

    for i, m in enumerate(modes):
        Li = np.asarray(reordered[m]["length_factor"], dtype=float)
        Fi = np.asarray(reordered[m]["frequency_GHz"], dtype=float)

        if np.any(np.diff(Li) < 0):
            order = np.argsort(Li)
            Li = Li[order]
            Fi = Fi[order]

        if Li.min() > L.min() or Li.max() < L.max():
            raise ValueError(
                f"Mode {m} does not span the shared length axis range "
                f"[{L.min()}, {L.max()}]."
            )

        Fmat[:, i] = np.interp(L, Li, Fi)

    # ----------------------------
    # 3) Detect crossings, skipping same-technical pairs
    # ----------------------------
    crossings = {}
    all_modes_that_cross = set()
    skipped_pairs_same_technical = []

    # Precompute technical tuples for speed
    tech = {m: tech_tuple(m) for m in modes}

    for i in range(N):
        for j in range(i + 1, N):
            Mi, Mj = modes[i], modes[j]

            # Exclusion: if both have a technical tuple and they match, skip
            if tech[Mi] is not None and tech[Mi] == tech[Mj]:
                skipped_pairs_same_technical.append((Mi, Mj))
                continue

            fi = Fmat[:, i]
            fj = Fmat[:, j]
            g = fi - fj

            idxs = np.where(np.diff(np.sign(g)) != 0)[0]
            if idxs.size == 0:
                continue

            def gfun(x):
                return np.interp(x, L, fi) - np.interp(x, L, fj)

            for idx in idxs:
                L1, L2 = L[idx], L[idx + 1]
                try:
                    Lc = brentq(gfun, L1, L2)
                except ValueError:
                    continue

                Fc = np.interp(Lc, L, fi)

                key = f"{Mi}–{Mj}"
                crossings[key] = {
                    "mode_i": Mi,
                    "mode_j": Mj,
                    "length_factor": float(f"{Lc:.12g}"),
                    "frequency_GHz": float(f"{Fc:.12g}"),
                }
                all_modes_that_cross.add(Mi)
                all_modes_that_cross.add(Mj)

    return crossings, sorted(all_modes_that_cross, key=mode_index), ordered_modes, skipped_pairs_same_technical


def add_manual_crossing(crossings, all_modes_that_cross, ordered_modes,
                        mode_i, mode_j, length_factor, frequency_GHz):
    """
    Add a manual crossing entry and update the companion outputs produced by
    find_mode_crossings_non_consecutive(...).

    Parameters
    ----------
    crossings : dict
        The crossings dict returned by find_mode_crossings_non_consecutive.
    all_modes_that_cross : list[str]
        The list returned by find_mode_crossings_non_consecutive.
    ordered_modes : list[str]
        The physical-order list returned by find_mode_crossings_non_consecutive.
    mode_i, mode_j : str
        Mode names (must match the names used in `ordered_modes` / `crossings`).
        Example: "mode 4", "mode 11" (or relabeled names if relabel_in_order=True).
    length_factor : float
        Crossing x-location (Lc).
    frequency_GHz : float
        Crossing frequency (Fc).

    Returns
    -------
    crossings2, all_modes_that_cross2, ordered_modes2
        Updated versions (copies; inputs are not mutated).
    """
    import re

    def mode_index(name: str):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else name

    # Copy so we don't mutate caller's objects unexpectedly
    crossings2 = dict(crossings)

    # Build/update the crossing entry
    key = f"{mode_i}–{mode_j}"
    crossings2[key] = {
        "mode_i": mode_i,
        "mode_j": mode_j,
        "length_factor": float(f"{float(length_factor):.12g}"),
        "frequency_GHz": float(f"{float(frequency_GHz):.12g}"),
    }

    # Update all_modes_that_cross
    s = set(all_modes_that_cross)
    s.add(mode_i)
    s.add(mode_j)
    all_modes_that_cross2 = sorted(s, key=mode_index)

    # ordered_modes is a physical ordering; we shouldn't reorder it here.
    # But we *can* ensure the referenced modes exist in it (append if missing).
    ordered_modes2 = list(ordered_modes)
    if mode_i not in ordered_modes2:
        ordered_modes2.append(mode_i)
    if mode_j not in ordered_modes2:
        ordered_modes2.append(mode_j)

    return crossings2, all_modes_that_cross2, ordered_modes2


def find_mode_crossings(mode_dict):
    """
    Reassigns modes based on increasing frequency at x = 1.0
    and detects all mode crossings with high precision.

    Returns a dict with:
        {
            "mode i–mode j": {
                "mode_i": "mode i",
                "mode_j": "mode j",
                "length_factor": Lc,
                "frequency_GHz": Fc
            },
            ...
        }
    """

    # ---------------------------------------------------------------
    # 1. Reorder modes according to frequency at x = 1.0
    # ---------------------------------------------------------------

    target_x = 1.0
    rename_list = []

    for mode, data in mode_dict.items():
        # print(f"{mode = }")
        # print(f"{data.keys() = }")

        L = np.array(data["length_factor"])
        F = np.array(data["frequency_GHz"])

        # interpolate frequency at x=1.0
        if L.min() <= target_x <= L.max():
            f_interp = np.interp(target_x, L, F)
        else:
            raise ValueError(f"Mode {mode} does not span x=1.0")

        rename_list.append((mode, f_interp))

    # Sort by increasing frequency
    rename_list.sort(key=lambda t: t[1])

    # Build new ordered dictionary with new consistent names
    reordered = {}
    for i, (old_name, _) in enumerate(rename_list, start=1):
        new_name = f"mode {i}"
        reordered[new_name] = mode_dict[old_name]

    # ---------------------------------------------------------------
    # 2. Prepare arrays for crossing detection
    # ---------------------------------------------------------------
    modes = list(reordered.keys())  # already sorted physically
    N = len(modes)

    L = np.array(reordered[modes[0]]["length_factor"])  # shared length axis

    # Frequencies matrix: (num_lengths, num_modes)
    F = np.zeros((len(L), N))
    for i, m in enumerate(modes):
        F[:, i] = reordered[m]["frequency_GHz"]

    crossings = {}
    all_modes_that_cross = []
    # ---------------------------------------------------------------
    # 3. Detect crossings between consecutive modes (and non-consecutive)
    # ---------------------------------------------------------------
    for i in range(N):
        for j in range(i + 1, N):

            Mi, Mj = modes[i], modes[j]
            fi = F[:, i]
            fj = F[:, j]

            g = fi - fj  # difference function

            # sign change → potential crossing
            idxs = np.where(np.diff(np.sign(g)) != 0)[0]

            for idx in idxs:
                L1, L2 = L[idx], L[idx + 1]

                # interpolation-based root function
                def gfun(x):
                    fi_x = np.interp(x, L, fi)
                    fj_x = np.interp(x, L, fj)
                    return fi_x - fj_x

                try:
                    Lc = brentq(gfun, L1, L2)
                except ValueError:
                    continue

                Fc = np.interp(Lc, L, fi)

                crossings[f"{Mi}–{Mj}"] = {
                    "mode_i": Mi,
                    "mode_j": Mj,
                    "length_factor": float(f"{Lc:.12g}"),
                    "frequency_GHz": float(f"{Fc:.12g}"),
                }

                if Mi not in all_modes_that_cross:
                    all_modes_that_cross.append(Mi)
                if Mj not in all_modes_that_cross:
                    all_modes_that_cross.append(Mj)

    return crossings, all_modes_that_cross


def redistribute_modes_for_continuity(mode_dict, alpha=0.3, beta=1.0):
    """
    Reorder modes to maximize continuity.

    Fixes:
      1) Numeric sorting of 'mode N' keys (critical for N>=10)
      2) Predictive tracking to reduce swaps at close approaches:
         cost = beta*|predicted - curr| + alpha*|prev - curr|
    """

    def mode_num(k: str) -> int:
        return int(k.split()[1])

    # --- assemble matrix with NUMERIC mode ordering ---
    keys = sorted(mode_dict.keys(), key=mode_num)
    lengths = np.array(mode_dict[keys[0]]["length_factor"])
    freq_matrix = np.vstack(
        [mode_dict[k]["frequency_GHz"] for k in keys]
    ).T  # (steps, modes)

    T, M = freq_matrix.shape
    tracked = np.zeros_like(freq_matrix)

    # Step 0: keep given ordering
    tracked[0, :] = freq_matrix[0, :]

    # Step 1: simple match to step 0
    cost01 = np.abs(tracked[0, :][:, None] - freq_matrix[1, :][None, :])
    _, col = linear_sum_assignment(cost01)
    tracked[1, :] = freq_matrix[1, col]

    # Steps 2..T-1: predictive assignment
    for t in range(2, T):
        prev = tracked[t - 1, :]
        prevprev = tracked[t - 2, :]
        pred = prev + (prev - prevprev)  # linear extrapolation

        curr = freq_matrix[t, :]
        cost = beta * np.abs(pred[:, None] - curr[None, :]) + alpha * np.abs(
            prev[:, None] - curr[None, :]
        )

        _, col = linear_sum_assignment(cost)
        tracked[t, :] = curr[col]

    # --- rebuild dict ---
    new_mode_dict = {}
    for m in range(M):
        new_mode_dict[f"mode {m + 1}"] = {
            "length_factor": lengths.tolist(),
            "frequency_GHz": tracked[:, m].tolist(),
        }

    return new_mode_dict

def add_normalised_freqs_to_dict(data_dict, f010):
    for key in data_dict.keys():
        data_dict[key]["frequency_normalised"] = [i/f010 for i in data_dict[key]["frequency_GHz"]]

    return data_dict

def plot_modes_without_legend_WIDE(
    renumbered_modes,
    crossings,
    savepath,
    savename,
    inspect: bool = False,
    clusters: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    loglog: bool = False,
    xlim_min = 0.5,
    xlim_max = 1.5,
        normalised=True,
        # clusters=clusters_3=True,
):
    """
    Wide version:
      - data x-range strictly [0.5, 1.5]
      - extra canvas space to the RIGHT of x=1.5 for mode labels
      - optional log–log plotting with log grid
    """

    fig, ax = plt.subplots(figsize=(18, 8))

    # ------------------------------------------------------------------
    # 1. PLOT MODE CURVES
    # ------------------------------------------------------------------
    mode_list = list(renumbered_modes.keys())
    # mode_cmap = plt.cm.get_cmap("nipy_spectral", max(1, len(mode_list)))
    mode_cmap = plt.cm.get_cmap("viridis", max(1, len(mode_list)))
    mode_colors = {mode: mode_cmap(i) for i, mode in enumerate(mode_list)}

    for mode in mode_list:

        xs = np.asarray(renumbered_modes[mode]["length_factor"], dtype=float)
        if normalised:
            ys = np.asarray(renumbered_modes[mode]["frequency_normalised"], dtype=float)
        else:
            ys = np.asarray(renumbered_modes[mode]["frequency_GHz"], dtype=float)

        if loglog:
            mask = (xs > 0) & (ys > 0)
            xs, ys = xs[mask], ys[mask]

        ax.plot(xs, ys, lw=1.0, alpha=0.80, color=mode_colors[mode])

    # ------------------------------------------------------------------
    # 2. PLOT CROSSINGS
    # ------------------------------------------------------------------
    # cross_cmap = plt.cm.get_cmap("tab20", max(4, len(crossings)))
    cross_cmap = plt.cm.get_cmap("copper", len(crossings))
    cross_colors = [cross_cmap(i) for i in range(len(crossings))]

    for idx, (_, c) in enumerate(crossings.items()):

        xs = np.asarray(c["length_factor"], dtype=float)
        if normalised:
            ys = np.asarray(c["frequency_GHz"]/1.3, dtype=float)
        else:
            ys = np.asarray(c["frequency_GHz"], dtype=float)

        if loglog:
            mask = (xs > 0) & (ys > 0)
            xs, ys = xs[mask], ys[mask]

        ax.scatter(
            xs,
            ys,
            s=140,
            facecolors="none",
            edgecolors=cross_colors[idx],
            linewidths=2.0,
            label=idx
        )

    # ------------------------------------------------------------------
    # 3. AXIS SCALE + DATA RANGE
    # ------------------------------------------------------------------
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(xlim_min, xlim_max)

    # Reserve right-side gutter for labels
    fig.subplots_adjust(right=0.80)

    # Label x-position outside data range
    label_x = 1.52 if not loglog else 1.52 * 1.02

    # ------------------------------------------------------------------
    # 3.5 OPTIONAL: CLUSTER RECTANGLES
    # ------------------------------------------------------------------
    if clusters:
        for cluster_id, b in clusters.items():
            try:
                x0, x1 = float(b["x_min"]), float(b["x_max"])
                y0, y1 = float(b["y_min"]), float(b["y_max"])
            except (KeyError, TypeError, ValueError):
                continue

            if loglog and (x0 <= 0 or y0 <= 0):
                continue

            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="k",
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
            )
            ax.add_patch(rect)

            ax.text(
                x0 * (1.01 if loglog else 1.0) + (0 if loglog else 0.002),
                y1 / (1.01 if loglog else 1.0) - (0 if loglog else 0.002),
                str(cluster_id),
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
            )

    # ------------------------------------------------------------------
    # 4. DISTRIBUTED MODE LABELS
    # ------------------------------------------------------------------
    min_sep = 0.004
    repulsion = 0.0015

    mode_positions = []
    for mode in mode_list:

        L = np.asarray(renumbered_modes[mode]["length_factor"], dtype=float)
        if normalised:
            F = np.asarray(renumbered_modes[mode]["frequency_normalised"], dtype=float)
        else:
            F = np.asarray(renumbered_modes[mode]["frequency_GHz"], dtype=float)

        if loglog:
            mask = (L > 0) & (F > 0)
            L, F = L[mask], F[mask]

        if L.size == 0:
            continue

        if L.min() <= 1.5 <= L.max():
            f_interp = np.interp(1.5, L, F)
            mode_positions.append([mode, f_interp])

    mode_positions.sort(key=lambda t: t[1])

    adjusted = []
    for i, (_, f) in enumerate(mode_positions):
        if i == 0:
            adjusted.append(f)
        else:
            prev = adjusted[-1]
            new_y = f
            if not loglog and new_y - prev < min_sep:
                new_y = prev + min_sep + repulsion * (i % 5)
            adjusted.append(new_y)

    for (mode, _), f_adj in zip(mode_positions, adjusted):
        ax.text(
            label_x,
            f_adj,
            mode,
            fontsize=8,
            weight="bold",
            color=mode_colors[mode],
            ha="left",
            va="center",
            alpha=0.85,
            clip_on=False,
        )

    ax.axvline(1.5, color="gray", ls=":", lw=0.7, alpha=0.4)
    ax.axvline(1.0, ls="--", color="k", alpha=0.65, lw=1.0)

    # ------------------------------------------------------------------
    # 5. LABELS + LOG GRID
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=13)
    if normalised:
        ax.set_ylabel("$f_{mnp}/f_{010}$", fontsize=13)
    else:
        ax.set_ylabel("f [GHz]", fontsize=13)

    if loglog:
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.35, lw=0.6)
        ax.grid(which="minor", alpha=0.15, lw=0.4)
    else:
        ax.grid(alpha=0.25, lw=0.5)

    if inspect:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")





def load_mode_metadata_dict(py_path: str, dict_name: str = "TESLA_ALL_modes_dict") -> dict:
    """
    Loads TESLA_ALL_modes_dict from a .py file like TESLA_midcell_ALL_modes_dict.py
    """
    mod = SourceFileLoader("mode_meta_mod", py_path).load_module()
    d = getattr(mod, dict_name, None)
    if not isinstance(d, dict):
        raise ValueError(f"Did not find dict '{dict_name}' in {py_path}")
    return d

def plot_modes_with_mode_legend_and_crossing_labels(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    xlim=(0.93, 1.10),
    show=False,
):
    """
    Clean version:
      - viridis for modes
      - crossings as open circles
      - legend INSIDE plot (loc="best")
      - legend entries = $TM_{010}$ style only
      - legend de-duplicates identical technical names (keeps first occurrence)
      - crossing annotations show the two involved modes
      - crossing y-values divided by 1.3 (to match normalised scaling)
    """

    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    # print(f"{mode_meta['mode_61'] = }")

    def tech_name(mode_key: str) -> str:
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + build de-duplicated legend
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(k.split("_")[-1]) if str(k).split("_")[-1].isdigit() else 10**9
    )

    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    # De-dup legend by technical label
    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs = np.asarray(renumbered_modes[mk]["length_factor"], dtype=float)
        if normalised:
            ys = np.asarray(renumbered_modes[mk]["frequency_normalised"], dtype=float)
        else:
            ys = np.asarray(renumbered_modes[mk]["frequency_GHz"], dtype=float)

        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        lbl = tech_name(mk)

        # keep only one entry per duplicate technical name
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / 1.3  # match normalization

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors="k",
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)
        if len(pair) >= 2:
            t1 = tech_name(pair[0])
            t2 = tech_name(pair[1])
            txt = f"{t1}\n{t2}"

            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 - 0.0025,
                y0 -0.2,
                txt,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=13)
    ax.set_ylabel(r"$f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=13)

    ax.grid(alpha=0.25)

    # --------------------------
    # Legend inside plot (deduped)
    # --------------------------
    ax.legend(
        legend_handles,
        legend_labels,
        # loc="best",
        # loc="center right",
        loc="center left",
        fontsize=12,
        framealpha=0.9,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")


def plot_modes_without_legend_WIDE_mode_groups(
    renumbered_modes,

    savepath,
    savename,
    inspect: bool = False,
    clusters: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    loglog: bool = False,
    mode_groups: Optional[Dict[str, list]] = None,
    # e.g. {
    #   "p index = 0": [1, 19],
    #   "p index = 1": [4, 11, 28, 53],
    #   ...
    # }
    plot_only_grouped: bool = True,
):
    """
    Wide version:
      - data x-range strictly [0.5, 1.5]
      - extra canvas space to the RIGHT of x=1.5 for mode labels
      - optional log–log plotting with log grid
      - NEW: plot only modes belonging to user-provided integer groups, with legend.

    Modes are referenced by integers: 1 -> renumbered_modes["mode 1"] or ["mode_1"] etc.
    This function tolerates: "mode 1", "mode_1", "Mode 1", etc.
    """

    def _mode_key_from_int(i: int) -> str:
        # Try common key formats
        candidates = [
            f"mode {i}",
            f"mode_{i}",
            f"Mode {i}",
            f"Mode_{i}",
        ]
        for k in candidates:
            if k in renumbered_modes:
                return k
        # Fallback: try to find a key ending with the integer
        s = str(i)
        for k in renumbered_modes.keys():
            kk = k.strip().lower().replace("_", " ")
            if kk.startswith("mode ") and kk.split()[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for integer {i} (tried {candidates})")

    # Decide which modes to plot, and how to style them
    # mode_to_grouplabel: dict[mode_key] -> group label string
    mode_to_grouplabel = {}

    if mode_groups is None:
        # Default behavior: plot everything (no grouping legend)
        mode_list = list(renumbered_modes.keys())
    else:
        # Build mapping from modes to group labels
        for group_label, int_list in mode_groups.items():
            for mi in int_list:
                mk = _mode_key_from_int(int(mi))
                mode_to_grouplabel[mk] = group_label

        if plot_only_grouped:
            mode_list = list(mode_to_grouplabel.keys())
        else:
            mode_list = list(renumbered_modes.keys())

    # Stable order (numeric if possible)
    def _mode_num_safe(k: str) -> int:
        kk = k.strip().lower().replace("_", " ")
        try:
            return int(kk.split()[-1])
        except Exception:
            return 10**9

    mode_list = sorted(mode_list, key=_mode_num_safe)

    # ------------------------------------------------------------------
    # Figure / Axes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 8))

    # ------------------------------------------------------------------
    # Styles by group (colors + linestyles)
    # ------------------------------------------------------------------
    if mode_groups is None:
        # Original behavior: one color per mode (no legend)
        mode_cmap = plt.cm.get_cmap("nipy_spectral", max(1, len(mode_list)))
        mode_style = {m: {"color": mode_cmap(i), "ls": "-", "lw": 1.0, "alpha": 0.80}
                      for i, m in enumerate(mode_list)}
        legend_handles = None
    else:
        # One color + linestyle per group, legend shows groups
        group_labels = list(mode_groups.keys())
        group_cmap = plt.cm.get_cmap("tab10", max(1, len(group_labels)))
        linestyles = ["-", "--", ":", "-."]

        group_style = {}
        for gi, glab in enumerate(group_labels):
            group_style[glab] = {
                "color": group_cmap(gi),
                "ls": linestyles[gi % len(linestyles)],
                "lw": 1.4,
                "alpha": 0.90,
            }

        # Apply group style; ungrouped modes (if plot_only_grouped=False) get gray
        mode_style = {}
        for m in mode_list:
            glab = mode_to_grouplabel.get(m, None)
            if glab is None:
                mode_style[m] = {"color": "0.6", "ls": "-", "lw": 0.8, "alpha": 0.35}
            else:
                mode_style[m] = dict(group_style[glab])

        # Legend handles for groups
        legend_handles = []
        for glab in group_labels:
            h = plt.Line2D([0], [0],
                           color=group_style[glab]["color"],
                           linestyle=group_style[glab]["ls"],
                           linewidth=2.0,
                           label=glab)
            legend_handles.append(h)

    # ------------------------------------------------------------------
    # 1. PLOT MODE CURVES (filtered/grouped)
    # ------------------------------------------------------------------
    for mode in mode_list:
        xs = np.asarray(renumbered_modes[mode]["length_factor"], dtype=float)
        ys = np.asarray(renumbered_modes[mode]["frequency_GHz"], dtype=float)

        if loglog:
            mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[mask], ys[mask]

        st = mode_style[mode]
        # ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"], alpha=st["alpha"])
        ax.scatter(xs, ys, color=st["color"], marker='o', s=1, alpha=st["alpha"])

    # # ------------------------------------------------------------------
    # # 2. PLOT CROSSINGS (unchanged)
    # # ------------------------------------------------------------------
    # cross_cmap = plt.cm.get_cmap("tab20", max(4, len(crossings)))
    # cross_colors = [cross_cmap(i) for i in range(len(crossings))]
    #
    # for idx, (_, c) in enumerate(crossings.items()):
    #     xs = np.asarray(c["length_factor"], dtype=float)
    #     ys = np.asarray(c["frequency_GHz"], dtype=float)
    #
    #     if loglog:
    #         mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    #         xs, ys = xs[mask], ys[mask]
    #
    #     ax.scatter(
    #         xs,
    #         ys,
    #         s=140,
    #         facecolors="none",
    #         edgecolors=cross_colors[idx],
    #         linewidths=2.0,
    #     )

    # ------------------------------------------------------------------
    # 3. AXIS SCALE + DATA RANGE
    # ------------------------------------------------------------------
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(0.9, 1.1)

    # Reserve right-side gutter for labels
    fig.subplots_adjust(right=0.80)

    # Label x-position outside data range (only meaningful for modes that reach x=1.5)
    label_x = 1.52 if not loglog else 1.52 * 1.02

    # ------------------------------------------------------------------
    # 3.5 OPTIONAL: CLUSTER RECTANGLES
    # ------------------------------------------------------------------
    if clusters:
        for cluster_id, b in clusters.items():
            try:
                x0, x1 = float(b["x_min"]), float(b["x_max"])
                y0, y1 = float(b["y_min"]), float(b["y_max"])
            except (KeyError, TypeError, ValueError):
                continue

            if loglog and (x0 <= 0 or y0 <= 0):
                continue

            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="k",
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
            )
            ax.add_patch(rect)

            ax.text(
                x0 * (1.01 if loglog else 1.0) + (0 if loglog else 0.002),
                y1 / (1.01 if loglog else 1.0) - (0 if loglog else 0.002),
                str(cluster_id),
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
            )

    # ------------------------------------------------------------------
    # 4. DISTRIBUTED MODE LABELS (only for plotted modes)
    # ------------------------------------------------------------------
    min_sep = 0.004
    repulsion = 0.0015

    mode_positions = []
    for mode in mode_list:
        L = np.asarray(renumbered_modes[mode]["length_factor"], dtype=float)
        F = np.asarray(renumbered_modes[mode]["frequency_GHz"], dtype=float)

        if loglog:
            mask = (L > 0) & (F > 0) & np.isfinite(L) & np.isfinite(F)
            L, F = L[mask], F[mask]

        if L.size == 0:
            continue

        if L.min() <= 1.5 <= L.max():
            f_interp = np.interp(1.5, L, F)
            mode_positions.append([mode, f_interp])

    mode_positions.sort(key=lambda t: t[1])

    adjusted = []
    for i, (_, f) in enumerate(mode_positions):
        if i == 0:
            adjusted.append(f)
        else:
            prev = adjusted[-1]
            new_y = f
            if not loglog and new_y - prev < min_sep:
                new_y = prev + min_sep + repulsion * (i % 5)
            adjusted.append(new_y)

    for (mode, _), f_adj in zip(mode_positions, adjusted):
        ax.text(
            label_x,
            f_adj,
            mode,
            fontsize=8,
            weight="bold",
            color=mode_style[mode]["color"],
            ha="left",
            va="center",
            alpha=0.90,
            clip_on=False,
        )

    ax.axvline(1.5, color="gray", ls=":", lw=0.7, alpha=0.4)
    ax.axvline(1.0, ls="--", color="k", alpha=0.65, lw=1.0)

    # ------------------------------------------------------------------
    # 5. LABELS + GRID
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=13)
    ax.set_ylabel("f [GHz]", fontsize=13)

    if loglog:
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.35, lw=0.6)
        ax.grid(which="minor", alpha=0.15, lw=0.4)
    else:
        ax.grid(alpha=0.25, lw=0.5)

    # ------------------------------------------------------------------
    # 6. LEGEND (grouped only)
    # ------------------------------------------------------------------
    if legend_handles is not None and len(legend_handles) > 0:
        ax.legend(handles=legend_handles, loc="upper left", framealpha=0.85)

    if inspect:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")


def extract_cluster_data(
    mode_dict, crossings, x_min, x_max, y_min, y_max, resolution=200
):
    """
    Extract modes and crossings inside a cluster bounding box.
    Ensures mode curves are interpolated over full [x_min, x_max].
    """

    # ---------------------------------------------------------------
    # 1. Extract crossings inside region
    # ---------------------------------------------------------------
    crossings_in_cluster = {}

    for key, c in crossings.items():
        Lc = c["length_factor"]
        Fc = c["frequency_GHz"]

        if (x_min <= Lc <= x_max) and (y_min <= Fc <= y_max):
            crossings_in_cluster[key] = c

    # If no crossings in cluster, nothing to plot
    if len(crossings_in_cluster) == 0:
        return {}, {}

    # ---------------------------------------------------------------
    # 2. Identify which modes appear in these crossings
    # ---------------------------------------------------------------
    modes_needed = set()
    for c in crossings_in_cluster.values():
        modes_needed.add(c["mode_i"])
        modes_needed.add(c["mode_j"])

    # ---------------------------------------------------------------
    # 3. Interpolate EACH relevant mode over dense grid
    # ---------------------------------------------------------------
    xs_dense = np.linspace(x_min, x_max, resolution)
    modes_in_cluster = {}

    for mode in modes_needed:
        L = np.array(mode_dict[mode]["length_factor"])
        F = np.array(mode_dict[mode]["frequency_GHz"])

        # Always interpolate curve over the entire window
        F_interp = np.interp(xs_dense, L, F)

        modes_in_cluster[mode] = {
            "length_factor": xs_dense.tolist(),
            "frequency_GHz": F_interp.tolist(),
        }

    return modes_in_cluster, crossings_in_cluster


def plot_cluster(
    modes,
    crossings,
    x_min,
    x_max,
    y_min,
    y_max,
    savepath,
    savename,
    title="Cluster Plot",
):

    plt.figure(figsize=(10, 6))

    # ---------------------------------------------------------------
    # Plot modes
    # ---------------------------------------------------------------
    legend_entries = []
    legend_labels = []

    for mode, data in modes.items():
        (line,) = plt.plot(  # ← unpack the single Line2D
            data["length_factor"],
            data["frequency_GHz"],
            lw=1.4,
        )

        legend_entries.append(line)
        legend_labels.append(mode)

    # ---------------------------------------------------------------
    # Plot crossings (unique colours)
    # ---------------------------------------------------------------
    cmap = plt.cm.get_cmap("tab10", len(crossings))
    colors = [cmap(i) for i in range(len(crossings))]

    for idx, (_, c) in enumerate(crossings.items()):
        s = plt.scatter(
            c["length_factor"],
            c["frequency_GHz"],
            s=150,
            facecolors="none",
            edgecolors=colors[idx],
            linewidths=2,
        )

        legend_entries.append(s)
        legend_labels.append(f"{c['mode_i']} × {c['mode_j']}")

    # ---------------------------------------------------------------
    # Axis bounds
    # ---------------------------------------------------------------
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # ---------------------------------------------------------------
    # Labels + legend
    # ---------------------------------------------------------------
    plt.xlabel(r"$\ell=2d/\lambda$", fontsize=12)
    plt.ylabel("f [GHz]", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.25)

    plt.legend(
        legend_entries,
        legend_labels,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(f"{savepath}\\{savename}.png")
    plt.close("all")


def build_savename(prefix: str, mode_keys) -> str:
    """
    Replicates your savename logic:
    savename = prefix + ''.join(k.split()) for k in modes_sub.keys()

    NOTE: dict key iteration order can be non-obvious depending on upstream creation,
    so we sort keys for stable filenames. If you want original order, remove sorted().
    """
    return prefix + "".join("".join(k.split()) for k in sorted(mode_keys))


def run_cluster_plots(
    clusters: Dict[int, Dict[str, Any]],
    renumbered_modes,
    crossings,
    savepath: str,
    *,
    savename_prefix: str = "zoom_crossings_",
    resolution_default: int = 200,
    print_keys: bool = True,
):
    for cluster_id, cfg in clusters.items():
        x_min, x_max = cfg["x_min"], cfg["x_max"]
        y_min, y_max = cfg["y_min"], cfg["y_max"]
        resolution = cfg.get("resolution", resolution_default)

        modes_sub, cross_sub = extract_cluster_data(
            renumbered_modes,
            crossings,
            x_min,
            x_max,
            y_min,
            y_max,
            resolution=resolution,
        )

        if print_keys:
            print(f"[cluster {cluster_id}] {modes_sub.keys() = }")

        # If no crossings/modes in region, skip (extract_cluster_data returns {}, {})
        if not modes_sub or not cross_sub:
            continue

        savename = build_savename(savename_prefix, modes_sub.keys())
        title = f"Crossing Cluster {cluster_id}: {x_min} < x < {x_max}, {y_min} < f < {y_max} GHz"

        plot_cluster(
            modes_sub,
            cross_sub,
            x_min,
            x_max,
            y_min,
            y_max,
            savepath,
            savename=savename,
            title=title,
        )


####################################################################################################################
#                                                                                                                  #
#                                        MODAL PAIR ANALYSIS FUNCTIONS                                             #
#                                                                                                                  #
####################################################################################################################


def read_3D_CST_field_data(
    path: str,
    field_kind: Optional[Literal["E", "H"]] = None,
    coord_unit: Literal["mm", "m"] = "mm",
) -> Dict[str, np.ndarray]:
    """
    Read CST 3D ASCII field export and return 3D numpy arrays.

    Returns:
    data = {
        "x": x,                 # 1D array (meters)
        "y": y,                 # 1D array (meters)
        "z": z,                 # 1D array (meters)
        "Fieldx": Fx,           # complex (Nx,Ny,Nz)
        "Fieldy": Fy,           # complex (Nx,Ny,Nz)
        "Fieldz": Fz,           # complex (Nx,Ny,Nz)
        "absField": absField,   # float   (Nx,Ny,Nz)
    }
    """

    # --------------------------------------------------
    # Read file
    # --------------------------------------------------
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find dashed separator
    sep_idx = None
    for i, ln in enumerate(lines[:2000]):
        s = ln.strip()
        if len(s) >= 20 and set(s) == {"-"}:
            sep_idx = i
            break
    if sep_idx is None:
        raise ValueError("Could not find CST dashed separator line.")

    header = "".join(lines[:sep_idx]).lower()

    # Auto-detect field kind if needed
    if field_kind is None:
        if "exre" in header or "eyre" in header or "ezre" in header:
            field_kind = "E"
        elif "hxre" in header or "hyre" in header or "hzre" in header:
            field_kind = "H"
        else:
            raise ValueError("Could not auto-detect field type (E/H).")

    field_kind = field_kind.upper()
    if field_kind not in ("E", "H"):
        raise ValueError("field_kind must be 'E' or 'H'")

    # --------------------------------------------------
    # Load numeric table
    # --------------------------------------------------
    df = pd.read_csv(StringIO("".join(lines[sep_idx + 1 :])), sep=r"\s+", header=None)

    if df.shape[1] < 9:
        raise ValueError("Expected at least 9 columns in CST field table.")

    df = df.iloc[:, :9]

    df.columns = ["x_u", "y_u", "z_u", "FxRe", "FxIm", "FyRe", "FyIm", "FzRe", "FzIm"]

    # --------------------------------------------------
    # Coordinates (meters)
    # --------------------------------------------------
    x_raw = df["x_u"].astype(float).to_numpy()
    y_raw = df["y_u"].astype(float).to_numpy()
    z_raw = df["z_u"].astype(float).to_numpy()

    if coord_unit == "mm":
        x_raw *= 1e-3
        y_raw *= 1e-3
        z_raw *= 1e-3
    elif coord_unit != "m":
        raise ValueError("coord_unit must be 'mm' or 'm'")

    # Unique sorted coordinates
    x = np.unique(x_raw)
    y = np.unique(y_raw)
    z = np.unique(z_raw)

    Nx, Ny, Nz = len(x), len(y), len(z)

    # --------------------------------------------------
    # Build index maps
    # --------------------------------------------------
    xi = {v: i for i, v in enumerate(x)}
    yi = {v: i for i, v in enumerate(y)}
    zi = {v: i for i, v in enumerate(z)}

    # --------------------------------------------------
    # Allocate 3D arrays
    # --------------------------------------------------
    Fx = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Fy = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Fz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # --------------------------------------------------
    # Fill arrays
    # --------------------------------------------------
    Fx_vals = df["FxRe"].to_numpy() + 1j * df["FxIm"].to_numpy()
    Fy_vals = df["FyRe"].to_numpy() + 1j * df["FyIm"].to_numpy()
    Fz_vals = df["FzRe"].to_numpy() + 1j * df["FzIm"].to_numpy()

    for xv, yv, zv, fx, fy, fz in zip(x_raw, y_raw, z_raw, Fx_vals, Fy_vals, Fz_vals):
        i = xi[xv]
        j = yi[yv]
        k = zi[zv]
        Fx[i, j, k] = fx
        Fy[i, j, k] = fy
        Fz[i, j, k] = fz

    # --------------------------------------------------
    # Magnitude
    # --------------------------------------------------
    absField = np.sqrt(np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + np.abs(Fz) ** 2)

    return {
        "x": x,
        "y": y,
        "z": z,
        "Fieldx": Fx,
        "Fieldy": Fy,
        "Fieldz": Fz,
        "absField": absField,
    }


def mode_string(s: str, underscore=False) -> str:
    """
    Convert strings like:
      'mode 4'   -> 'mode004'
      'mode 10'  -> 'mode010'
      'mode 201' -> 'mode201'
    for mode numbers 1..999.
    """
    s = s.strip().lower()
    if not s.startswith("mode"):
        raise ValueError("Input must start with 'mode'")

    # extract the integer after 'mode'
    if underscore:
        n = int(s.replace("mode_", "").strip())
    else:
        n = int(s.replace("mode", "").strip())

    if not (1 <= n <= 999):
        raise ValueError("Mode number must be between 1 and 999")

    return f"mode{n:03d}"


def app_makedirs_no_movies(
    savepath,
    array_path,
):
    os.makedirs(savepath)
    os.makedirs(array_path)


def add_frequency_at_design_length(mode_dict):
    for mode in range(len(mode_dict.keys())):
        idx = mode_dict[f"mode {mode + 1}"]["length_factor"].index(1.0)
        design_freq = list(mode_dict[f"mode {mode + 1}"]["frequency_GHz"])[idx]
        mode_dict[f"mode {mode + 1}"]["design_freq_GHz"] = design_freq

    return mode_dict


def plot_degeneracy(
    E1_length_factor,
    E1_frequency_GHz,
    E2_length_factor,
    E2_frequency_GHz,
    E1_name,
    E2_name,
    interp_resolution,
    savepath,
    extrapolate=False,
):
    E1_data_CINT_factor, E1_data_CINT_freq = pmm.cubic_spline_interpolation(
        E1_length_factor, E1_frequency_GHz, interp_resolution
    )
    E2_data_CINT_factor, E2_data_CINT_freq = pmm.cubic_spline_interpolation(
        E2_length_factor, E2_frequency_GHz, interp_resolution
    )

    if extrapolate:
        m_1, c_1 = pmm.best_fit(E1_data_CINT_factor, E1_data_CINT_freq)
        m_2, c_2 = pmm.best_fit(E2_data_CINT_factor, E2_data_CINT_freq)
        factor_at_degeneracy, intersection_freq = line_intersection(c_1, m_1, c_2, m_2)

        plt.plot(E1_data_CINT_factor, E1_data_CINT_freq)
        plt.plot(E2_data_CINT_factor, E2_data_CINT_freq)
        plt.scatter(factor_at_degeneracy, intersection_freq)
        plt.show()
        #
        # input(f"{factor_at_degeneracy = }\n{intersection_freq = }")

    else:
        """Find Intersection"""

        intersection_idx, intersection_freq = pmm.find_idx_val_intersection(
            E1_data_CINT_freq, E2_data_CINT_freq
        )

        factor_at_degeneracy = E1_data_CINT_factor[intersection_idx]
        print(f"{factor_at_degeneracy = }")
        print(f"{intersection_freq = }")
        print(f"Normalised {intersection_freq / 1.3}")

    """ plot data """

    xdelta = 0.1
    ydelta = -0.1

    plt.scatter(E1_length_factor, E1_frequency_GHz, s=5)
    plt.plot(E1_data_CINT_factor, E1_data_CINT_freq, label=f"{E1_name}")
    plt.text(
        E1_length_factor[-1] + xdelta, E1_data_CINT_freq[-1] + ydelta, f"{E1_name}"
    )

    plt.scatter(E2_length_factor, E2_frequency_GHz, s=5)
    plt.plot(E2_data_CINT_factor, E2_data_CINT_freq, label=f"{E2_name}")
    plt.text(
        E2_length_factor[-1] + xdelta, E2_data_CINT_freq[-1] + ydelta, f"{E2_name}"
    )

    plt.scatter(
        factor_at_degeneracy,
        intersection_freq,
        marker="o",
        s=120,
        facecolors="none",
        edgecolors="red",
        label=f"Degeneracy at {factor_at_degeneracy:1.6f}",
    )

    plt.text(1., min([min(E1_frequency_GHz), min(E2_frequency_GHz)]), "Design\nLength")

    global_y_min = min(E1_frequency_GHz + E2_frequency_GHz)
    global_y_max = max(E1_frequency_GHz + E2_frequency_GHz)
    plt.vlines(1.0, global_y_min, global_y_max, ls="--", lw=0.6, color="k")
    plt.xlabel(r"$\ell=2d/\lambda$")
    plt.ylabel("$f_{mnp} / f_{010}$")
    plt.legend(loc="upper right")
    plt.savefig(f"{savepath}\\{E1_name}_{E2_name}_degeneracy.png")
    plt.close("all")

    return factor_at_degeneracy, intersection_freq


def load_3d_field(filename):
    """
    Loads the field file and returns:
        field_3d : (nx, ny, nz, 6)
        xs, ys, zs : 1D coordinate arrays
    """

    data = np.loadtxt(filename, skiprows=2)

    xyz = data[:, :3]
    fields = data[:, 3:]  # ExRe...EzIm

    xs = np.unique(xyz[:, 0])
    ys = np.unique(xyz[:, 1])
    zs = np.unique(xyz[:, 2])

    nx, ny, nz = len(xs), len(ys), len(zs)

    # sort by x, y, z
    idx = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    fields_sorted = fields[idx]

    field_3d = fields_sorted.reshape(nx, ny, nz, 6)

    return field_3d, xs, ys, zs


def load_2d_field_yz_x0(filename, *, tol=1e-12):
    """
    Loads a 2D y-z plane field file where x is present as a column (typically all zeros).

    Expected columns per row:
        x, y, z, ExRe, ExIm, EyRe, EyIm, EzRe, EzIm
    Returns:
        field_yz : (ny, nz, 6)
        ys, zs   : 1D coordinate arrays
        x_plane  : the (constant) x value found (e.g. 0.0)

    Notes:
      - Sorts by y then z before reshaping.
      - Raises if x is not (approximately) constant.
    """
    data = np.loadtxt(filename, skiprows=2)

    xyz = data[:, :3]
    fields = data[:, 3:]  # ExRe...EzIm (6 cols)

    if fields.shape[1] != 6:
        raise ValueError(
            f"Expected 6 field columns after x,y,z; got {fields.shape[1]}."
        )

    x_col = xyz[:, 0]
    x_plane = float(np.median(x_col))

    # Verify x is (almost) constant, e.g. all zeros
    if not np.allclose(x_col, x_plane, atol=tol, rtol=0):
        # Provide helpful diagnostics
        raise ValueError(
            f"x column is not constant within tol={tol}. "
            f"min={x_col.min()}, max={x_col.max()}, median={x_plane}"
        )

    yz = xyz[:, 1:3]  # keep y,z only

    ys = np.unique(yz[:, 0])
    zs = np.unique(yz[:, 1])
    ny, nz = len(ys), len(zs)

    # sort by y, then z
    idx = np.lexsort((yz[:, 1], yz[:, 0]))
    fields_sorted = fields[idx]

    # reshape into (ny, nz, 6)
    field_yz = fields_sorted.reshape(ny, nz, 6)

    return field_yz, ys, zs, x_plane


def get_3D_data(
    field_map_filename_E1: str,
    field_map_filename_E2: str,
    array_path: str,
    create_data: bool = True,
    coord_unit: str = "mm",
):
    """
    Uses read_3D_CST_field_data() (must be defined/imported) to read two CST 3D E-field maps,
    then saves/loads ALL arrays as 3D arrays.

    Returned dict matches your original keys, but with the NECESSARY change:
      - it now ALSO returns the x coordinate vectors as xs1 and xs2
        (you already return y and z as ys*/zs*).
    """

    os.makedirs(array_path, exist_ok=True)

    if create_data:
        # --- Read full 3D field dictionaries ---
        E1 = read_3D_CST_field_data(
            field_map_filename_E1, field_kind="E", coord_unit=coord_unit
        )
        E2 = read_3D_CST_field_data(
            field_map_filename_E2, field_kind="E", coord_unit=coord_unit
        )

        # --- Extract 3D components ---
        E1_Ex, E1_Ey, E1_Ez = E1["Fieldx"], E1["Fieldy"], E1["Fieldz"]
        E2_Ex, E2_Ey, E2_Ez = E2["Fieldx"], E2["Fieldy"], E2["Fieldz"]

        # coordinate vectors (1D)
        x1, y1, z1 = E1["x"], E1["y"], E1["z"]
        x2, y2, z2 = E2["x"], E2["y"], E2["z"]


        # sanity: require identical grids (same shape + coords)
        if E1_Ex.shape != E2_Ex.shape:
            raise ValueError(
                f"E1 and E2 grid shapes differ: {E1_Ex.shape} vs {E2_Ex.shape}"
            )
        if not (np.allclose(x1, x2) and np.allclose(y1, y2) and np.allclose(z1, z2)):
            raise ValueError("E1 and E2 coordinate vectors differ (x/y/z).")

        # --- Plus/minus combos (3D) ---
        Ex_plus = E1_Ex + E2_Ex
        Ey_plus = E1_Ey + E2_Ey
        Ez_plus = E1_Ez + E2_Ez
        Ex_minus = E1_Ex - E2_Ex
        Ey_minus = E1_Ey - E2_Ey
        Ez_minus = E1_Ez - E2_Ez

        # --- Magnitudes (3D) ---
        abs_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2 + np.abs(E1_Ez) ** 2)
        abs_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2 + np.abs(E2_Ez) ** 2)
        abs_add = np.sqrt(
            np.abs(Ex_plus) ** 2 + np.abs(Ey_plus) ** 2 + np.abs(Ez_plus) ** 2
        )
        abs_sub = np.sqrt(
            np.abs(Ex_minus) ** 2 + np.abs(Ey_minus) ** 2 + np.abs(Ez_minus) ** 2
        )

        # --- Transverse Fields (3D) ---
        trans_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2)
        trans_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2)

        # --- Save everything (3D arrays + coordinate vectors) ---
        np.save(os.path.join(array_path, "abs_E1.npy"), abs_E1)
        np.save(os.path.join(array_path, "E1_Ex.npy"), E1_Ex)
        np.save(os.path.join(array_path, "E1_Ey.npy"), E1_Ey)
        np.save(os.path.join(array_path, "E1_Ez.npy"), E1_Ez)
        np.save(os.path.join(array_path, "x1.npy"), x1)
        np.save(os.path.join(array_path, "y1.npy"), y1)
        np.save(os.path.join(array_path, "z1.npy"), z1)

        np.save(os.path.join(array_path, "abs_E2.npy"), abs_E2)
        np.save(os.path.join(array_path, "E2_Ex.npy"), E2_Ex)
        np.save(os.path.join(array_path, "E2_Ey.npy"), E2_Ey)
        np.save(os.path.join(array_path, "E2_Ez.npy"), E2_Ez)
        np.save(os.path.join(array_path, "x2.npy"), x2)
        np.save(os.path.join(array_path, "y2.npy"), y2)
        np.save(os.path.join(array_path, "z2.npy"), z2)

        np.save(os.path.join(array_path, "trans_E1.npy"), trans_E1)
        np.save(os.path.join(array_path, "trans_E2.npy"), trans_E2)

        np.save(os.path.join(array_path, "abs_add.npy"), abs_add)
        np.save(os.path.join(array_path, "Ex_plus.npy"), Ex_plus)
        np.save(os.path.join(array_path, "Ey_plus.npy"), Ey_plus)
        np.save(os.path.join(array_path, "Ez_plus.npy"), Ez_plus)

        np.save(os.path.join(array_path, "abs_sub.npy"), abs_sub)
        np.save(os.path.join(array_path, "Ex_minus.npy"), Ex_minus)
        np.save(os.path.join(array_path, "Ey_minus.npy"), Ey_minus)
        np.save(os.path.join(array_path, "Ez_minus.npy"), Ez_minus)

    else:
        # --- Load everything (3D arrays + coordinate vectors) ---
        abs_E1 = np.load(os.path.join(array_path, "abs_E1.npy"))
        E1_Ex = np.load(os.path.join(array_path, "E1_Ex.npy"))
        E1_Ey = np.load(os.path.join(array_path, "E1_Ey.npy"))
        E1_Ez = np.load(os.path.join(array_path, "E1_Ez.npy"))
        x1 = np.load(os.path.join(array_path, "x1.npy"))
        y1 = np.load(os.path.join(array_path, "y1.npy"))
        z1 = np.load(os.path.join(array_path, "z1.npy"))

        abs_E2 = np.load(os.path.join(array_path, "abs_E2.npy"))
        E2_Ex = np.load(os.path.join(array_path, "E2_Ex.npy"))
        E2_Ey = np.load(os.path.join(array_path, "E2_Ey.npy"))
        E2_Ez = np.load(os.path.join(array_path, "E2_Ez.npy"))
        x2 = np.load(os.path.join(array_path, "x2.npy"))
        y2 = np.load(os.path.join(array_path, "y2.npy"))
        z2 = np.load(os.path.join(array_path, "z2.npy"))

        trans_E1 = np.load(os.path.join(array_path, "trans_E1.npy"))
        trans_E2 = np.load(os.path.join(array_path, "trans_E2.npy"))

        abs_add = np.load(os.path.join(array_path, "abs_add.npy"))
        Ex_plus = np.load(os.path.join(array_path, "Ex_plus.npy"))
        Ey_plus = np.load(os.path.join(array_path, "Ey_plus.npy"))
        Ez_plus = np.load(os.path.join(array_path, "Ez_plus.npy"))

        abs_sub = np.load(os.path.join(array_path, "abs_sub.npy"))
        Ex_minus = np.load(os.path.join(array_path, "Ex_minus.npy"))
        Ey_minus = np.load(os.path.join(array_path, "Ey_minus.npy"))
        Ez_minus = np.load(os.path.join(array_path, "Ez_minus.npy"))

    print(f"{E1_Ex.shape = }")
    print(f"{E1_Ey.shape = }")
    print(f"{E1_Ez.shape = }")

    # Keep original naming scheme; add xs1/xs2 as the necessary change.
    return {
        "abs_E1": abs_E1,
        "E1_Ex": E1_Ex,
        "E1_Ey": E1_Ey,
        "E1_Ez": E1_Ez,
        "xs1": x1,   # <-- added (necessary)
        "ys1": y1,
        "zs1": z1,
        "abs_E2": abs_E2,
        "E2_Ex": E2_Ex,
        "E2_Ey": E2_Ey,
        "E2_Ez": E2_Ez,
        "xs2": x2,   # <-- added (necessary)
        "ys2": y2,
        "zs2": z2,
        "abs_add": abs_add,
        "Ex_plus": Ex_plus,
        "Ey_plus": Ey_plus,
        "Ez_plus": Ez_plus,
        "trans_E1": trans_E1,
        "trans_E2": trans_E2,
        "abs_sub": abs_sub,
        "Ex_minus": Ex_minus,
        "Ey_minus": Ey_minus,
        "Ez_minus": Ez_minus,
    }


import numpy as np
from scipy.interpolate import RegularGridInterpolator

def _rotate_cartesian_3d_field_about_z(field_xyz, x, y, z, angle_deg, *, fill_value=0.0):
    """
    Rotate a 3D field defined on a rectilinear Cartesian grid about the z-axis by angle_deg,
    resampling back onto the SAME native (x,y,z) grid.

    field_xyz must be shaped (Nx, Ny, Nz) corresponding to (x, y, z).
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    F = np.asarray(field_xyz)

    if F.shape != (len(x), len(y), len(z)):
        raise ValueError(f"Field shape {F.shape} does not match (len(x),len(y),len(z))="
                         f"{(len(x),len(y),len(z))}.")

    interp = RegularGridInterpolator((x, y, z), F, bounds_error=False, fill_value=fill_value)

    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    a = np.radians(angle_deg)

    # inverse map to sample source coordinates
    Xs = Xg*np.cos(-a) - Yg*np.sin(-a)
    Ys = Xg*np.sin(-a) + Yg*np.cos(-a)
    xy_src = np.column_stack([Xs.ravel(), Ys.ravel()])

    F_rot = np.empty_like(F, dtype=np.float64)
    for k, z0 in enumerate(z):
        pts = np.column_stack([xy_src, np.full(xy_src.shape[0], z0, dtype=np.float64)])
        F_rot[:, :, k] = interp(pts).reshape(len(x), len(y))

    return F_rot


def _alignment_angle_to_center_x_plane(abs_E, x, y, z, *, center_x_mode="middle"):
    """
    Find global peak of abs_E and return angle_deg that rotates it into abs_E[center_x,:,:].

    Arrays are assumed shaped [x,y,z]. Aligning to abs_E[center_x,:,:] means x'≈0 plane.
    """
    x = np.asarray(x); y = np.asarray(y)
    abs_E = np.asarray(abs_E)

    nx = abs_E.shape[0]
    ix_center = nx // 2 if center_x_mode == "middle" else int(np.argmin(np.abs(x)))

    ixp, iyp, izp = np.unravel_index(np.argmax(abs_E), abs_E.shape)
    x_peak = float(x[ixp])
    y_peak = float(y[iyp])

    # if on axis, azimuth undefined; no meaningful rotation
    if np.isclose(x_peak, 0.0) and np.isclose(y_peak, 0.0):
        return 0.0, (ixp, iyp, izp), (ixp, iyp, izp), ix_center

    phi_peak_deg = np.degrees(np.arctan2(y_peak, x_peak))
    angle_deg = 90.0 - phi_peak_deg
    return angle_deg, (ixp, iyp, izp), None, ix_center

def align_E1_E2_abs_to_vertical_plane(
    data_dict,
    *,
    center_x_mode="closest_to_zero",   # recommended for real CST grids
    fill_value=0.0,
    verify=True,
):
    """
    Replaces the following keys in the returned dict with rotated/aligned versions:

        field_keys = [
            "abs_E1","abs_E2","abs_add","abs_sub",
            "E1_Ex","E1_Ey","E1_Ez",
            "E2_Ex","E2_Ey","E2_Ez",
            "trans_E1","trans_E2",
            "Ex_plus","Ey_plus","Ez_plus",
            "Ex_minus","Ey_minus","Ez_minus",
        ]

    Alignment rule:
      - Find global peak of abs_E1
      - Rotate about z-axis so the peak lies in the vertical plane abs_E1[x_center, :, :]
        where x_center corresponds to coordinate x≈0 (not index midpoint by default).

    Handles CST axis-order mismatch by auto-detecting whether arrays are (Nx,Ny,Nz) or (Ny,Nx,Nz).
    """

    field_keys = [
        "abs_E1",
        "abs_E2",
        "abs_add",
        "abs_sub",
        "E1_Ex",
        "E1_Ey",
        "E1_Ez",
        "E2_Ex",
        "E2_Ey",
        "E2_Ez",
        "trans_E1",
        "trans_E2",
        "Ex_plus",
        "Ey_plus",
        "Ez_plus",
        "Ex_minus",
        "Ey_minus",
        "Ez_minus",
    ]

    # --- Require coordinates ---
    for k in ("xs1", "ys1", "zs1"):
        if k not in data_dict:
            raise KeyError(f'Missing "{k}" in data_dict. Add x1 as "xs1" in get_3D_data().')

    x = np.asarray(data_dict["xs1"])
    y = np.asarray(data_dict["ys1"])
    z = np.asarray(data_dict["zs1"])

    # --- Sanity: required fields exist ---
    missing = [k for k in field_keys if k not in data_dict]
    if missing:
        raise KeyError(f"data_dict missing required field keys: {missing}")

    # --- Detect array axis order using abs_E1 ---
    abs_E1_in = np.asarray(data_dict["abs_E1"])
    if abs_E1_in.ndim != 3:
        raise ValueError(f'Expected 3D arrays. Got abs_E1 shape={abs_E1_in.shape}')

    # We want internal order (Nx,Ny,Nz).
    # CST sometimes stores (Ny,Nx,Nz). Detect via matching coordinate lengths.
    if abs_E1_in.shape == (len(x), len(y), len(z)):
        to_internal = lambda A: np.asarray(A)               # already (Nx,Ny,Nz)
        from_internal = lambda A: A                         # no-op
        internal_order = "Nx,Ny,Nz"
    elif abs_E1_in.shape == (len(y), len(x), len(z)):
        to_internal = lambda A: np.asarray(A).transpose(1, 0, 2)   # (Ny,Nx,Nz)->(Nx,Ny,Nz)
        from_internal = lambda A: A.transpose(1, 0, 2)             # back to (Ny,Nx,Nz)
        internal_order = "Ny,Nx,Nz (transposed)"
    else:
        raise ValueError(
            f"abs_E1 shape {abs_E1_in.shape} matches neither "
            f"(len(x),len(y),len(z))={(len(x),len(y),len(z))} nor "
            f"(len(y),len(x),len(z))={(len(y),len(x),len(z))}. "
            "Check CST reader / coordinate vectors."
        )

    # --- Choose center x-index by coordinate (x≈0) ---
    if center_x_mode == "closest_to_zero":
        ix_center = int(np.argmin(np.abs(x)))
    else:
        ix_center = len(x) // 2

    # --- Compute alignment angle from abs_E1 peak (in internal order) ---
    abs_E1 = to_internal(abs_E1_in)
    ixp, iyp, izp = np.unravel_index(np.argmax(abs_E1), abs_E1.shape)
    x_peak = float(x[ixp])
    y_peak = float(y[iyp])

    # If peak is on axis, azimuth undefined => no meaningful rotation
    if np.isclose(x_peak, 0.0) and np.isclose(y_peak, 0.0):
        angle_deg = 0.0
    else:
        # rotate so x'≈0 for the peak => peak lies in x_center plane
        phi_peak_deg = np.degrees(np.arctan2(y_peak, x_peak))
        angle_deg = 90.0 - phi_peak_deg

    # --- Rotation helper: rotate one 3D field in internal order (Nx,Ny,Nz) ---
    def rotate_field_internal(F_internal):
        interp = RegularGridInterpolator((x, y, z), F_internal, bounds_error=False, fill_value=fill_value)

        Xg, Yg = np.meshgrid(x, y, indexing="ij")
        a = np.radians(angle_deg)

        # inverse-map sampling
        Xs = Xg*np.cos(-a) - Yg*np.sin(-a)
        Ys = Xg*np.sin(-a) + Yg*np.cos(-a)
        xy_src = np.column_stack([Xs.ravel(), Ys.ravel()])

        F_rot = np.empty_like(F_internal, dtype=np.float64)
        for k, z0 in enumerate(z):
            pts = np.column_stack([xy_src, np.full(xy_src.shape[0], z0, dtype=np.float64)])
            F_rot[:, :, k] = interp(pts).reshape(len(x), len(y))
        return F_rot

    # --- Rotate + replace listed fields ---
    out = dict(data_dict)
    for k in field_keys:
        F_in = np.asarray(out[k])
        F_int = to_internal(F_in)
        F_rot_int = rotate_field_internal(F_int)
        out[k] = from_internal(F_rot_int)

    # --- Verify: abs_E1 peak moved into x_center plane (internal check) ---
    if verify and angle_deg != 0.0:
        abs_E1_rot = to_internal(out["abs_E1"])
        ixr, iyr, izr = np.unravel_index(np.argmax(abs_E1_rot), abs_E1_rot.shape)
        if ixr != ix_center:
            raise RuntimeError(
                f"Alignment check failed. After rotation, abs_E1 peak x-index={ixr}, "
                f"expected center_x_index={ix_center}. "
                f"(Angle was {angle_deg:.6g} deg; internal order was {internal_order}.)"
            )

    # Metadata (optional, doesn’t affect your listed keys)
    out["align_angle_deg"] = float(angle_deg)
    out["align_center_x_index"] = int(ix_center)
    out["align_internal_order_detected"] = internal_order

    return out




def parse_mode_number(mode_str: str) -> int:
    """
    Convert strings like:
      'mode 4', 'mode4', 'mode 004', 'MODE 10'
    into an integer mode number.

    Returns int in range 1..999.
    """
    if not isinstance(mode_str, str):
        raise TypeError("mode_str must be a string")

    match = re.search(r"mode\s*0*([1-9]\d{0,2})", mode_str.lower())
    if not match:
        raise ValueError(f"Could not parse mode number from '{mode_str}'")

    return int(match.group(1))


def extract_3d_field_slices(
    data: Dict[str, np.ndarray],
    field_keys: List[str],
) -> Dict[str, np.ndarray]:
    """
    From a data dict containing multiple 3D fields, extract three slices
    from each field:

      iris             = field[:, :, 0]
      transverse_mid   = field[:, :, mid_pixel]
      longitudinal_mid = field[mid_pixel, :, :]

    Parameters
    ----------
    data : dict
        Dictionary containing 3D numpy arrays.
    field_keys : list of str
        Keys in `data` corresponding to 3D fields to slice.

    Returns
    -------
    slices : dict
        Flat dictionary with keys like:
          '<field>_iris'
          '<field>_transverse_mid'
          '<field>_longitudinal_mid'
    """

    slices: Dict[str, np.ndarray] = {}

    for key in field_keys:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in data dict")

        field = data[key]
        if not isinstance(field, np.ndarray) or field.ndim != 3:
            raise ValueError(
                f"'{key}' must be a 3D numpy array, got shape {getattr(field, 'shape', None)}"
            )

        Nx, Ny, Nz = field.shape
        mid_trans_pixel = Nx // 2  # consistent with earlier convention
        mid_longit_pixel = Nz // 2  # consistent with earlier convention

        slices[f"{key}_iris_1"] = field[:, :, 0]
        slices[f"{key}_iris_2"] = field[:, :, Nz-1]
        slices[f"{key}_transverse_mid"] = field[:, :, mid_longit_pixel]
        slices[f"{key}_longitudinal_mid"] = field[mid_trans_pixel, :, :]

    return slices


def extract_abs_slices(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    From a data dict containing 3D abs field maps:
      abs_E1, abs_E2, abs_add, abs_sub

    extract three slices from each:
      iris             = map[:, :, 0]
      transverse_mid   = map[:, :, mid_pixel]
      longitudinal_mid = map[mid_pixel, :, :]

    Returns a flat dict with 12 entries.
    """

    abs_keys = ["abs_E1", "abs_E2", "abs_add", "abs_sub"]
    out: Dict[str, np.ndarray] = {}

    for key in abs_keys:
        if key not in data:
            raise KeyError(f"Missing '{key}' in data dict")

        field = data[key]
        if field.ndim != 3:
            raise ValueError(f"{key} must be a 3D array, got shape {field.shape}")

        Nx, Ny, Nz = field.shape
        mid_pixel = Nz // 2  # consistent with earlier usage

        out[f"{key}_iris"] = field[:, :, 0]
        out[f"{key}_transverse_mid"] = field[:, :, mid_pixel]
        out[f"{key}_longitudinal_mid"] = field[mid_pixel, :, :]

    return out


def get_data(
    field_map_filename_E1,
    field_map_filename_E2,
    array_path,
    create_data=True,
):
    if create_data:
        E1_field, ys1, zs1, x_plane_1 = load_2d_field_yz_x0(f"{field_map_filename_E1}")
        E2_field, ys2, zs2, x_plane_2 = load_2d_field_yz_x0(f"{field_map_filename_E2}")

        E1_Ex = E1_field[:, :, 0]
        E1_Ey = E1_field[:, :, 2]
        E1_Ez = E1_field[:, :, 4]
        E2_Ex = E2_field[:, :, 0]
        E2_Ey = E2_field[:, :, 2]
        E2_Ez = E2_field[:, :, 4]

        Ex_plus = E1_Ex + E2_Ex
        Ey_plus = E1_Ey + E2_Ey
        Ez_plus = E1_Ez + E2_Ez
        Ex_minus = E1_Ex - E2_Ex
        Ey_minus = E1_Ey - E2_Ey
        Ez_minus = E1_Ez - E2_Ez

        abs_E1 = np.sqrt(E1_Ex**2.0 + E1_Ey**2.0 + E1_Ez**2.0)
        abs_E2 = np.sqrt(E2_Ex**2.0 + E2_Ey**2.0 + E2_Ez**2.0)
        abs_add = np.sqrt(Ex_plus**2.0 + Ey_plus**2.0 + Ez_plus**2.0)
        abs_sub = np.sqrt(Ex_minus**2.0 + Ey_minus**2.0 + Ez_minus**2.0)

        np.save(f"{array_path}\\abs_E1.npy", abs_E1)
        np.save(f"{array_path}\\E1_Ex.npy", E1_Ex)
        np.save(f"{array_path}\\E1_Ey.npy", E1_Ey)
        np.save(f"{array_path}\\E1_Ez.npy", E1_Ez)
        # np.save(f"{array_path}\\xs1.npy", xs1)
        np.save(f"{array_path}\\ys1.npy", ys1)
        np.save(f"{array_path}\\zs1.npy", zs1)
        np.save(f"{array_path}\\abs_E2.npy", abs_E2)
        np.save(f"{array_path}\\E2_Ex.npy", E2_Ex)
        np.save(f"{array_path}\\E2_Ey.npy", E2_Ey)
        np.save(f"{array_path}\\E2_Ez.npy", E2_Ez)
        # np.save(f"{array_path}\\xs2.npy", xs2)
        np.save(f"{array_path}\\ys2.npy", ys2)
        np.save(f"{array_path}\\zs2.npy", zs2)
        np.save(f"{array_path}\\abs_add.npy", abs_add)
        np.save(f"{array_path}\\Ex_plus.npy", Ex_plus)
        np.save(f"{array_path}\\Ey_plus.npy", Ey_plus)
        np.save(f"{array_path}\\Ez_plus.npy", Ez_plus)
        np.save(f"{array_path}\\abs_sub.npy", abs_sub)
        np.save(f"{array_path}\\Ex_minus.npy", Ex_minus)
        np.save(f"{array_path}\\Ey_minus.npy", Ey_minus)
        np.save(f"{array_path}\\Ez_minus.npy", Ez_minus)

    else:
        abs_E1 = np.load(f"{array_path}\\abs_E1.npy")
        E1_Ex = np.load(f"{array_path}\\E1_Ex.npy")
        E1_Ey = np.load(f"{array_path}\\E1_Ey.npy")
        E1_Ez = np.load(f"{array_path}\\E1_Ez.npy")
        # xs1 = np.load(f"{array_path}\\xs1.npy")
        ys1 = np.load(f"{array_path}\\ys1.npy")
        zs1 = np.load(f"{array_path}\\zs1.npy")
        abs_E2 = np.load(f"{array_path}\\abs_E2.npy")
        E2_Ex = np.load(f"{array_path}\\E2_Ex.npy")
        E2_Ey = np.load(f"{array_path}\\E2_Ey.npy")
        E2_Ez = np.load(f"{array_path}\\E2_Ez.npy")
        # xs2 = np.load(f"{array_path}\\xs2.npy")
        ys2 = np.load(f"{array_path}\\ys2.npy")
        zs2 = np.load(f"{array_path}\\zs2.npy")
        abs_add = np.load(f"{array_path}\\abs_add.npy")
        Ex_plus = np.load(f"{array_path}\\Ex_plus.npy")
        Ey_plus = np.load(f"{array_path}\\Ey_plus.npy")
        Ez_plus = np.load(f"{array_path}\\Ez_plus.npy")
        abs_sub = np.load(f"{array_path}\\abs_sub.npy")
        Ex_minus = np.load(f"{array_path}\\Ex_minus.npy")
        Ey_minus = np.load(f"{array_path}\\Ey_minus.npy")
        Ez_minus = np.load(f"{array_path}\\Ez_minus.npy")

    print(f"{E1_Ex.shape = }")
    print(f"{E1_Ey.shape = }")
    print(f"{E1_Ez.shape = }")

    return {
        "abs_E1": abs_E1,
        "E1_Ex": E1_Ex,
        "E1_Ey": E1_Ey,
        "E1_Ez": E1_Ez,
        # "xs1": xs1,
        "ys1": ys1,
        "zs1": zs1,
        "abs_E2": abs_E2,
        "E2_Ex": E2_Ex,
        "E2_Ey": E2_Ey,
        "E2_Ez": E2_Ez,
        # "xs2": xs2,
        "ys2": ys2,
        "zs2": zs2,
        "abs_add": abs_add,
        "Ex_plus": Ex_plus,
        "Ey_plus": Ey_plus,
        "Ez_plus": Ez_plus,
        "abs_sub": abs_sub,
        "Ex_minus": Ex_minus,
        "Ey_minus": Ey_minus,
        "Ez_minus": Ez_minus,
    }


def plot_abs_fields_3D(
    abs_add,
    abs_sub,
    xs,
    ys,
    zs,
    savepath,
):
    """
    Create true 3D scatter plots of abs_add and abs_sub.

    Parameters
    ----------
    abs_add : ndarray (nx, ny, nz)
        Magnitude of the PLUS combination field.
    abs_sub : ndarray (nx, ny, nz)
        Magnitude of the MINUS combination field.
    xs, ys, zs : 1D coordinate arrays
    point_size : float
        Scatter point size
    """
    point_size = 20
    # Build coordinate grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Flatten everything
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()

    add_f = abs_add.flatten()
    sub_f = abs_sub.flatten()

    # Mask out zero values
    mask_add = add_f != 0
    mask_sub = sub_f != 0

    # --- Plot abs_add ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc1 = ax.scatter(
        Xf[mask_add],
        Yf[mask_add],
        Zf[mask_add],
        c=add_f[mask_add],
        s=point_size,
        cmap="viridis",
    )

    plt.colorbar(sc1, shrink=0.65, label="|E_plus|")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("True 3D Scatter of Field Magnitude (PLUS)")
    plt.tight_layout()
    plt.savefig(f"{savepath}\\additive_3D.png")
    plt.close("all")

    # --- Plot abs_sub ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc2 = ax.scatter(
        Xf[mask_sub],
        Yf[mask_sub],
        Zf[mask_sub],
        c=sub_f[mask_sub],
        s=point_size,
        cmap="viridis",
    )

    plt.colorbar(sc2, shrink=0.65, label="|E_minus|")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("True 3D Scatter of Field Magnitude (MINUS)")
    plt.tight_layout()
    plt.savefig(f"{savepath}\\subtractive_3D.png")
    plt.close("all")


def get_cavity_profile_for_current_xsection(array):

    array = np.array(array)
    z = range(array.shape[0])
    y = range(array.shape[1])
    Z, Y = np.meshgrid(z, y)
    z_indices = []
    y_indices = []
    profile_2D = np.empty(
        (
            array.shape[0],
            array.shape[1],
        )
    )
    # profile = np.zeros((abs_E_2D_array.shape[1], abs_E_2D_array.shape[0],), dtype=np.float64)
    # print(f"{array.shape = }")
    # print(f"{len(Z) = }")
    # print(f"{len(Y) = }")
    for i in range(1, array.shape[0] - 1, 1):
        for j in range(1, array.shape[1] - 1, 1):
            if array[i, j] == float(0.0):
                if any(
                    [
                        array[i - 1, j],
                        array[i + 1, j],
                        array[i, j - 1],
                        array[i, j + 1],
                        array[i - 1, j - 1],
                        array[i - 1, j + 1],
                        array[i + 1, j - 1],
                        array[i + 1, j + 1],
                    ],
                ):
                    profile_2D[i, j] = 1.0
                    z_indices.append(Z[i, j])
                    y_indices.append(Y[i, j])
                else:
                    pass


def save_out_iris_slice_image(array, savepath, savename):
    global_max = array.max()
    global_max_MVm = global_max * 1.0e-6
    plane = array[:, :, 0]
    iris_max = plane.max()
    iris_max_MVm = iris_max * 1.0e-6
    print(f"\n{savename}")
    print(f"{plane.shape = }")
    print(f"{global_max_MVm = }")
    print(f"{iris_max_MVm = }")
    plane_norm = plane / global_max
    profile_2D, z_profile, y_profile = get_cavity_profile_for_current_xsection(plane)
    iris = plt.imshow(plane_norm, origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(iris)
    plt.text(
        5,
        array.shape[1] - 20,
        f"Transverse Iris\nglobal max. = {global_max_MVm:1.1f} MV/m\niris max. = {iris_max_MVm:1.1f} MV/m",
        color="white",
    )
    plt.scatter(z_profile, y_profile, marker=".", s=3.0, color="r")
    plt.savefig(f"{savepath}\\{savename}_TransverseIris.png")
    plt.close("all")


def line_intersection(m1, c1, m2, c2):
    """
    Returns the intersection point (x, y) of two lines:
    y = m1*x + c1
    y = m2*x + c2

    Returns:
        (x, y) tuple if lines intersect
        None if lines are parallel but distinct
        "infinite" if lines are identical
    """
    if m1 == m2:
        if c1 == c2:
            return "infinite"  # same line
        else:
            return None  # parallel, no intersection

    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return (x, y)


def plot_field_grid(
    E1_Ex,
    E2_Ex,
    E1_Ey,
    E2_Ey,
    E1_Ez,
    E2_Ez,
    abs_E1,
    abs_E2,
    abs_add,
    abs_sub,
    savepath,
    savename,
    fontsize=20,
):
    # Stack images in plotting order
    imgs = [
        [E1_Ex, E2_Ex],
        [E1_Ey, E2_Ey],
        [E1_Ez, E2_Ez],
        [abs_E1, abs_E2],
        [abs_add, abs_sub],
    ]

    # Row labels (top → bottom)
    row_ylabels = [r"$E_x$", r"$E_y$", r"$E_z$", r"$|E|$"]

    # Global color scale
    all_vals = np.concatenate([np.ravel(np.asarray(im)) for row in imgs for im in row])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    fig, axes = plt.subplots(
        5, 2, sharex=True, sharey=True, figsize=(8, 14), constrained_layout=True
    )

    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    cm_signed = "RdBu_r"
    cm_mag = "viridis"

    for r in range(5):
        for c in range(2):
            ax = axes[r, c]
            data = imgs[r][c]

            cmap = cm_signed if r < 3 else cm_mag

            im = ax.imshow(
                data,
                # vmin=vmin,
                # vmax=vmax,
                cmap=cmap,
                origin="upper",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # One shared colorbar per row
            if c == 1:
                cb = fig.colorbar(im, ax=axes[r, :], fraction=0.03, pad=0.01)
                cb.ax.tick_params(labelsize=8)

        # Left-column y-labels for top 4 rows
        if r < 4:
            axes[r, 0].set_ylabel(
                row_ylabels[r], rotation=0, labelpad=25, fontsize=fontsize, va="center"
            )

    # Bottom row: independent y-labels
    axes[4, 0].set_ylabel(
        r"$E_+$", rotation=0, labelpad=25, va="center", fontsize=fontsize
    )
    axes[4, 1].set_ylabel(
        r"$E_{-}$", rotation=0, labelpad=25, va="center", fontsize=fontsize
    )

    # Ensure output directory exists
    os.makedirs(savepath, exist_ok=True)
    # plt.show()
    plt.savefig(f"{savepath}\\{savename}.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def filter_modes_with_m_zero(mode_dict):
    return {
        mode_name: mode_data
        for mode_name, mode_data in mode_dict.items()
        if mode_data.get("m") == 0
    }


def filter_modes_with_m_one(mode_dict):
    return {
        mode_name: mode_data
        for mode_name, mode_data in mode_dict.items()
        if mode_data.get("m") == 1
    }

def filter_modes_with_m_two(mode_dict):
    return {
        mode_name: mode_data
        for mode_name, mode_data in mode_dict.items()
        if mode_data.get("m") == 2
    }


def crossings_xy_lists(
    xs_list: List[List[float]],
    ys_list: List[List[float]],
    *,
    tol: float = 1e-9,
    dedup_tol: float = 1e-8,
) -> Tuple[List[float], List[float]]:
    """
    Find intersection points between any pair of datasets (polylines) and return:
      (x_crossings, y_crossings)

    Each dataset is treated as piecewise-linear between consecutive points.
    """
    if len(xs_list) != len(ys_list):
        raise ValueError("xs_list and ys_list must have the same outer length.")

    n = len(xs_list)
    for k in range(n):
        if len(xs_list[k]) != len(ys_list[k]):
            raise ValueError(f"Dataset {k}: x and y lengths differ.")

    def line_params(x0, y0, x1, y1):
        dx = x1 - x0
        if abs(dx) <= tol:
            return None, None
        m = (y1 - y0) / dx
        b = y0 - m * x0
        return m, b

    def within(a, b, x):
        lo, hi = (a, b) if a <= b else (b, a)
        return (x >= lo - tol) and (x <= hi + tol)

    def add_point(points: List[Tuple[float, float]], x: float, y: float):
        for px, py in points:
            if abs(px - x) <= dedup_tol and abs(py - y) <= dedup_tol:
                return
        points.append((x, y))

    points: List[Tuple[float, float]] = []

    for i in range(n):
        xi, yi = xs_list[i], ys_list[i]
        if len(xi) < 2:
            continue

        for j in range(i + 1, n):
            xj, yj = xs_list[j], ys_list[j]
            if len(xj) < 2:
                continue

            a = 0
            b = 0
            while a < len(xi) - 1 and b < len(xj) - 1:
                x0a, y0a = xi[a], yi[a]
                x1a, y1a = xi[a + 1], yi[a + 1]
                x0b, y0b = xj[b], yj[b]
                x1b, y1b = xj[b + 1], yj[b + 1]

                a_lo, a_hi = (x0a, x1a) if x0a <= x1a else (x1a, x0a)
                b_lo, b_hi = (x0b, x1b) if x0b <= x1b else (x1b, x0b)

                lo = max(a_lo, b_lo)
                hi = min(a_hi, b_hi)

                if hi < lo - tol:
                    if a_hi < b_hi:
                        a += 1
                    else:
                        b += 1
                    continue

                ma, ba = line_params(x0a, y0a, x1a, y1a)
                mb, bb = line_params(x0b, y0b, x1b, y1b)

                # Skip vertical/degenerate segments (common assumption: x is monotonic & unique)
                if ma is None or mb is None:
                    if a_hi < b_hi:
                        a += 1
                    else:
                        b += 1
                    continue

                # Parallel or (near-)parallel
                if abs(ma - mb) <= tol:
                    # Colinear overlap -> infinite intersections; return overlap endpoints as representatives
                    if abs(ba - bb) <= tol:
                        for x_int in (lo, hi):
                            y_int = ma * x_int + ba
                            add_point(points, x_int, y_int)

                    if a_hi < b_hi:
                        a += 1
                    else:
                        b += 1
                    continue

                # Proper intersection
                x_int = (bb - ba) / (ma - mb)
                if (
                    within(lo, hi, x_int)
                    and within(x0a, x1a, x_int)
                    and within(x0b, x1b, x_int)
                ):
                    y_int = ma * x_int + ba
                    add_point(points, x_int, y_int)

                if a_hi < b_hi:
                    a += 1
                else:
                    b += 1

    x_cross = [p[0] for p in points]
    y_cross = [p[1] for p in points]
    return x_cross, y_cross

def combine_mode_and_sweep_dicts_non_sequential(mode_dict, param_sweep_dict):
    """
    Combine mode_dict and param_sweep_dict when keys differ as:
      mode_dict:        'mode_1'
      param_sweep_dict: 'mode 1'

    The output dictionary contains ONLY the keys present in mode_dict
    (even if they are non-sequential like mode_1, mode_9, mode_4).
    """

    # Normalize param_sweep_dict keys: "mode 1" -> "mode_1"
    normalized_sweep = {
        key.replace(" ", "_"): val for key, val in param_sweep_dict.items()
    }

    combined_dict = {}

    for mode_key, mode_data in mode_dict.items():
        sweep_data = normalized_sweep.get(mode_key)

        if sweep_data is not None:
            combined_dict[mode_key] = {**mode_data, **sweep_data}
        else:
            combined_dict[mode_key] = mode_data.copy()

    return combined_dict

def combine_mode_and_sweep_dicts(mode_dict, param_sweep_dict):
    """
    Combine mode_dict and param_sweep_dict when keys differ as:
      mode_dict:        'mode_1'
      param_sweep_dict: 'mode 1'

    Output keys are always 'mode_1'.
    """

    # Normalize param_sweep_dict keys: "mode 1" -> "mode_1"
    normalized_sweep = {
        key.replace(" ", "_"): val for key, val in param_sweep_dict.items()
    }

    combined_dict = {}

    for mode_key, mode_data in mode_dict.items():
        if mode_key not in normalized_sweep:
            continue

        sweep_data = normalized_sweep[mode_key]

        combined_dict[mode_key] = {**mode_data, **sweep_data}

    return combined_dict


####################################################################################################################
#                                                                                                                  #
#                                            MODE IDENTIFICATION                                                   #
#                                                                                                                  #
####################################################################################################################


# ============================================================
# Reader for TESLA/CST ASCII field tables
# ============================================================


def _find_separator_line(lines, max_scan=800) -> int:
    for i, ln in enumerate(lines[:max_scan]):
        s = ln.strip()
        if len(s) >= 20 and set(s) == {"-"}:
            return i
    raise ValueError("Could not find dashed separator line (-----).")


def read_field_table(path: str, kind: str) -> pd.DataFrame:
    """
    Expects 9 numeric columns after dashed separator:
      E: x y z ExRe ExIm EyRe EyIm EzRe EzIm
      H: x y z HxRe HxIm HyRe HyIm HzRe HzIm

    Coordinates are assumed to be in mm -> converted to meters.
    Returns DataFrame with x,y,z (meters) and complex Ez or Hz.
    """
    kind = kind.upper().strip()
    if kind not in ("E", "H"):
        raise ValueError("kind must be 'E' or 'H'")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    sep_idx = _find_separator_line(lines)
    data = "".join(lines[sep_idx + 1 :])

    df = pd.read_csv(StringIO(data), sep=r"\s+", header=None)
    if df.shape[1] < 9:
        raise ValueError(f"{path}: expected >= 9 numeric columns, got {df.shape[1]}")

    df = df.iloc[:, :9].copy()

    if kind == "E":
        df.columns = [
            "x_u",
            "y_u",
            "z_u",
            "ExRe",
            "ExIm",
            "EyRe",
            "EyIm",
            "EzRe",
            "EzIm",
        ]
        df["val"] = df["EzRe"].to_numpy() + 1j * df["EzIm"].to_numpy()
        valname = "Ez"
    else:
        df.columns = [
            "x_u",
            "y_u",
            "z_u",
            "HxRe",
            "HxIm",
            "HyRe",
            "HyIm",
            "HzRe",
            "HzIm",
        ]
        df["val"] = df["HzRe"].to_numpy() + 1j * df["HzIm"].to_numpy()
        valname = "Hz"

    # Coordinates: assume mm -> m
    df["x"] = df["x_u"].astype(float) * 1e-3
    df["y"] = df["y_u"].astype(float) * 1e-3
    df["z"] = df["z_u"].astype(float) * 1e-3

    out = df[["x", "y", "z", "val"]].copy()
    out.rename(columns={"val": valname}, inplace=True)
    return out


# ============================================================
# Structured-grid utilities (robust “merge”)
# ============================================================


def build_3d_array_from_df(
    df: pd.DataFrame, value_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a structured-grid DataFrame into a 3D array arr[ix,iy,iz].
    Returns (arr, x_vals, y_vals, z_vals).
    """
    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())
    z_vals = np.sort(df["z"].unique())

    Nx, Ny, Nz = len(x_vals), len(y_vals), len(z_vals)
    arr = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}
    zi = {v: i for i, v in enumerate(z_vals)}

    for _, row in df.iterrows():
        arr[xi[row["x"]], yi[row["y"]], zi[row["z"]]] = row[value_col]

    return arr, x_vals, y_vals, z_vals


def grids_match(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-12, rtol: float = 0.0
) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def merge_by_grid(dfE: pd.DataFrame, dfH: pd.DataFrame, debug: bool = False):
    """
    Robust replacement for pandas merge:
    - build Ez_array and Hz_array separately
    - verify coordinate vectors match (within tolerance)
    - return aligned arrays and coordinate vectors
    """
    Ez_array, xE, yE, zE = build_3d_array_from_df(dfE, "Ez")
    Hz_array, xH, yH, zH = build_3d_array_from_df(dfH, "Hz")

    if debug:
        print(
            f"[DEBUG] E grid: {Ez_array.shape}  x[{xE[0]:.3e},{xE[-1]:.3e}] y[{yE[0]:.3e},{yE[-1]:.3e}] z[{zE[0]:.3e},{zE[-1]:.3e}]"
        )
        print(
            f"[DEBUG] H grid: {Hz_array.shape}  x[{xH[0]:.3e},{xH[-1]:.3e}] y[{yH[0]:.3e},{yH[-1]:.3e}] z[{zH[0]:.3e},{zH[-1]:.3e}]"
        )

    if not (grids_match(xE, xH) and grids_match(yE, yH) and grids_match(zE, zH)):
        raise ValueError(
            "E and H grids do not match.\n"
            f"  len(xE,xH)=({len(xE)},{len(xH)}) len(yE,yH)=({len(yE)},{len(yH)}) len(zE,zH)=({len(zE)},{len(zH)})\n"
            "Likely causes: unit mismatch (mm vs m), column shift, or wrong pairing."
        )

    return Ez_array, Hz_array, (xE, yE, zE)


# ============================================================
# Plot feature (unchanged)
# ============================================================


def plot_Ez_midplane(Ez_array: np.ndarray, title: str):
    mid_pixel = Ez_array.shape[2] // 2
    plt.figure()
    plt.imshow(np.real(Ez_array[:, :, mid_pixel]).T, origin="lower", aspect="equal")
    plt.colorbar(label="Re(Ez)")
    plt.title(f"{title} | mid_pixel={mid_pixel}")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.show()


# ============================================================
# New classifier core (voxel-centric sampling)
# ============================================================


def align_global_phase(v: np.ndarray) -> np.ndarray:
    mag = np.abs(v)
    if not np.any(np.isfinite(mag)) or np.nanmax(mag) == 0:
        return v
    idx = int(np.nanargmax(mag))
    v2 = v * np.exp(-1j * np.angle(v[idx]))
    if np.real(v2[idx]) < 0:
        v2 = -v2
    return v2


def deadband_sign(x: np.ndarray, dead_frac: float = 0.07) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    amp = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else 0.0
    if amp == 0.0 or not np.isfinite(amp):
        return np.zeros_like(x, dtype=int)
    dead = dead_frac * amp
    s = np.sign(x).astype(int)
    s[np.abs(x) < dead] = 0
    return s


def count_sign_flips(signs: np.ndarray) -> int:
    nz = signs[signs != 0]
    if len(nz) <= 1:
        return 0
    return int(np.sum(nz[1:] * nz[:-1] < 0))


def classify_family(
    Ez: np.ndarray,
    Hz: np.ndarray,
    ratio_hi: float = 5.0,
    ratio_lo: float = 5.0,
    eps: float = 1e-30,
) -> Tuple[str, float, str]:
    """
    Same logic as before, but default thresholds are ratio_hi=5.0 and ratio_lo=5.0.

    Interprets:
      R = sum|Ez|^2 / sum|Hz|^2
      if R >= ratio_hi -> TM
      if R <= 1/ratio_lo -> TE
      else -> HYBRID (label by dominant)
    """
    EzE = float(np.sum(np.abs(Ez) ** 2) + eps)
    HzE = float(np.sum(np.abs(Hz) ** 2) + eps)
    R = EzE / HzE
    log10R = float(np.log10(R))

    if log10R >= ratio_hi:
        return "TM", log10R, "TM"
    if log10R <= ratio_lo:
        return "TE", log10R, "TE"
    return "HYBRID", log10R, ("TM" if R >= 1.0 else "TE")


def find_peak_voxel(arr: np.ndarray) -> Tuple[int, int, int]:
    """Return (ix, iy, iz) of max |arr|."""
    idx = np.unravel_index(int(np.argmax(np.abs(arr))), arr.shape)
    return int(idx[0]), int(idx[1]), int(idx[2])


def sample_radial_line(
    arr: np.ndarray, ix0: int, iy0: int, iz0: int, axis: str = "x"
) -> np.ndarray:
    """
    Sample along a Cartesian line through the voxel (ix0,iy0,iz0).
    axis: "x" -> arr[:, iy0, iz0]
          "y" -> arr[ix0, :, iz0]
    """
    if axis == "x":
        return arr[:, iy0, iz0]
    if axis == "y":
        return arr[ix0, :, iz0]
    raise ValueError("axis must be 'x' or 'y'")


def sample_azimuthal_ring(
    arr: np.ndarray, ix0: int, iy0: int, iz0: int, nphi: int = 72
) -> np.ndarray:
    """
    Sample around a ring (approximately) at fixed iz0 and at radius r0 from center
    where r0 is the distance of the peak voxel from the grid center.

    Uses nearest-neighbor sampling on the structured grid.
    Returns complex samples f(phi_k).
    """
    Nx, Ny, Nz = arr.shape
    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0

    dx0 = ix0 - cx
    dy0 = iy0 - cy
    r0 = float(np.hypot(dx0, dy0))

    # If peak is at/near center, ring degenerates. Use a small r instead.
    if r0 < 1.0:
        r0 = 2.0

    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    samples = np.zeros(nphi, dtype=np.complex128)

    for k, phi in enumerate(phis):
        ix = int(np.rint(cx + r0 * np.cos(phi)))
        iy = int(np.rint(cy + r0 * np.sin(phi)))
        ix = max(0, min(Nx - 1, ix))
        iy = max(0, min(Ny - 1, iy))
        samples[k] = arr[ix, iy, iz0]

    return samples


def estimate_m_from_ring(
    ring_samples: np.ndarray, m_max: int = 10, signif_frac: float = 0.30
) -> Tuple[int, np.ndarray]:
    """
    Compute DFT-like harmonic amplitudes on ring:
      a_m = mean( f(phi) * exp(-i m phi) )
    Choose smallest significant m (>= signif_frac*max).
    """
    nphi = len(ring_samples)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    f = align_global_phase(ring_samples)
    amps = np.zeros(m_max + 1, dtype=float)

    for m in range(m_max + 1):
        a_m = np.mean(f * np.exp(-1j * m * phis))
        amps[m] = float(np.abs(a_m))

    amax = float(np.max(amps)) if len(amps) else 0.0
    if amax == 0.0:
        return 0, amps

    for m in range(m_max + 1):
        if amps[m] >= signif_frac * amax:
            return int(m), amps
    return int(np.argmax(amps)), amps


def estimate_n_from_radial_line(line: np.ndarray) -> int:
    """
    Estimate n as (# sign flips of Re(line)) + 1 after phase alignment and smoothing.
    """
    v = align_global_phase(line)
    xr = np.real(v)
    xr_s = np.convolve(xr, np.ones(3) / 3, mode="same")
    s = deadband_sign(xr_s, dead_frac=0.07)
    flips = count_sign_flips(s)
    return max(flips + 1, 1)


def estimate_p_from_z_line(zline: np.ndarray) -> int:
    """
    DEFINITE p from z-line: (# sign flips of Re(zline)) after alignment/smoothing.
    """
    v = align_global_phase(zline)
    zr = np.real(v)
    zr_s = np.convolve(zr, np.ones(3) / 3, mode="same")
    s = deadband_sign(zr_s, dead_frac=0.07)
    flips = count_sign_flips(s)
    return int(max(flips, 0))


# ============================================================
# Mode result + identification (new voxel-centric indices)
# ============================================================


@dataclass
class ModeResult:
    TM_TE: str
    m: int
    n: int
    p: int
    meta: Dict


def identify_mode(
    E_path: str,
    H_path: str,
    ratio_hi: float = 5.0,
    ratio_lo: float = 5.0,
    m_max: int = 10,
    signif_frac: float = 0.30,
    prefer_m0_for_TE: bool = True,
    debug: bool = False,
) -> Tuple[ModeResult, np.ndarray]:
    """
    New algorithm:
      - TM/TE via global ratio as before (but default ratio_hi=ratio_lo=5)
      - choose the peak voxel in Ez (TM-like) or Hz (TE-like)
      - sample through that voxel:
          azimuthal ring at iz0 -> m
          radial line through voxel (max of x- or y-line) -> n
          z-line through voxel -> p
    Returns (ModeResult, Ez_array) for optional plotting.
    """
    dfE = read_field_table(E_path, "E")
    dfH = read_field_table(H_path, "H")
    Ez_array, Hz_array, (xv, yv, zv) = merge_by_grid(dfE, dfH, debug=debug)

    Ez_flat = Ez_array.ravel()
    Hz_flat = Hz_array.ravel()

    raw_family, log10R, family_like = classify_family(
        Ez_flat, Hz_flat, ratio_hi=ratio_hi, ratio_lo=ratio_lo
    )

    use = Ez_array if family_like == "TM" else Hz_array
    use_name = "Ez" if family_like == "TM" else "Hz"

    ix0, iy0, iz0 = find_peak_voxel(use)

    # ---- m from azimuthal ring at same z ----
    ring = sample_azimuthal_ring(use, ix0, iy0, iz0, nphi=72)
    m, m_amps = estimate_m_from_ring(ring, m_max=m_max, signif_frac=signif_frac)

    # TE-family preference: if m=0 harmonic is significant, force m=0
    if prefer_m0_for_TE and family_like == "TE":
        if m_amps[0] >= signif_frac * float(np.max(m_amps)):
            m = 0

    # ---- n from radial line through peak voxel ----
    xline = sample_radial_line(use, ix0, iy0, iz0, axis="x")
    yline = sample_radial_line(use, ix0, iy0, iz0, axis="y")

    # choose the line with stronger peak-to-peak to be more informative
    def p2p(a):
        return float(
            np.nanmax(np.real(align_global_phase(a)))
            - np.nanmin(np.real(align_global_phase(a)))
        )

    line = xline if p2p(xline) >= p2p(yline) else yline
    n = estimate_n_from_radial_line(line)

    # ---- p from z-line through peak voxel ----
    zline = use[ix0, iy0, :]
    p = estimate_p_from_z_line(zline)

    if debug:
        print(
            f"[DEBUG] {os.path.basename(E_path)} / {os.path.basename(H_path)} "
            f"raw={raw_family} like={family_like} log10R={log10R:.2f} "
            f"peak=({ix0},{iy0},{iz0}) using={use_name} -> m={m} n={n} p={p}"
        )

    return (
        ModeResult(
            TM_TE=family_like,
            m=int(m),
            n=int(n),
            p=int(p),
            meta={
                "raw_family": raw_family,
                "log10R": log10R,
                "peak_voxel": (ix0, iy0, iz0),
                "using_component": use_name,
                "m_amps": [float(a) for a in m_amps],
            },
        ),
        Ez_array,
    )


def _to_real_image(arr, *, component="real", abs_for_absE=True, rowname=None):
    """
    Convert possibly-complex arrays to real-valued 2D arrays for plotting.

    component: "real" | "imag" | "abs" | "phase"
      - for Ex/Ey/Ez rows, default uses 'component' (real/imag/abs/phase)
      - for absE row, default uses abs if abs_for_absE=True
    """
    a = np.asarray(arr)

    # Special-case absE row: typically magnitude
    if rowname == "absE" and abs_for_absE:
        return np.abs(a).astype(float)

    if np.iscomplexobj(a):
        if component == "real":
            return np.real(a).astype(float)
        if component == "imag":
            return np.imag(a).astype(float)
        if component == "abs":
            return np.abs(a).astype(float)
        if component == "phase":
            return np.angle(a).astype(float)
        raise ValueError("component must be one of: 'real', 'imag', 'abs', 'phase'")

    # Already real
    return a.astype(float)


def plot_slice_field_grids(
    slice_dict,
    save_directory_fname,
    slice_types=("iris", "transverse_mid", "longitudinal_mid"),
    ops=("plus", "minus"),
    *,
    extent_by_type=None,
    cmap_div="RdBu_r",
    cmap_abs="viridis",
    vlim_xyz=None,
    vlim_abs=None,
    complex_component="real",
    abs_for_absE=True,
    figsize=(11, 10),
    tight=True,
    save_fig=True,
):
    rows = ["Ex", "Ey", "Ez", "absE"]

    def _need(k):
        if k not in slice_dict:
            raise KeyError(f"Missing key in slice_dict: {k}")

    def _get_vlim_xyz_for_row(stype, rowname, keys_for_row):
        if isinstance(vlim_xyz, (int, float)):
            v = float(vlim_xyz)
            return (-v, v)
        if isinstance(vlim_xyz, dict) and rowname in vlim_xyz:
            v = float(vlim_xyz[rowname])
            return (-v, v)

        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=abs_for_absE,
                rowname=rowname,
            )
            for k in keys_for_row
        ]
        m = max(np.nanmax(np.abs(im)) for im in imgs)
        return (-m, m) if np.isfinite(m) and m > 0 else (-1.0, 1.0)

    def _get_vlim_abs(stype, abs_keys):
        if isinstance(vlim_abs, tuple) and len(vlim_abs) == 2:
            return (float(vlim_abs[0]), float(vlim_abs[1]))
        if isinstance(vlim_abs, dict) and stype in vlim_abs:
            t = vlim_abs[stype]
            return (float(t[0]), float(t[1]))

        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=True,
                rowname="absE",
            )
            for k in abs_keys
        ]
        vmin = min(np.nanmin(im) for im in imgs)
        vmax = max(np.nanmax(im) for im in imgs)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            return (0.0, 1.0)
        return (float(vmin), float(vmax))

    figs = {}
    axes_out = {}
    max_dict = {}  # <- NEW: subplot_id -> max value

    for stype in slice_types:
        extent = extent_by_type.get(stype) if extent_by_type else None

        for op in ops:
            op = op.lower().strip()
            if op not in {"plus", "minus"}:
                raise ValueError("ops must be drawn from {'plus','minus'}")

            abs_op_key = "abs_add" if op == "plus" else "abs_sub"
            comp_op_suffix = "plus" if op == "plus" else "minus"

            key_map = {
                ("Ex", "E1"): f"E1_Ex_{stype}",
                ("Ex", "E2"): f"E2_Ex_{stype}",
                ("Ex", op): f"Ex_{comp_op_suffix}_{stype}",
                ("Ey", "E1"): f"E1_Ey_{stype}",
                ("Ey", "E2"): f"E2_Ey_{stype}",
                ("Ey", op): f"Ey_{comp_op_suffix}_{stype}",
                ("Ez", "E1"): f"E1_Ez_{stype}",
                ("Ez", "E2"): f"E2_Ez_{stype}",
                ("Ez", op): f"Ez_{comp_op_suffix}_{stype}",
                ("absE", "E1"): f"abs_E1_{stype}",
                ("absE", "E2"): f"abs_E2_{stype}",
                ("absE", op): f"{abs_op_key}_{stype}",
            }

            # validate
            for r in rows:
                for c in ("E1", "E2", op):
                    _need(key_map[(r, c)])

            # limits
            vlims_xyz = {}
            for r in ["Ex", "Ey", "Ez"]:
                keys_for_row = [
                    key_map[(r, "E1")],
                    key_map[(r, "E2")],
                    key_map[(r, op)],
                ]
                vlims_xyz[r] = _get_vlim_xyz_for_row(stype, r, keys_for_row)

            abs_keys = [
                key_map[("absE", "E1")],
                key_map[("absE", "E2")],
                key_map[("absE", op)],
            ]
            vmin_abs, vmax_abs = _get_vlim_abs(stype, abs_keys)

            # plot
            fig, ax = plt.subplots(4, 3, figsize=figsize, sharex=True, sharey=True)
            fig.suptitle(f"{stype} — E1 {'+' if op=='plus' else '-'} E2", y=0.98)

            col_labels = ["E1", "E2", op]
            for ci, clab in enumerate(col_labels):
                ax[0, ci].set_title(clab)

            for ri, rname in enumerate(rows):
                ax[ri, 0].set_ylabel(rname)

                for ci, cname in enumerate(["E1", "E2", op]):
                    a = ax[ri, ci]
                    raw = slice_dict[key_map[(rname, cname)]]

                    img = _to_real_image(
                        raw,
                        component=complex_component,
                        abs_for_absE=abs_for_absE,
                        rowname=rname,
                    )

                    # Decide max metric:
                    # - if subplot contains negative values -> use max(abs(img))
                    # - otherwise use max(img)
                    has_negative = np.nanmin(img) < 0
                    if has_negative:
                        mval = float(np.nanmax(np.abs(img)))
                    else:
                        mval = float(np.nanmax(img))
                    if not np.isfinite(mval):
                        mval = float("nan")

                    subplot_id = (
                        f"{op}_{stype}_{cname}_{rname}"  # e.g. minus_iris_E1_Ex
                    )
                    max_dict[subplot_id] = mval

                    # Plot
                    if rname in {"Ex", "Ey", "Ez"}:
                        vmin, vmax = vlims_xyz[rname]
                        im = a.imshow(
                            img,
                            origin="lower",
                            extent=extent,
                            cmap=cmap_div,
                            vmin=vmin,
                            vmax=vmax,
                            aspect="auto",
                        )
                    else:
                        im = a.imshow(
                            img,
                            origin="lower",
                            extent=extent,
                            cmap=cmap_abs,
                            vmin=vmin_abs,
                            vmax=vmax_abs,
                            aspect="auto",
                        )

                    # Annotate max value on every subplot
                    a.text(
                        0.02,
                        0.98,
                        f"max={mval:.3g}",
                        transform=a.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox=dict(
                            facecolor="white", alpha=0.65, edgecolor="none", pad=2.0
                        ),
                    )

                    # Colorbar on the rightmost column
                    if ci == 2:
                        divider = make_axes_locatable(a)
                        cax = divider.append_axes("right", size="4%", pad=0.05)
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.ax.tick_params(labelsize=8)

                mid = ax[ri, 1]
                mid.text(
                    1.02,
                    1.02,
                    f"{'+' if op=='plus' else '-'}  =",
                    transform=mid.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )

            if tight:
                plt.tight_layout()

            figs[(stype, op)] = fig
            axes_out[(stype, op)] = ax

            if save_fig:

                plt.savefig(f"{save_directory_fname}_{op}_{stype}.png")
                plt.close("all")
                # input(f"{save_directory_fname}_{op}_{stype}.png")
            else:
                plt.show()

    return figs, axes_out, max_dict


def plot_slice_field_grids_no_txt_old(
    slice_dict,
    save_directory_fname,
    slice_types=("iris", "transverse_mid", "longitudinal_mid"),
    ops=("plus", "minus"),
    *,
    extent_by_type=None,
    cmap_div="RdBu_r",
    cmap_abs="viridis",
    vlim_xyz=None,
    vlim_abs=None,
    complex_component="real",  # <- NEW: how to render complex Ex/Ey/Ez
    abs_for_absE=True,  # <- NEW: absE row uses |.| by default
    figsize=(11, 10),
    tight=True,
    save_fig=True,
):
    rows = ["Ex", "Ey", "Ez", "absE"]

    def _need(k):
        if k not in slice_dict:
            raise KeyError(f"Missing key in slice_dict: {k}")

    def _get_vlim_xyz_for_row(stype, rowname, keys_for_row):
        # explicit limits
        if isinstance(vlim_xyz, (int, float)):
            v = float(vlim_xyz)
            return (-v, v)
        if isinstance(vlim_xyz, dict) and rowname in vlim_xyz:
            v = float(vlim_xyz[rowname])
            return (-v, v)

        # auto symmetric limits per-row across 3 panels
        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=abs_for_absE,
                rowname=rowname,
            )
            for k in keys_for_row
        ]
        m = max(np.nanmax(np.abs(im)) for im in imgs)
        return (-m, m) if np.isfinite(m) and m > 0 else (-1.0, 1.0)

    def _get_vlim_abs(stype, abs_keys):
        if isinstance(vlim_abs, tuple) and len(vlim_abs) == 2:
            return (float(vlim_abs[0]), float(vlim_abs[1]))
        if isinstance(vlim_abs, dict) and stype in vlim_abs:
            t = vlim_abs[stype]
            return (float(t[0]), float(t[1]))

        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=True,
                rowname="absE",
            )
            for k in abs_keys
        ]
        vmin = min(np.nanmin(im) for im in imgs)
        vmax = max(np.nanmax(im) for im in imgs)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            return (0.0, 1.0)
        return (float(vmin), float(vmax))

    figs = {}
    axes_out = {}

    for stype in slice_types:
        extent = extent_by_type.get(stype) if extent_by_type else None

        for op in ops:
            op = op.lower().strip()
            if op not in {"plus", "minus"}:
                raise ValueError("ops must be drawn from {'plus','minus'}")

            abs_op_key = "abs_add" if op == "plus" else "abs_sub"
            comp_op_suffix = "plus" if op == "plus" else "minus"

            key_map = {
                ("Ex", "E1"): f"E1_Ex_{stype}",
                ("Ex", "E2"): f"E2_Ex_{stype}",
                ("Ex", op): f"Ex_{comp_op_suffix}_{stype}",
                ("Ey", "E1"): f"E1_Ey_{stype}",
                ("Ey", "E2"): f"E2_Ey_{stype}",
                ("Ey", op): f"Ey_{comp_op_suffix}_{stype}",
                ("Ez", "E1"): f"E1_Ez_{stype}",
                ("Ez", "E2"): f"E2_Ez_{stype}",
                ("Ez", op): f"Ez_{comp_op_suffix}_{stype}",
                ("absE", "E1"): f"abs_E1_{stype}",
                ("absE", "E2"): f"abs_E2_{stype}",
                ("absE", op): f"{abs_op_key}_{stype}",
            }

            # validate
            for r in rows:
                for c in ("E1", "E2", op):
                    _need(key_map[(r, c)])

            # limits
            vlims_xyz = {}
            for r in ["Ex", "Ey", "Ez"]:
                keys_for_row = [
                    key_map[(r, "E1")],
                    key_map[(r, "E2")],
                    key_map[(r, op)],
                ]
                vlims_xyz[r] = _get_vlim_xyz_for_row(stype, r, keys_for_row)

            abs_keys = [
                key_map[("absE", "E1")],
                key_map[("absE", "E2")],
                key_map[("absE", op)],
            ]
            vmin_abs, vmax_abs = _get_vlim_abs(stype, abs_keys)

            # plot
            fig, ax = plt.subplots(4, 3, figsize=figsize, sharex=True, sharey=True)
            fig.suptitle(f"{stype} — E1 {'+' if op=='plus' else '-'} E2", y=0.98)

            col_labels = ["E1", "E2", op]
            for ci, clab in enumerate(col_labels):
                ax[0, ci].set_title(clab)

            for ri, rname in enumerate(rows):
                ax[ri, 0].set_ylabel(rname)
                for ci, cname in enumerate(["E1", "E2", op]):
                    a = ax[ri, ci]
                    raw = slice_dict[key_map[(rname, cname)]]

                    img = _to_real_image(
                        raw,
                        component=complex_component,
                        abs_for_absE=abs_for_absE,
                        rowname=rname,
                    )

                    if rname in {"Ex", "Ey", "Ez"}:
                        vmin, vmax = vlims_xyz[rname]
                        im = a.imshow(
                            img,
                            origin="lower",
                            extent=extent,
                            cmap=cmap_div,
                            vmin=vmin,
                            vmax=vmax,
                            aspect="auto",
                        )
                    else:
                        im = a.imshow(
                            img,
                            origin="lower",
                            extent=extent,
                            cmap=cmap_abs,
                            vmin=vmin_abs,
                            vmax=vmax_abs,
                            aspect="auto",
                        )

                    if ci == 2:
                        divider = make_axes_locatable(a)
                        cax = divider.append_axes("right", size="4%", pad=0.05)
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.ax.tick_params(labelsize=8)

                mid = ax[ri, 1]
                mid.text(
                    1.02,
                    1.02,
                    f"{'+' if op=='plus' else '-'}  =",
                    transform=mid.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )

            if tight:
                plt.tight_layout()

            figs[(stype, op)] = fig
            axes_out[(stype, op)] = ax

            if save_fig:
                plt.savefig(f"{save_directory_fname}_{ops}_{stype}.png")
                plt.close("all")
            else:
                plt.show()

    return figs, axes_out


# Convenience wrappers
def plot_all_plus_old(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids(
        slice_dict, save_directory_fname, ops=("plus",), **kwargs
    )


def plot_all_minus_old(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids(
        slice_dict, save_directory_fname, ops=("minus",), **kwargs
    )


def plot_slice_field_grids_with_txt(
    slice_dict,
    save_directory_fname,
    slice_types=("iris_1", "iris_2", "transverse_mid", "longitudinal_mid"),
    ops=("plus", "minus"),
    *,
    extent_by_type=None,
    cmap_div="RdBu_r",
    cmap_abs="viridis",
    vlim_xyz=None,
    vlim_abs=None,
    complex_component="real",
    abs_for_absE=True,
    figsize=(11, 10),
    tight=True,
    save_fig=True,
):
    """
    Plots 4x3 grids for each slice type and op, annotates each subplot with the
    maximum magnitude value (max(|pixel|)) of that subplot, and returns those
    maxima in a dict with keys like: 'minus_iris_1_E1_Ex'.
    """
    rows = ["Ex", "Ey", "Ez", "absE"]

    def _need(k):
        if k not in slice_dict:
            raise KeyError(f"Missing key in slice_dict: {k}")

    def _get_vlim_xyz_for_row(rowname, keys_for_row):
        # explicit limits
        if isinstance(vlim_xyz, (int, float)):
            v = float(vlim_xyz)
            return (-v, v)
        if isinstance(vlim_xyz, dict) and rowname in vlim_xyz:
            v = float(vlim_xyz[rowname])
            return (-v, v)

        # auto symmetric limits per-row across 3 panels
        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=abs_for_absE,
                rowname=rowname,
            )
            for k in keys_for_row
        ]
        m = max(np.nanmax(np.abs(im)) for im in imgs)
        return (-m, m) if np.isfinite(m) and m > 0 else (-1.0, 1.0)

    def _get_vlim_abs(stype, abs_keys):
        if isinstance(vlim_abs, tuple) and len(vlim_abs) == 2:
            return (float(vlim_abs[0]), float(vlim_abs[1]))
        if isinstance(vlim_abs, dict) and stype in vlim_abs:
            t = vlim_abs[stype]
            return (float(t[0]), float(t[1]))

        imgs = [
            _to_real_image(
                slice_dict[k],
                component=complex_component,
                abs_for_absE=True,
                rowname="absE",
            )
            for k in abs_keys
        ]
        vmin = min(np.nanmin(im) for im in imgs)
        vmax = max(np.nanmax(im) for im in imgs)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            return (0.0, 1.0)
        return (float(vmin), float(vmax))

    def _max_magnitude(img: np.ndarray) -> float:
        """Max magnitude for any real/complex image; NaN-safe."""
        m = np.nanmax(np.abs(img))
        return float(m) if np.isfinite(m) else float("nan")

    figs = {}
    axes_out = {}
    max_dict = {}  # <- NEW: subplot_id -> max(|pixel|)

    for stype in slice_types:
        extent = extent_by_type.get(stype) if extent_by_type else None

        for op in ops:
            op = op.lower().strip()
            if op not in {"plus", "minus"}:
                raise ValueError("ops must be drawn from {'plus','minus'}")

            abs_op_key = "abs_add" if op == "plus" else "abs_sub"
            comp_op_suffix = "plus" if op == "plus" else "minus"

            key_map = {
                ("Ex", "E1"): f"E1_Ex_{stype}",
                ("Ex", "E2"): f"E2_Ex_{stype}",
                ("Ex", op):   f"Ex_{comp_op_suffix}_{stype}",

                ("Ey", "E1"): f"E1_Ey_{stype}",
                ("Ey", "E2"): f"E2_Ey_{stype}",
                ("Ey", op):   f"Ey_{comp_op_suffix}_{stype}",

                ("Ez", "E1"): f"E1_Ez_{stype}",
                ("Ez", "E2"): f"E2_Ez_{stype}",
                ("Ez", op):   f"Ez_{comp_op_suffix}_{stype}",

                ("absE", "E1"): f"abs_E1_{stype}",
                ("absE", "E2"): f"abs_E2_{stype}",
                ("absE", op):   f"{abs_op_key}_{stype}",
            }

            # validate keys exist
            for r in rows:
                for c in ("E1", "E2", op):
                    _need(key_map[(r, c)])

            # limits
            vlims_xyz = {}
            for r in ["Ex", "Ey", "Ez"]:
                keys_for_row = [key_map[(r, "E1")], key_map[(r, "E2")], key_map[(r, op)]]
                vlims_xyz[r] = _get_vlim_xyz_for_row(r, keys_for_row)

            abs_keys = [key_map[("absE", "E1")], key_map[("absE", "E2")], key_map[("absE", op)]]
            vmin_abs, vmax_abs = _get_vlim_abs(stype, abs_keys)

            # plot
            fig, ax = plt.subplots(4, 3, figsize=figsize, sharex=True, sharey=True)
            fig.suptitle(f"{stype} — E1 {'+' if op=='plus' else '-'} E2", y=0.98)

            col_labels = ["E1", "E2", op]
            for ci, clab in enumerate(col_labels):
                ax[0, ci].set_title(clab)

            for ri, rname in enumerate(rows):
                ax[ri, 0].set_ylabel(rname)

                for ci, cname in enumerate(["E1", "E2", op]):
                    a = ax[ri, ci]
                    raw = slice_dict[key_map[(rname, cname)]]

                    img = _to_real_image(
                        raw,
                        component=complex_component,
                        abs_for_absE=abs_for_absE,
                        rowname=rname,
                    )


                    #TODO remove if deemed incorrect
                    # img = img.T

                    # max magnitude + annotate + store
                    mval = _max_magnitude(img)
                    subplot_id = f"{op}_{stype}_{cname}_{rname}"  # e.g. minus_iris_1_E1_Ex
                    max_dict[subplot_id] = mval

                    a.text(
                        0.02, 0.98, f"max|·|={mval:.3g}",
                        transform=a.transAxes,
                        ha="left", va="top",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0),
                    )

                    # draw
                    if rname in {"Ex", "Ey", "Ez"}:
                        vmin, vmax = vlims_xyz[rname]
                        im = a.imshow(
                            img, origin="lower", extent=extent,
                            cmap=cmap_div, vmin=vmin, vmax=vmax, aspect="auto"
                        )
                    else:
                        im = a.imshow(
                            img, origin="lower", extent=extent,
                            cmap=cmap_abs, vmin=vmin_abs, vmax=vmax_abs, aspect="auto"
                        )

                    # colorbar on rightmost col
                    if ci == 2:
                        divider = make_axes_locatable(a)
                        cax = divider.append_axes("right", size="4%", pad=0.05)
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.ax.tick_params(labelsize=8)

                mid = ax[ri, 1]
                mid.text(
                    1.02, 1.02, f"{'+' if op=='plus' else '-'}  =",
                    transform=mid.transAxes, ha="left", va="bottom", fontsize=10
                )

            if tight:
                plt.tight_layout()

            figs[(stype, op)] = fig
            axes_out[(stype, op)] = ax

            if save_fig:
                plt.savefig(f"{save_directory_fname}_{op}_{stype}.png")
                plt.close("all")
            else:
                plt.show()

    return figs, axes_out, max_dict


# Convenience wrappers
def plot_all_plus(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids_with_txt(
        slice_dict, save_directory_fname, ops=("plus",), **kwargs
    )

def plot_all_minus(slice_dict, save_directory_fname, **kwargs):
    return plot_slice_field_grids_with_txt(
        slice_dict, save_directory_fname, ops=("minus",), **kwargs
    )


# ============================================================
# Loop framework: modes 1..60 -> modes_dict (usage unchanged)
# ============================================================


def build_modes_dict(
    folder: str,
    start: int = 1,
    stop: int = 60,
    e_pattern: str = "Field_TESLA_21x21x21_E_mode{mode:03d}.txt",
    h_pattern: str = "Field_TESLA_21x21x21_H_mode{mode:03d}.txt",
    debug: bool = True,
    plot_midplane: bool = False,
    plot_every: int = 1,
) -> Dict[str, Dict]:
    modes_dict: Dict[str, Dict] = {}

    for mode in range(start, stop + 1):
        key = f"mode_{mode}"
        e_path = os.path.join(folder, e_pattern.format(mode=mode))
        h_path = os.path.join(folder, h_pattern.format(mode=mode))

        if not (os.path.exists(e_path) and os.path.exists(h_path)):
            if debug:
                print(
                    f"[WARN] Missing files for {key}: {e_path} or {h_path} -> skipped"
                )
            continue

        res, Ez_array = identify_mode(e_path, h_path, debug=debug)

        if plot_midplane and ((mode - start) % max(plot_every, 1) == 0):
            plot_Ez_midplane(Ez_array, title=f"{key} Ez midplane")

        modes_dict[key] = {"TM_TE": res.TM_TE, "m": res.m, "n": res.n, "p": res.p}

    return modes_dict



####################################################################################################################
#                                                                                                                  #
#                                            PARAMETER SWEEP ANALYSIS FUNCTIONS                                             #
#                                                                                                                  #
####################################################################################################################




def fit_power_law(
    x_data,
    y_data,
    *,
    weights=None,
    print_r2=True,
    return_r2=False,
    return_cov=False,
    filter_positive=True,
    make_plots=True,
    savepath=None,
    savename="power_law_fit",
    show_plots=False,
    dpi=150
):
    """
    Fit y = A * x^(-p) using log-log least squares and optionally
    generate & save diagnostic plots.

    Parameters
    ----------
    x_data, y_data : array-like
        Input data (must be positive)
    weights : array-like, optional
        Weights in log space
    print_r2 : bool
        Print fit summary and R^2
    return_r2 : bool
        Return R^2
    return_cov : bool
        Return covariance matrix
    filter_positive : bool
        Remove non-positive values
    make_plots : bool
        Generate diagnostic plots
    savepath : str or Path, optional
        Directory to save plots
    savename : str
        Base filename (no extension)
    show_plots : bool
        Show plots interactively
    dpi : int
        Resolution for saved figures

    Returns
    -------
    A : float
    p : float
    r2 : float, optional
    cov : ndarray, optional
    """

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    if filter_positive:
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        if weights is not None:
            weights = np.asarray(weights)[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid data points to fit.")

    logx = np.log(x)
    logy = np.log(y)

    # Linear regression in log space
    if weights is not None:
        coeffs, cov = np.polyfit(logx, logy, deg=1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(logx, logy, deg=1, cov=True)

    slope, intercept = coeffs
    A = np.exp(intercept)
    p = -slope

    # R^2 in log-log space
    logy_pred = intercept + slope * logx
    ss_res = np.sum((logy - logy_pred) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    if print_r2:
        print(f"Power-law fit:")
        print(f"  y = {A:.6g} * x^(-{p:.6g})")
        print(f"  R^2 (log-log) = {r2:.6f}")

    # ---------- Plotting ----------
    if make_plots:
        x_fit = np.linspace(x.min(), x.max(), 400)
        y_fit = A * x_fit ** (-p)

        if savepath is not None:
            savepath = Path(savepath)
            savepath.mkdir(parents=True, exist_ok=True)

        # Linear-scale plot
        plt.figure()
        plt.plot(x, y, "o", label="Data")
        plt.plot(x_fit, y_fit, "-", label="Power-law fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Power-law fit (linear scale)")

        if savepath is not None:
            plt.savefig(savepath / f"{savename}_linear.png", dpi=dpi, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()

        # Log-log plot
        plt.figure()
        plt.loglog(x, y, "o", label="Data")
        plt.loglog(x_fit, y_fit, "-", label="Power-law fit")
        plt.xlabel("log(x)")
        plt.ylabel("log(y)")
        plt.legend()
        plt.title("Power-law fit (log-log)")

        if savepath is not None:
            plt.savefig(savepath / f"{savename}_loglog.png", dpi=dpi, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()

    # ---------- Outputs ----------
    outputs = [A, p]
    if return_r2:
        outputs.append(r2)
    if return_cov:
        outputs.append(cov)

    return tuple(outputs)


def add_power_law_fits_to_pickle_dict(
    data_dict,
    *,
    x_key="length_factor",
    y_key="frequency_GHz",
    store_key="power_law_fit",
    print_r2=False,
    savepath,
    savename,
):
    """
    Mutates and returns data_dict by adding fits wherever x_key/y_key exist.

    Ensures the dict stores:
      entry[store_key]["success"] = True/False
      entry[store_key]["result"]  = {"a": A, "p": p, "r2": r2}   (always)

    Notes
    -----
    - We call fit_power_law(..., return_r2=True) so r2 is ALWAYS produced
      in LOG-LOG space (matching the fit).
    ² reporting).
    - We do NOT recompute r2 elsewhere, preventing negative linear-space r2 issues.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("Expected data_dict to be a dict")

    def _try_fit_entry(entry, key):
        if not isinstance(entry, dict):
            return

        if x_key not in entry or y_key not in entry:
            return

        x = np.asarray(entry[x_key], dtype=float)
        y = np.asarray(entry[y_key], dtype=float)

        # log-safe / finite filter
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x_fit = x[mask]
        y_fit = y[mask]
        if x_fit.size < 2:
            return

        try:
            A, p, r2 = fit_power_law(
                x_fit,
                y_fit,
                print_r2=print_r2,
                return_r2=True,
                filter_positive=False,  # already filtered above
                savepath=savepath,
                savename=savename+f"_{key}",
            )

            entry[store_key] = {
                "success": True,
                "result": {
                    "a": float(A),
                    "p": float(p),
                    "r2": float(r2),  # LOG-LOG r2
                },
            }

        except Exception as err:
            entry[store_key] = {
                "success": False,
                "result": {},
                "error": str(err),
            }

    # Fit at top-level and one nested level (as before)
    for key, v in data_dict.items():
        _try_fit_entry(v, key)
        if isinstance(v, dict):
            for vv in v.values():
                _try_fit_entry(vv, key)

    return data_dict


def plot_power_law_params_filtered(
    data_dict,
    *,
    r2_min=0.9,
    savepath=".",
    savename="powerlaw",
    fit_key="power_law_fit",
):
    """
    Plots ONLY a and p values whose stored r2 >= r2_min.

    Expects:
      entry[fit_key]["success"] == True
      entry[fit_key]["result"]  == {"a":..., "p":..., "r2":...}

    Saves:
      f"{savepath}\\{savename}_a.png"
      f"{savepath}\\{savename}_p.png"
    """
    a_vals = []
    p_vals = []

    def _collect(entry):
        if not isinstance(entry, dict):
            return

        plf = entry.get(fit_key)
        if not isinstance(plf, dict) or not plf.get("success", False):
            return

        res = plf.get("result")
        if not isinstance(res, dict):
            return

        r2 = res.get("r2", None)
        if r2 is None or not np.isfinite(r2) or float(r2) < float(r2_min):
            return

        a = res.get("a", None)
        p = res.get("p", None)

        if a is not None and np.isfinite(a):
            a_vals.append(float(a))
        if p is not None and np.isfinite(p):
            p_vals.append(float(p))

    # Walk dict (top + one nested level)
    for v in data_dict.values():
        _collect(v)
        if isinstance(v, dict):
            for vv in v.values():
                _collect(vv)

    # ---- Plot a ----
    if a_vals:
        plt.figure(figsize=(8, 5))
        plt.plot(a_vals, "o", alpha=0.75)
        plt.ylabel("Power-law prefactor a")
        plt.xlabel("Index")
        plt.title(f"a values with r² ≥ {r2_min}")
        plt.grid(alpha=0.3)
        plt.savefig(f"{savepath}\\{savename}_a.png", dpi=300)
        plt.close("all")

    # ---- Plot p ----
    if p_vals:
        plt.figure(figsize=(8, 5))
        plt.plot(p_vals, "o", alpha=0.75)
        plt.ylabel("Power-law exponent p")
        plt.xlabel("Index")
        plt.title(f"p values with r² ≥ {r2_min}")
        plt.grid(alpha=0.3)
        plt.savefig(f"{savepath}\\{savename}_p.png", dpi=300)
        plt.close("all")

    # ---- Plot a vs p ----
    if a_vals and p_vals:
        plt.figure(figsize=(7, 6))
        plt.plot(a_vals, p_vals, "o", alpha=0.75)
        plt.xlabel("Power-law prefactor a")
        plt.ylabel("Power-law exponent p")
        plt.title(f"a vs p with r² ≥ {r2_min}")
        plt.grid(alpha=0.3)
        plt.savefig(f"{savepath}\\{savename}_a_vs_p.png", dpi=300)
        plt.close("all")


def cavity_frequency(m, n, p, l,
                     *,
                     a=1.0,
                     c=299792458.0,
                     normalised=True):
    """
    Compute cylindrical cavity frequency f_mnp.

    Parameters
    ----------
    m : int
        Azimuthal index
    n : int
        Radial root index (1-based)
    p : int
        Axial index
    l : float
        length_factor (L = l * lambda / 2)
    a : float, optional
        Cavity radius (meters). Only used if normalised=False.
    c : float, optional
        Speed of light (m/s)
    normalised : bool, optional
        If True, return f / f_010.
        If False, return absolute frequency (Hz).

    Returns
    -------
    float
        Frequency (normalised or absolute)
    """

    # ---- Get Bessel root ν_mn ----
    # jn_zeros gives first k zeros of J_m
    nu_mn = jn_zeros(m, n)[-1]

    # Fundamental root ν_01
    nu_01 = jn_zeros(0, 1)[0]  # 2.404825...

    # ---- If normalised ----
    if normalised:

        B_mn = nu_mn / nu_01

        # Using simplified collapse form:
        # f̂ = sqrt( B_mn^2 + (K * p / l)^2 )
        # K = π a / (ν_01 L) absorbed into geometry.
        # For length_factor definition L = l * λ/2,
        # axial term simplifies to (p / l) form.

        return np.sqrt(B_mn**2 + (p / l)**2)

    # ---- If absolute frequency ----
    else:

        # Physical cavity length
        # L = l * lambda / 2
        # But lambda depends on frequency, so use full formula:

        # Use dispersion relation directly:
        # f = c/(2π) * sqrt( (ν_mn/a)^2 + (pπ/L)^2 )
        # We must compute L from length factor and λ.
        # Since λ = c/f, this is implicit.
        # Instead assume geometric L given directly:
        # L = l * a  (simplified geometry scaling)

        L = l * a

        return (c / (2 * np.pi)) * np.sqrt((nu_mn / a)**2 +
                                           (p * np.pi / L)**2)



def pillbox_freq_from_radius(radius, GHz=False):
    """

    :param radius:
    :return:
    """
    radius = float(radius)
    csol = 299792458.0
    f_Hz = (2.4048 * csol) / (2.0 * np.pi * radius)
    f_GHz = f_Hz / 1e9
    if GHz:
        return f_GHz
    else:
        return f_Hz


def pillbox_radius_from_freq(f_Hz):
    f_Hz = float(f_Hz)
    csol = 299792458.0
    radius_m = (2.4048 * csol) / (2.0 * np.pi * f_Hz)

    return radius_m

def beta_from_energy(E_eV, rest_mass_energy_eV):
    """
    Returns relativistic beta (v/c) for a particle.

    Parameters
    ----------
    E_eV : float or array-like
        Kinetic energy in electron volts (eV)
    rest_mass_energy_eV : float
        Rest mass energy (m c^2) in eV

    Returns
    -------
    beta : float or numpy array
        Relativistic beta (v/c)
    """
    gamma = 1 + np.array(E_eV) / rest_mass_energy_eV
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_modes_with_mode_legend_and_crossing_labels_extrap(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    normalisation_factor=1.3,
    xlim=(0.8, 2.0),
    show=False,
    plotXdelta=-0.0025,
    plotYdelta=-0.2,
    verbose_labels: bool = False,
    # --- extrapolated crossings ---
    extrapolate_pairs=(),
    left_pairs_list=[],
    extrapolate_window=12,
    extrapolate_margin=0.70,
    extrapolate_line_ls="--",
    extrapolate_point_kwargs=None,
    tuning_error=0.03,
):
    """
    Plots modes + known crossings, and adds extrapolated crossings for specified mode pairs.

    Also adds two dashed vertical red lines at:
      x = 1 - tuning_error
      x = 1 + tuning_error

    One of these lines is included in the legend as:
      "Error Margin"
    """

    if extrapolate_point_kwargs is None:
        extrapolate_point_kwargs = dict(
            s=140, facecolors="none", edgecolors="k", linewidths=2.0
        )

    # --------------------------
    # Mode metadata loader (optional)
    # --------------------------
    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    def tech_name(mode_key: str) -> str:
        mode_key = mode_key.replace(" ", "_")
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    def _mode_key(i: int) -> str:
        candidates = [f"mode_{i}", f"mode {i}", f"Mode_{i}", f"Mode {i}"]
        for k in candidates:
            if k in renumbered_modes:
                return k
        s = str(i)
        for k in renumbered_modes.keys():
            kk = str(k).strip().lower().replace(" ", "_")
            if kk.startswith("mode_") and kk.split("_")[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for {i}")

    def _get_xy(mode_key: str):
        x = np.asarray(renumbered_modes[mode_key]["length_factor"], dtype=float)
        if normalised:
            y = np.asarray(
                renumbered_modes[mode_key]["frequency_normalised"], dtype=float
            )
        else:
            y = np.asarray(renumbered_modes[mode_key]["frequency_GHz"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        return x[o], y[o]

    def _fit_powerlaw_end(x, y, n=12, end="right"):
        """
        Fit y = A * x^k using least squares in log-log space
        on first/last n samples.
        Returns (A, k).
        """
        n = int(max(2, min(n, x.size)))
        if end == "right":
            xx, yy = x[-n:], y[-n:]
        else:
            xx, yy = x[:n], y[:n]

        m = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < 2:
            return None

        lx = np.log(xx)
        ly = np.log(yy)

        coeffs = np.polyfit(lx, ly, deg=1)
        k = float(coeffs[0])
        logA = float(coeffs[1])
        A = float(np.exp(logA))
        return A, k

    def _solve_powerlaw_intersection(A1, k1, A2, k2):
        den = (k1 - k2)
        if abs(den) < 1e-12:
            return None
        ratio = A2 / A1
        if ratio <= 0:
            return None
        return float(ratio ** (1.0 / den))

    def _color_for_mode(mk):
        try:
            idx = mode_keys.index(mk)
            return cmap(idx)
        except Exception:
            return "0.25"

    # --------------------------
    # Plot setup
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + legend (dedup by tech name or verbose label)
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(str(k).replace(" ", "_").split("_")[-1])
        if str(k).replace(" ", "_").split("_")[-1].isdigit()
        else 10**9,
    )
    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs, ys = _get_xy(mk)
        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        if verbose_labels:
            lbl = f"{mk}: {tech_name(mk)}"
        else:
            lbl = tech_name(mk)

        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot existing crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / normalisation_factor

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors="k",
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)

        if len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_64":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.25

        elif len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_60":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.25

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_43":
            plotXdelta_use = plotXdelta - 0.01
            plotYdelta_use = plotYdelta

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.45
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta


        if len(pair) >= 2:
            txt = f"{tech_name(pair[0])}\n{tech_name(pair[1])}"
            print(f"Plot text = {txt}")
            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 + plotXdelta_use,
                y0 + plotYdelta_use,
                txt,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Extrapolated crossings
    # --------------------------
    left_pairs = {tuple(sorted(pair)) for pair in left_pairs_list}

    for (i1, i2) in extrapolate_pairs:
        m1 = _mode_key(int(i1))
        m2 = _mode_key(int(i2))

        x1, y1 = _get_xy(m1)
        x2, y2 = _get_xy(m2)
        if x1.size < 2 or x2.size < 2:
            continue

        min_curr = float(max(np.min(x1), np.min(x2)))
        max_curr = float(min(np.max(x1), np.max(x2)))

        want_left = tuple(sorted((int(i1), int(i2)))) in left_pairs
        end = "left" if want_left else "right"

        fit1 = _fit_powerlaw_end(x1, y1, n=extrapolate_window, end=end)
        fit2 = _fit_powerlaw_end(x2, y2, n=extrapolate_window, end=end)
        if fit1 is None or fit2 is None:
            continue

        A1, k1 = fit1
        A2, k2 = fit2

        x_star = _solve_powerlaw_intersection(A1, k1, A2, k2)
        if x_star is None or not np.isfinite(x_star) or x_star <= 0:
            continue

        if want_left:
            if not (x_star < min_curr and x_star >= (min_curr - extrapolate_margin)):
                continue
            x_from = min_curr
        else:
            if not (x_star > max_curr and x_star <= (max_curr + extrapolate_margin)):
                continue
            x_from = max_curr

        y_star = A1 * (x_star ** k1)
        if not (np.isfinite(y_star) and y_star > 0):
            continue

        y1_from = A1 * (x_from ** k1)
        y2_from = A2 * (x_from ** k2)

        c1 = _color_for_mode(m1)
        c2 = _color_for_mode(m2)

        ax.plot(
            [x_from, x_star],
            [y1_from, y_star],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c1,
            zorder=4,
        )
        ax.plot(
            [x_from, x_star],
            [y2_from, A2 * (x_star ** k2)],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c2,
            zorder=4,
        )

        ax.scatter([x_star], [y_star], zorder=6, **extrapolate_point_kwargs)

        print(f"plotter {m1} {m2}")

        if m1 == "mode_84" and m2 == "mode_95":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_79" and m2 == "mode_96":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_84" and m2 == "mode_93":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_34" and m2 == "mode_37":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_11" and m2 == "mode_12":
            plotXdelta_use = 0.025
            plotYdelta_use = plotYdelta
        elif m1 == "mode_6" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta
        elif m1 == "mode_6" and m2 == "mode_10":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta

        elif m1 == "mode_39" and m2 == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2. * plotYdelta

            # input(f"{m1} {m2}")
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        txt = f"{tech_name(m1)}\n{tech_name(m2)}"
        ax.text(
            x_star + plotXdelta_use,
            y_star + plotYdelta_use,
            txt,
            fontsize=8,
            ha="left" if not want_left else "right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=7,
        )

        new_key = f"extrap_{i1}_{i2}"
        if new_key not in crossings:
            crossings[new_key] = {
                "length_factor": float(x_star),
                "frequency_GHz": float(y_star * normalisation_factor),
                "modes": [m1, m2],
                "extrapolated": True,
                "fit": {
                    m1: {"A": float(A1), "k": float(k1), "end": end},
                    m2: {"A": float(A2), "k": float(k2), "end": end},
                },
            }

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)

    # central reference line
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    # error margin lines
    ax.axvline(1.0 - tuning_error, ls="--", color="red", alpha=0.8)
    ax.axvline(1.0 + tuning_error, ls="--", color="red", alpha=0.8)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=13)
    ax.set_ylabel(r"$f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=13)
    ax.grid(alpha=0.25)

    # --------------------------
    # Legend
    # --------------------------
    error_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="Error Margin")

    ax.legend(
        legend_handles + [error_handle],
        legend_labels + ["Error Margin"],
        loc="upper left",
        # loc="lower left",
        fontsize=12,
        framealpha=0.9,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")



def plot_modes_with_mode_legend_and_crossing_labels_extrap_MonoDiQuad(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    normalisation_factor=1.3,
    xlim=(0.8, 2.0),
    show=False,
    plotXdelta=-0.0025,
    plotYdelta=-0.2,
    verbose_labels: bool = False,
    # --- extrapolated crossings ---
    extrapolate_pairs=(),
    left_pairs_list=[],
    extrapolate_window=12,
    extrapolate_margin=0.70,
    extrapolate_line_ls="--",
    extrapolate_point_kwargs=None,
    tuning_error=0.03,
):
    """
    Plots modes + known crossings, and adds extrapolated crossings for specified mode pairs.

    Also adds two dashed vertical red lines at:
      x = 1 - tuning_error
      x = 1 + tuning_error

    One of these lines is included in the legend as:
      "Error Margin"
    """

    if extrapolate_point_kwargs is None:
        extrapolate_point_kwargs = dict(
            s=140, facecolors="none", edgecolors="k", linewidths=2.0
        )

    # --------------------------
    # Mode metadata loader (optional)
    # --------------------------
    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    def tech_name(mode_key: str) -> str:
        mode_key = mode_key.replace(" ", "_")
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    def _mode_key(i: int) -> str:
        candidates = [f"mode_{i}", f"mode {i}", f"Mode_{i}", f"Mode {i}"]
        for k in candidates:
            if k in renumbered_modes:
                return k
        s = str(i)
        for k in renumbered_modes.keys():
            kk = str(k).strip().lower().replace(" ", "_")
            if kk.startswith("mode_") and kk.split("_")[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for {i}")

    def _get_xy(mode_key: str):
        x = np.asarray(renumbered_modes[mode_key]["length_factor"], dtype=float)
        if normalised:
            y = np.asarray(
                renumbered_modes[mode_key]["frequency_normalised"], dtype=float
            )
        else:
            y = np.asarray(renumbered_modes[mode_key]["frequency_GHz"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        return x[o], y[o]

    def _fit_powerlaw_end(x, y, n=12, end="right"):
        """
        Fit y = A * x^k using least squares in log-log space
        on first/last n samples.
        Returns (A, k).
        """
        n = int(max(2, min(n, x.size)))
        if end == "right":
            xx, yy = x[-n:], y[-n:]
        else:
            xx, yy = x[:n], y[:n]

        m = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < 2:
            return None

        lx = np.log(xx)
        ly = np.log(yy)

        coeffs = np.polyfit(lx, ly, deg=1)
        k = float(coeffs[0])
        logA = float(coeffs[1])
        A = float(np.exp(logA))
        return A, k

    def _solve_powerlaw_intersection(A1, k1, A2, k2):
        den = (k1 - k2)
        if abs(den) < 1e-12:
            return None
        ratio = A2 / A1
        if ratio <= 0:
            return None
        return float(ratio ** (1.0 / den))

    def _color_for_mode(mk):
        try:
            idx = mode_keys.index(mk)
            return cmap(idx)
        except Exception:
            return "0.25"

    # --------------------------
    # Plot setup
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + legend (dedup by tech name or verbose label)
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(str(k).replace(" ", "_").split("_")[-1])
        if str(k).replace(" ", "_").split("_")[-1].isdigit()
        else 10**9,
    )
    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs, ys = _get_xy(mk)
        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        if verbose_labels:
            lbl = f"{mk}: {tech_name(mk)}"
        else:
            lbl = tech_name(mk)

        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot existing crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / normalisation_factor

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors="k",
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)

        if len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_64":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.25

        elif len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_60":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.25

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_43":
            plotXdelta_use = plotXdelta - 0.01
            plotYdelta_use = plotYdelta

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.45

        elif len(pair) >= 2 and pair[0] == "mode_94" and pair[1] == "mode_97":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta


        if len(pair) >= 2:
            txt = f"{tech_name(pair[0])}\n{tech_name(pair[1])}"
            print(f"Plot text = {txt}")
            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 + plotXdelta_use,
                y0 + plotYdelta_use,
                txt,
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Extrapolated crossings
    # --------------------------
    left_pairs = {tuple(sorted(pair)) for pair in left_pairs_list}

    for (i1, i2) in extrapolate_pairs:
        m1 = _mode_key(int(i1))
        m2 = _mode_key(int(i2))

        x1, y1 = _get_xy(m1)
        x2, y2 = _get_xy(m2)
        if x1.size < 2 or x2.size < 2:
            continue

        min_curr = float(max(np.min(x1), np.min(x2)))
        max_curr = float(min(np.max(x1), np.max(x2)))

        want_left = tuple(sorted((int(i1), int(i2)))) in left_pairs
        end = "left" if want_left else "right"

        fit1 = _fit_powerlaw_end(x1, y1, n=extrapolate_window, end=end)
        fit2 = _fit_powerlaw_end(x2, y2, n=extrapolate_window, end=end)
        if fit1 is None or fit2 is None:
            continue

        A1, k1 = fit1
        A2, k2 = fit2

        x_star = _solve_powerlaw_intersection(A1, k1, A2, k2)
        if x_star is None or not np.isfinite(x_star) or x_star <= 0:
            continue

        if want_left:
            if not (x_star < min_curr and x_star >= (min_curr - extrapolate_margin)):
                continue
            x_from = min_curr
        else:
            if not (x_star > max_curr and x_star <= (max_curr + extrapolate_margin)):
                continue
            x_from = max_curr

        y_star = A1 * (x_star ** k1)
        if not (np.isfinite(y_star) and y_star > 0):
            continue

        y1_from = A1 * (x_from ** k1)
        y2_from = A2 * (x_from ** k2)

        c1 = _color_for_mode(m1)
        c2 = _color_for_mode(m2)

        ax.plot(
            [x_from, x_star],
            [y1_from, y_star],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c1,
            zorder=4,
        )
        ax.plot(
            [x_from, x_star],
            [y2_from, A2 * (x_star ** k2)],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c2,
            zorder=4,
        )

        ax.scatter([x_star], [y_star], zorder=6, **extrapolate_point_kwargs)

        print(f"plotter {m1} {m2}")

        if m1 == "mode_84" and m2 == "mode_95":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_79" and m2 == "mode_96":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_84" and m2 == "mode_93":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_34" and m2 == "mode_37":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_11" and m2 == "mode_12":
            plotXdelta_use = 0.025
            plotYdelta_use = plotYdelta
        elif m1 == "mode_6" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta
        elif m1 == "mode_6" and m2 == "mode_10":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta

        elif m1 == "mode_39" and m2 == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2. * plotYdelta

        elif m1 == "mode_25" and m2 == "mode_27":
            plotXdelta_use = plotXdelta +0.02
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_27" and m2 == "mode_28":
            plotXdelta_use = plotXdelta + 0.03
            plotYdelta_use = plotYdelta

        elif m1 == "mode_68" and m2 == "mode_71":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_8" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta




            # (25, 27), (25, 28), (27, 28), (58, 68), (68, 71)
            # input(f"{m1} {m2}")
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        txt = f"{tech_name(m1)}\n{tech_name(m2)}"
        ax.text(
            x_star + plotXdelta_use,
            y_star + plotYdelta_use,
            txt,
            fontsize=10,
            ha="left" if not want_left else "right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=7,
        )

        new_key = f"extrap_{i1}_{i2}"
        if new_key not in crossings:
            crossings[new_key] = {
                "length_factor": float(x_star),
                "frequency_GHz": float(y_star * normalisation_factor),
                "modes": [m1, m2],
                "extrapolated": True,
                "fit": {
                    m1: {"A": float(A1), "k": float(k1), "end": end},
                    m2: {"A": float(A2), "k": float(k2), "end": end},
                },
            }

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)

    # central reference line
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    # error margin lines
    ax.axvline(1.0 - tuning_error, ls="--", color="red", alpha=0.8)
    ax.axvline(1.0 + tuning_error, ls="--", color="red", alpha=0.8)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=15)
    ax.set_ylabel(r"$\hat{f} = f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=15)
    ax.grid(alpha=0.25)

    # --------------------------
    # Legend
    # --------------------------
    error_handle_design = Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Design")
    error_handle_error = Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label=fr"Design $\pm$ {tuning_error*100:1.0f}%")

    ax.legend(
        legend_handles + [error_handle_design]+ [error_handle_error],
        legend_labels + ["$\ell=1$"] + [fr"$\ell=1$ $\pm$ {tuning_error*100:1.0f}%"],
        loc="upper left",
        # loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")


def plot_modes_with_mode_legend_and_crossing_labels_extrap_MonoDiQuad_TESLA(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    normalisation_factor=1.3,
    xlim=(0.8, 2.0),
    show=False,
    plotXdelta=-0.0025,
    plotYdelta=-0.2,
    verbose_labels: bool = False,
    # --- extrapolated crossings ---
    extrapolate_pairs=(),
    left_pairs_list=[],
    extrapolate_window=12,
    extrapolate_margin=0.70,
    extrapolate_line_ls="--",
    extrapolate_point_kwargs=None,
    tuning_error=0.03,
):
    """
    Plots modes + known crossings, and adds extrapolated crossings for specified mode pairs.

    Also adds two dashed vertical red lines at:
      x = 1 - tuning_error
      x = 1 + tuning_error

    One of these lines is included in the legend as:
      "Error Margin"
    """

    if extrapolate_point_kwargs is None:
        extrapolate_point_kwargs = dict(
            s=140, facecolors="none", edgecolors="k", linewidths=2.0
        )

    # --------------------------
    # Mode metadata loader (optional)
    # --------------------------
    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    def tech_name(mode_key: str) -> str:
        mode_key = mode_key.replace(" ", "_")
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    def _mode_key(i: int) -> str:
        candidates = [f"mode_{i}", f"mode {i}", f"Mode_{i}", f"Mode {i}"]
        for k in candidates:
            if k in renumbered_modes:
                return k
        s = str(i)
        for k in renumbered_modes.keys():
            kk = str(k).strip().lower().replace(" ", "_")
            if kk.startswith("mode_") and kk.split("_")[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for {i}")

    def _get_xy(mode_key: str):
        x = np.asarray(renumbered_modes[mode_key]["length_factor"], dtype=float)
        if normalised:
            y = np.asarray(
                renumbered_modes[mode_key]["frequency_normalised"], dtype=float
            )
        else:
            y = np.asarray(renumbered_modes[mode_key]["frequency_GHz"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        return x[o], y[o]

    def _fit_powerlaw_end(x, y, n=12, end="right"):
        """
        Fit y = A * x^k using least squares in log-log space
        on first/last n samples.
        Returns (A, k).
        """
        n = int(max(2, min(n, x.size)))
        if end == "right":
            xx, yy = x[-n:], y[-n:]
        else:
            xx, yy = x[:n], y[:n]

        m = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < 2:
            return None

        lx = np.log(xx)
        ly = np.log(yy)

        coeffs = np.polyfit(lx, ly, deg=1)
        k = float(coeffs[0])
        logA = float(coeffs[1])
        A = float(np.exp(logA))
        return A, k

    def _solve_powerlaw_intersection(A1, k1, A2, k2):
        den = (k1 - k2)
        if abs(den) < 1e-12:
            return None
        ratio = A2 / A1
        if ratio <= 0:
            return None
        return float(ratio ** (1.0 / den))

    def _color_for_mode(mk):
        try:
            idx = mode_keys.index(mk)
            return cmap(idx)
        except Exception:
            return "0.25"

    # --------------------------
    # Plot setup
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + legend (dedup by tech name or verbose label)
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(str(k).replace(" ", "_").split("_")[-1])
        if str(k).replace(" ", "_").split("_")[-1].isdigit()
        else 10**9,
    )
    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs, ys = _get_xy(mk)
        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        if verbose_labels:
            lbl = f"{mk}: {tech_name(mk)}"
        else:
            lbl = tech_name(mk)

        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot existing crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        # print(f"{c.keys() = }")
        # print(f"{c['mode_i'] = }")
        # input(f"{c['mode_j'] = }")
        if c['mode_i'] == 'mode_39' and c['mode_j'] == 'mode_43':
            colour = 'r'
        elif c['mode_i'] == 'mode_43' and c['mode_j'] == 'mode_46':
            colour = "darkorange"
        else:
            colour = 'k'

        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / normalisation_factor

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors=colour,
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)

        # if len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_64":
        #     plotXdelta_use = plotXdelta -0.05
        #     plotYdelta_use = -1.0

        if len(pair) >= 2 and pair[0] == "mode_43" and pair[1] == "mode_46":
            plotXdelta_use = plotXdelta - 0.1
            plotYdelta_use = -1.25

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_43":
            plotXdelta_use = plotXdelta - 0.01
            plotYdelta_use = plotYdelta

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.45

        elif len(pair) >= 2 and pair[0] == "mode_94" and pair[1] == "mode_97":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta


        if len(pair) >= 2:
            txt = f"{tech_name(pair[0])}\n{tech_name(pair[1])}"
            print(f"Plot text = {txt}")
            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 + plotXdelta_use,
                y0 + plotYdelta_use,
                txt,
                fontsize=16,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Extrapolated crossings
    # --------------------------
    left_pairs = {tuple(sorted(pair)) for pair in left_pairs_list}

    for (i1, i2) in extrapolate_pairs:
        m1 = _mode_key(int(i1))
        m2 = _mode_key(int(i2))

        x1, y1 = _get_xy(m1)
        x2, y2 = _get_xy(m2)
        if x1.size < 2 or x2.size < 2:
            continue

        min_curr = float(max(np.min(x1), np.min(x2)))
        max_curr = float(min(np.max(x1), np.max(x2)))

        want_left = tuple(sorted((int(i1), int(i2)))) in left_pairs
        end = "left" if want_left else "right"

        fit1 = _fit_powerlaw_end(x1, y1, n=extrapolate_window, end=end)
        fit2 = _fit_powerlaw_end(x2, y2, n=extrapolate_window, end=end)
        if fit1 is None or fit2 is None:
            continue

        A1, k1 = fit1
        A2, k2 = fit2

        x_star = _solve_powerlaw_intersection(A1, k1, A2, k2)
        if x_star is None or not np.isfinite(x_star) or x_star <= 0:
            continue

        if want_left:
            if not (x_star < min_curr and x_star >= (min_curr - extrapolate_margin)):
                continue
            x_from = min_curr
        else:
            if not (x_star > max_curr and x_star <= (max_curr + extrapolate_margin)):
                continue
            x_from = max_curr

        y_star = A1 * (x_star ** k1)
        if not (np.isfinite(y_star) and y_star > 0):
            continue

        y1_from = A1 * (x_from ** k1)
        y2_from = A2 * (x_from ** k2)

        c1 = _color_for_mode(m1)
        c2 = _color_for_mode(m2)

        ax.plot(
            [x_from, x_star],
            [y1_from, y_star],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c1,
            zorder=4,
        )
        ax.plot(
            [x_from, x_star],
            [y2_from, A2 * (x_star ** k2)],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c2,
            zorder=4,
        )

        ax.scatter([x_star], [y_star], zorder=6, **extrapolate_point_kwargs)

        print(f"plotter {m1} {m2}")

        if m1 == "mode_43" and m2 == "mode_46":
            plotYdelta_use = -plotYdelta * 2.4
            plotXdelta_use = -plotXdelta * 8.0
        elif m1 == "mode_79" and m2 == "mode_96":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_84" and m2 == "mode_93":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_34" and m2 == "mode_37":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_11" and m2 == "mode_12":
            plotXdelta_use = 0.025
            plotYdelta_use = plotYdelta
        elif m1 == "mode_6" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta
        elif m1 == "mode_6" and m2 == "mode_10":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta

        elif m1 == "mode_39" and m2 == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2. * plotYdelta

        elif m1 == "mode_25" and m2 == "mode_27":
            plotXdelta_use = plotXdelta +0.02
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_27" and m2 == "mode_28":
            plotXdelta_use = plotXdelta + 0.03
            plotYdelta_use = plotYdelta

        elif m1 == "mode_68" and m2 == "mode_71":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_8" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta




            # (25, 27), (25, 28), (27, 28), (58, 68), (68, 71)
            # input(f"{m1} {m2}")
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        txt = f"{tech_name(m1)}\n{tech_name(m2)}"
        ax.text(
            x_star + plotXdelta_use,
            y_star + plotYdelta_use,
            txt,
            fontsize=16,
            ha="left" if not want_left else "right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=7,
        )

        new_key = f"extrap_{i1}_{i2}"
        if new_key not in crossings:
            crossings[new_key] = {
                "length_factor": float(x_star),
                "frequency_GHz": float(y_star * normalisation_factor),
                "modes": [m1, m2],
                "extrapolated": True,
                "fit": {
                    m1: {"A": float(A1), "k": float(k1), "end": end},
                    m2: {"A": float(A2), "k": float(k2), "end": end},
                },
            }

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)

    # central reference line
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    # error margin lines
    # ax.axvline(1.0 - tuning_error, ls="--", color="red", alpha=0.8)
    # ax.axvline(1.0 + tuning_error, ls="--", color="red", alpha=0.8)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=20)
    ax.set_ylabel(r"$\hat{f} = f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=20)
    ax.grid(alpha=0.25)

    ## --------------------------
    # Legend
    # --------------------------
    error_handle_design = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$\ell=1$"
    )

    ax.legend(
        legend_handles + [error_handle_design],
        legend_labels + [r"$\ell=1$"],
        loc="upper left",
        fontsize=14,          # increase/decrease for paper readability
        ncol=2,               # two-column legend
        framealpha=0.9,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.6,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")
        
        
        

def plot_modes_with_mode_legend_and_crossing_labels_extrap_MonoDiQuad_PIPII(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    normalisation_factor=1.3,
    xlim=(0.8, 2.0),
    show=False,
    plotXdelta=-0.0025,
    plotYdelta=-0.2,
    verbose_labels: bool = False,
    # --- extrapolated crossings ---
    extrapolate_pairs=(),
    left_pairs_list=[],
    extrapolate_window=12,
    extrapolate_margin=0.70,
    extrapolate_line_ls="--",
    extrapolate_point_kwargs=None,
    tuning_error=0.03,
):
    """
    Plots modes + known crossings, and adds extrapolated crossings for specified mode pairs.

    Also adds two dashed vertical red lines at:
      x = 1 - tuning_error
      x = 1 + tuning_error

    One of these lines is included in the legend as:
      "Error Margin"
    """

    if extrapolate_point_kwargs is None:
        extrapolate_point_kwargs = dict(
            s=140, facecolors="none", edgecolors="k", linewidths=2.0
        )

    # --------------------------
    # Mode metadata loader (optional)
    # --------------------------
    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    def tech_name(mode_key: str) -> str:
        mode_key = mode_key.replace(" ", "_")
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    def _mode_key(i: int) -> str:
        candidates = [f"mode_{i}", f"mode {i}", f"Mode_{i}", f"Mode {i}"]
        for k in candidates:
            if k in renumbered_modes:
                return k
        s = str(i)
        for k in renumbered_modes.keys():
            kk = str(k).strip().lower().replace(" ", "_")
            if kk.startswith("mode_") and kk.split("_")[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for {i}")

    def _get_xy(mode_key: str):
        x = np.asarray(renumbered_modes[mode_key]["length_factor"], dtype=float)
        if normalised:
            y = np.asarray(
                renumbered_modes[mode_key]["frequency_normalised"], dtype=float
            )
        else:
            y = np.asarray(renumbered_modes[mode_key]["frequency_GHz"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        return x[o], y[o]

    def _fit_powerlaw_end(x, y, n=12, end="right"):
        """
        Fit y = A * x^k using least squares in log-log space
        on first/last n samples.
        Returns (A, k).
        """
        n = int(max(2, min(n, x.size)))
        if end == "right":
            xx, yy = x[-n:], y[-n:]
        else:
            xx, yy = x[:n], y[:n]

        m = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < 2:
            return None

        lx = np.log(xx)
        ly = np.log(yy)

        coeffs = np.polyfit(lx, ly, deg=1)
        k = float(coeffs[0])
        logA = float(coeffs[1])
        A = float(np.exp(logA))
        return A, k

    def _solve_powerlaw_intersection(A1, k1, A2, k2):
        den = (k1 - k2)
        if abs(den) < 1e-12:
            return None
        ratio = A2 / A1
        if ratio <= 0:
            return None
        return float(ratio ** (1.0 / den))

    def _color_for_mode(mk):
        try:
            idx = mode_keys.index(mk)
            return cmap(idx)
        except Exception:
            return "0.25"

    # --------------------------
    # Plot setup
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + legend (dedup by tech name or verbose label)
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(str(k).replace(" ", "_").split("_")[-1])
        if str(k).replace(" ", "_").split("_")[-1].isdigit()
        else 10**9,
    )
    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs, ys = _get_xy(mk)
        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        if verbose_labels:
            lbl = f"{mk}: {tech_name(mk)}"
        else:
            lbl = tech_name(mk)

        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot existing crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        # print(f"{c.keys() = }")
        # print(f"{c['mode_i'] = }")
        # input(f"{c['mode_j'] = }")
        if c['mode_i'] == 'mode_39' and c['mode_j'] == 'mode_43':
            colour = 'r'
        elif c['mode_i'] == 'mode_43' and c['mode_j'] == 'mode_46':
            colour = "darkorange"
        else:
            colour = 'k'

        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / normalisation_factor

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors=colour,
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)

        # if len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_64":
        #     plotXdelta_use = plotXdelta -0.05
        #     plotYdelta_use = -1.0

        if len(pair) >= 2 and pair[0] == "mode_43" and pair[1] == "mode_46":
            plotXdelta_use = plotXdelta - 0.1
            plotYdelta_use = -1.25

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_43":
            plotXdelta_use = plotXdelta - 0.01
            plotYdelta_use = plotYdelta

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.45

        elif len(pair) >= 2 and pair[0] == "mode_94" and pair[1] == "mode_97":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta


        if len(pair) >= 2:
            txt = f"{tech_name(pair[0])}\n{tech_name(pair[1])}"
            print(f"Plot text = {txt}")
            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 + plotXdelta_use,
                y0 + plotYdelta_use,
                txt,
                fontsize=16,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Extrapolated crossings
    # --------------------------
    left_pairs = {tuple(sorted(pair)) for pair in left_pairs_list}

    for (i1, i2) in extrapolate_pairs:
        m1 = _mode_key(int(i1))
        m2 = _mode_key(int(i2))

        x1, y1 = _get_xy(m1)
        x2, y2 = _get_xy(m2)
        if x1.size < 2 or x2.size < 2:
            continue

        min_curr = float(max(np.min(x1), np.min(x2)))
        max_curr = float(min(np.max(x1), np.max(x2)))

        want_left = tuple(sorted((int(i1), int(i2)))) in left_pairs
        end = "left" if want_left else "right"

        fit1 = _fit_powerlaw_end(x1, y1, n=extrapolate_window, end=end)
        fit2 = _fit_powerlaw_end(x2, y2, n=extrapolate_window, end=end)
        if fit1 is None or fit2 is None:
            continue

        A1, k1 = fit1
        A2, k2 = fit2

        x_star = _solve_powerlaw_intersection(A1, k1, A2, k2)
        if x_star is None or not np.isfinite(x_star) or x_star <= 0:
            continue

        if want_left:
            if not (x_star < min_curr and x_star >= (min_curr - extrapolate_margin)):
                continue
            x_from = min_curr
        else:
            if not (x_star > max_curr and x_star <= (max_curr + extrapolate_margin)):
                continue
            x_from = max_curr

        y_star = A1 * (x_star ** k1)
        if not (np.isfinite(y_star) and y_star > 0):
            continue

        y1_from = A1 * (x_from ** k1)
        y2_from = A2 * (x_from ** k2)
        
        
        
        c1 = _color_for_mode(m1)
        c2 = _color_for_mode(m2)

        ax.plot(
            [x_from, x_star],
            [y1_from, y_star],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c1,
            zorder=4,
        )
        ax.plot(
            [x_from, x_star],
            [y2_from, A2 * (x_star ** k2)],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c2,
            zorder=4,
        )
        print(f"{m1 = }\n{m2 = }")
        if m1 == 'mode_46' and m2 == 'mode_50':
            colour = 'r'
        else:
            colour = 'k'
       
        ax.scatter([x_star], [y_star], zorder=6, color=colour,  s=140, facecolors="none", edgecolors=colour, linewidths=2.0)

        print(f"plotter {m1} {m2}")

        if m1 == "mode_43" and m2 == "mode_46":
            plotYdelta_use = -plotYdelta * 2.4
            plotXdelta_use = -plotXdelta * 8.0
        elif m1 == "mode_79" and m2 == "mode_96":
            plotYdelta_use = -0.25
            plotXdelta_use = plotXdelta
        elif m1 == "mode_84" and m2 == "mode_93":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_34" and m2 == "mode_37":
            plotYdelta_use = -plotYdelta * 2.0
            plotXdelta_use = plotXdelta
        elif m1 == "mode_11" and m2 == "mode_12":
            plotXdelta_use = 0.025
            plotYdelta_use = plotYdelta
        elif m1 == "mode_6" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta
        elif m1 == "mode_6" and m2 == "mode_10":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.*plotYdelta

        elif m1 == "mode_39" and m2 == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2. * plotYdelta

        elif m1 == "mode_25" and m2 == "mode_27":
            plotXdelta_use = plotXdelta +0.02
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_27" and m2 == "mode_28":
            plotXdelta_use = plotXdelta + 0.03
            plotYdelta_use = plotYdelta

        elif m1 == "mode_68" and m2 == "mode_71":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta

        elif m1 == "mode_8" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -1.5 * plotYdelta




            # (25, 27), (25, 28), (27, 28), (58, 68), (68, 71)
            # input(f"{m1} {m2}")
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        txt = f"{tech_name(m1)}\n{tech_name(m2)}"
        ax.text(
            x_star + plotXdelta_use,
            y_star + plotYdelta_use,
            txt,
            fontsize=16,
            ha="left" if not want_left else "right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=7,
        )

        new_key = f"extrap_{i1}_{i2}"
        if new_key not in crossings:
            crossings[new_key] = {
                "length_factor": float(x_star),
                "frequency_GHz": float(y_star * normalisation_factor),
                "modes": [m1, m2],
                "extrapolated": True,
                "fit": {
                    m1: {"A": float(A1), "k": float(k1), "end": end},
                    m2: {"A": float(A2), "k": float(k2), "end": end},
                },
            }

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)

    # central reference line
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    # error margin lines
    # ax.axvline(1.0 - tuning_error, ls="--", color="red", alpha=0.8)
    # ax.axvline(1.0 + tuning_error, ls="--", color="red", alpha=0.8)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=20)
    ax.set_ylabel(r"$\hat{f} = f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=20)
    ax.grid(alpha=0.25)

    # --------------------------
    # Legend
    # --------------------------
    error_handle_design = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$\ell=1$"
    )

    ax.legend(
        legend_handles + [error_handle_design],
        legend_labels + [r"$\ell=1$"],
        loc="upper left",
        fontsize=14,  # increase/decrease for paper readability
        ncol=2,  # two-column legend
        framealpha=0.9,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.6,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")


def plot_modes_with_mode_legend_and_crossing_labels_extrap_MonoDiQuad_3HC(
    renumbered_modes: dict,
    crossings: dict,
    savepath: str,
    savename: str,
    *,
    mode_meta: dict | None = None,
    mode_meta_py: str | None = None,
    normalised: bool = True,
    normalisation_factor=1.3,
    xlim=(0.8, 2.0),
    show=False,
    plotXdelta=-0.0025,
    plotYdelta=-0.2,
    verbose_labels: bool = False,
    # --- extrapolated crossings ---
    extrapolate_pairs=(),
    left_pairs_list=[],
    extrapolate_window=12,
    extrapolate_margin=0.70,
    extrapolate_line_ls="--",
    extrapolate_point_kwargs=None,
    tuning_error=0.03,
):
    """
    Plots modes + known crossings, and adds extrapolated crossings for specified mode pairs.

    Also adds two dashed vertical red lines at:
      x = 1 - tuning_error
      x = 1 + tuning_error

    One of these lines is included in the legend as:
      "Error Margin"
    """

    if extrapolate_point_kwargs is None:
        extrapolate_point_kwargs = dict(
            s=140, facecolors="none", edgecolors="k", linewidths=2.0
        )

    # --------------------------
    # Mode metadata loader (optional)
    # --------------------------
    if mode_meta is None:
        if mode_meta_py is None:
            raise ValueError("Provide either mode_meta or mode_meta_py")
        mode_meta = load_mode_metadata_dict(mode_meta_py)

    def tech_name(mode_key: str) -> str:
        mode_key = mode_key.replace(" ", "_")
        info = mode_meta.get(mode_key)
        if not isinstance(info, dict):
            return mode_key
        pol = str(info.get("TM_TE", "")).upper()
        m = info.get("m")
        n = info.get("n")
        p = info.get("p")
        if pol in {"TM", "TE"} and m is not None and n is not None and p is not None:
            return rf"${pol}_{{{int(m)}{int(n)}{int(p)}}}$"
        return mode_key

    def extract_cross_modes(c: dict):
        if "modes" in c:
            return c["modes"]
        if "mode_pair" in c:
            return c["mode_pair"]
        if "mode_i" in c and "mode_j" in c:
            return [c["mode_i"], c["mode_j"]]
        return []

    def _mode_key(i: int) -> str:
        candidates = [f"mode_{i}", f"mode {i}", f"Mode_{i}", f"Mode {i}"]
        for k in candidates:
            if k in renumbered_modes:
                return k
        s = str(i)
        for k in renumbered_modes.keys():
            kk = str(k).strip().lower().replace(" ", "_")
            if kk.startswith("mode_") and kk.split("_")[-1] == s:
                return k
        raise KeyError(f"Could not find mode key for {i}")

    def _get_xy(mode_key: str):
        x = np.asarray(renumbered_modes[mode_key]["length_factor"], dtype=float)
        if normalised:
            y = np.asarray(
                renumbered_modes[mode_key]["frequency_normalised"], dtype=float
            )
        else:
            y = np.asarray(renumbered_modes[mode_key]["frequency_GHz"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        return x[o], y[o]

    def _fit_powerlaw_end(x, y, n=12, end="right"):
        """
        Fit y = A * x^k using least squares in log-log space
        on first/last n samples.
        Returns (A, k).
        """
        n = int(max(2, min(n, x.size)))
        if end == "right":
            xx, yy = x[-n:], y[-n:]
        else:
            xx, yy = x[:n], y[:n]

        m = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < 2:
            return None

        lx = np.log(xx)
        ly = np.log(yy)

        coeffs = np.polyfit(lx, ly, deg=1)
        k = float(coeffs[0])
        logA = float(coeffs[1])
        A = float(np.exp(logA))
        return A, k

    def _solve_powerlaw_intersection(A1, k1, A2, k2):
        den = (k1 - k2)
        if abs(den) < 1e-12:
            return None
        ratio = A2 / A1
        if ratio <= 0:
            return None
        return float(ratio ** (1.0 / den))

    def _color_for_mode(mk):
        try:
            idx = mode_keys.index(mk)
            return cmap(idx)
        except Exception:
            return "0.25"

    # --------------------------
    # Plot setup
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------
    # Plot modes + legend (dedup by tech name or verbose label)
    # --------------------------
    mode_keys = sorted(
        renumbered_modes.keys(),
        key=lambda k: int(str(k).replace(" ", "_").split("_")[-1])
        if str(k).replace(" ", "_").split("_")[-1].isdigit()
        else 10**9,
    )
    cmap = plt.cm.get_cmap("viridis", max(1, len(mode_keys)))

    seen_labels = set()
    legend_handles = []
    legend_labels = []

    for i, mk in enumerate(mode_keys):
        xs, ys = _get_xy(mk)
        h, = ax.plot(xs, ys, lw=1.2, color=cmap(i), alpha=0.9)

        if verbose_labels:
            lbl = f"{mk}: {tech_name(mk)}"
        else:
            lbl = tech_name(mk)

        if lbl not in seen_labels:
            seen_labels.add(lbl)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # --------------------------
    # Plot existing crossings + annotate
    # --------------------------
    for _, c in crossings.items():
        # print(f"{c.keys() = }")
        # print(f"{c['mode_i'] = }")
        # input(f"{c['mode_j'] = }")
        if c['mode_i'] == 'mode_39' and c['mode_j'] == 'mode_43':
            colour = 'r'
        elif c['mode_i'] == 'mode_43' and c['mode_j'] == 'mode_46':
            colour = "darkorange"
        else:
            colour = 'k'

        cx = np.asarray(c["length_factor"], dtype=float)
        cy = np.asarray(c["frequency_GHz"], dtype=float) / normalisation_factor

        ax.scatter(
            cx,
            cy,
            s=140,
            facecolors="none",
            edgecolors=colour,
            linewidths=2.0,
            zorder=5,
        )

        pair = extract_cross_modes(c)

        # if len(pair) >= 2 and pair[0] == "mode_56" and pair[1] == "mode_64":
        #     plotXdelta_use = plotXdelta -0.05
        #     plotYdelta_use = -1.0

        if len(pair) >= 2 and pair[0] == "mode_43" and pair[1] == "mode_46":
            plotXdelta_use = plotXdelta - 0.1
            plotYdelta_use = -1.25

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_43":
            plotXdelta_use = plotXdelta - 0.01
            plotYdelta_use = plotYdelta

        elif len(pair) >= 2 and pair[0] == "mode_39" and pair[1] == "mode_45":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -0.45

        elif len(pair) >= 2 and pair[0] == "mode_94" and pair[1] == "mode_97":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.0 * plotYdelta
        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta


        if len(pair) >= 2:
            txt = f"{tech_name(pair[0])}\n{tech_name(pair[1])}"
            print(f"Plot text = {txt}")
            x0 = float(np.atleast_1d(cx)[0])
            y0 = float(np.atleast_1d(cy)[0])

            ax.text(
                x0 + plotXdelta_use,
                y0 + plotYdelta_use,
                txt,
                fontsize=16,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    # --------------------------
    # Extrapolated crossings
    # --------------------------
    left_pairs = {tuple(sorted(pair)) for pair in left_pairs_list}

    for (i1, i2) in extrapolate_pairs:
        m1 = _mode_key(int(i1))
        m2 = _mode_key(int(i2))

        x1, y1 = _get_xy(m1)
        x2, y2 = _get_xy(m2)
        if x1.size < 2 or x2.size < 2:
            continue

        min_curr = float(max(np.min(x1), np.min(x2)))
        max_curr = float(min(np.max(x1), np.max(x2)))

        want_left = tuple(sorted((int(i1), int(i2)))) in left_pairs
        end = "left" if want_left else "right"

        fit1 = _fit_powerlaw_end(x1, y1, n=extrapolate_window, end=end)
        fit2 = _fit_powerlaw_end(x2, y2, n=extrapolate_window, end=end)
        if fit1 is None or fit2 is None:
            continue

        A1, k1 = fit1
        A2, k2 = fit2

        x_star = _solve_powerlaw_intersection(A1, k1, A2, k2)
        if x_star is None or not np.isfinite(x_star) or x_star <= 0:
            continue

        if want_left:
            if not (x_star < min_curr and x_star >= (min_curr - extrapolate_margin)):
                continue
            x_from = min_curr
        else:
            if not (x_star > max_curr and x_star <= (max_curr + extrapolate_margin)):
                continue
            x_from = max_curr

        y_star = A1 * (x_star ** k1)
        if not (np.isfinite(y_star) and y_star > 0):
            continue

        y1_from = A1 * (x_from ** k1)
        y2_from = A2 * (x_from ** k2)
        
        
        
        c1 = _color_for_mode(m1)
        c2 = _color_for_mode(m2)

        ax.plot(
            [x_from, x_star],
            [y1_from, y_star],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c1,
            zorder=4,
        )
        ax.plot(
            [x_from, x_star],
            [y2_from, A2 * (x_star ** k2)],
            ls=extrapolate_line_ls,
            lw=1.2,
            alpha=0.9,
            color=c2,
            zorder=4,
        )
        print(f"{m1 = }\n{m2 = }")
        if m1 == 'mode_58' and m2 == 'mode_68':
            colour = 'r'
        else:
            colour = 'k'
       
        ax.scatter([x_star], [y_star], zorder=6, color=colour,  s=140, facecolors="none", edgecolors=colour, linewidths=2.0)

        print(f"plotter {m1} {m2}")

        # default
        plotXdelta_use = plotXdelta
        plotYdelta_use = plotYdelta

        if m1 == "mode_8" and m2 == "mode_9":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.1 * plotYdelta

        elif m1 == "mode_25" and m2 == "mode_27":
            plotXdelta_use = plotXdelta + 0.04
            plotYdelta_use = -2.1 * plotYdelta

        elif m1 == "mode_25" and m2 == "mode_28":
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        elif m1 == "mode_27" and m2 == "mode_28":
            plotXdelta_use = plotXdelta + 0.04
            plotYdelta_use = plotYdelta

        elif m1 == "mode_58" and m2 == "mode_68":
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        elif m1 == "mode_68" and m2 == "mode_71":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.1 * plotYdelta

        elif m1 == "mode_85" and m2 == "mode_94":
            plotXdelta_use = plotXdelta 
            plotYdelta_use = plotYdelta

        elif m1 == "mode_70" and m2 == "mode_71":
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        elif m1 == "mode_28" and m2 == "mode_83":
            plotXdelta_use = plotXdelta
            plotYdelta_use = -2.0 * plotYdelta

        else:
            plotXdelta_use = plotXdelta
            plotYdelta_use = plotYdelta

        txt = f"{tech_name(m1)}\n{tech_name(m2)}"
        ax.text(
            x_star + plotXdelta_use,
            y_star + plotYdelta_use,
            txt,
            fontsize=16,
            ha="left" if not want_left else "right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=7,
        )

        new_key = f"extrap_{i1}_{i2}"
        if new_key not in crossings:
            crossings[new_key] = {
                "length_factor": float(x_star),
                "frequency_GHz": float(y_star * normalisation_factor),
                "modes": [m1, m2],
                "extrapolated": True,
                "fit": {
                    m1: {"A": float(A1), "k": float(k1), "end": end},
                    m2: {"A": float(A2), "k": float(k2), "end": end},
                },
            }

    # --------------------------
    # Styling
    # --------------------------
    ax.set_xlim(*xlim)

    # central reference line
    ax.axvline(1.0, ls="--", color="k", alpha=0.65)

    # error margin lines
    # ax.axvline(1.0 - tuning_error, ls="--", color="red", alpha=0.8)
    # ax.axvline(1.0 + tuning_error, ls="--", color="red", alpha=0.8)

    ax.set_xlabel(r"$\ell=2d/\lambda$", fontsize=20)
    ax.set_ylabel(r"$\hat{f} = f_{mnp}/f_{010}$" if normalised else "f [GHz]", fontsize=20)
    ax.grid(alpha=0.25)

    # --------------------------
    # Legend
    # --------------------------
    error_handle_design = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$\ell=1$"
    )

    ax.legend(
        legend_handles + [error_handle_design],
        legend_labels + [r"$\ell=1$"],
        loc="upper left",
        fontsize=14,  # increase/decrease for paper readability
        ncol=2,  # two-column legend
        framealpha=0.9,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.6,
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{savepath}\\{savename}.png", dpi=300)
        plt.close("all")


def fix_crossing_with_target_x(
    mode_dict,
    mode_a="",
    mode_b="",
    *,
    crossing_x=0.,
    allow_break_radius=2,
    prefer_swap_after=True,
):
    """
    Reassign datapoints between two modes, but **force** the identity swap (crossing)
    to happen near a required length_factor (crossing_x).

    Assumptions
    -----------
    - mode_a and mode_b share the same length_factor grid
    - The "best" reassignment is a *single* identity swap at some breakpoint index k:
        for x <= x[k]   : keep (A,B)
        for x >  x[k]   : swap (A,B)  (or the opposite, depending on prefer_swap_after)

    This enforces a physically plausible single crossing near crossing_x, rather than
    allowing scattered pointwise swaps.

    Parameters
    ----------
    mode_dict : dict
        Your modes dict (loaded from .pkl).
    mode_a, mode_b : str
        Keys for the two modes to fix.
    crossing_x : float
        Required/expected crossing location in length_factor (e.g. ~1.025).
    allow_break_radius : int
        How many grid indices around the nearest crossing index are allowed as breakpoints.
        (0 means force the nearest grid breakpoint only.)
    prefer_swap_after : bool
        If True (default): swap identity AFTER the breakpoint (x > break).
        If False: swap identity BEFORE the breakpoint.

    Returns
    -------
    fixed_dict : dict
        Updated dict with reassigned data between the two modes.
    info : dict
        Metadata about the chosen breakpoint and how many points were swapped.
    """
    # shallow copy dict + per-mode dicts
    md = {k: (v.copy() if isinstance(v, dict) else v) for k, v in mode_dict.items()}

    if mode_a not in md or mode_b not in md:
        raise KeyError(f"Need both {mode_a} and {mode_b} in mode_dict")

    A = md[mode_a].copy()
    B = md[mode_b].copy()

    xA = np.asarray(A["length_factor"], dtype=float)
    xB = np.asarray(B["length_factor"], dtype=float)
    if len(xA) != len(xB) or not np.allclose(xA, xB):
        raise ValueError("Modes do not share the same length_factor grid; cannot reassign pointwise.")
    x = xA

    yA = np.asarray(A["frequency_GHz"], dtype=float)
    yB = np.asarray(B["frequency_GHz"], dtype=float)

    # Find the nearest segment index to crossing_x: want x[k] <= crossing_x < x[k+1]
    k0 = int(np.clip(np.searchsorted(x, crossing_x) - 1, 0, len(x) - 2))

    # Candidate breakpoints around k0
    kmin = max(0, k0 - int(allow_break_radius))
    kmax = min(len(x) - 2, k0 + int(allow_break_radius))
    candidates = list(range(kmin, kmax + 1))

    def apply_break_swap(y1, y2, k_break, swap_after=True):
        """
        Build reassigned streams (out1, out2) with a single swap at k_break.
        If swap_after=True: swap for indices > k_break.
        If swap_after=False: swap for indices <= k_break.
        """
        out1 = y1.copy()
        out2 = y2.copy()
        if swap_after:
            mask = np.zeros(len(x), dtype=bool)
            mask[k_break + 1 :] = True
        else:
            mask = np.ones(len(x), dtype=bool)
            mask[k_break + 1 :] = False

        out1[mask] = y2[mask]
        out2[mask] = y1[mask]
        return out1, out2, mask

    def sse_piecewise_linear(out1, out2):
        """
        Score assignment by how 'line-like' each curve is (simple global linear fit SSE).
        This is the same spirit as your "use extremes to interpolate" requirement, but
        more robust than forcing endpoints only.
        """
        # fit out1 ~ a1*x+b1, out2 ~ a2*x+b2
        a1, b1 = np.polyfit(x, out1, 1)
        a2, b2 = np.polyfit(x, out2, 1)
        r1 = out1 - (a1 * x + b1)
        r2 = out2 - (a2 * x + b2)
        return float(np.sum(r1 * r1) + np.sum(r2 * r2))

    # Evaluate candidates with both swap directions (after vs before),
    # and choose best while staying near crossing_x by construction.
    best = None
    for k in candidates:
        for swap_after in (True, False):
            outA, outB, mask = apply_break_swap(yA, yB, k, swap_after=swap_after)
            score = sse_piecewise_linear(outA, outB)

            # optional preference: if tie-ish, prefer_swap_after wins
            tie_bias = 0.0
            if prefer_swap_after and not swap_after:
                tie_bias = 1e-12
            if (not prefer_swap_after) and swap_after:
                tie_bias = 1e-12

            total = score + tie_bias

            if best is None or total < best["total"]:
                best = {
                    "k_break": k,
                    "swap_after": swap_after,
                    "mask": mask,
                    "score": score,
                    "total": total,
                }

    # Apply best mask to frequency_GHz and frequency_normalised (if present)
    k_break = best["k_break"]
    swap_after = best["swap_after"]
    mask = best["mask"]

    def apply_mask_list(arr_a, arr_b, mask):
        aa = np.asarray(arr_a)
        bb = np.asarray(arr_b)
        out_a = aa.copy()
        out_b = bb.copy()
        out_a[mask] = bb[mask]
        out_b[mask] = aa[mask]
        return out_a.tolist(), out_b.tolist()

    A["frequency_GHz"], B["frequency_GHz"] = apply_mask_list(A["frequency_GHz"], B["frequency_GHz"], mask)

    if "frequency_normalised" in A and "frequency_normalised" in B:
        A["frequency_normalised"], B["frequency_normalised"] = apply_mask_list(
            A["frequency_normalised"], B["frequency_normalised"], mask
        )

    md[mode_a] = A
    md[mode_b] = B

    info = {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "crossing_x_target": float(crossing_x),
        "k_break": int(k_break),
        "x_break": float(x[k_break]),
        "x_break_next": float(x[k_break + 1]),
        "swap_after": bool(swap_after),
        "num_points_swapped": int(np.sum(mask)),
        "score_linear_fit_sse": float(best["score"]),
        "swapmask": mask,   # numpy bool array
        "x_grid": x,        # numpy array
    }
    return md, info

def pickle_dump(dictionary, dir_fname):
    with open(dir_fname, "wb") as handle:

        pkl.dump(dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"Saved pickle out: {dir_fname}")

def pickle_load(dir_fname):
    with open(dir_fname, 'rb') as handle:
        dictionary = pkl.load(handle)

    return dictionary



def get_3D_data_rotated(
    field_map_filename_E1: str,
    field_map_filename_E2: str,
    array_path: str,
    plot: bool = False,
    create_data: bool = True,
    coord_unit: str = "mm",
):
    """
    Uses read_3D_CST_field_data() (must be defined/imported) to read two CST 3D E-field maps,
    then saves/loads ALL arrays as 3D arrays.

    Returned dict matches your original keys, but with the NECESSARY change:
      - it now ALSO returns the x coordinate vectors as xs1 and xs2
        (you already return y and z as ys*/zs*).
    """

    os.makedirs(array_path, exist_ok=True)

    if create_data:
        # --- Read full 3D field dictionaries ---
        E1 = cylindrical_rotate_align_and_plot(field_map_filename_E1, plot=plot, n_r=120, n_phi=360, n_z=200, r_max=None)
        E2 = cylindrical_rotate_align_and_plot(field_map_filename_E2, plot=plot, n_r=120, n_phi=360, n_z=200, r_max=None)

        # --- Extract 3D components ---
        E1_Ex, E1_Ey, E1_Ez = E1["Ex"], E1["Ey"], E1["Ez"]
        E2_Ex, E2_Ey, E2_Ez = E2["Ex"], E2["Ey"], E2["Ez"]

        # coordinate vectors (1D)
        x1, y1, z1 = E1["x"], E1["y"], E1["z"]
        x2, y2, z2 = E2["x"], E2["y"], E2["z"]

        # sanity: require identical grids (same shape + coords)
        if E1_Ex.shape != E2_Ex.shape:
            raise ValueError(
                f"E1 and E2 grid shapes differ: {E1_Ex.shape} vs {E2_Ex.shape}"
            )
        if not (np.allclose(x1, x2) and np.allclose(y1, y2) and np.allclose(z1, z2)):
            raise ValueError("E1 and E2 coordinate vectors differ (x/y/z).")

        # --- Plus/minus combos (3D) ---
        Ex_plus = E1_Ex + E2_Ex
        Ey_plus = E1_Ey + E2_Ey
        Ez_plus = E1_Ez + E2_Ez
        Ex_minus = E1_Ex - E2_Ex
        Ey_minus = E1_Ey - E2_Ey
        Ez_minus = E1_Ez - E2_Ez

        # --- Magnitudes (3D) ---
        abs_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2 + np.abs(E1_Ez) ** 2)
        abs_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2 + np.abs(E2_Ez) ** 2)
        abs_add = np.sqrt(
            np.abs(Ex_plus) ** 2 + np.abs(Ey_plus) ** 2 + np.abs(Ez_plus) ** 2
        )
        abs_sub = np.sqrt(
            np.abs(Ex_minus) ** 2 + np.abs(Ey_minus) ** 2 + np.abs(Ez_minus) ** 2
        )

        # --- Transverse Fields (3D) ---
        trans_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2)
        trans_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2)

        # --- Save everything (3D arrays + coordinate vectors) ---
        np.save(os.path.join(array_path, "abs_E1.npy"), abs_E1)
        np.save(os.path.join(array_path, "E1_Ex.npy"), E1_Ex)
        np.save(os.path.join(array_path, "E1_Ey.npy"), E1_Ey)
        np.save(os.path.join(array_path, "E1_Ez.npy"), E1_Ez)
        np.save(os.path.join(array_path, "x1.npy"), x1)
        np.save(os.path.join(array_path, "y1.npy"), y1)
        np.save(os.path.join(array_path, "z1.npy"), z1)

        np.save(os.path.join(array_path, "abs_E2.npy"), abs_E2)
        np.save(os.path.join(array_path, "E2_Ex.npy"), E2_Ex)
        np.save(os.path.join(array_path, "E2_Ey.npy"), E2_Ey)
        np.save(os.path.join(array_path, "E2_Ez.npy"), E2_Ez)
        np.save(os.path.join(array_path, "x2.npy"), x2)
        np.save(os.path.join(array_path, "y2.npy"), y2)
        np.save(os.path.join(array_path, "z2.npy"), z2)

        np.save(os.path.join(array_path, "trans_E1.npy"), trans_E1)
        np.save(os.path.join(array_path, "trans_E2.npy"), trans_E2)

        np.save(os.path.join(array_path, "abs_add.npy"), abs_add)
        np.save(os.path.join(array_path, "Ex_plus.npy"), Ex_plus)
        np.save(os.path.join(array_path, "Ey_plus.npy"), Ey_plus)
        np.save(os.path.join(array_path, "Ez_plus.npy"), Ez_plus)

        np.save(os.path.join(array_path, "abs_sub.npy"), abs_sub)
        np.save(os.path.join(array_path, "Ex_minus.npy"), Ex_minus)
        np.save(os.path.join(array_path, "Ey_minus.npy"), Ey_minus)
        np.save(os.path.join(array_path, "Ez_minus.npy"), Ez_minus)

    else:
        # --- Load everything (3D arrays + coordinate vectors) ---
        abs_E1 = np.load(os.path.join(array_path, "abs_E1.npy"))
        E1_Ex = np.load(os.path.join(array_path, "E1_Ex.npy"))
        E1_Ey = np.load(os.path.join(array_path, "E1_Ey.npy"))
        E1_Ez = np.load(os.path.join(array_path, "E1_Ez.npy"))
        x1 = np.load(os.path.join(array_path, "x1.npy"))
        y1 = np.load(os.path.join(array_path, "y1.npy"))
        z1 = np.load(os.path.join(array_path, "z1.npy"))

        abs_E2 = np.load(os.path.join(array_path, "abs_E2.npy"))
        E2_Ex = np.load(os.path.join(array_path, "E2_Ex.npy"))
        E2_Ey = np.load(os.path.join(array_path, "E2_Ey.npy"))
        E2_Ez = np.load(os.path.join(array_path, "E2_Ez.npy"))
        x2 = np.load(os.path.join(array_path, "x2.npy"))
        y2 = np.load(os.path.join(array_path, "y2.npy"))
        z2 = np.load(os.path.join(array_path, "z2.npy"))

        trans_E1 = np.load(os.path.join(array_path, "trans_E1.npy"))
        trans_E2 = np.load(os.path.join(array_path, "trans_E2.npy"))

        abs_add = np.load(os.path.join(array_path, "abs_add.npy"))
        Ex_plus = np.load(os.path.join(array_path, "Ex_plus.npy"))
        Ey_plus = np.load(os.path.join(array_path, "Ey_plus.npy"))
        Ez_plus = np.load(os.path.join(array_path, "Ez_plus.npy"))

        abs_sub = np.load(os.path.join(array_path, "abs_sub.npy"))
        Ex_minus = np.load(os.path.join(array_path, "Ex_minus.npy"))
        Ey_minus = np.load(os.path.join(array_path, "Ey_minus.npy"))
        Ez_minus = np.load(os.path.join(array_path, "Ez_minus.npy"))

    print(f"{E1_Ex.shape = }")
    print(f"{E1_Ey.shape = }")
    print(f"{E1_Ez.shape = }")

    # Keep original naming scheme; add xs1/xs2 as the necessary change.
    return {
        "abs_E1": abs_E1,
        "E1_Ex": E1_Ex,
        "E1_Ey": E1_Ey,
        "E1_Ez": E1_Ez,
        "xs1": x1,   # <-- added (necessary)
        "ys1": y1,
        "zs1": z1,
        "abs_E2": abs_E2,
        "E2_Ex": E2_Ex,
        "E2_Ey": E2_Ey,
        "E2_Ez": E2_Ez,
        "xs2": x2,   # <-- added (necessary)
        "ys2": y2,
        "zs2": z2,
        "abs_add": abs_add,
        "Ex_plus": Ex_plus,
        "Ey_plus": Ey_plus,
        "Ez_plus": Ez_plus,
        "trans_E1": trans_E1,
        "trans_E2": trans_E2,
        "abs_sub": abs_sub,
        "Ex_minus": Ex_minus,
        "Ey_minus": Ey_minus,
        "Ez_minus": Ez_minus,
    }

def get_3D_data_monopole(
    field_map_filename_E1: str,
    field_map_filename_E2: str,
    array_path: str,
    plot: bool = False,
    create_data: bool = True,
    coord_unit: str = "mm",
):
    """
    Uses read_3D_CST_field_data() (must be defined/imported) to read two CST 3D E-field maps,
    then saves/loads ALL arrays as 3D arrays.

    Returned dict matches your original keys, but with the NECESSARY change:
      - it now ALSO returns the x coordinate vectors as xs1 and xs2
        (you already return y and z as ys*/zs*).
    """

    cols = [
        "x [pixels]",
        "y [pixels]",
        "z [pixels]",
        "ExRe [V/m]",
        "ExIm [V/m]",
        "EyRe [V/m]",
        "EyIm [V/m]",
        "EzRe [V/m]",
        "EzIm [V/m]",
    ]

    os.makedirs(array_path, exist_ok=True)

    if create_data:
        # --- Read full 3D field dictionaries ---
        E1 = pd.read_csv(field_map_filename_E1, sep=r"\s+", skiprows=2, names=cols, engine="python")
        E2 = pd.read_csv(field_map_filename_E2, sep=r"\s+", skiprows=2, names=cols, engine="python")

        # --- Extract 3D components ---
        E1_Ex, E1_Ey, E1_Ez = E1["Ex"], E1["Ey"], E1["Ez"]
        E2_Ex, E2_Ey, E2_Ez = E2["Ex"], E2["Ey"], E2["Ez"]

        # coordinate vectors (1D)
        x1, y1, z1 = E1["x"], E1["y"], E1["z"]
        x2, y2, z2 = E2["x"], E2["y"], E2["z"]

        # sanity: require identical grids (same shape + coords)
        if E1_Ex.shape != E2_Ex.shape:
            raise ValueError(
                f"E1 and E2 grid shapes differ: {E1_Ex.shape} vs {E2_Ex.shape}"
            )
        if not (np.allclose(x1, x2) and np.allclose(y1, y2) and np.allclose(z1, z2)):
            raise ValueError("E1 and E2 coordinate vectors differ (x/y/z).")

        # --- Plus/minus combos (3D) ---
        Ex_plus = E1_Ex + E2_Ex
        Ey_plus = E1_Ey + E2_Ey
        Ez_plus = E1_Ez + E2_Ez
        Ex_minus = E1_Ex - E2_Ex
        Ey_minus = E1_Ey - E2_Ey
        Ez_minus = E1_Ez - E2_Ez

        # --- Magnitudes (3D) ---
        abs_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2 + np.abs(E1_Ez) ** 2)
        abs_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2 + np.abs(E2_Ez) ** 2)
        abs_add = np.sqrt(
            np.abs(Ex_plus) ** 2 + np.abs(Ey_plus) ** 2 + np.abs(Ez_plus) ** 2
        )
        abs_sub = np.sqrt(
            np.abs(Ex_minus) ** 2 + np.abs(Ey_minus) ** 2 + np.abs(Ez_minus) ** 2
        )

        # --- Transverse Fields (3D) ---
        trans_E1 = np.sqrt(np.abs(E1_Ex) ** 2 + np.abs(E1_Ey) ** 2)
        trans_E2 = np.sqrt(np.abs(E2_Ex) ** 2 + np.abs(E2_Ey) ** 2)

        # --- Save everything (3D arrays + coordinate vectors) ---
        np.save(os.path.join(array_path, "abs_E1.npy"), abs_E1)
        np.save(os.path.join(array_path, "E1_Ex.npy"), E1_Ex)
        np.save(os.path.join(array_path, "E1_Ey.npy"), E1_Ey)
        np.save(os.path.join(array_path, "E1_Ez.npy"), E1_Ez)
        np.save(os.path.join(array_path, "x1.npy"), x1)
        np.save(os.path.join(array_path, "y1.npy"), y1)
        np.save(os.path.join(array_path, "z1.npy"), z1)

        np.save(os.path.join(array_path, "abs_E2.npy"), abs_E2)
        np.save(os.path.join(array_path, "E2_Ex.npy"), E2_Ex)
        np.save(os.path.join(array_path, "E2_Ey.npy"), E2_Ey)
        np.save(os.path.join(array_path, "E2_Ez.npy"), E2_Ez)
        np.save(os.path.join(array_path, "x2.npy"), x2)
        np.save(os.path.join(array_path, "y2.npy"), y2)
        np.save(os.path.join(array_path, "z2.npy"), z2)

        np.save(os.path.join(array_path, "trans_E1.npy"), trans_E1)
        np.save(os.path.join(array_path, "trans_E2.npy"), trans_E2)

        np.save(os.path.join(array_path, "abs_add.npy"), abs_add)
        np.save(os.path.join(array_path, "Ex_plus.npy"), Ex_plus)
        np.save(os.path.join(array_path, "Ey_plus.npy"), Ey_plus)
        np.save(os.path.join(array_path, "Ez_plus.npy"), Ez_plus)

        np.save(os.path.join(array_path, "abs_sub.npy"), abs_sub)
        np.save(os.path.join(array_path, "Ex_minus.npy"), Ex_minus)
        np.save(os.path.join(array_path, "Ey_minus.npy"), Ey_minus)
        np.save(os.path.join(array_path, "Ez_minus.npy"), Ez_minus)

    else:
        # --- Load everything (3D arrays + coordinate vectors) ---
        abs_E1 = np.load(os.path.join(array_path, "abs_E1.npy"))
        E1_Ex = np.load(os.path.join(array_path, "E1_Ex.npy"))
        E1_Ey = np.load(os.path.join(array_path, "E1_Ey.npy"))
        E1_Ez = np.load(os.path.join(array_path, "E1_Ez.npy"))
        x1 = np.load(os.path.join(array_path, "x1.npy"))
        y1 = np.load(os.path.join(array_path, "y1.npy"))
        z1 = np.load(os.path.join(array_path, "z1.npy"))

        abs_E2 = np.load(os.path.join(array_path, "abs_E2.npy"))
        E2_Ex = np.load(os.path.join(array_path, "E2_Ex.npy"))
        E2_Ey = np.load(os.path.join(array_path, "E2_Ey.npy"))
        E2_Ez = np.load(os.path.join(array_path, "E2_Ez.npy"))
        x2 = np.load(os.path.join(array_path, "x2.npy"))
        y2 = np.load(os.path.join(array_path, "y2.npy"))
        z2 = np.load(os.path.join(array_path, "z2.npy"))

        trans_E1 = np.load(os.path.join(array_path, "trans_E1.npy"))
        trans_E2 = np.load(os.path.join(array_path, "trans_E2.npy"))

        abs_add = np.load(os.path.join(array_path, "abs_add.npy"))
        Ex_plus = np.load(os.path.join(array_path, "Ex_plus.npy"))
        Ey_plus = np.load(os.path.join(array_path, "Ey_plus.npy"))
        Ez_plus = np.load(os.path.join(array_path, "Ez_plus.npy"))

        abs_sub = np.load(os.path.join(array_path, "abs_sub.npy"))
        Ex_minus = np.load(os.path.join(array_path, "Ex_minus.npy"))
        Ey_minus = np.load(os.path.join(array_path, "Ey_minus.npy"))
        Ez_minus = np.load(os.path.join(array_path, "Ez_minus.npy"))

    print(f"{E1_Ex.shape = }")
    print(f"{E1_Ey.shape = }")
    print(f"{E1_Ez.shape = }")

    # Keep original naming scheme; add xs1/xs2 as the necessary change.
    return {
        "abs_E1": abs_E1,
        "E1_Ex": E1_Ex,
        "E1_Ey": E1_Ey,
        "E1_Ez": E1_Ez,
        "xs1": x1,   # <-- added (necessary)
        "ys1": y1,
        "zs1": z1,
        "abs_E2": abs_E2,
        "E2_Ex": E2_Ex,
        "E2_Ey": E2_Ey,
        "E2_Ez": E2_Ez,
        "xs2": x2,   # <-- added (necessary)
        "ys2": y2,
        "zs2": z2,
        "abs_add": abs_add,
        "Ex_plus": Ex_plus,
        "Ey_plus": Ey_plus,
        "Ez_plus": Ez_plus,
        "trans_E1": trans_E1,
        "trans_E2": trans_E2,
        "abs_sub": abs_sub,
        "Ex_minus": Ex_minus,
        "Ey_minus": Ey_minus,
        "Ez_minus": Ez_minus,
    }



def cylindrical_rotate_align_and_plot(
    filepath, plot, n_r=120, n_phi=360, n_z=200, r_max=None
):


    # ------------------------------------------------------------
    # Load Cartesian field
    # ------------------------------------------------------------
    cols = [
        "x [pixels]",
        "y [pixels]",
        "z [pixels]",
        "ExRe [V/m]",
        "ExIm [V/m]",
        "EyRe [V/m]",
        "EyIm [V/m]",
        "EzRe [V/m]",
        "EzIm [V/m]",
    ]
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=2, names=cols, engine="python")

    xs = np.sort(df["x [pixels]"].unique())
    ys = np.sort(df["y [pixels]"].unique())
    zs = np.sort(df["z [pixels]"].unique())

    nx, ny, nz0 = len(xs), len(ys), len(zs)

    xi = pd.Categorical(df["x [pixels]"], categories=xs, ordered=True).codes
    yi = pd.Categorical(df["y [pixels]"], categories=ys, ordered=True).codes
    zi = pd.Categorical(df["z [pixels]"], categories=zs, ordered=True).codes

    Ex = np.zeros((nx, ny, nz0))
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)

    Ex[xi, yi, zi] = df["ExRe [V/m]"].to_numpy()
    Ey[xi, yi, zi] = df["EyRe [V/m]"].to_numpy()
    Ez[xi, yi, zi] = df["EzRe [V/m]"].to_numpy()

    Eabs = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # ------------------------------------------------------------
    # Cylindrical grid
    # ------------------------------------------------------------
    if r_max is None:
        r_max = min(np.max(np.abs(xs)), np.max(np.abs(ys)))

    r = np.linspace(0, r_max, n_r)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    phi_deg = np.degrees(phi)

    z = np.linspace(zs.min(), zs.max(), n_z)
    z_center = n_z // 2

    # ------------------------------------------------------------
    # Cylindrical interpolation of Eabs (before rotation)
    # Eabs_cyl[z,r,phi]
    # ------------------------------------------------------------
    interp_Eabs = RegularGridInterpolator(
        (xs, ys, zs), Eabs, bounds_error=False, fill_value=0.0
    )

    RR, PP = np.meshgrid(r, phi, indexing="ij")
    X = RR * np.cos(PP)
    Y = RR * np.sin(PP)
    base_xy = np.column_stack([X.ravel(), Y.ravel()])

    Eabs_cyl = np.zeros((n_z, n_r, n_phi))
    for k, z0 in enumerate(z):
        pts = np.column_stack([base_xy, np.full(base_xy.shape[0], z0)])
        Eabs_cyl[k] = interp_Eabs(pts).reshape(n_r, n_phi)

    # ------------------------------------------------------------
    # Plot BEFORE rotation
    # ------------------------------------------------------------
    slice0 = Eabs_cyl[z_center]
    r0, p0 = np.unravel_index(np.argmax(slice0), slice0.shape)

    if plot:
        plt.figure()
        plt.pcolormesh(phi_deg, r, slice0, shading="auto")
        plt.scatter(
            phi_deg[p0], r[r0], s=120, facecolors="none", edgecolors="red", linewidths=2
        )
        plt.xlabel("phi (deg)")
        plt.ylabel("r (pixels)")
        plt.title("Before rotation: Eabs[z_center,:,:]")
        plt.show()

    # ------------------------------------------------------------
    # Compute rotation angle to move peak into vertical plane x=0
    # i.e. abs[mid_pixel, :, :]
    # ------------------------------------------------------------
    phi_peak = phi[p0]

    candidate_targets = np.array([np.pi / 2, 3 * np.pi / 2])
    candidate_rotations = candidate_targets - phi_peak

    # wrap to [-pi, pi)
    candidate_rotations = (candidate_rotations + np.pi) % (2 * np.pi) - np.pi

    best_idx = np.argmin(np.abs(candidate_rotations))
    angle_rad = candidate_rotations[best_idx]
    angle_deg = np.degrees(angle_rad)

    # ------------------------------------------------------------
    # Rotate Cartesian field (original resolution)
    # ------------------------------------------------------------
    interp_Ex = RegularGridInterpolator(
        (xs, ys, zs), Ex, bounds_error=False, fill_value=0.0
    )
    interp_Ey = RegularGridInterpolator(
        (xs, ys, zs), Ey, bounds_error=False, fill_value=0.0
    )
    interp_Ez = RegularGridInterpolator(
        (xs, ys, zs), Ez, bounds_error=False, fill_value=0.0
    )

    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    a = np.radians(angle_deg)
    Xs = Xg * np.cos(-a) - Yg * np.sin(-a)
    Ys = Xg * np.sin(-a) + Yg * np.cos(-a)
    base_xy_cart = np.column_stack([Xs.ravel(), Ys.ravel()])

    Ex_rot = np.zeros_like(Ex)
    Ey_rot = np.zeros_like(Ey)
    Ez_rot = np.zeros_like(Ez)

    for k, z0 in enumerate(zs):
        pts = np.column_stack([base_xy_cart, np.full(base_xy_cart.shape[0], z0)])
        Ex_rot[:, :, k] = interp_Ex(pts).reshape(nx, ny)
        Ey_rot[:, :, k] = interp_Ey(pts).reshape(nx, ny)
        Ez_rot[:, :, k] = interp_Ez(pts).reshape(nx, ny)

    Eabs_rot = np.sqrt(Ex_rot**2 + Ey_rot**2 + Ez_rot**2)

    # ------------------------------------------------------------
    # Cylindrical interpolation AFTER rotation
    # ------------------------------------------------------------
    interp_Eabs_rot = RegularGridInterpolator(
        (xs, ys, zs), Eabs_rot, bounds_error=False, fill_value=0.0
    )

    Eabs_cyl_rot = np.zeros_like(Eabs_cyl)
    for k, z0 in enumerate(z):
        pts = np.column_stack([base_xy, np.full(base_xy.shape[0], z0)])
        Eabs_cyl_rot[k] = interp_Eabs_rot(pts).reshape(n_r, n_phi)

    # ------------------------------------------------------------
    # Plot AFTER rotation
    # ------------------------------------------------------------
    slice1 = Eabs_cyl_rot[z_center]
    r1, p1 = np.unravel_index(np.argmax(slice1), slice1.shape)

    if plot:
        plt.figure()
        plt.pcolormesh(phi_deg, r, slice1, shading="auto")
        plt.scatter(
            phi_deg[p1], r[r1], s=120, facecolors="none", edgecolors="red", linewidths=2
        )
        plt.xlabel("phi (deg)")
        plt.ylabel("r (pixels)")
        plt.title("After rotation: Eabs[z_center,:,:]")
        plt.show()

    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "Ex": Ex_rot,
        "Ey": Ey_rot,
        "Ez": Ez_rot,
        "Eabs": Eabs_rot,
        "angle_deg": angle_deg,
    }

def pad_mode(s):
    return re.sub(r'_(\d+)$', lambda m: f"{int(m.group(1)):03d}", s)