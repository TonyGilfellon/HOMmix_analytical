#!/usr/bin/env python3
"""
compare_Ez_loss_kick_focus_parent_degen_freqs.py

Independent comparison of Ez-derived loss, dipole kick and quadrupole
focusing/defocusing for four Ez field maps: E1, E2, E+ and E-.

Expected pickle format
----------------------
{
    "E1":    np.ndarray[nx, ny, nz],
    "E2":    np.ndarray[nx, ny, nz],
    "plus":  np.ndarray[nx, ny, nz],
    "minus": np.ndarray[nx, ny, nz],
}

Method 1: Taylor / modal decomposition method
---------------------------------------------
1. Build the complex transit-time voltage map

       Vz(x,y) = integral Ez(x,y,z) exp(i omega z/(beta c)) dz.

2. Fit the near-axis voltage map to

       Vz = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2.

3. Calculate

       k_parallel = |a0|^2/(4U),
       k_perp     = |(c/omega) [ax, ay]|^2/(4U),
       K          = (c/omega) [[2*axx, axy], [axy, 2*ayy]] / sqrt(4U).

Method 2: local quadratic least-squares validation method
---------------------------------------------------------
1. Build the same complex transit-time voltage map Vz(x,y).
2. Sample a small local stencil around the true cylinder axis.
3. Fit a local quadratic surface to that small stencil,

       Vz = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2,

   and analytically differentiate the local fit at the axis.
4. Calculate the same k_parallel, k_perp and K quantities.

This is intentionally separate from Method 1: Method 1 uses the wider
near-axis modal/Taylor aperture, while Method 2 uses only a small local
validation stencil.

Important geometry convention
-----------------------------
The cylinder axis is explicitly array[75, 75, :]. The cylinder radius Req_m is
therefore radius_pixels=75.0 pixels from the axis by default, so

       dx = dy = Req_m / 75.0.

This replaces the older assumption dx = 2*Req_m/(nx-1), which only happens to
match when nx=151 and the axis/radius convention is exact.

Per-field frequency/length rules
--------------------------------
E1:
    frequency = f_E1, the native/parent mode frequency evaluated at design length
    length    = d0 = lambda_010/2

E2:
    frequency = f_E2, the native/parent mode frequency evaluated at design length
    length    = d0 = lambda_010/2

plus, minus:
    frequency = f_degen, the degenerate/mixed-field frequency
    length    = d = d0 * ell

Here f_010 is retained only to define the design half-wavelength d0 and, if
Req_m is not supplied, the default pillbox radius. It is not used as the E1/E2
analysis frequency unless you explicitly set f_E1=f_010 or f_E2=f_010.

Stored energy
-------------
U is calculated separately for every field. Because the pickle only contains Ez,
the default is an Ez-only electric-energy proxy:

    U_Ez = eps0/2 * integral_cylinder |Ez|^2 dV.

If you later have full E and H maps, replace stored_energy_from_Ez_only() with
the full electromagnetic stored-energy integral.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import map_coordinates as _scipy_map_coordinates
except Exception:  # pragma: no cover - scipy may not be installed on all systems
    _scipy_map_coordinates = None

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
PC = 1e-12
DEFAULT_FIELD_NAMES = ("E1", "E2", "plus", "minus")


@dataclass(frozen=True)
class FieldParams:
    frequency_Hz: float
    length_m: float
    Req_m: float
    beta: float = 1.0

    # Explicit axis convention for 151x151 maps.
    axis_i: float = 75.0
    axis_j: float = 75.0
    radius_pixels: float = 75.0

    # Method controls.
    # Method 1: wider near-axis/modal fit radius.
    fit_pixels: int = 8

    # Method 2: small local validation fit radius.
    local_fit_pixels: int = 3

    # Retained for backward compatibility; no longer used by Method 2.
    fd_step_pixels: float = 2.0
    interp_order: int = 1  # used only if fractional-axis interpolation is required

    # Filled in per field by with_calculated_energy().
    U_J: float | None = None

    @property
    def dx_m(self) -> float:
        if self.Req_m <= 0:
            raise ValueError("Req_m must be > 0.")
        if self.radius_pixels <= 0:
            raise ValueError("radius_pixels must be > 0.")
        return float(self.Req_m) / float(self.radius_pixels)

    @property
    def dy_m(self) -> float:
        return self.dx_m

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * float(self.frequency_Hz)


# -----------------------------------------------------------------------------
# Loading and grids
# -----------------------------------------------------------------------------

def load_fields(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected pickle to contain a dict, got {type(data)!r}.")

    fields: dict[str, np.ndarray] = {}
    for name in DEFAULT_FIELD_NAMES:
        if name not in data:
            raise KeyError(f"Missing field {name!r}; found keys {list(data.keys())!r}.")
        arr = np.asarray(data[name], dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Field {name!r} is not 3D; shape={arr.shape}.")
        fields[name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return fields


def z_grid(length_m: float, nz: int, *, centre_z: bool = False) -> np.ndarray:
    if nz < 2:
        raise ValueError("Need at least two longitudinal samples.")
    if centre_z:
        return np.linspace(-0.5 * length_m, 0.5 * length_m, nz)
    return np.linspace(0.0, length_m, nz)


def xy_coordinate_arrays(shape: tuple[int, int, int], params: FieldParams) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, _ = shape
    x = (np.arange(nx, dtype=float) - float(params.axis_i)) * params.dx_m
    y = (np.arange(ny, dtype=float) - float(params.axis_j)) * params.dy_m
    return x, y


def validate_axis_inside(Ez: np.ndarray, params: FieldParams, margin_pixels: float = 0.0) -> None:
    nx, ny, _ = Ez.shape
    if not (margin_pixels <= params.axis_i <= nx - 1 - margin_pixels):
        raise ValueError(f"axis_i={params.axis_i} is outside map bounds for nx={nx}.")
    if not (margin_pixels <= params.axis_j <= ny - 1 - margin_pixels):
        raise ValueError(f"axis_j={params.axis_j} is outside map bounds for ny={ny}.")


# -----------------------------------------------------------------------------
# Stored energy
# -----------------------------------------------------------------------------

def stored_energy_from_Ez_only(Ez: np.ndarray, params: FieldParams) -> float:
    """Ez-only electric stored-energy proxy over cylindrical aperture r <= Req_m."""
    Ez = np.asarray(Ez, dtype=float)
    nx, ny, nz = Ez.shape
    x, y = xy_coordinate_arrays(Ez.shape, params)
    dz = params.length_m / (nz - 1)

    X, Y = np.meshgrid(x, y, indexing="ij")
    mask_xy = (X * X + Y * Y) <= params.Req_m * params.Req_m
    dV = params.dx_m * params.dy_m * dz
    U = 0.5 * EPS0 * float(np.sum(Ez * Ez * mask_xy[:, :, None])) * dV
    if not np.isfinite(U) or U <= 0:
        raise ValueError(f"Calculated non-positive stored energy U={U!r}.")
    return U


def with_calculated_energy(Ez: np.ndarray, params: FieldParams) -> FieldParams:
    return replace(params, U_J=stored_energy_from_Ez_only(Ez, params))


def require_U(params: FieldParams) -> float:
    if params.U_J is None or params.U_J <= 0:
        raise ValueError("FieldParams.U_J must be calculated before metric evaluation.")
    return float(params.U_J)


# -----------------------------------------------------------------------------
# Transit-time voltage map and interpolation
# -----------------------------------------------------------------------------

def accelerating_voltage_map(Ez: np.ndarray, params: FieldParams, *, centre_z: bool = False) -> np.ndarray:
    """
    Return Vz_map[i,j] = integral Ez[i,j,z] exp(i omega z/(beta c)) dz.

    This is vectorised over x,y and is used by both methods, ensuring the same
    transit-time convention is used throughout.
    """
    Ez = np.asarray(Ez, dtype=float)
    nz = Ez.shape[2]
    z = z_grid(params.length_m, nz, centre_z=centre_z)
    phase = np.exp(1j * params.omega * z / (float(params.beta) * C0))
    return np.trapezoid(Ez * phase[None, None, :], z, axis=2)


def _bilinear_sample_real(A: np.ndarray, i: float, j: float) -> float:
    nx, ny = A.shape
    if i < 0 or j < 0 or i > nx - 1 or j > ny - 1:
        raise ValueError(f"Interpolation point ({i}, {j}) outside array shape {A.shape}.")
    i0 = int(np.floor(i)); j0 = int(np.floor(j))
    i1 = min(i0 + 1, nx - 1); j1 = min(j0 + 1, ny - 1)
    ti = i - i0; tj = j - j0
    return float(
        (1 - ti) * (1 - tj) * A[i0, j0]
        + ti * (1 - tj) * A[i1, j0]
        + (1 - ti) * tj * A[i0, j1]
        + ti * tj * A[i1, j1]
    )


def sample_complex_map(Vz_map: np.ndarray, i: float, j: float, *, order: int = 1) -> complex:
    """
    Interpolate a complex Vz map at fractional array coordinates (i,j).

    Uses scipy.ndimage.map_coordinates when available. Falls back to bilinear
    interpolation for order=1.
    """
    Vz_map = np.asarray(Vz_map, dtype=complex)
    if _scipy_map_coordinates is not None:
        coords = np.array([[float(i)], [float(j)]])
        real = _scipy_map_coordinates(Vz_map.real, coords, order=order, mode="nearest")[0]
        imag = _scipy_map_coordinates(Vz_map.imag, coords, order=order, mode="nearest")[0]
        return complex(real, imag)

    if order != 1:
        raise RuntimeError("scipy is not available; only bilinear interpolation order=1 is supported.")
    return complex(_bilinear_sample_real(Vz_map.real, i, j), _bilinear_sample_real(Vz_map.imag, i, j))


# -----------------------------------------------------------------------------
# Shared metric utilities
# -----------------------------------------------------------------------------

def phase_align_complex_matrix(M: np.ndarray) -> tuple[np.ndarray, float]:
    M = np.asarray(M, dtype=complex)
    idx = np.unravel_index(np.argmax(np.abs(M)), M.shape)
    phase = float(np.angle(M[idx]))
    return M * np.exp(-1j * phase), phase


def phase_align_complex_vector(v: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.asarray(v, dtype=complex)
    idx = int(np.argmax(np.abs(v)))
    phase = float(np.angle(v[idx]))
    return v * np.exp(-1j * phase), phase


def loss_factor_from_V(V: complex, U_J: float) -> float:
    return float(abs(V) ** 2 / (4.0 * U_J))


def dipole_kick_factor_from_gradient(grad: np.ndarray, params: FieldParams) -> float:
    U_J = require_U(params)
    grad_norm = float(np.linalg.norm(np.asarray(grad, dtype=complex)))  # V/m
    Vperp_per_m = (C0 / params.omega) * grad_norm
    return float(abs(Vperp_per_m) ** 2 / (4.0 * U_J))


def K_matrix_energy_normalised(H: np.ndarray, params: FieldParams) -> np.ndarray:
    U_J = require_U(params)
    K_raw = (C0 / params.omega) * np.asarray(H, dtype=complex)
    return K_raw / np.sqrt(4.0 * U_J)


def quadrupole_invariants(K_real: np.ndarray) -> dict[str, float]:
    """Return rotation-tolerant diagnostics for a real 2x2 quadrupole matrix."""
    Kxx = float(K_real[0, 0])
    Kxy = float(K_real[0, 1])
    Kyy = float(K_real[1, 1])
    evals = np.linalg.eigvalsh(K_real)
    quad_strength = float(np.sqrt((Kxx - Kyy) ** 2 + 4.0 * Kxy ** 2))
    return {
        "K_eig_min_U_norm": float(evals[0]),
        "K_eig_max_U_norm": float(evals[1]),
        "K_quad_strength_U_norm": quad_strength,
        "K_frobenius_U_norm": float(np.linalg.norm(K_real)),
    }


def metrics_from_voltage_derivatives(
    *,
    V0: complex,
    grad: np.ndarray,
    H: np.ndarray,
    params: FieldParams,
    method_family: str,
    fit_pixels_used: int | float | None = None,
    fd_step_pixels: float | None = None,
) -> dict[str, Any]:
    U_J = require_U(params)
    kpar = loss_factor_from_V(V0, U_J)
    kperp = dipole_kick_factor_from_gradient(grad, params)

    K_complex = K_matrix_energy_normalised(H, params)
    K_phase, K_phase_rad = phase_align_complex_matrix(K_complex)
    K_real = K_phase.real

    _, lin_phase_rad = phase_align_complex_vector(np.asarray(grad, dtype=complex))

    Kxx = float(K_real[0, 0])
    Kxy = float(K_real[0, 1])
    Kyy = float(K_real[1, 1])

    electron_x = "focusing" if Kxx > 0 else "defocusing" if Kxx < 0 else "neutral"
    electron_y = "focusing" if Kyy > 0 else "defocusing" if Kyy < 0 else "neutral"

    out: dict[str, Any] = {
        "method_family": method_family,
        "k_parallel_V_per_C": kpar,
        "k_parallel_V_per_pC": kpar * PC,
        "k_perp_V_per_C_per_m2": kperp,
        "k_perp_V_per_pC_per_m2": kperp * PC,
        "Kxx_U_norm": Kxx,
        "Kxy_U_norm": Kxy,
        "Kyy_U_norm": Kyy,
        "trace_U_norm": float(np.trace(K_real)),
        "determinant_U_norm": float(np.linalg.det(K_real)),
        "electron_x": electron_x,
        "electron_y": electron_y,
        "phase_K_rad": K_phase_rad,
        "phase_linear_rad": lin_phase_rad,
        "Vacc_axis_abs_V": float(abs(V0)),
        "grad_x_abs_V_per_m": float(abs(grad[0])),
        "grad_y_abs_V_per_m": float(abs(grad[1])),
        "fit_pixels_used": fit_pixels_used,
        "fd_step_pixels": fd_step_pixels,
    }
    out.update(quadrupole_invariants(K_real))
    return out


# -----------------------------------------------------------------------------
# Method 1: Taylor/modal polynomial fit
# -----------------------------------------------------------------------------

def fit_voltage_polynomial(Vz_map: np.ndarray, params: FieldParams, *, circular_aperture: bool = True) -> dict[str, Any]:
    """Method 1 fit: Vz = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2."""
    nx, ny = Vz_map.shape
    validate_axis_inside(np.zeros((nx, ny, 2)), params)

    ixc = int(round(params.axis_i))
    iyc = int(round(params.axis_j))
    max_px = min(int(params.fit_pixels), ixc - 1, iyc - 1, nx - 2 - ixc, ny - 2 - iyc)
    if max_px < 2:
        raise ValueError("fit_pixels leaves too few points around the axis.")

    points: list[tuple[float, float]] = []
    values: list[complex] = []
    aperture_r = max_px * min(params.dx_m, params.dy_m)

    for i in range(ixc - max_px, ixc + max_px + 1):
        for j in range(iyc - max_px, iyc + max_px + 1):
            x = (float(i) - params.axis_i) * params.dx_m
            y = (float(j) - params.axis_j) * params.dy_m
            if circular_aperture and np.hypot(x, y) > aperture_r:
                continue
            points.append((x, y))
            values.append(complex(Vz_map[i, j]))

    pts = np.asarray(points, dtype=float)
    Vc = np.asarray(values, dtype=complex)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

    coeff_real, *_ = np.linalg.lstsq(A, Vc.real, rcond=None)
    coeff_imag, *_ = np.linalg.lstsq(A, Vc.imag, rcond=None)
    coeff = coeff_real + 1j * coeff_imag

    a0, ax, ay, axx, axy, ayy = coeff
    H = np.array([[2.0 * axx, axy], [axy, 2.0 * ayy]], dtype=complex)
    grad = np.array([ax, ay], dtype=complex)

    return {
        "points_xy_m": pts,
        "Vz_values_V": Vc,
        "coeff": {"a0": a0, "ax": ax, "ay": ay, "axx": axx, "axy": axy, "ayy": ayy},
        "V0": a0,
        "grad_V_per_m": grad,
        "hessian_V_per_m2": H,
        "fit_pixels_used": max_px,
    }


def method1_taylor_modal_metrics(Vz_map: np.ndarray, params: FieldParams, nz: int) -> dict[str, Any]:
    fit = fit_voltage_polynomial(Vz_map, params)
    out = metrics_from_voltage_derivatives(
        V0=fit["V0"],
        grad=fit["grad_V_per_m"],
        H=fit["hessian_V_per_m2"],
        params=params,
        method_family="Taylor/modal fit of transit-time Vz",
        fit_pixels_used=fit["fit_pixels_used"],
        fd_step_pixels=None,
    )
    out.update({
        "transverse_pixel_m": params.dx_m,
        "longitudinal_pixel_m": params.length_m / (nz - 1),
    })
    return out


# -----------------------------------------------------------------------------
# Method 2: local quadratic least-squares fit validation
# -----------------------------------------------------------------------------

def local_quadratic_least_squares_derivatives(Vz_map: np.ndarray, params: FieldParams) -> dict[str, Any]:
    """
    Calculate V0, grad(Vz) and Hessian(Vz) at the axis from a small local
    quadratic least-squares fit.

    This replaces the previous central finite-difference Method 2. It is still
    independent of Method 1 because it uses a deliberately small local stencil
    rather than the wider modal/Taylor fit aperture used by Method 1.

    Fit model about the true axis:

        Vz(x,y) = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2

    so

        grad = [ax, ay]
        H    = [[2*axx, axy], [axy, 2*ayy]]
    """
    nx, ny = Vz_map.shape
    r = int(params.local_fit_pixels)
    if r < 1:
        raise ValueError("local_fit_pixels must be >= 1.")
    if r == 1:
        # A 3x3 stencil gives 9 points for 6 coefficients. This is valid but
        # not very smoothing; r=2 or r=3 is usually better.
        pass

    validate_axis_inside(np.zeros((nx, ny, 2)), params, margin_pixels=r)

    i0 = float(params.axis_i)
    j0 = float(params.axis_j)
    ixc = int(round(i0))
    iyc = int(round(j0))

    points: list[tuple[float, float]] = []
    values: list[complex] = []

    # Use a compact circular stencil to reduce square-corner bias while still
    # remaining a local validation method.
    aperture_r_m = r * min(params.dx_m, params.dy_m)
    fractional_axis = (abs(i0 - ixc) > 1e-12) or (abs(j0 - iyc) > 1e-12)

    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            x = di * params.dx_m
            y = dj * params.dy_m
            if np.hypot(x, y) > aperture_r_m:
                continue

            # If the axis is exactly a pixel centre, use exact array values.
            # If not, sample around the fractional axis using interpolation.
            if fractional_axis:
                V = sample_complex_map(
                    Vz_map,
                    i0 + di,
                    j0 + dj,
                    order=int(params.interp_order),
                )
            else:
                V = complex(Vz_map[ixc + di, iyc + dj])

            points.append((x, y))
            values.append(V)

    pts = np.asarray(points, dtype=float)
    Vc = np.asarray(values, dtype=complex)
    if pts.shape[0] < 6:
        raise ValueError(
            f"Local quadratic fit needs at least 6 points; got {pts.shape[0]}. "
            "Increase local_fit_pixels."
        )

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

    coeff_real, *_ = np.linalg.lstsq(A, Vc.real, rcond=None)
    coeff_imag, *_ = np.linalg.lstsq(A, Vc.imag, rcond=None)
    coeff = coeff_real + 1j * coeff_imag

    a0, ax, ay, axx, axy, ayy = coeff
    H = np.array([[2.0 * axx, axy], [axy, 2.0 * ayy]], dtype=complex)
    grad = np.array([ax, ay], dtype=complex)

    return {
        "V0": a0,
        "grad_V_per_m": grad,
        "hessian_V_per_m2": H,
        "local_fit_pixels": r,
        "local_fit_points": int(pts.shape[0]),
        "interp_order": int(params.interp_order) if fractional_axis else 0,
    }


def method2_local_quadratic_ls_metrics(Vz_map: np.ndarray, params: FieldParams, nz: int) -> dict[str, Any]:
    deriv = local_quadratic_least_squares_derivatives(Vz_map, params)
    out = metrics_from_voltage_derivatives(
        V0=deriv["V0"],
        grad=deriv["grad_V_per_m"],
        H=deriv["hessian_V_per_m2"],
        params=params,
        method_family="Local quadratic least-squares fit of transit-time Vz",
        fit_pixels_used=deriv["local_fit_pixels"],
        fd_step_pixels=None,
    )
    out.update({
        "transverse_pixel_m": params.dx_m,
        "longitudinal_pixel_m": params.length_m / (nz - 1),
        "local_fit_pixels": deriv["local_fit_pixels"],
        "local_fit_points": deriv["local_fit_points"],
        "interp_order": deriv["interp_order"],
    })
    return out


# -----------------------------------------------------------------------------
# Per-field parameter construction and comparison table
# -----------------------------------------------------------------------------

def make_params_by_field(
    *,
    f_010: float,
    f_E1: float,
    f_E2: float,
    f_degen: float,
    ell: float,
    Req_m: float,
    beta: float = 1.0,
    axis_i: float = 75.0,
    axis_j: float = 75.0,
    radius_pixels: float = 75.0,
    fit_pixels: int = 8,
    fd_step_pixels: float = 2.0,
    interp_order: int = 1,
    local_fit_pixels: int = 3,
) -> dict[str, FieldParams]:
    if f_010 <= 0 or f_E1 <= 0 or f_E2 <= 0 or f_degen <= 0:
        raise ValueError("f_010, f_E1, f_E2 and f_degen must be > 0.")
    if ell <= 0:
        raise ValueError("ell must be > 0.")

    d0 = (C0 / float(f_010)) / 2.0
    d = d0 * float(ell)

    common = dict(
        Req_m=Req_m,
        beta=beta,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
        fit_pixels=fit_pixels,
        fd_step_pixels=fd_step_pixels,
        interp_order=interp_order,
        local_fit_pixels=local_fit_pixels,
    )
    return {
        "E1": FieldParams(frequency_Hz=f_E1, length_m=d0, **common),
        "E2": FieldParams(frequency_Hz=f_E2, length_m=d0, **common),
        "plus": FieldParams(frequency_Hz=f_degen, length_m=d, **common),
        "minus": FieldParams(frequency_Hz=f_degen, length_m=d, **common),
    }


def safe_ratio(numer: Any, denom: Any) -> float:
    try:
        numer_f = float(numer)
        denom_f = float(denom)
    except Exception:
        return np.nan
    return np.nan if denom_f == 0 else numer_f / denom_f


def build_comparison(fields: dict[str, np.ndarray], params_by_field: dict[str, FieldParams]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for field_name in DEFAULT_FIELD_NAMES:
        Ez = fields[field_name]
        params = with_calculated_energy(Ez, params_by_field[field_name])
        Vz_map = accelerating_voltage_map(Ez, params)
        nz = Ez.shape[2]

        for method_name, metric_func in (
            ("1_Taylor_modal_polynomial_fit", method1_taylor_modal_metrics),
            ("2_local_quadratic_least_squares", method2_local_quadratic_ls_metrics),
        ):
            metrics = metric_func(Vz_map, params, nz)
            rows.append({
                "field": field_name,
                "method": method_name,
                **metrics,
                "frequency_Hz": params.frequency_Hz,
                "U_J_calculated_from_Ez_only": params.U_J,
                "length_m": params.length_m,
                "Req_m": params.Req_m,
                "axis_i": params.axis_i,
                "axis_j": params.axis_j,
                "radius_pixels": params.radius_pixels,
                "beta": params.beta,
            })

    df = pd.DataFrame(rows)

    ratio_cols = [
        "k_parallel_V_per_pC",
        "k_perp_V_per_pC_per_m2",
        "Kxx_U_norm",
        "Kxy_U_norm",
        "Kyy_U_norm",
        "trace_U_norm",
        "K_eig_min_U_norm",
        "K_eig_max_U_norm",
        "K_quad_strength_U_norm",
        "K_frobenius_U_norm",
    ]
    ratio_rows = []
    for field_name, g in df.groupby("field", sort=False):
        r1 = g[g["method"] == "1_Taylor_modal_polynomial_fit"].iloc[0]
        r2 = g[g["method"] == "2_local_quadratic_least_squares"].iloc[0]
        row: dict[str, Any] = {"field": field_name, "method": "ratio_method1_over_method2"}
        for col in ratio_cols:
            row[col] = safe_ratio(r1.get(col, np.nan), r2.get(col, np.nan))
        ratio_rows.append(row)

    if ratio_rows:
        df = pd.concat([df, pd.DataFrame(ratio_rows)], ignore_index=True)

    return df


# -----------------------------------------------------------------------------
# Helpers and main entry point
# -----------------------------------------------------------------------------

def pillbox_radius_from_freq(f_Hz: float) -> float:
    """TM010 pillbox radius from f_010, using J0 first root 2.4048."""
    return (2.4048 * C0) / (2.0 * np.pi * float(f_Hz))


def main(
    *,
    pickle_path: str | Path = "Ez_fields_E1_E2_Epl_Emin.pkl",
    out_csv: str | Path = "Ez_method_comparison.csv",
    out_xlsx: str | Path | None = None,
    f_010: float = 1.3e9,
    f_E1: float | None = None,
    f_E2: float | None = None,
    f_degen: float | None = None,
    fhat_degen: float | None = None,
    ell: float = 1.0,
    Req_m: float | None = None,
    beta: float = 1.0,
    axis_i: float = 75.0,
    axis_j: float = 75.0,
    radius_pixels: float = 75.0,
    fit_pixels: int = 8,
    fd_step_pixels: float = 2.0,
    interp_order: int = 1,
    local_fit_pixels: int = 3,
) -> pd.DataFrame:
    """
    Calculate the comparison table.

    E1 uses frequency f_E1 and length d0=lambda_010/2.
    E2 uses frequency f_E2 and length d0=lambda_010/2.
    plus/minus use frequency f_degen and length d=d0*ell.

    f_010 is retained only to define the design length d0 and the default
    pillbox radius if Req_m is omitted.

    Axis/radius default: axis=array[75,75,:], radius=75 pixels.
    """
    if f_E1 is None:
        raise ValueError("Supply f_E1: native/parent E1 frequency at design length.")
    if f_E2 is None:
        raise ValueError("Supply f_E2: native/parent E2 frequency at design length.")
    if f_degen is None:
        if fhat_degen is None:
            raise ValueError("Supply either f_degen or fhat_degen for E+ and E-.")
        f_degen = float(fhat_degen) * float(f_010)

    if Req_m is None:
        Req_m = pillbox_radius_from_freq(f_010)

    fields = load_fields(pickle_path)
    params_by_field = make_params_by_field(
        f_010=f_010,
        f_E1=float(f_E1),
        f_E2=float(f_E2),
        f_degen=float(f_degen),
        ell=ell,
        Req_m=Req_m,
        beta=beta,
        axis_i=axis_i,
        axis_j=axis_j,
        radius_pixels=radius_pixels,
        fit_pixels=fit_pixels,
        fd_step_pixels=fd_step_pixels,
        interp_order=interp_order,
        local_fit_pixels=local_fit_pixels,
    )

    df = build_comparison(fields, params_by_field)

    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False)

    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        with pd.ExcelWriter(out_xlsx) as writer:
            df.to_excel(writer, index=False, sheet_name="comparison")

    compact_cols = [
        "field", "method", "frequency_Hz", "length_m", "U_J_calculated_from_Ez_only",
        "k_parallel_V_per_pC", "k_perp_V_per_pC_per_m2",
        "Kxx_U_norm", "Kyy_U_norm", "Kxy_U_norm", "K_quad_strength_U_norm",
        "electron_x", "electron_y",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    print(df[compact_cols].to_string(index=False))
    print(f"\nSaved: {out_csv.resolve()}")
    if out_xlsx is not None:
        print(f"Saved: {out_xlsx.resolve()}")

    return df


if __name__ == "__main__":
    f_013 = 4110960958.218893
    f_031 = 4855302811.7021
    f_112 = 3324223278.358419
    f_120 = 3792484091.65239
    f_220 = 4550191737.401735
    f_213 = 4787208055.648925


    main(
        pickle_path=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_monopoles\TM013_TM031\Ez_fields_E1_E2_Epl_Emin.pkl",
        out_csv=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_monopoles\TM013_TM031\Ez_method_comparison.csv",
        f_010=1.3e9,
        f_E1=f_013,
        f_E2=4855302811.7021,
        fhat_degen=3.8004,
        ell=0.8182,
        Req_m=None,
        beta=1.0,
        axis_i=75.0,
        axis_j=75.0,
        radius_pixels=75.0,
        fit_pixels=8,
        fd_step_pixels=2.0,
        interp_order=1,
    )

    # dipoles
    main(
        pickle_path=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles\TM112_TM120\Ez_fields_E1_E2_Epl_Emin.pkl",
        out_csv=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles\TM112_TM120\Ez_method_comparison.csv",
        f_010=1.3e9,
        f_E1=f_112,
        f_E2=f_120,
        fhat_degen=2.9173,
        ell=0.8184,
        Req_m=None,
        beta=1.0,
        axis_i=75.0,
        axis_j=75.0,
        radius_pixels=75.0,
        fit_pixels=8,
        fd_step_pixels=2.0,
        interp_order=1,
    )

    # quadrupoles
    main(
        pickle_path=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles\TM220_TM213\Ez_fields_E1_E2_Epl_Emin.pkl",
        out_csv=r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_quadrupoles\TM220_TM213\Ez_method_comparison.csv",
        f_010=1.3e9,
        f_E1=f_220,
        f_E2=f_213,
        fhat_degen=2.9173,
        ell=0.8184,
        Req_m=None,
        beta=1.0,
        axis_i=75.0,
        axis_j=75.0,
        radius_pixels=75.0,
        fit_pixels=8,
        fd_step_pixels=2.0,
        interp_order=1,
    )
