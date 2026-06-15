import numpy as np
from scipy.constants import c, e, m_e, mu_0
from scipy.special import jv, jvp, jnp_zeros
from scipy.integrate import solve_ivp


def te_pillbox_params(R, L, m, n, p):
    """
    Pure pillbox TE_mnp mode.

    Boundary condition:
        J_m'(k_c R) = 0

    p = longitudinal index, usually >= 1 for closed pillbox TE modes.
    """
    xmn = jnp_zeros(m, n)[-1]      # nth zero of J_m'
    kc = xmn / R
    kz = p * np.pi / L
    omega = c * np.sqrt(kc**2 + kz**2)
    freq = omega / (2 * np.pi)
    return kc, kz, omega, freq


def cyl_from_xyz(x, y):
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi


def cyl_vec_to_cart(Fr, Fphi, phi):
    Fx = Fr * np.cos(phi) - Fphi * np.sin(phi)
    Fy = Fr * np.sin(phi) + Fphi * np.cos(phi)
    return Fx, Fy


def te_pillbox_fields_xyz(
    x, y, z, t,
    R, L, m, n, p,
    H0=1.0,
    phase=0.0,
    azimuth="cos",
):
    """
    Time-domain fields for a pure TE_mnp pillbox mode.

    Returns:
        E = [Ex, Ey, Ez]
        B = [Bx, By, Bz]

    Uses phasors with exp(i omega t), then takes real part.
    """
    kc, kz, omega, freq = te_pillbox_params(R, L, m, n, p)

    r, phi = cyl_from_xyz(x, y)
    kr = kc * r

    if azimuth == "cos":
        Cphi = np.cos(m * phi)
        dC_dphi = -m * np.sin(m * phi)
    elif azimuth == "sin":
        Cphi = np.sin(m * phi)
        dC_dphi = m * np.cos(m * phi)
    else:
        raise ValueError("azimuth must be 'cos' or 'sin'.")

    Cz = np.cos(kz * z)

    J = jv(m, kr)
    dJ_dr = kc * jvp(m, kr, 1)

    Hz = H0 * J * Cphi * Cz
    dHz_dr = H0 * dJ_dr * Cphi * Cz

    # Avoid singular r=0 division.
    if r < 1e-14:
        inv_r_dHz_dphi = 0.0
    else:
        inv_r_dHz_dphi = H0 * J * dC_dphi * Cz / r

    # TE phasor fields from H_z.
    Er_ph = 1j * omega * mu_0 / kc**2 * inv_r_dHz_dphi
    Ephi_ph = -1j * omega * mu_0 / kc**2 * dHz_dr
    Ez_ph = 0.0

    Hr_ph = -1j * kz / kc**2 * dHz_dr
    Hphi_ph = -1j * kz / kc**2 * inv_r_dHz_dphi
    Hz_ph = Hz

    Ex_ph, Ey_ph = cyl_vec_to_cart(Er_ph, Ephi_ph, phi)
    Hx_ph, Hy_ph = cyl_vec_to_cart(Hr_ph, Hphi_ph, phi)

    time_factor = np.exp(1j * (omega * t + phase))

    E = np.real(np.array([Ex_ph, Ey_ph, Ez_ph]) * time_factor)
    H = np.real(np.array([Hx_ph, Hy_ph, Hz_ph]) * time_factor)
    B = mu_0 * H

    return E, B


def track_te_lorentz(
    x0, y0,
    R, L, m, n, p,
    gamma,
    H0=1.0,
    phase=0.0,
    azimuth="cos",
    q=-e,
    nsteps=2000,
):
    """
    Direct Lorentz-force tracking through the TE pillbox mode.

    Initial particle:
        x=x0, y=y0, z=0
        px=py=0
        pz set by gamma

    Returns final state and transverse momentum kick.
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p0 = gamma * m_e * beta * c
    pz0 = p0

    y_init = np.array([x0, y0, 0.0, 0.0, 0.0, pz0])

    def rhs(t, Y):
        x, y, z, px, py, pz = Y
        pvec = np.array([px, py, pz])
        gamma_now = np.sqrt(1.0 + np.dot(pvec, pvec) / (m_e * c)**2)
        v = pvec / (gamma_now * m_e)

        Efield, Bfield = te_pillbox_fields_xyz(
            x, y, z, t,
            R, L, m, n, p,
            H0=H0,
            phase=phase,
            azimuth=azimuth,
        )

        F = q * (Efield + np.cross(v, Bfield))

        return np.array([v[0], v[1], v[2], F[0], F[1], F[2]])

    t_end = L / (beta * c)

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y_init,
        rtol=1e-8,
        atol=1e-11,
        max_step=t_end / nsteps,
    )

    final = sol.y[:, -1]
    dpx = final[3] - y_init[3]
    dpy = final[4] - y_init[4]

    return {
        "final_state": final,
        "delta_p": np.array([dpx, dpy]),
        "delta_xp": np.array([dpx / pz0, dpy / pz0]),
        "solution": sol,
    }


def te_vector_potential_perp_xyz(
    x, y, z, t,
    R, L, m, n, p,
    H0=1.0,
    phase=0.0,
    azimuth="cos",
):
    """
    For TE mode with scalar potential zero:
        E_perp = -dA_perp/dt

    With exp(i omega t) convention:
        A_phasor = i E_phasor / omega

    This reconstructs A_perp by reusing the field function approximately.
    """
    kc, kz, omega, freq = te_pillbox_params(R, L, m, n, p)

    # Get E at phase and at phase + pi/2 to reconstruct phasor-like A.
    E, _ = te_pillbox_fields_xyz(
        x, y, z, t,
        R, L, m, n, p,
        H0=H0,
        phase=phase,
        azimuth=azimuth,
    )

    E90, _ = te_pillbox_fields_xyz(
        x, y, z, t,
        R, L, m, n, p,
        H0=H0,
        phase=phase + np.pi / 2,
        azimuth=azimuth,
    )

    # Since A is 90 degrees from E and divided by omega.
    A = -E90 / omega
    return A[:2]


def ponderomotive_potential(
    x, y,
    R, L, m, n, p,
    gamma,
    H0=1.0,
    phase=0.0,
    azimuth="cos",
    q=-e,
    nz=2000,
):
    """
    Computes:
        U_p(x,y) = - q^2 / pz * integral A_perp^2 dz

    The transverse kick estimate is:
        delta_p_perp = grad_perp U_p
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    pz = gamma * m_e * beta * c

    zs = np.linspace(0.0, L, nz)
    vals = np.empty_like(zs)

    for i, z in enumerate(zs):
        t = z / (beta * c)
        Axy = te_vector_potential_perp_xyz(
            x, y, z, t,
            R, L, m, n, p,
            H0=H0,
            phase=phase,
            azimuth=azimuth,
        )
        vals[i] = np.dot(Axy, Axy)

    integral = np.trapz(vals, zs)
    return -(q**2 / pz) * integral


def ponderomotive_kick(
    x, y,
    R, L, m, n, p,
    gamma,
    H0=1.0,
    phase=0.0,
    azimuth="cos",
    q=-e,
    h=1e-6,
    nz=2000,
):
    """
    Finite-difference gradient of the TE ponderomotive potential.

    Returns:
        delta_p = [dpx, dpy]
        delta_xp = [dpx/pz, dpy/pz]
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    pz = gamma * m_e * beta * c

    U_xp = ponderomotive_potential(
        x + h, y, R, L, m, n, p, gamma,
        H0=H0, phase=phase, azimuth=azimuth, q=q, nz=nz,
    )
    U_xm = ponderomotive_potential(
        x - h, y, R, L, m, n, p, gamma,
        H0=H0, phase=phase, azimuth=azimuth, q=q, nz=nz,
    )
    U_yp = ponderomotive_potential(
        x, y + h, R, L, m, n, p, gamma,
        H0=H0, phase=phase, azimuth=azimuth, q=q, nz=nz,
    )
    U_ym = ponderomotive_potential(
        x, y - h, R, L, m, n, p, gamma,
        H0=H0, phase=phase, azimuth=azimuth, q=q, nz=nz,
    )

    dpx = (U_xp - U_xm) / (2 * h)
    dpy = (U_yp - U_ym) / (2 * h)

    return {
        "delta_p": np.array([dpx, dpy]),
        "delta_xp": np.array([dpx / pz, dpy / pz]),
    }

def gamma_from_kinetic_energy(
    kinetic_energy,
    unit="GeV",
    rest_energy_MeV=0.510998950,
):
    """
    Convert kinetic energy to relativistic gamma.

    Parameters
    ----------
    kinetic_energy : float
        Kinetic energy.

    unit : str
        'eV', 'keV', 'MeV', 'GeV', or 'TeV'

    Returns
    -------
    gamma : float
    """
    factors = {
        "eV": 1e-6,
        "keV": 1e-3,
        "MeV": 1.0,
        "GeV": 1e3,
        "TeV": 1e6,
    }

    K_MeV = kinetic_energy * factors[unit]

    return 1.0 + K_MeV / rest_energy_MeV

if __name__ == "__main__":
    gamma_tony = gamma_from_kinetic_energy(8.0)
    print(f"{gamma_tony = }")
    R = 0.04  # m
    L = 0.10  # m
    m, n, p = 1, 1, 1
    gamma = gamma_tony
    H0 = 1e4  # A/m

    x0 = 1e-3
    y0 = 0.0

    kc, kz, omega, freq = te_pillbox_params(R, L, m, n, p)
    print(freq / 1e9, "GHz")

    direct = track_te_lorentz(x0, y0, R, L, m, n, p, gamma, H0=H0)
    pond = ponderomotive_kick(x0, y0, R, L, m, n, p, gamma, H0=H0)

    print("Direct Lorentz delta p:", direct["delta_p"])
    print("Direct Lorentz delta x':", direct["delta_xp"])

    print("Ponderomotive delta p:", pond["delta_p"])
    print("Ponderomotive delta x':", pond["delta_xp"])