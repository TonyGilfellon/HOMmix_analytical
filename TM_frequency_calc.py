import numpy as np
from scipy.special import jn_zeros

C0 = 299792458.0

def tm_root_v_mn(m: int, n: int) -> float:
    """nth positive zero of J_m(x); n is 1-based."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    return float(jn_zeros(int(m), int(n))[-1])

def f_tm(m: int, n: int, p: int, R: float, L: float, c: float = C0) -> float:
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    v = tm_root_v_mn(m, n)
    return float(c / (2.0 * np.pi) * np.sqrt((v / R) ** 2 + (p * np.pi / L) ** 2))

def pillbox_radius_from_freq(f_Hz: float, c: float = C0) -> float:
    """Radius for TM010 at f_Hz."""
    return tm_root_v_mn(0, 1) * c / (2.0 * np.pi * float(f_Hz))

if __name__ == "__main__":
    f_010 =1.3e9
    lambda_010 = C0/f_010
    d_0 = lambda_010/2.
    R = pillbox_radius_from_freq(f_010)


    m=0
    n=1
    p=3
    f_013 = f_tm(m, n, p, R, d_0)
    print(f"{f_013 = }")

    m = 0
    n = 3
    p = 1
    f_031 = f_tm(m, n, p, R, d_0)
    print(f"{f_031 = }")

    m = 1
    n = 1
    p = 2
    f_112 = f_tm(m, n, p, R, d_0)
    print(f"{f_112 = }")

    m = 1
    n = 2
    p = 0
    f_120 = f_tm(m, n, p, R, d_0)
    print(f"{f_120 = }")

    m = 2
    n = 2
    p = 0
    f_220 = f_tm(m, n, p, R, d_0)
    print(f"{f_220 = }")

    m = 2
    n = 1
    p = 3
    f_213 = f_tm(m, n, p, R, d_0)
    print(f"{f_213 = }")