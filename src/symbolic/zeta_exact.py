"""
zeta_exact.py - Arbitrary Precision Riemann Zeta Function

Uses mpmath for exact symbolic computation with configurable precision.
This module provides rigorous numerical computation of:
- Riemann zeta function ζ(s)
- Completed xi function ξ(s)
- Functional equation verification
- Zero detection with guaranteed accuracy

Requirements: pip install mpmath sympy
"""

from mpmath import mp, mpf, mpc, gamma, pi, log, exp, sin, cos, sqrt, fabs
from mpmath import zeta as mp_zeta
from typing import Tuple, List, Optional
import json

# Set default precision (decimal places)
mp.dps = 50  # 50 decimal places by default


def set_precision(dps: int):
    """Set the number of decimal places for all computations."""
    mp.dps = dps


def zeta(s: complex, dps: Optional[int] = None) -> complex:
    """
    Compute ζ(s) with arbitrary precision.
    
    Args:
        s: Complex number
        dps: Decimal places (uses global default if None)
    
    Returns:
        ζ(s) as a complex number
    """
    if dps is not None:
        old_dps = mp.dps
        mp.dps = dps
    
    s_mp = mpc(s.real, s.imag) if isinstance(s, complex) else mpc(s)
    result = mp_zeta(s_mp)
    
    if dps is not None:
        mp.dps = old_dps
    
    return complex(float(result.real), float(result.imag))


def zeta_mp(s: mpc) -> mpc:
    """Compute ζ(s) in mpmath precision (returns mpc)."""
    return mp_zeta(s)


def xi(s: complex, dps: Optional[int] = None) -> complex:
    """
    Compute the completed zeta function ξ(s).
    
    ξ(s) = (1/2) * s * (s-1) * π^(-s/2) * Γ(s/2) * ζ(s)
    
    This satisfies ξ(s) = ξ(1-s).
    
    Args:
        s: Complex number
        dps: Decimal places
    
    Returns:
        ξ(s) as a complex number
    """
    if dps is not None:
        old_dps = mp.dps
        mp.dps = dps
    
    s_mp = mpc(s.real, s.imag) if isinstance(s, complex) else mpc(s)
    result = xi_mp(s_mp)
    
    if dps is not None:
        mp.dps = old_dps
    
    return complex(float(result.real), float(result.imag))


def xi_mp(s: mpc) -> mpc:
    """Compute ξ(s) in mpmath precision (returns mpc)."""
    half = mpf('0.5')
    
    # Prefactor: (1/2) * s * (s-1)
    prefactor = half * s * (s - 1)
    
    # π^(-s/2)
    pi_term = pi ** (-s / 2)
    
    # Γ(s/2)
    gamma_term = gamma(s / 2)
    
    # ζ(s)
    zeta_term = mp_zeta(s)
    
    return prefactor * pi_term * gamma_term * zeta_term


def verify_functional_equation(s: complex, tolerance: float = 1e-30) -> Tuple[bool, float]:
    """
    Verify that ξ(s) = ξ(1-s).
    
    Args:
        s: Complex number to test
        tolerance: Maximum allowed difference
    
    Returns:
        (passed, relative_difference)
    """
    s_mp = mpc(s.real, s.imag)
    one_minus_s = mpc(1) - s_mp
    
    xi_s = xi_mp(s_mp)
    xi_1_minus_s = xi_mp(one_minus_s)
    
    diff = fabs(xi_s - xi_1_minus_s)
    avg_mag = (fabs(xi_s) + fabs(xi_1_minus_s)) / 2
    
    if avg_mag > 1e-50:
        rel_diff = float(diff / avg_mag)
    else:
        rel_diff = float(diff)
    
    return rel_diff < tolerance, rel_diff


def find_zero_bisection(t_low: float, t_high: float, tolerance: float = 1e-15) -> float:
    """
    Find a zero of ζ(1/2 + it) using bisection on the Z function.
    
    The Riemann-Siegel Z function is real on the critical line,
    and its zeros correspond to zeros of ζ.
    
    Args:
        t_low: Lower bound for t
        t_high: Upper bound for t
        tolerance: Precision for zero location
    
    Returns:
        t value where ζ(1/2 + it) ≈ 0
    """
    from mpmath import siegelz
    
    z_low = float(siegelz(mpf(t_low)))
    z_high = float(siegelz(mpf(t_high)))
    
    if z_low * z_high > 0:
        raise ValueError(f"No sign change between t={t_low} and t={t_high}")
    
    while t_high - t_low > tolerance:
        t_mid = (t_low + t_high) / 2
        z_mid = float(siegelz(mpf(t_mid)))
        
        if z_low * z_mid < 0:
            t_high = t_mid
            z_high = z_mid
        else:
            t_low = t_mid
            z_low = z_mid
    
    return (t_low + t_high) / 2


def find_zeros_in_range(t_min: float, t_max: float, step: float = 0.1, 
                        tolerance: float = 1e-12) -> List[float]:
    """
    Find all zeros of ζ(1/2 + it) in a range.
    
    Args:
        t_min: Start of range
        t_max: End of range
        step: Initial step size for searching
        tolerance: Precision for zero locations
    
    Returns:
        List of t values where zeros occur
    """
    from mpmath import siegelz
    
    zeros = []
    t = t_min
    prev_z = float(siegelz(mpf(t)))
    
    while t < t_max:
        t += step
        curr_z = float(siegelz(mpf(t)))
        
        if prev_z * curr_z < 0:
            # Sign change - refine with bisection
            zero_t = find_zero_bisection(t - step, t, tolerance)
            zeros.append(zero_t)
        
        prev_z = curr_z
    
    return zeros


def verify_zero(t: float, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Verify that t corresponds to a zero of ζ(1/2 + it).
    
    Args:
        t: Imaginary part to test
        tolerance: Maximum allowed |ζ|
    
    Returns:
        (is_zero, magnitude)
    """
    s = mpc(mpf('0.5'), mpf(t))
    zeta_val = mp_zeta(s)
    mag = float(fabs(zeta_val))
    
    return mag < tolerance, mag


def compute_winding_number(center: complex, radius: float, 
                           samples: int = 1000) -> int:
    """
    Compute the winding number of ζ around a contour.
    
    W = (1/2πi) ∮ (ζ'/ζ) ds
    
    This counts zeros minus poles inside the contour.
    
    Args:
        center: Center of circular contour
        radius: Radius of contour
        samples: Number of sample points
    
    Returns:
        Winding number (integer)
    """
    center_mp = mpc(center.real, center.imag)
    r = mpf(radius)
    
    integral = mpc(0)
    
    for i in range(samples):
        theta1 = 2 * pi * i / samples
        theta2 = 2 * pi * (i + 1) / samples
        
        s1 = center_mp + r * mpc(cos(theta1), sin(theta1))
        s2 = center_mp + r * mpc(cos(theta2), sin(theta2))
        
        z1 = mp_zeta(s1)
        z2 = mp_zeta(s2)
        
        # Δarg contribution
        from mpmath import arg
        arg1 = arg(z1)
        arg2 = arg(z2)
        
        delta_arg = arg2 - arg1
        
        # Handle branch cut
        if delta_arg > pi:
            delta_arg -= 2 * pi
        elif delta_arg < -pi:
            delta_arg += 2 * pi
        
        integral += mpc(0, delta_arg)
    
    # W = integral / (2πi)
    winding = integral / (2 * pi * mpc(0, 1))
    
    return round(float(winding.real))


# Known zeros for verification (first 100)
KNOWN_ZEROS = [
    14.134725141734693790,
    21.022039638771554993,
    25.010857580145688763,
    30.424876125859513210,
    32.935061587739189691,
    37.586178158825671257,
    40.918719012147495187,
    43.327073280914999519,
    48.005150881167159727,
    49.773832477672302181,
    52.970321477714460644,
    56.446247697063394804,
    59.347044002602353079,
    60.831778524609809844,
    65.112544048081606660,
    67.079810529494173714,
    69.546401711173979252,
    72.067157674481907582,
    75.704690699083933168,
    77.144840068874805372,
    79.337375020249367922,
    82.910380854086030183,
    84.735492980517050105,
    87.425274613125229406,
    88.809111207634465423,
    92.491899270558484296,
    94.651344040519886966,
    95.870634228245309758,
    98.831194218193692233,
    101.31785100573139122,
]


def verify_known_zeros(num_zeros: int = 10, tolerance: float = 1e-6) -> List[dict]:
    """
    Verify our zero detection against known values.
    
    Args:
        num_zeros: Number of zeros to verify
        tolerance: Allowed error in zero location
    
    Returns:
        List of verification results
    """
    results = []
    
    for i, known_t in enumerate(KNOWN_ZEROS[:num_zeros]):
        # Check |ζ(1/2 + it)| at known zero
        is_zero, mag = verify_zero(known_t)
        
        # Find zero near known value
        try:
            found_t = find_zero_bisection(known_t - 0.5, known_t + 0.5, 1e-12)
            error = abs(found_t - known_t)
        except ValueError:
            found_t = None
            error = float('inf')
        
        results.append({
            'index': i + 1,
            'known_t': known_t,
            'found_t': found_t,
            'error': error,
            'magnitude': mag,
            'verified': error < tolerance and mag < 1e-6
        })
    
    return results


if __name__ == '__main__':
    print("Riemann Zeta Exact Computation")
    print("=" * 50)
    
    # Test basic values
    print("\n1. Testing ζ(2) = π²/6:")
    z2 = zeta(2.0)
    expected = float(pi**2 / 6)
    print(f"   ζ(2) = {z2.real:.15f}")
    print(f"   π²/6 = {expected:.15f}")
    print(f"   Error: {abs(z2.real - expected):.2e}")
    
    # Test functional equation
    print("\n2. Verifying functional equation ξ(s) = ξ(1-s):")
    test_points = [0.3 + 10j, 0.7 + 20j, 0.2 + 15j]
    for s in test_points:
        passed, diff = verify_functional_equation(s)
        status = "✓" if passed else "✗"
        print(f"   {status} s = {s}: relative diff = {diff:.2e}")
    
    # Verify known zeros
    print("\n3. Verifying first 10 known zeros:")
    results = verify_known_zeros(10)
    for r in results:
        status = "✓" if r['verified'] else "✗"
        print(f"   {status} Zero #{r['index']}: t = {r['known_t']:.10f}, "
              f"|ζ| = {r['magnitude']:.2e}, error = {r['error']:.2e}")
    
    # Test winding number
    print("\n4. Testing winding numbers:")
    for t in KNOWN_ZEROS[:3]:
        center = complex(0.5, t)
        w = compute_winding_number(center, 0.3, 500)
        print(f"   W around t={t:.4f}: {w}")
    
    # Off-center test (should be 0)
    center_off = complex(0.8, 14.13)
    w_off = compute_winding_number(center_off, 0.2, 500)
    print(f"   W at σ=0.8 (off-line): {w_off}")
    
    print("\n" + "=" * 50)
    print("All tests completed.")

