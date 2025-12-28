"""
xi_function.py - Completed Zeta Function (Xi Function)

The xi function ξ(s) is defined as:
    ξ(s) = (1/2) * s * (s-1) * π^(-s/2) * Γ(s/2) * ζ(s)

Key property: ξ(s) = ξ(1-s) (functional equation)

This module provides:
- High-precision xi computation
- Functional equation verification
- Symmetry analysis
- Zero correspondence with zeta
"""

from mpmath import mp, mpf, mpc, gamma, pi, log, exp, fabs, sqrt
from mpmath import zeta as mp_zeta, arg
from typing import Tuple, List, Dict
import json

# Set high precision
mp.dps = 50


class XiFunction:
    """Class for computing and analyzing the completed zeta function."""
    
    def __init__(self, precision: int = 50):
        """Initialize with specified precision (decimal places)."""
        self.precision = precision
        mp.dps = precision
    
    def __call__(self, s: complex) -> complex:
        """Compute ξ(s)."""
        return self.compute(s)
    
    def compute(self, s: complex) -> complex:
        """
        Compute ξ(s) = (1/2) * s * (s-1) * π^(-s/2) * Γ(s/2) * ζ(s)
        
        Args:
            s: Complex number
        
        Returns:
            ξ(s)
        """
        s_mp = mpc(s.real, s.imag) if isinstance(s, complex) else mpc(s)
        result = self._compute_mp(s_mp)
        return complex(float(result.real), float(result.imag))
    
    def _compute_mp(self, s: mpc) -> mpc:
        """Internal computation in mpmath precision."""
        half = mpf('0.5')
        
        # Special cases
        if s == 0 or s == 1:
            # ξ(0) = ξ(1) = -1/2
            return mpc(-half)
        
        # (1/2) * s * (s-1)
        prefactor = half * s * (s - 1)
        
        # π^(-s/2)
        pi_term = pi ** (-s / 2)
        
        # Γ(s/2)
        try:
            gamma_term = gamma(s / 2)
        except:
            # Handle potential poles
            return mpc(float('inf'))
        
        # ζ(s)
        zeta_term = mp_zeta(s)
        
        return prefactor * pi_term * gamma_term * zeta_term
    
    def verify_functional_equation(self, s: complex) -> Dict:
        """
        Verify ξ(s) = ξ(1-s).
        
        Args:
            s: Complex point to test
        
        Returns:
            Dictionary with verification results
        """
        s_mp = mpc(s.real, s.imag)
        one_minus_s = mpc(1) - s_mp
        
        xi_s = self._compute_mp(s_mp)
        xi_1_minus_s = self._compute_mp(one_minus_s)
        
        diff = fabs(xi_s - xi_1_minus_s)
        
        xi_s_mag = fabs(xi_s)
        xi_1_minus_s_mag = fabs(xi_1_minus_s)
        avg_mag = (xi_s_mag + xi_1_minus_s_mag) / 2
        
        if avg_mag > 1e-100:
            rel_diff = float(diff / avg_mag)
        else:
            rel_diff = float(diff)
        
        return {
            's': s,
            '1-s': complex(1 - s.real, -s.imag),
            'xi_s': complex(float(xi_s.real), float(xi_s.imag)),
            'xi_1_minus_s': complex(float(xi_1_minus_s.real), float(xi_1_minus_s.imag)),
            'absolute_diff': float(diff),
            'relative_diff': rel_diff,
            'verified': rel_diff < 1e-30
        }
    
    def magnitude_at(self, sigma: float, t: float) -> float:
        """
        Compute |ξ(σ + it)|.
        
        Args:
            sigma: Real part
            t: Imaginary part
        
        Returns:
            Magnitude
        """
        s = mpc(sigma, t)
        xi_val = self._compute_mp(s)
        return float(fabs(xi_val))
    
    def magnitude_symmetry_test(self, t: float, offsets: List[float] = None) -> Dict:
        """
        Test that |ξ(σ + it)| = |ξ((1-σ) + it)| for various σ.
        
        This follows from the functional equation.
        
        Args:
            t: Imaginary part
            offsets: Offsets from σ = 0.5 to test
        
        Returns:
            Symmetry verification results
        """
        if offsets is None:
            offsets = [0.1, 0.2, 0.3, 0.4]
        
        results = []
        
        for offset in offsets:
            sigma_right = 0.5 + offset
            sigma_left = 0.5 - offset
            
            mag_right = self.magnitude_at(sigma_right, t)
            mag_left = self.magnitude_at(sigma_left, t)
            
            if max(mag_right, mag_left) > 1e-100:
                ratio = mag_left / mag_right if mag_right > 1e-100 else float('inf')
                diff = abs(mag_right - mag_left)
                rel_diff = diff / max(mag_right, mag_left)
            else:
                ratio = 1.0
                rel_diff = 0.0
            
            results.append({
                'sigma_right': sigma_right,
                'sigma_left': sigma_left,
                'mag_right': mag_right,
                'mag_left': mag_left,
                'ratio': ratio,
                'relative_diff': rel_diff,
                'symmetric': rel_diff < 0.01
            })
        
        return {
            't': t,
            'results': results,
            'all_symmetric': all(r['symmetric'] for r in results)
        }
    
    def is_zero_at(self, s: complex, tolerance: float = 1e-10) -> Tuple[bool, float]:
        """
        Check if ξ(s) ≈ 0.
        
        Args:
            s: Point to check
            tolerance: Maximum magnitude for "zero"
        
        Returns:
            (is_zero, magnitude)
        """
        s_mp = mpc(s.real, s.imag)
        xi_val = self._compute_mp(s_mp)
        mag = float(fabs(xi_val))
        
        return mag < tolerance, mag
    
    def analyze_critical_line(self, t_values: List[float]) -> List[Dict]:
        """
        Analyze ξ along the critical line σ = 1/2.
        
        Args:
            t_values: List of imaginary parts to analyze
        
        Returns:
            Analysis results
        """
        results = []
        
        for t in t_values:
            s = mpc(mpf('0.5'), t)
            xi_val = self._compute_mp(s)
            
            results.append({
                't': t,
                'xi_real': float(xi_val.real),
                'xi_imag': float(xi_val.imag),
                'magnitude': float(fabs(xi_val)),
                'phase': float(arg(xi_val)),
                'is_small': float(fabs(xi_val)) < 0.1
            })
        
        return results


def batch_verify_functional_equation(points: List[complex], precision: int = 50) -> Dict:
    """
    Verify functional equation at multiple points.
    
    Args:
        points: List of complex points
        precision: Decimal places
    
    Returns:
        Summary of verification
    """
    xi = XiFunction(precision)
    
    results = []
    for s in points:
        result = xi.verify_functional_equation(s)
        results.append(result)
    
    num_verified = sum(1 for r in results if r['verified'])
    
    return {
        'total_points': len(points),
        'verified': num_verified,
        'all_passed': num_verified == len(points),
        'details': results
    }


if __name__ == '__main__':
    print("Xi Function Analysis")
    print("=" * 60)
    
    xi = XiFunction(precision=50)
    
    # Test functional equation
    print("\n1. Functional Equation ξ(s) = ξ(1-s):")
    test_points = [
        0.3 + 10j,
        0.7 + 15j,
        0.2 + 20j,
        0.8 + 25j,
        0.4 + 30j
    ]
    
    for s in test_points:
        result = xi.verify_functional_equation(s)
        status = "✓" if result['verified'] else "✗"
        print(f"   {status} s = {s}: rel_diff = {result['relative_diff']:.2e}")
    
    # Test magnitude symmetry
    print("\n2. Magnitude Symmetry |ξ(σ+it)| = |ξ((1-σ)+it)|:")
    for t in [15.0, 20.0, 25.0]:
        sym_result = xi.magnitude_symmetry_test(t)
        status = "✓" if sym_result['all_symmetric'] else "✗"
        print(f"   {status} t = {t}: all symmetric = {sym_result['all_symmetric']}")
        for r in sym_result['results'][:2]:  # Show first two
            print(f"      σ={r['sigma_left']:.1f}: {r['mag_left']:.4e}, "
                  f"σ={r['sigma_right']:.1f}: {r['mag_right']:.4e}, "
                  f"ratio={r['ratio']:.4f}")
    
    # Test at known zero
    print("\n3. Xi at Known Zeta Zeros:")
    known_zeros_t = [14.134725, 21.022040, 25.010858]
    for t in known_zeros_t:
        s = complex(0.5, t)
        is_zero, mag = xi.is_zero_at(s)
        status = "✓" if is_zero else "✗"
        print(f"   {status} t = {t:.6f}: |ξ| = {mag:.4e}")
    
    # Analyze along critical line
    print("\n4. Critical Line Analysis:")
    t_range = [10 + i*2 for i in range(10)]
    analysis = xi.analyze_critical_line(t_range)
    for a in analysis[:5]:
        print(f"   t = {a['t']:.1f}: |ξ| = {a['magnitude']:.4e}, "
              f"phase = {a['phase']:.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete.")

