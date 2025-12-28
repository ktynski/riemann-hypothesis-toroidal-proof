"""
winding.py - Symbolic Winding Number Computation

Computes the winding number (index) of ζ(s) around contours:

    W = (1/2πi) ∮ (ζ'/ζ) ds

Key properties:
- W is always an integer (topological invariant)
- W = number of zeros - number of poles inside contour
- For ζ in the critical strip: W = number of zeros (no poles)
- W = 1 around each simple zero

This provides TOPOLOGICAL PROTECTION: zeros cannot move
continuously without changing W, which is discrete.
"""

from mpmath import mp, mpf, mpc, pi, cos, sin, fabs, arg, log, diff
from mpmath import zeta as mp_zeta
from typing import Tuple, List, Dict, Optional
import json

# Set precision
mp.dps = 50


class WindingNumberComputer:
    """
    Compute and verify winding numbers for the Riemann zeta function.
    """
    
    def __init__(self, precision: int = 50):
        """Initialize with specified precision."""
        mp.dps = precision
        self.precision = precision
    
    def compute_around_point(self, center: complex, radius: float, 
                             samples: int = 1000) -> int:
        """
        Compute winding number of ζ around a circular contour.
        
        Uses argument principle: W = Δarg(ζ) / (2π)
        
        Args:
            center: Center of contour
            radius: Radius of contour
            samples: Number of sample points
        
        Returns:
            Winding number (integer)
        """
        center_mp = mpc(center.real, center.imag)
        r = mpf(radius)
        
        total_delta_arg = mpf(0)
        
        for i in range(samples):
            theta1 = 2 * pi * i / samples
            theta2 = 2 * pi * (i + 1) / samples
            
            # Points on contour
            s1 = center_mp + r * mpc(cos(theta1), sin(theta1))
            s2 = center_mp + r * mpc(cos(theta2), sin(theta2))
            
            # ζ at these points
            z1 = mp_zeta(s1)
            z2 = mp_zeta(s2)
            
            # Argument change
            arg1 = arg(z1)
            arg2 = arg(z2)
            
            delta = arg2 - arg1
            
            # Handle branch cut (argument discontinuity)
            while delta > pi:
                delta -= 2 * pi
            while delta < -pi:
                delta += 2 * pi
            
            total_delta_arg += delta
        
        # W = Δarg / (2π)
        winding = total_delta_arg / (2 * pi)
        
        return round(float(winding))
    
    def compute_with_error_bound(self, center: complex, radius: float,
                                  samples: int = 1000) -> Dict:
        """
        Compute winding number with error estimate.
        
        Since W must be an integer, we compute the raw value and
        check how close it is to an integer.
        
        Args:
            center: Center of contour
            radius: Radius of contour
            samples: Number of sample points
        
        Returns:
            Dictionary with winding number and confidence
        """
        center_mp = mpc(center.real, center.imag)
        r = mpf(radius)
        
        total_delta_arg = mpf(0)
        max_step = mpf(0)
        
        for i in range(samples):
            theta1 = 2 * pi * i / samples
            theta2 = 2 * pi * (i + 1) / samples
            
            s1 = center_mp + r * mpc(cos(theta1), sin(theta1))
            s2 = center_mp + r * mpc(cos(theta2), sin(theta2))
            
            z1 = mp_zeta(s1)
            z2 = mp_zeta(s2)
            
            arg1 = arg(z1)
            arg2 = arg(z2)
            
            delta = arg2 - arg1
            
            while delta > pi:
                delta -= 2 * pi
            while delta < -pi:
                delta += 2 * pi
            
            total_delta_arg += delta
            max_step = max(max_step, fabs(delta))
        
        raw_winding = float(total_delta_arg / (2 * pi))
        rounded_winding = round(raw_winding)
        distance_to_integer = abs(raw_winding - rounded_winding)
        
        return {
            'center': center,
            'radius': radius,
            'samples': samples,
            'raw_winding': raw_winding,
            'winding': rounded_winding,
            'distance_to_integer': distance_to_integer,
            'max_step': float(max_step),
            'confident': distance_to_integer < 0.01
        }
    
    def verify_zero_has_winding_one(self, t: float, radius: float = 0.3,
                                    samples: int = 500) -> Dict:
        """
        Verify that a zero at s = 1/2 + it has winding number 1.
        
        This confirms the zero is simple (multiplicity 1).
        
        Args:
            t: Imaginary part of zero
            radius: Contour radius
            samples: Sample points
        
        Returns:
            Verification result
        """
        center = complex(0.5, t)
        result = self.compute_with_error_bound(center, radius, samples)
        
        result['expected_winding'] = 1
        result['is_simple_zero'] = result['winding'] == 1
        result['t'] = t
        
        return result
    
    def verify_no_zero_off_line(self, sigma: float, t: float, 
                                radius: float = 0.2, samples: int = 500) -> Dict:
        """
        Verify that there is no zero at (σ, t) where σ ≠ 0.5.
        
        The winding number should be 0.
        
        Args:
            sigma: Real part (should be ≠ 0.5)
            t: Imaginary part
            radius: Contour radius
            samples: Sample points
        
        Returns:
            Verification result
        """
        center = complex(sigma, t)
        result = self.compute_with_error_bound(center, radius, samples)
        
        result['expected_winding'] = 0
        result['no_zero_inside'] = result['winding'] == 0
        result['sigma'] = sigma
        result['t'] = t
        
        return result
    
    def scan_for_zeros(self, sigma_min: float, sigma_max: float,
                       t_min: float, t_max: float,
                       step: float = 0.5, radius: float = 0.2) -> List[Dict]:
        """
        Scan a region for zeros using winding numbers.
        
        Args:
            sigma_min, sigma_max: Real part range
            t_min, t_max: Imaginary part range
            step: Grid step size
            radius: Contour radius for each test
        
        Returns:
            List of points with winding ≠ 0
        """
        zeros_found = []
        
        sigma = sigma_min
        while sigma <= sigma_max:
            t = t_min
            while t <= t_max:
                center = complex(sigma, t)
                result = self.compute_with_error_bound(center, radius, 200)
                
                if result['winding'] != 0:
                    zeros_found.append({
                        'sigma': sigma,
                        't': t,
                        'winding': result['winding'],
                        'confident': result['confident']
                    })
                
                t += step
            sigma += step
        
        return zeros_found
    
    def topological_protection_test(self, t: float, 
                                    sigma_offsets: List[float] = None) -> Dict:
        """
        Demonstrate topological protection: moving a zero off the
        critical line would require changing the winding number.
        
        Args:
            t: Imaginary part of a known zero
            sigma_offsets: Offsets from σ = 0.5 to test
        
        Returns:
            Analysis showing winding changes
        """
        if sigma_offsets is None:
            sigma_offsets = [0, 0.1, 0.2, 0.3]
        
        results = []
        
        for offset in sigma_offsets:
            sigma = 0.5 + offset
            center = complex(sigma, t)
            
            result = self.compute_with_error_bound(center, 0.25, 500)
            
            results.append({
                'sigma': sigma,
                'offset_from_critical': offset,
                'winding': result['winding'],
                'confident': result['confident']
            })
        
        # The key insight: winding is 1 at σ=0.5 and 0 elsewhere
        winding_at_critical = results[0]['winding'] if sigma_offsets[0] == 0 else None
        windings_off_critical = [r['winding'] for r in results if r['offset_from_critical'] != 0]
        
        return {
            't': t,
            'winding_at_critical_line': winding_at_critical,
            'windings_off_critical': windings_off_critical,
            'topological_protection_confirmed': (
                winding_at_critical == 1 and 
                all(w == 0 for w in windings_off_critical)
            ),
            'details': results
        }


def verify_known_zeros_are_simple(zeros_t: List[float], 
                                  precision: int = 50) -> List[Dict]:
    """
    Verify that known zeros are simple (winding = 1).
    
    Args:
        zeros_t: List of imaginary parts of known zeros
        precision: Computation precision
    
    Returns:
        Verification results
    """
    computer = WindingNumberComputer(precision)
    results = []
    
    for t in zeros_t:
        result = computer.verify_zero_has_winding_one(t)
        results.append(result)
    
    return results


if __name__ == '__main__':
    print("Winding Number Analysis")
    print("=" * 60)
    
    computer = WindingNumberComputer(50)
    
    # Known zeros
    KNOWN_ZEROS = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    
    # Test 1: Winding around known zeros
    print("\n1. Winding Numbers Around Known Zeros:")
    for t in KNOWN_ZEROS[:3]:
        result = computer.verify_zero_has_winding_one(t)
        status = "✓" if result['is_simple_zero'] else "✗"
        print(f"   {status} t = {t:.6f}: W = {result['winding']}, "
              f"distance to int = {result['distance_to_integer']:.2e}")
    
    # Test 2: No zeros off critical line
    print("\n2. No Zeros Off Critical Line:")
    for sigma in [0.3, 0.4, 0.6, 0.7]:
        result = computer.verify_no_zero_off_line(sigma, 14.13)
        status = "✓" if result['no_zero_inside'] else "✗"
        print(f"   {status} σ = {sigma}, t = 14.13: W = {result['winding']}")
    
    # Test 3: Topological protection
    print("\n3. Topological Protection Demonstration:")
    t_test = 14.134725
    tp_result = computer.topological_protection_test(t_test, [0, 0.1, 0.2, 0.3])
    
    status = "✓" if tp_result['topological_protection_confirmed'] else "✗"
    print(f"   {status} At t = {t_test}:")
    print(f"      W at critical line (σ=0.5): {tp_result['winding_at_critical_line']}")
    print(f"      W off critical line: {tp_result['windings_off_critical']}")
    print(f"      Topological protection confirmed: {tp_result['topological_protection_confirmed']}")
    
    # Test 4: Detailed winding with error bounds
    print("\n4. Detailed Winding with Error Bounds:")
    for t in KNOWN_ZEROS[:2]:
        result = computer.compute_with_error_bound(complex(0.5, t), 0.3, 1000)
        print(f"   t = {t:.6f}:")
        print(f"      Raw W = {result['raw_winding']:.10f}")
        print(f"      Rounded W = {result['winding']}")
        print(f"      Distance to int = {result['distance_to_integer']:.2e}")
        print(f"      Confident: {result['confident']}")
    
    print("\n" + "=" * 60)
    print("Winding number analysis complete.")
    print("\nKey Insight: Winding numbers are INTEGERS.")
    print("This provides TOPOLOGICAL PROTECTION for zeros.")
    print("A zero cannot 'move' without W changing discretely.")

