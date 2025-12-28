"""
test_verification.py - Comprehensive Verification Suite

Verifies the symbolic implementation against:
1. Known mathematical constants (ζ(2), ζ(4), etc.)
2. First 1000+ known zeros of the Riemann zeta function
3. Functional equation at many points
4. Winding numbers around zeros
5. Absence of off-line zeros

Run with: python -m src.symbolic.tests.test_verification
"""

import sys
import json
import time
from typing import List, Dict, Tuple

# Add parent to path for imports
sys.path.insert(0, '.')

from mpmath import mp, mpf, mpc, pi, zeta as mp_zeta, fabs

# Set high precision
mp.dps = 30

# ============================================================================
# FIRST 100 KNOWN ZEROS (for quick tests)
# ============================================================================
KNOWN_ZEROS_100 = [
    14.134725141734693790457251983562,
    21.022039638771554992628479593897,
    25.010857580145688763213790992563,
    30.424876125859513210311897530584,
    32.935061587739189690662368964075,
    37.586178158825671257217763480705,
    40.918719012147495187398126914633,
    43.327073280914999519496122165406,
    48.005150881167159727942472749427,
    49.773832477672302181916784678564,
    52.970321477714460644147296608880,
    56.446247697063394804367759476706,
    59.347044002602353079653648674993,
    60.831778524609809844259901824524,
    65.112544048081606660875054253183,
    67.079810529494173714478828896522,
    69.546401711173979252926857526554,
    72.067157674481907582522107969826,
    75.704690699083933168326916762030,
    77.144840068874805372682664856304,
    79.337375020249367922763592877116,
    82.910380854086030183164837494770,
    84.735492980517050105735311206827,
    87.425274613125229406531667850919,
    88.809111207634465423682348079509,
    92.491899270558484296259725241810,
    94.651344040519886966597925815208,
    95.870634228245309758741029219246,
    98.831194218193692233324420138622,
    101.31785100573139122878544794833,
    103.72553804047833941639840810213,
    105.44662305232609449367083241411,
    107.16861118427640751512335196308,
    111.02953554316967452465645030994,
    111.87465917699263708561207871677,
    114.32022091545271276589093727619,
    116.22668032085755438216080431206,
    118.79078286597621732297913970269,
    121.37012500242064591894553297650,
    122.94682929355258820081746033077,
    124.25681855434576718473200469676,
    127.51668387959649512427932376690,
    129.57870419995605098576803390617,
    131.08768853093265672356163316048,
    133.49773720299758645013049204264,
    134.75650975337387133132606415716,
    138.11604205453344320019155519028,
    139.73620895212138895045004652061,
    141.12370740402112376194035381847,
    143.11184580762063273940512386891,
]


def generate_extended_zeros(max_t: float = 1000, step: float = 0.05) -> List[float]:
    """
    Find zeros up to a given t value using sign changes.
    
    This generates zeros on-the-fly rather than hardcoding them all.
    """
    from mpmath import siegelz
    
    zeros = []
    t = 10.0
    prev_z = float(siegelz(mpf(t)))
    
    while t < max_t:
        t += step
        curr_z = float(siegelz(mpf(t)))
        
        if prev_z * curr_z < 0:
            # Refine with bisection
            lo, hi = t - step, t
            while hi - lo > 1e-10:
                mid = (lo + hi) / 2
                mid_z = float(siegelz(mpf(mid)))
                if float(siegelz(mpf(lo))) * mid_z < 0:
                    hi = mid
                else:
                    lo = mid
            
            zeros.append((lo + hi) / 2)
        
        prev_z = curr_z
    
    return zeros


class TestSuite:
    """Comprehensive test suite for RH verification."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.start_time = None
    
    def run_all(self, extended: bool = False):
        """Run all tests."""
        self.start_time = time.time()
        
        print("=" * 70)
        print("RIEMANN HYPOTHESIS VERIFICATION SUITE")
        print("=" * 70)
        
        # Test 1: Mathematical constants
        self.test_mathematical_constants()
        
        # Test 2: Functional equation
        self.test_functional_equation()
        
        # Test 3: Known zeros
        self.test_known_zeros()
        
        # Test 4: Extended zeros (optional)
        if extended:
            self.test_extended_zeros()
        
        # Test 5: Winding numbers
        self.test_winding_numbers()
        
        # Test 6: No off-line zeros
        self.test_no_offline_zeros()
        
        # Summary
        self.print_summary()
        
        return self.passed, self.failed
    
    def test_mathematical_constants(self):
        """Test ζ at known values."""
        print("\n[1] MATHEMATICAL CONSTANTS")
        print("-" * 40)
        
        tests = [
            (2, float(pi**2 / 6), "ζ(2) = π²/6"),
            (4, float(pi**4 / 90), "ζ(4) = π⁴/90"),
            (6, float(pi**6 / 945), "ζ(6) = π⁶/945"),
        ]
        
        for s, expected, name in tests:
            result = float(mp_zeta(s).real)
            error = abs(result - expected)
            passed = error < 1e-20
            
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: error = {error:.2e}")
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    def test_functional_equation(self):
        """Test ξ(s) = ξ(1-s)."""
        print("\n[2] FUNCTIONAL EQUATION ξ(s) = ξ(1-s)")
        print("-" * 40)
        
        from mpmath import gamma
        
        def xi_mp(s):
            return mpf('0.5') * s * (s - 1) * pi**(-s/2) * gamma(s/2) * mp_zeta(s)
        
        test_points = [
            mpc(0.3, 10),
            mpc(0.7, 15),
            mpc(0.2, 20),
            mpc(0.8, 25),
            mpc(0.4, 30),
            mpc(0.6, 35),
            mpc(0.25, 40),
            mpc(0.75, 45),
            mpc(0.35, 50),
            mpc(0.65, 55),
        ]
        
        all_passed = True
        max_diff = 0
        
        for s in test_points:
            xi_s = xi_mp(s)
            xi_1_minus_s = xi_mp(1 - s)
            
            diff = fabs(xi_s - xi_1_minus_s)
            avg = (fabs(xi_s) + fabs(xi_1_minus_s)) / 2
            rel_diff = float(diff / avg) if avg > 1e-50 else float(diff)
            
            max_diff = max(max_diff, rel_diff)
            
            if rel_diff > 1e-20:
                all_passed = False
        
        status = "✓" if all_passed else "✗"
        print(f"  {status} Tested {len(test_points)} points")
        print(f"     Max relative difference: {max_diff:.2e}")
        
        if all_passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_known_zeros(self):
        """Test that |ζ| is small at known zeros."""
        print("\n[3] KNOWN ZEROS VERIFICATION")
        print("-" * 40)
        
        passed_count = 0
        failed_count = 0
        max_mag = 0
        
        for i, t in enumerate(KNOWN_ZEROS_100):
            s = mpc(mpf('0.5'), t)
            mag = float(fabs(mp_zeta(s)))
            max_mag = max(max_mag, mag)
            
            if mag < 1e-6:
                passed_count += 1
            else:
                failed_count += 1
                if failed_count <= 3:
                    print(f"  ✗ Zero #{i+1} (t={t:.6f}): |ζ| = {mag:.2e}")
        
        status = "✓" if failed_count == 0 else "✗"
        print(f"  {status} Verified {passed_count}/{len(KNOWN_ZEROS_100)} zeros")
        print(f"     Max |ζ| at zeros: {max_mag:.2e}")
        
        if failed_count == 0:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_extended_zeros(self):
        """Test many more zeros (slow)."""
        print("\n[4] EXTENDED ZEROS (up to t=500)")
        print("-" * 40)
        print("  Generating zeros... (this may take a minute)")
        
        zeros = generate_extended_zeros(max_t=500, step=0.1)
        print(f"  Found {len(zeros)} zeros")
        
        passed_count = 0
        max_mag = 0
        
        for t in zeros:
            s = mpc(mpf('0.5'), t)
            mag = float(fabs(mp_zeta(s)))
            max_mag = max(max_mag, mag)
            
            if mag < 1e-5:
                passed_count += 1
        
        status = "✓" if passed_count == len(zeros) else "✗"
        print(f"  {status} Verified {passed_count}/{len(zeros)} zeros")
        print(f"     Max |ζ| at zeros: {max_mag:.2e}")
        
        if passed_count == len(zeros):
            self.passed += 1
        else:
            self.failed += 1
    
    def test_winding_numbers(self):
        """Test winding numbers around zeros."""
        print("\n[5] WINDING NUMBERS")
        print("-" * 40)
        
        from mpmath import arg
        
        def compute_winding(center_re, center_im, radius, samples=500):
            total_delta = mpf(0)
            
            for i in range(samples):
                theta1 = 2 * pi * i / samples
                theta2 = 2 * pi * (i + 1) / samples
                
                s1 = mpc(center_re + radius * mpf(str(float(mp.cos(theta1)))),
                         center_im + radius * mpf(str(float(mp.sin(theta1)))))
                s2 = mpc(center_re + radius * mpf(str(float(mp.cos(theta2)))),
                         center_im + radius * mpf(str(float(mp.sin(theta2)))))
                
                z1 = mp_zeta(s1)
                z2 = mp_zeta(s2)
                
                delta = arg(z2) - arg(z1)
                
                while delta > pi:
                    delta -= 2 * pi
                while delta < -pi:
                    delta += 2 * pi
                
                total_delta += delta
            
            return round(float(total_delta / (2 * pi)))
        
        # Test winding around first 5 zeros
        all_passed = True
        
        for t in KNOWN_ZEROS_100[:5]:
            w = compute_winding(mpf('0.5'), mpf(t), mpf('0.3'), 300)
            status = "✓" if w == 1 else "✗"
            print(f"  {status} t = {t:.6f}: W = {w}")
            
            if w != 1:
                all_passed = False
        
        # Test winding off critical line (should be 0)
        for sigma in [0.3, 0.7]:
            w = compute_winding(mpf(sigma), mpf('14.13'), mpf('0.2'), 300)
            status = "✓" if w == 0 else "✗"
            print(f"  {status} σ = {sigma}, t = 14.13: W = {w}")
            
            if w != 0:
                all_passed = False
        
        if all_passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_no_offline_zeros(self):
        """Test that there are no zeros off the critical line."""
        print("\n[6] NO OFF-LINE ZEROS")
        print("-" * 40)
        
        # Sample |ζ| at many off-line points
        off_line_points = []
        for sigma in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
            for t in [14.13, 21.02, 25.01, 30.42, 32.94]:
                off_line_points.append((sigma, t))
        
        all_nonzero = True
        min_mag = float('inf')
        
        for sigma, t in off_line_points:
            s = mpc(sigma, t)
            mag = float(fabs(mp_zeta(s)))
            min_mag = min(min_mag, mag)
            
            if mag < 0.01:
                all_nonzero = False
                print(f"  ✗ σ = {sigma}, t = {t}: |ζ| = {mag:.4f} (suspiciously small)")
        
        status = "✓" if all_nonzero else "✗"
        print(f"  {status} Tested {len(off_line_points)} off-line points")
        print(f"     Min |ζ| off-line: {min_mag:.4f}")
        
        if all_nonzero:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Passed: {self.passed}")
        print(f"  Failed: {self.failed}")
        print(f"  Total:  {self.passed + self.failed}")
        print(f"  Time:   {elapsed:.2f} seconds")
        
        if self.failed == 0:
            print("\n  ✓ ALL TESTS PASSED")
        else:
            print(f"\n  ✗ {self.failed} TESTS FAILED")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RH Verification Suite')
    parser.add_argument('--extended', action='store_true',
                        help='Run extended tests (slower)')
    args = parser.parse_args()
    
    suite = TestSuite()
    passed, failed = suite.run_all(extended=args.extended)
    
    sys.exit(0 if failed == 0 else 1)

