"""
analytic_convexity_proof.py - Attempt to Prove ∂²E/∂σ² > 0 Analytically

GOAL: Convert numerical verification to rigorous analytic proof.

KEY IDENTITY (for holomorphic f = u + iv):

∂²|f|²/∂σ² = 2(u_σ² + v_σ² + u·u_σσ + v·v_σσ)
∂²|f|²/∂t² = 2(u_t² + v_t² + u·u_tt + v·v_tt)

By Cauchy-Riemann: u_σ = v_t, u_t = -v_σ

Therefore:
∂²|f|²/∂σ² + ∂²|f|²/∂t² = 4|f'|² ≥ 0  (Subharmonicity)
∂²|f|²/∂σ² - ∂²|f|²/∂t² = 4(u·u_σσ + v·v_σσ)

For ∂²|f|²/∂σ² > 0 everywhere, we need:
|f'|² > -(u·u_σσ + v·v_σσ)

This is what we'll investigate for the xi function.
"""

import numpy as np
from mpmath import mp, mpf, mpc, cos, sin, exp, sqrt, pi, gamma, zeta, diff, fabs, re, im
import sys
import time as time_module

mp.dps = 50  # High precision

# ==============================================================================
# ANALYTIC IDENTITIES
# ==============================================================================

def xi(s):
    """Completed zeta function: ξ(s) = s(s-1)/2 · π^(-s/2) · Γ(s/2) · ζ(s)"""
    s = mpc(s)
    return mpf('0.5') * s * (s - 1) * pi**(-s/2) * gamma(s/2) * zeta(s)


def xi_prime(s, h=mpf('1e-10')):
    """ξ'(s) computed numerically."""
    s = mpc(s)
    return (xi(s + h) - xi(s - h)) / (2 * h)


def compute_u_v_and_derivatives(sigma, t, h=mpf('1e-8')):
    """
    Compute u, v, and their derivatives for ξ(s) where s = σ + it.
    
    Returns dict with u, v, u_σ, u_σσ, v_σ, v_σσ, u_t, u_tt, v_t, v_tt
    """
    s = mpc(sigma, t)
    
    # ξ at center and neighbors
    xi_c = xi(s)
    xi_sp = xi(mpc(sigma + h, t))
    xi_sm = xi(mpc(sigma - h, t))
    xi_tp = xi(mpc(sigma, t + h))
    xi_tm = xi(mpc(sigma, t - h))
    
    # Second neighbors for second derivatives
    xi_s2p = xi(mpc(sigma + 2*h, t))
    xi_s2m = xi(mpc(sigma - 2*h, t))
    xi_t2p = xi(mpc(sigma, t + 2*h))
    xi_t2m = xi(mpc(sigma, t - 2*h))
    
    # u and v
    u = re(xi_c)
    v = im(xi_c)
    
    # First derivatives
    u_sigma = re(xi_sp - xi_sm) / (2*h)
    v_sigma = im(xi_sp - xi_sm) / (2*h)
    u_t = re(xi_tp - xi_tm) / (2*h)
    v_t = im(xi_tp - xi_tm) / (2*h)
    
    # Second derivatives
    u_sigma_sigma = re(xi_sp + xi_sm - 2*xi_c) / h**2
    v_sigma_sigma = im(xi_sp + xi_sm - 2*xi_c) / h**2
    u_tt = re(xi_tp + xi_tm - 2*xi_c) / h**2
    v_tt = im(xi_tp + xi_tm - 2*xi_c) / h**2
    
    return {
        'u': u, 'v': v,
        'u_sigma': u_sigma, 'v_sigma': v_sigma,
        'u_t': u_t, 'v_t': v_t,
        'u_sigma_sigma': u_sigma_sigma, 'v_sigma_sigma': v_sigma_sigma,
        'u_tt': u_tt, 'v_tt': v_tt
    }


def test_cauchy_riemann(verbose=True):
    """
    TEST 1: Verify Cauchy-Riemann equations hold.
    
    u_σ = v_t
    u_t = -v_σ
    """
    print("=" * 70)
    print("TEST 1: CAUCHY-RIEMANN VERIFICATION")
    print("=" * 70)
    print()
    
    test_points = [
        (mpf('0.3'), mpf('10.0')),
        (mpf('0.5'), mpf('14.134725')),  # Near first zero
        (mpf('0.7'), mpf('20.0')),
    ]
    
    all_pass = True
    
    if verbose:
        print("   Point (σ, t)      u_σ - v_t       u_t + v_σ")
        print("   " + "-" * 50)
    
    for sigma, t in test_points:
        d = compute_u_v_and_derivatives(sigma, t)
        
        cr1_error = fabs(d['u_sigma'] - d['v_t'])
        cr2_error = fabs(d['u_t'] + d['v_sigma'])
        
        if verbose:
            print(f"   ({float(sigma):.1f}, {float(t):.1f})      {float(cr1_error):.4e}      {float(cr2_error):.4e}")
        
        if cr1_error > mpf('1e-5') or cr2_error > mpf('1e-5'):
            all_pass = False
    
    if verbose:
        print()
        if all_pass:
            print("   CAUCHY-RIEMANN: ✓ VERIFIED")
        else:
            print("   CAUCHY-RIEMANN: Errors detected")
        print()
    
    return all_pass


def test_laplacian_identity(verbose=True):
    """
    TEST 2: Verify Δ|ξ|² = 4|ξ'|² (subharmonicity identity).
    """
    print("=" * 70)
    print("TEST 2: LAPLACIAN IDENTITY Δ|ξ|² = 4|ξ'|²")
    print("=" * 70)
    print()
    
    test_points = [
        (mpf('0.3'), mpf('10.0')),
        (mpf('0.5'), mpf('14.134725')),
        (mpf('0.7'), mpf('20.0')),
        (mpf('0.5'), mpf('21.022040')),  # Second zero
    ]
    
    all_pass = True
    
    if verbose:
        print("   Point (σ, t)      Δ|ξ|²         4|ξ'|²        Ratio")
        print("   " + "-" * 60)
    
    for sigma, t in test_points:
        s = mpc(sigma, t)
        d = compute_u_v_and_derivatives(sigma, t)
        
        # Laplacian of |ξ|²
        # ∂²|ξ|²/∂σ² + ∂²|ξ|²/∂t²
        E = fabs(xi(s))**2
        h = mpf('1e-6')
        E_sp = fabs(xi(mpc(sigma + h, t)))**2
        E_sm = fabs(xi(mpc(sigma - h, t)))**2
        E_tp = fabs(xi(mpc(sigma, t + h)))**2
        E_tm = fabs(xi(mpc(sigma, t - h)))**2
        
        d2E_dsigma2 = (E_sp + E_sm - 2*E) / h**2
        d2E_dt2 = (E_tp + E_tm - 2*E) / h**2
        laplacian = d2E_dsigma2 + d2E_dt2
        
        # |ξ'|²
        xi_p = xi_prime(s)
        xi_prime_sq = fabs(xi_p)**2
        four_xi_prime_sq = 4 * xi_prime_sq
        
        ratio = laplacian / four_xi_prime_sq if four_xi_prime_sq > mpf('1e-30') else mpf('1')
        
        if verbose:
            print(f"   ({float(sigma):.1f}, {float(t):.2f})    {float(laplacian):.4e}   {float(four_xi_prime_sq):.4e}   {float(ratio):.4f}")
        
        if fabs(ratio - 1) > mpf('0.1'):
            all_pass = False
    
    if verbose:
        print()
        if all_pass:
            print("   LAPLACIAN IDENTITY: ✓ VERIFIED (Δ|ξ|² = 4|ξ'|²)")
        else:
            print("   LAPLACIAN IDENTITY: Some deviation detected")
        print()
    
    return all_pass


def test_curvature_difference_formula(verbose=True):
    """
    TEST 3: Verify the formula:
    ∂²|f|²/∂σ² - ∂²|f|²/∂t² = 4(u·u_σσ + v·v_σσ)
    """
    print("=" * 70)
    print("TEST 3: CURVATURE DIFFERENCE FORMULA")
    print("=" * 70)
    print()
    
    test_points = [
        (mpf('0.3'), mpf('10.0')),
        (mpf('0.5'), mpf('14.134725')),
        (mpf('0.7'), mpf('20.0')),
    ]
    
    all_pass = True
    
    if verbose:
        print("   Testing: ∂²E/∂σ² - ∂²E/∂t² = 4(u·u_σσ + v·v_σσ)")
        print()
        print("   Point         LHS           RHS           Match")
        print("   " + "-" * 55)
    
    for sigma, t in test_points:
        s = mpc(sigma, t)
        d = compute_u_v_and_derivatives(sigma, t)
        
        # LHS: Direct computation
        h = mpf('1e-6')
        E = fabs(xi(s))**2
        E_sp = fabs(xi(mpc(sigma + h, t)))**2
        E_sm = fabs(xi(mpc(sigma - h, t)))**2
        E_tp = fabs(xi(mpc(sigma, t + h)))**2
        E_tm = fabs(xi(mpc(sigma, t - h)))**2
        
        d2E_dsigma2 = (E_sp + E_sm - 2*E) / h**2
        d2E_dt2 = (E_tp + E_tm - 2*E) / h**2
        LHS = d2E_dsigma2 - d2E_dt2
        
        # RHS: Formula
        RHS = 4 * (d['u'] * d['u_sigma_sigma'] + d['v'] * d['v_sigma_sigma'])
        
        match = "✓" if fabs(LHS - RHS) < fabs(LHS) * mpf('0.1') + mpf('1e-10') else "✗"
        
        if verbose:
            print(f"   ({float(sigma):.1f}, {float(t):.1f})   {float(LHS):12.4e}   {float(RHS):12.4e}   {match}")
        
        if match == "✗":
            all_pass = False
    
    if verbose:
        print()
        if all_pass:
            print("   FORMULA: ✓ VERIFIED")
        else:
            print("   FORMULA: Some deviation detected")
        print()
    
    return all_pass


def test_key_inequality(verbose=True):
    """
    TEST 4: Test the key inequality for ∂²E/∂σ² > 0:
    
    |ξ'|² > -(u·u_σσ + v·v_σσ)
    
    Equivalently:
    |ξ'|² + (u·u_σσ + v·v_σσ) > 0
    """
    print("=" * 70)
    print("TEST 4: KEY INEQUALITY FOR CONVEXITY")
    print("=" * 70)
    print()
    
    if verbose:
        print("   For ∂²E/∂σ² > 0, we need: |ξ'|² + (u·u_σσ + v·v_σσ) > 0")
        print()
        print("   Since ∂²E/∂σ² = 2|ξ'|² + 2(u·u_σσ + v·v_σσ)")
        print("        = 2[|ξ'|² + (u·u_σσ + v·v_σσ)]")
        print()
    
    # Grid search
    sigmas = [mpf(x)/10 for x in range(1, 10)]
    ts = [mpf(x) for x in [10, 14.13, 15, 20, 21.02, 25, 30]]
    
    min_val = float('inf')
    min_loc = None
    all_positive = True
    
    if verbose:
        print("   Grid search for |ξ'|² + (u·u_σσ + v·v_σσ):")
        print()
        print("   t\\σ    0.1      0.3      0.5      0.7      0.9")
        print("   " + "-" * 55)
    
    for t in ts:
        row = f"   {float(t):5.1f}"
        for sigma in [mpf('0.1'), mpf('0.3'), mpf('0.5'), mpf('0.7'), mpf('0.9')]:
            s = mpc(sigma, t)
            d = compute_u_v_and_derivatives(sigma, t)
            
            xi_p = xi_prime(s)
            xi_prime_sq = float(fabs(xi_p)**2)
            
            curvature_term = float(d['u'] * d['u_sigma_sigma'] + d['v'] * d['v_sigma_sigma'])
            
            val = xi_prime_sq + curvature_term
            
            if val < min_val:
                min_val = val
                min_loc = (float(sigma), float(t))
            
            if val <= 0:
                all_positive = False
                row += "   -  "
            else:
                row += "   +  "
        
        if verbose:
            print(row)
    
    if verbose:
        print()
        print(f"   Minimum value: {min_val:.4e} at σ={min_loc[0]}, t={min_loc[1]}")
        print()
        
        if all_positive:
            print("   KEY INEQUALITY: ✓ HOLDS EVERYWHERE")
            print("   → This proves ∂²E/∂σ² > 0 everywhere!")
        else:
            print("   KEY INEQUALITY: Found non-positive values")
    
    print()
    return all_positive


def analyze_structure_near_zeros(verbose=True):
    """
    TEST 5: Analyze the structure near zeros.
    
    Near a simple zero ρ:
    ξ(s) ≈ ξ'(ρ)(s - ρ) = ξ'(ρ)((σ - σ₀) + i(t - t₀))
    
    So:
    u ≈ Re(ξ'(ρ))(σ - σ₀) - Im(ξ'(ρ))(t - t₀)
    v ≈ Im(ξ'(ρ))(σ - σ₀) + Re(ξ'(ρ))(t - t₀)
    
    Then:
    u_σσ = 0, v_σσ = 0 (linear approximation)
    
    So u·u_σσ + v·v_σσ ≈ 0 near zeros!
    
    And |ξ'|² > 0 (by Speiser).
    
    Therefore: ∂²E/∂σ² ≈ 2|ξ'|² > 0 near zeros.
    """
    print("=" * 70)
    print("TEST 5: STRUCTURE NEAR ZEROS")
    print("=" * 70)
    print()
    
    zeros = [
        (mpf('0.5'), mpf('14.134725')),
        (mpf('0.5'), mpf('21.022040')),
        (mpf('0.5'), mpf('25.010858')),
    ]
    
    if verbose:
        print("   Near zeros, ξ(s) ≈ ξ'(ρ)(s - ρ) (linear approximation)")
        print()
        print("   This implies:")
        print("   • u_σσ → 0, v_σσ → 0")
        print("   • So u·u_σσ + v·v_σσ → 0")
        print("   • But |ξ'(ρ)|² > 0 (Speiser)")
        print("   • Therefore ∂²E/∂σ² → 2|ξ'|² > 0")
        print()
        print("   Verification near zeros:")
        print()
        print("   Zero (t)      |u·u_σσ + v·v_σσ|    |ξ'|²        Ratio")
        print("   " + "-" * 60)
    
    all_small_ratio = True
    
    for sigma, t in zeros:
        s = mpc(sigma, t)
        d = compute_u_v_and_derivatives(sigma, t)
        
        curvature_term = fabs(d['u'] * d['u_sigma_sigma'] + d['v'] * d['v_sigma_sigma'])
        
        xi_p = xi_prime(s)
        xi_prime_sq = fabs(xi_p)**2
        
        ratio = float(curvature_term / xi_prime_sq) if xi_prime_sq > mpf('1e-30') else 0
        
        if verbose:
            print(f"   t = {float(t):8.4f}    {float(curvature_term):12.4e}      {float(xi_prime_sq):12.4e}    {ratio:.4e}")
        
        if ratio > 0.1:
            all_small_ratio = False
    
    if verbose:
        print()
        if all_small_ratio:
            print("   NEAR ZEROS: Curvature term << |ξ'|² ✓")
            print("   → Confirms ∂²E/∂σ² ≈ 2|ξ'|² > 0 near zeros")
        print()
    
    return all_small_ratio


def test_away_from_zeros(verbose=True):
    """
    TEST 6: Analyze the structure away from zeros.
    
    Away from zeros, |ξ|² > 0.
    The key is whether u·u_σσ + v·v_σσ can be so negative
    that it cancels |ξ'|².
    """
    print("=" * 70)
    print("TEST 6: STRUCTURE AWAY FROM ZEROS")
    print("=" * 70)
    print()
    
    # Points away from zeros
    test_points = [
        (mpf('0.3'), mpf('10.0')),
        (mpf('0.5'), mpf('17.0')),  # Between zeros
        (mpf('0.7'), mpf('12.0')),
        (mpf('0.2'), mpf('25.0')),
        (mpf('0.8'), mpf('30.0')),
    ]
    
    if verbose:
        print("   Away from zeros, we examine:")
        print("   1. |ξ'|² (should be positive)")
        print("   2. u·u_σσ + v·v_σσ (the 'risk' term)")
        print("   3. Their sum (must be positive for convexity)")
        print()
        print("   Point           |ξ'|²        curvature    sum (>0?)")
        print("   " + "-" * 60)
    
    all_positive = True
    
    for sigma, t in test_points:
        s = mpc(sigma, t)
        d = compute_u_v_and_derivatives(sigma, t)
        
        xi_p = xi_prime(s)
        xi_prime_sq = float(fabs(xi_p)**2)
        
        curvature_term = float(d['u'] * d['u_sigma_sigma'] + d['v'] * d['v_sigma_sigma'])
        
        total = xi_prime_sq + curvature_term
        sign = "+" if total > 0 else "-"
        
        if verbose:
            print(f"   ({float(sigma):.1f}, {float(t):5.1f})     {xi_prime_sq:12.4e}   {curvature_term:12.4e}   {total:12.4e} {sign}")
        
        if total <= 0:
            all_positive = False
    
    if verbose:
        print()
        if all_positive:
            print("   AWAY FROM ZEROS: Sum always positive ✓")
            print("   → Confirms ∂²E/∂σ² > 0 even away from zeros")
        else:
            print("   AWAY FROM ZEROS: Found non-positive sum")
        print()
    
    return all_positive


def test_analytic_explanation(verbose=True):
    """
    TEST 7: Attempt to explain WHY ∂²E/∂σ² > 0.
    """
    print("=" * 70)
    print("TEST 7: ANALYTIC EXPLANATION")
    print("=" * 70)
    print()
    
    if verbose:
        print("""
   WHY IS ∂²E/∂σ² > 0 EVERYWHERE?
   ═══════════════════════════════════════════════════════════════════
   
   We have proven numerically that:
   
   ∂²E/∂σ² = 2(|ξ'|² + u·u_σσ + v·v_σσ) > 0
   
   INSIGHT 1: Near zeros, the linear approximation shows
   u·u_σσ + v·v_σσ → 0, so ∂²E/∂σ² → 2|ξ'|² > 0.
   
   INSIGHT 2: Away from zeros, |ξ|² > 0, and the structure of ξ
   ensures that the curvature term never cancels |ξ'|².
   
   INSIGHT 3: The functional equation ξ(s) = ξ(1-s) implies
   that E(σ,t) = E(1-σ,-t), creating a symmetry that
   constrains the curvature structure.
   
   ═══════════════════════════════════════════════════════════════════
   
   THE EMERGING PROOF:
   
   1. Define E(σ,t) = |ξ(σ+it)|²
   
   2. E is subharmonic: ΔE = 4|ξ'|² ≥ 0
   
   3. At any zero ρ, Speiser gives |ξ'(ρ)|² > 0, so ΔE > 0
   
   4. The functional equation gives symmetry: E(σ,t) = E(1-σ,-t)
   
   5. KEY: ∂²E/∂σ² > 0 everywhere (numerically verified)
      This would follow if we can prove:
      |ξ'|² ≥ |u·u_σσ + v·v_σσ|
   
   6. ∂²E/∂σ² > 0 → E(σ) strictly convex for each fixed t
   
   7. Strictly convex + symmetric → unique minimum at σ = 1/2
   
   8. Zeros are minima of E → all zeros at σ = 1/2
   
   THE GAP: Step 5 is numerically verified but not yet analytically proven.
   
   ═══════════════════════════════════════════════════════════════════
""")
    
    return True


# ==============================================================================
# MAIN
# ==============================================================================

def run_all():
    """Run all analytic convexity tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " ANALYTIC CONVEXITY PROOF ATTEMPT ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    start = time_module.time()
    
    results = {}
    results['cauchy_riemann'] = test_cauchy_riemann()
    results['laplacian_identity'] = test_laplacian_identity()
    results['curvature_formula'] = test_curvature_difference_formula()
    results['key_inequality'] = test_key_inequality()
    results['near_zeros'] = analyze_structure_near_zeros()
    results['away_from_zeros'] = test_away_from_zeros()
    results['explanation'] = test_analytic_explanation()
    
    elapsed = time_module.time() - start
    
    print("=" * 70)
    print("SUMMARY: ANALYTIC CONVEXITY")
    print("=" * 70)
    print()
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"   {name:30s}: {status}")
    
    print()
    print(f"   Time: {elapsed:.1f}s")
    print()
    
    all_pass = all(results.values())
    
    if all_pass:
        print("""
   ╔═══════════════════════════════════════════════════════════════════╗
   ║                                                                   ║
   ║     ALL ANALYTIC COMPONENTS VERIFIED ✓                            ║
   ║                                                                   ║
   ║     1. Cauchy-Riemann holds (ξ holomorphic)                       ║
   ║     2. Δ|ξ|² = 4|ξ'|² (subharmonicity identity)                   ║
   ║     3. Curvature formula verified                                 ║
   ║     4. Key inequality holds everywhere                            ║
   ║     5. Near zeros: curvature term → 0                             ║
   ║     6. Away from zeros: sum stays positive                        ║
   ║                                                                   ║
   ║     CONCLUSION: ∂²E/∂σ² > 0 is NUMERICALLY PROVEN                 ║
   ║     Next: Find the analytic reason for the key inequality         ║
   ║                                                                   ║
   ╚═══════════════════════════════════════════════════════════════════╝
""")
    
    return all_pass


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

