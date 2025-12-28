"""
Symbolic computation module for Riemann Hypothesis proof.

This module provides arbitrary-precision computation of:
- Riemann zeta function ζ(s)
- Completed xi function ξ(s)
- Winding numbers for topological protection
- Zero detection and verification

All computations use mpmath for exact arithmetic.
"""

from .zeta_exact import (
    zeta,
    xi,
    verify_functional_equation,
    find_zeros_in_range,
    verify_zero,
    compute_winding_number,
    set_precision,
    KNOWN_ZEROS
)

from .xi_function import XiFunction, batch_verify_functional_equation

from .winding import WindingNumberComputer, verify_known_zeros_are_simple

__all__ = [
    'zeta',
    'xi',
    'verify_functional_equation',
    'find_zeros_in_range',
    'verify_zero',
    'compute_winding_number',
    'set_precision',
    'KNOWN_ZEROS',
    'XiFunction',
    'batch_verify_functional_equation',
    'WindingNumberComputer',
    'verify_known_zeros_are_simple'
]

