import Lake
open Lake DSL

package «rh» where
  -- add package configuration options here
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩, -- Pretty-print anonymous functions using `fun`
    ⟨`autoImplicit, false⟩   -- Disable auto-implicit arguments
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «RiemannHypothesis» where
  -- add library configuration options here
  globs := #[.submodules `RiemannHypothesis]

