# FFPEDDIC Python Version

PyTorch implementation of the fast quadrature/windowing solver for the fractional Fokker-Planck equation with Dirac-delta initial conditions.

Run commands from the repository root so the scripts can find shared configs in `configs/`.

## Contents

- `lib/solver.py`: reusable `FFPESolver`, initialization routines, scaling fallbacks, and cached window quadrature.
- `lib/quadrature.py`: Gauss-Legendre, Gauss-Jacobi, Chebfun-style large-`n` Legendre nodes, and fractional quadrature helpers.
- `lib/windowing.py`: smooth compact-support windowing functions used by far-field quadrature.
- `lib/special_cases.py`: closed-form special cases for `alpha = 1 / 2`, `alpha = 1 / 3`, `alpha = 2 / 3`, rational-alpha cases, and `erfcx` helpers.
- `lib/legendre_polynomial.py`: Legendre polynomial evaluation and coefficient helpers.
- `lib/config.py`: JSON config loader.
- `lib/defaults.py`: default solver parameters used by scripts and rational-alpha fallback.
- `test_step01_accuracy_verification.py`: compare general solver values with rational-alpha special-case formulas.
- `test_step02_plot.py`: generate a general-solver surface plot.
- `test_extra01_special_case_plot.py`: generate an `alpha = 1 / 2` special-case surface plot.
- `requirements.txt`: Python package dependencies.

## Dependencies

```bash
python3 -m pip install -r python_version/requirements.txt
```

Main arrays use `torch.float64`. SciPy supplies quadrature nodes and Bessel functions; NumPy handles grids and dense arrays; mpmath handles high-precision hypergeometric formulas in special cases.

## Shared Configs

Script defaults point to root-level configs:

- `configs/test_step01_accuracy_verification.json`
- `configs/test_step02_plot.json`
- `configs/test_extra01_special_case_plot.json`

Override with `--config` when needed:

```bash
python3 python_version/test_step01_accuracy_verification.py --config configs/test_step01_accuracy_verification.json
```

## Running Scripts

Accuracy check:

```bash
python3 python_version/test_step01_accuracy_verification.py
```

General solver plot:

```bash
python3 python_version/test_step02_plot.py
```

Half-alpha special-case plot:

```bash
python3 python_version/test_extra01_special_case_plot.py
```

Plot scripts write PDFs in `python_version/` by default. Use `--output` to choose another path.

## Solver Use

```python
from python_version.lib.solver import FFPESolver

solver = FFPESolver(
    d = 1,
    alpha = 3 / 23,
    D_o = 0,
    D_f = 8,
    t = 0.04
)
solver.general_initialization()
result = solver.get_value( 1.92 )
print( result.value )
```

`general_initialization()` builds the near-origin fractional quadrature, far-field window quadrature, and reusable kernel values. Call it before `get_value`.

## Solver Parameters

Defaults live in `lib/defaults.py` and can be overridden through config keys:

- `L`: near-origin split point.
- `M_ini`: first far-field window size.
- `M_lim`: final far-field window limit; windows double by `M = 2 M`.
- `gamma`: window transition fraction.
- `d_tol`: stopping tolerance for successive far-field window contributions.
- `window_multiplier`: sets window quadrature points by `N = round( window_multiplier * M )`.
- `window_points_cap`: optional upper bound for `N`; use `None` for no cap.
- `device`: PyTorch device name.

Cached data can be cleared with `clear_solver_caches()` from `lib.solver` and `clear_quadrature_caches()` from `lib.quadrature`.

## Citation

Please cite our paper if you find this repo useful:

```bibtex
@article{ye2026fast,
    title={A Fast and Accurate Solver for the Fractional {F}okker--{P}lanck Equation with {D}irac-Delta Initial Conditions},
    author={Ye, Qihao and Tian, Xiaochuan and Wang, Dong},
    journal={SIAM Journal on Scientific Computing},
    volume={48},
    number={2},
    pages={A1050--A1074},
    year={2026},
    publisher={SIAM}
}
```

Manuscript: [[Journal Version](https://doi.org/10.1137/24M1682907), [ArXiv Version](https://arxiv.org/abs/2407.15315)]
