# FFPEDDIC

Solver for the Fractional Fokker-Planck Equation with Dirac-Delta Initial Conditions.

This repository implements the fast quadrature/windowing solver described in the paper below, together with special-case reference solvers for selected fractional exponents. The bundled `chebfun-master` directory is from [Chebfun](https://github.com/chebfun/chebfun).

## What Is Included

- `C_FFPESolver.m`: primary adaptive quadrature/windowing solver for the fractional Fokker-Planck density.
- `C_FractionalQuadrature.m`: fractional-weight quadrature rules and expansion-based quadrature helpers.
- `C_LegendrePolynomial.m`: Legendre polynomial helpers used by quadrature construction.
- `C_WindowingFunction.m` and `windowing_functions/`: reusable windowing-function base class and concrete test window functions.
- `C_FFPEHalfAlpha.m`: reference/special-case solver for `alpha = 1 / 2`.
- `C_FFPEOneThirdAlpha.m`: reference/special-case solver for `alpha = 1 / 3`.
- `C_FFPETwoThirdsAlpha.m`: reference/special-case solver for `alpha = 2 / 3`.
- `C_FFPERationalAlpha.m`: reference/special-case solver for rational `alpha = p / q` when `D_o = 0`.
- `configs/`: JSON inputs for runnable experiment and plotting scripts.
- `test_step01_accuracy_verification.m`: accuracy comparison between the general solver and special-case solvers.
- `test_step02_plot.m`: graph-generation script for the general solver.
- `test_extra01_special_case_plot.m`: graph-generation script for the `alpha = 1 / 2` special-case solver.

## Requirements

- MATLAB with class validation syntax support.
- Symbolic Math Toolbox for the special-case solvers that use `vpa`, `digits`, and `hypergeom`.
- The bundled `chebfun-master` folder for `legpts` and `jacpts`.

Run commands from the repository root unless you adjust paths manually.

## Quick Start

```matlab
addpath( 'chebfun-master' );
addpath( 'windowing_functions' );

d = 1;
alpha = 3 / 23;
D_o = 0;
D_f = 8;
delta_t = 0.04;
y = 1.92;

solver = C_FFPESolver( d, alpha, D_o, D_f, delta_t );
solver.general_initialization();
value = solver.get_value( y );
```

`general_initialization()` prepares the windowing-function samples, near-origin quadrature, and precomputed kernel values. Call it before `get_value`.

## Config-Driven Scripts

Script inputs live in `configs/*.json` and are loaded with `F_load_json_config.m`. Change dimensions, diffusion coefficients, grids, rational exponents, precision, and test modes in JSON files rather than editing scripts directly.

```matlab
config = F_load_json_config( 'configs/test_step01_accuracy_verification.json' );
```

Current config files:

- `configs/test_step01_accuracy_verification.json`
- `configs/test_step02_plot.json`
- `configs/test_extra01_special_case_plot.json`

## Running Examples

Accuracy verification:

```matlab
test_step01_accuracy_verification
```

General solver surface plot:

```matlab
test_step02_plot
```

Special-case surface plot:

```matlab
test_extra01_special_case_plot
```

Only add dependency paths when needed. Scripts that use the general solver need both `chebfun-master` and `windowing_functions`; scripts that only use closed-form special-case solvers do not need those paths unless they call shared helpers requiring them.

## Solver Notes

The general solver supports ordinary diffusion coefficient `D_o`, fractional diffusion coefficient `D_f`, dimension `d`, fractional exponent `alpha`, and time `t`. For zero displacement, the solver uses a dedicated branch. If direct evaluation does not converge within the windowing bounds, scaling fallbacks are used.

The special-case classes are useful for validation and high-precision comparison. Positive `D_o` is implemented for `C_FFPEHalfAlpha`; the `alpha = 1 / 3`, `alpha = 2 / 3`, and rational-alpha special cases currently raise descriptive errors when `D_o > 0`.

## Common Edits

- Change experiment parameters in `configs/*.json`.
- Tune general-solver windowing through `M_ini`, `M_lim`, `gamma`, and `d_tol`.
- Use `C_TestWindowingFunction01( M, gamma )`, `C_TestWindowingFunction02( M, gamma )`, or `C_TestWindowingFunction03( M, gamma, beta )` to inspect alternative window functions.
- Use `F_load_json_config( file_name )` for new runnable scripts.

## Troubleshooting

- Missing `legpts` or `jacpts`: run `addpath( 'chebfun-master' )`.
- Missing windowing class: run `addpath( 'windowing_functions' )`.
- Missing initialization error: call `general_initialization()` before `get_value`.
- Symbolic/MuPAD errors: verify Symbolic Math Toolbox is installed and working; special-case solvers depend on it.
- Unsupported ordinary diffusion error: use the general solver or `C_FFPEHalfAlpha` when `D_o > 0`.

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
