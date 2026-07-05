# FFPEDDIC MATLAB Version

MATLAB implementation of the fast quadrature/windowing solver for the fractional Fokker-Planck equation with Dirac-delta initial conditions.

## Contents

- `C_FFPESolver.m`: primary adaptive quadrature/windowing solver for the fractional Fokker-Planck density.
- `C_FractionalQuadrature.m`: fractional-weight quadrature rules and expansion-based quadrature helpers.
- `C_LegendrePolynomial.m`: Legendre polynomial helpers used by quadrature construction.
- `C_WindowingFunction.m` and `windowing_functions/`: reusable windowing-function base class and concrete test window functions.
- `C_FFPEHalfAlpha.m`: special-case solver for `alpha = 1 / 2`.
- `C_FFPEOneThirdAlpha.m`: special-case solver for `alpha = 1 / 3`.
- `C_FFPETwoThirdsAlpha.m`: special-case solver for `alpha = 2 / 3`.
- `C_FFPERationalAlpha.m`: special-case solver for rational `alpha = p / q` when `D_o = 0`.
- `F_load_json_config.m`: shared JSON config loader.
- `test_step01_accuracy_verification.m`: accuracy check against closed-form rational-alpha special cases.
- `test_step02_plot.m`: general solver plot script.
- `test_extra01_special_case_plot.m`: `alpha = 1 / 2` special-case plot script.
- `chebfun-master/`: bundled Chebfun dependency for `legpts` and `jacpts`.

## Requirements

- MATLAB with class validation syntax support.
- Symbolic Math Toolbox for special-case solvers that use `vpa`, `digits`, and `hypergeom`.
- Bundled `chebfun-master/` for `legpts` and `jacpts`.

## Configs

Shared JSON configs live in `../configs`. The MATLAB scripts use paths relative to `MATLAB_version`, so they can be run from repository root:

```matlab
run( 'MATLAB_version/test_step01_accuracy_verification.m' )
run( 'MATLAB_version/test_step02_plot.m' )
run( 'MATLAB_version/test_extra01_special_case_plot.m' )
```

## Quick Start

```matlab
script_folder = fullfile( pwd, 'MATLAB_version' );
addpath( script_folder );
addpath( fullfile( script_folder, 'chebfun-master' ) );
addpath( fullfile( script_folder, 'windowing_functions' ) );

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

`general_initialization()` prepares windowing-function samples, near-origin quadrature, and precomputed kernel values. Call it before `get_value`.

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
