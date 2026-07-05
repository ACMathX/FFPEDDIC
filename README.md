# FFPEDDIC

Solver for the Fractional Fokker-Planck Equation with Dirac-Delta Initial Conditions.

This repository contains MATLAB and Python implementations of the fast quadrature/windowing solver described in the paper below, together with special-case solvers for selected fractional exponents.

## Structure

- `MATLAB_version/`: MATLAB solver, MATLAB scripts, `windowing_functions/`, and bundled `chebfun-master/`.
- `python_version/`: Python solver, Python scripts, and Python package requirements.
- `configs/`: shared JSON inputs used by both implementations.

Each implementation has its own README:

- [MATLAB_version/README.md](MATLAB_version/README.md)
- [python_version/README.md](python_version/README.md)

## Shared Configs

Script inputs live in `configs/*.json`. Change dimensions, diffusion coefficients, grids, rational exponents, precision, and solver parameters in JSON files rather than editing scripts directly.

Current config files:

- `configs/test_step01_accuracy_verification.json`
- `configs/test_step02_plot.json`
- `configs/test_extra01_special_case_plot.json`

## Quick Start

MATLAB:

```matlab
run( 'MATLAB_version/test_step01_accuracy_verification.m' )
```

Python:

```bash
python3 python_version/test_step01_accuracy_verification.py
```

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
