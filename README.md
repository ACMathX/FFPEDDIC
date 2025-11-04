# FFPEDDIC
Solver for Fractional Fokker-Planck Equation with Dirac-Delta Initial Conditions

`chebfun-master` is from [chebfun](https://github.com/chebfun/chebfun)

The primary implementation resides in `C_FFPESolverV1M.m`, which contains all relevant method definitions. Example usages of key methods can be found in `test_step07a_accuracy_verification.m` and `test_step07b_graph_plot.m`.

The files `C_FFPEHalfAlpha.m`, `C_FFPEOneThirdAlpha.m`, `C_FFPETwoThirdsAlpha.m`, and `C_FFPERationalAlpha.m` implement specialized solver classes corresponding to the special cases $\alpha = 1 / 2$, $\alpha = 1 / 3$, $\alpha = 2 / 3$, and $\alpha = p / q$, respectively. Example scripts demonstrating their usage are provided in `test_extra03c_special_case_graph_plot.m` and `test_step07a_accuracy_verification.m`.

## Citation

Please cite our paper if you find this repo useful:
```
@article{ye2024fast,
    title={A fast and accurate solver for the fractional Fokker-Planck equation with Dirac-Delta initial conditions},
    author={Ye, Qihao and Tian, Xiaochuan and Wang, Dong},
    journal={arXiv preprint arXiv:2407.15315},
    year={2024}
}
```

Manuscript: [
[ArXiv Version](https://arxiv.org/abs/2407.15315)
]

Accepted by SIAM Journal on Scientific Computing (SISC)
