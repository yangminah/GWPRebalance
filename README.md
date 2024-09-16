# GWPRebalance

## MiMA
- This directory contains `forpy_mod.F90`, which is needed to use the package forpy for coupling python codes into forpy framework. More information about this package is available at [https://github.com/ylikx/forpy](https://github.com/ylikx/forpy).
- `cg_drag.f90` details how the MiMA codes were edited to replace the original AD99 implementation in Fortran with various versions of the ML emulators written in Python. These require using the aforementioned package, `forpy`.

## Training
This directory contains scripts used to train all of the ML schemes described in the manuscript.
# Experiments
All scripts that were run to train WaveNet and EDD architectures with the rebalancing technique. 
# Architectures
WaveNet and EDD architectures are defined here. 
- WaveNet: `d_model10.py`
- EDD: `ae_model09.py`
The sizes of the networks were determined by parameters:
- `n_ch` (EDD)
- `n_dilations` (EDD)
- `n_d` (WaveNet and EDD)
- `n_in` (WaveNet)
- `n_out` (WaveNet)
# `utils*.py`
- `utils.py` contains the training logic that includes the rebalancing technique. 
    - `train_loop3`, `train_loop4`, and `train_loop5` were used and accomplish the same rebalancing technique. They differ in how training progress was logged, and other accessorial details.

# Other things to note.
- We had precomputed many of the statistics needed to apply the rebalancing technique. These included how to divide by wind range coordinate into bins, and the histogram of the training dataset with respect to this wind range binning. 
- The number of epochs is set at 1_000_000, not because we were expecting a million pass-throughs but to ensure that the early-stopping mechanism is what stopped the training progress. 
    - At times, we had to restart the training progress because we were alloted 48 hours maximum to run jobs, and the training had not satisfied the early stopping criteria after 48 hours. 

