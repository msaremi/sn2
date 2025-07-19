# Structural System Neural Network (SN²) Solver CUDA Torch Extension

## Reference

Codes of the article [Neural Network Parameter-optimization of Gaussian pmDAGs](https://arxiv.org/abs/2309.14073)

## Installation

### Installation from pre-compiled .egg
Pre-compiled Python 3.9 and Python 3.12 Windows eggs are available [here](dist). You can install them using the following commands:
```shell
cd sn2/dist
wheel convert <egg file name>
pip3 install <created .whl file>
```
In place of `<egg file name>` the name of the .egg file should be written. It creates a .whl file, which is used in place of `<created .whl file>`.

### Installation from source

Installation requires the NVCC compiler with an ABI-compatible C++ compiler. 
On Windows, you will probably need to install [MSVC++](https://visualstudio.microsoft.com/downloads) and [the CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) in addition to a GPU-enabled [PyTorch](https://pytorch.org) package.

To install from source, first move to the package directory. Then, use the `pip3` command to install it:
```shell
cd sn2
pip3 install .
```
Alternatively, you may run `python setup.py install` to manually install the package.

## Examples

Examples:
- [Introduction](examples/introduction.ipynb)
- [Computation Method](examples/computation_method.ipynb)
- [Custom Loss Function](examples/custom_loss.ipynb)

Find more examples [here](examples).
