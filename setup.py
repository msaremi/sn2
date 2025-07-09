import torch, shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

class ExtendedBuildExtension(BuildExtension):
    def _check_nvcc_exists(self):
        if shutil.which("nvcc") is None:
            raise RuntimeError(
                "ERROR: CUDA compiler 'nvcc' not found in PATH.\n"
                "Please install the CUDA Toolkit and ensure 'nvcc' is accessible.\n"
            )

    def _check_cuda_available(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ERROR: Installed PyTorch does NOT have CUDA support.\n"
                "Please install a CUDA-enabled PyTorch (e.g., torch==1.10.1+cu102).\n"
            )

    def build_extensions(self) -> None:
        self._check_nvcc_exists()
        self._check_cuda_available()
        super().build_extensions()


setup(
    name='sn2_cuda',
    version='0.9.0',
    ext_modules=[
        CUDAExtension('sn2_cuda', [
            'src/sn2_cuda.cpp',
            'src/sn2_solver_kernel.cu',
        ], extra_compile_args={
            'cxx': ['/std:c++17', '/permissive-'],
        })
    ],
    cmdclass={
        'build_ext': ExtendedBuildExtension,
    },
    install_requires=[
       'torch>=1.10.1',
    ]
)
