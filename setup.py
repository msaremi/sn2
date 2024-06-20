from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
        'build_ext': BuildExtension,
    },
    install_requires=[
       'torch>=1.10.1+cu102',
    ]
)
