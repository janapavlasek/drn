#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    sources = ['drn/lib/src/batchnormp.c']
    headers = ['drn/lib/src/batchnormp.h']
    defines = []
    with_cuda = False

    abs_path = os.path.dirname(os.path.realpath(__file__))
    print(abs_path)
    extra_objects = [os.path.join(abs_path, 'drn/lib/dense/batchnormp_kernel.so')]
    extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

    extension = CppExtension
    extra_compile_args = {"cxx": []}

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        print('Including CUDA code.')
        sources += ['drn/lib/src/batchnormp_cuda.c']
        headers += ['drn/lib/src/batchnormp_cuda.h']
        defines += [('WITH_CUDA', None)]
        with_cuda = True
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    ext_modules = [
        extension(
            'drn.dense.batch_norm',
            headers=headers,
            sources=sources,
            define_macros=defines,
            relative_to=__file__,
            with_cuda=with_cuda,
            extra_objects=extra_objects,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="drn",
    version="0.1",
    description="Dilated Residual Networks",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
