#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "activeteacher", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    ext_modules = [
        # extension(
        #     "activeteacher._C",
        #     sources,
        #     include_dirs=include_dirs,
        #     define_macros=define_macros,
        #     extra_compile_args=extra_compile_args,
        # )
    ]

    return ext_modules


# For projects that are relative small and provide features that are very close
# to detectron2's core functionalities, we install them under detectron2.projects
PROJECTS = {}

setup(
    name="activeteacher",
    version=get_version(),
    author="Hunterj Lin",
    url="https://github.com/HunterJ-Lin/ActiveTeacher",
    description="SSOD is next-generation research "
    "platform for object detection and segmentation.",
    packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    package_dir=PROJECTS,
    package_data={},
    python_requires=">=3.6",
    install_requires=["detectron2>=0.5"],
    # ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
