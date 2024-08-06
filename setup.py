import copy, os
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

requirements = open("requirements.txt").read().splitlines()

has_cuda = True
if not torch.cuda.is_available():
    has_cuda = False

def CustomCUDAExtension(*args, **kwargs):
    if not os.name == "nt":
        FLAGS = ["-Wno-deprecated-declarations"]
        kwargs = copy.deepcopy(kwargs)
        if "extra_compile_args" in kwargs:
            kwargs["extra_compile_args"] += FLAGS
        else:
            kwargs["extra_compile_args"] = FLAGS

    return CUDAExtension(*args, **kwargs)


class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        # ninja is interfering with compiling separate extensions in parallel
        kwargs["use_ninja"] = False
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        if not os.name == "nt":
            FLAG_BLACKLIST = ["-Wstrict-prototypes"]
            FLAGS = ["-Wno-deprecated-declarations"]
            self.compiler.compiler_so = [
                x for x in self.compiler.compiler_so if x not in FLAG_BLACKLIST
            ] + FLAGS  # Covers non-cuda

        super().build_extensions()

def get_extensions():
    cuda_extensions = []
    return cuda_extensions

kwargs = {}
if has_cuda:
        kwargs_cuda = {
            "ext_modules": get_extensions(),
            "cmdclass": {"build_ext": CustomBuildExtension},
        }
        kwargs = {**kwargs, **kwargs_cuda}
setuptools.setup(
    name="LBM-based 2D fluid solver",
    description="PyTorch package for LBM-based 2D fluid solver",
    author="Ge Cao",
    author_email="gecao2@illinois.edu",
    version="2.0.0",
    packages=["src"],
    install_requires=requirements,
    python_requires=">=3.7",
    **kwargs,
)
