from setuptools import setup, Extension
import pybind11

module = Extension(
    "qpproblem_smo_solver",
    sources=["./module/src/main.cpp"],
    include_dirs=[
        pybind11.get_include(),
        "./module/src/",
        "./module/src/include/",
        "./module/src/source/",
    ],
    language="c++",
)

setup(
    name="qpproblem_smo_solver",
    version="0.1",
    description="C++ module solving QP problem appearing while training Support Vector Machine (binary classification) with regularization parametr C",
    ext_modules=[module],
)
