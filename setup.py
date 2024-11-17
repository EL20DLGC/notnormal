import os.path
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="notnormal",
    version="0.1.0",
    author="el20dlgc",
    author_email="el20dlgc@leeds.ac.uk",
    description="This package revolves around the NotNormal algorithm, which combines estimation and iteration to "
                "automatically extract events from (nano)electrochemical time series data.",
    long_description=long_description,
    packages=find_packages() + ["notnormal.data"],
    install_requires=[
        "numpy",
        "scipy",
        "stochastic",
        "ttkbootstrap",
        "matplotlib",
        "pyabf",
        "Cython"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Natural Language :: English"
    ],
    include_package_data=True,
    package_data={'': ['data/*']},
    ext_modules=cythonize(
        [
            Extension("notnormal.not_normal", ["notnormal/not_normal.py"],
                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]),
            Extension("notnormal.not_normal_gui", ["notnormal/not_normal_gui.py"],
                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]),
            Extension("notnormal.results", ["notnormal/results.py"],
                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
        ],
        compiler_directives={
            'language_level': 3,
        },
        quiet=True,
        show_all_warnings=False,
        annotate=False,
    )
)
