import os.path
from setuptools import find_packages, setup
from Cython.Build import cythonize


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="notnormal",
    version=get_version("version.py"),
    author="el20dlgc",
    author_email="el20dlgc@leeds.ac.uk",
    description="This package revolves around the NotNormal algorithm, which combines estimation and iteration to "
                "automatically extract events from (nano)electrochemical time series data.",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "stochastic",
        "ttkbootstrap",
        "matplotlib",
        "pyabf",
        "pandas",
        "Pillow",
        "Cython"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research'
    ],
    include_package_data=True,
    package_data={'': ['data/*']},
    ext_modules=cythonize(
        ["notnormal/not_normal.py", "notnormal/not_normal_gui.py"],
        compiler_directives={'language_level': 3},
        show_all_warnings=True,
        annotate=True
    ),
)
