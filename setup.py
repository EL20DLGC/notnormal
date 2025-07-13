import ast
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize


"""
Auto-generator for Cython extensions
"""

def uses_cython(file: Path) -> bool:
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "cython" for alias in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("cython"):
                    return True
    except Exception:
        pass
    return False


def discover_extensions():
    extensions = []
    for pyfile in Path("notnormal").rglob("*.py"):
        if pyfile.name == "__init__.py":
            continue
        if uses_cython(pyfile):
            modname = ".".join(pyfile.relative_to(".").with_suffix("").parts)
            extensions.append((modname, str(pyfile)))
    return extensions


ext_modules = cythonize(
    module_list=[
        Extension(name, [src], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
        for name, src in discover_extensions()
    ],
    compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'nonecheck': False,
        'cdivision': True,
        'infer_types': True,
        'annotation_typing': True
    },
    quiet=True,
    show_all_warnings=False,
    annotate=False,
    build_dir="build",
)


setup(ext_modules=ext_modules)
