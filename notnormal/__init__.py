# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
.. include:: ../README.md
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("notnormal")
except PackageNotFoundError:
    __version__ = "0.0.0"
