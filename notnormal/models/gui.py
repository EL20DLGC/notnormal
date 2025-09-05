# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides data models specifically for the GUI.
"""

from typing import Any
from dataclasses import dataclass, asdict
import cython

_COMPILED = cython.compiled


"""
Options classes (GUI usage)
"""

@dataclass
class FigureOptions:

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the FigureOptions object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the FigureOptions object.
        """

        return asdict(self)


    def reset(self):
        """
        Reset object.
        """

        self.__init__()
