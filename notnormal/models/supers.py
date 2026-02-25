# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides superclass data model for the entire package.
"""

from __future__ import annotations
from typing import Any, get_origin, get_args, Self
from dataclasses import dataclass, field, fields, MISSING
from copy import deepcopy
from numpy import empty
import cython

_COMPILED = cython.compiled


"""
Base dataclasses 
"""

@dataclass(slots=True)
class BaseDataclass:
    """
    A base dataclass for all arguments and results.

    Attributes:
        _is_empty (bool): Whether the data is empty. Default is False.
    """

    _is_empty: bool = field(default=False, init=False, repr=False, compare=False)


    def __post_init__(self):
        """
        Post initialisation check.
        """

        # Normal constructor path
        self.initialise()


    def initialise(self):
        """
        Initialise dataclass instance.
        """

        # Normalise input
        self._normalise()
        # Validate input
        self._validate()


    def _normalise(self):
        """
        Normalise input parameters.
        """


    def _validate(self):
        """
        Validate input parameters.
        """


    def to_dict(self) -> dict:
        """
        A method to serialise fields from a dataclass instance.

        Returns:
            dict: A dictionary of the dataclass fields.
        """

        result = {}
        for f in fields(self):
            if f.metadata.get("serialise") is False:
                continue
            if f.name.startswith("_"):
                continue
            result[f.name] = getattr(self, f.name)
        return result


    @classmethod
    def from_dict(cls, data: dict, **overrides: Any) -> Self:
        """
        A class method to create a dataclass instance with defaults derived from a dictionary.

        Attributes:
            data (dict): A dictionary of keyword arguments.
            **overrides (dict): A dictionary of keyword arguments.

        Returns:
            Self: A new dataclass instance filled with values derived from data.
        """

        # Bypass __init__ and __post_init__
        obj = cls.__new__(cls)

        # Fill all fields
        for f in fields(cls):
            if f.name == "_is_empty":
                setattr(obj, "_is_empty", False)
                continue
            if f.name in overrides:
                setattr(obj, f.name, overrides[f.name])
                continue

            if f.name in data:
                # Matching parameters
                val = deepcopy(data[f.name])
            else:
                # In dataclass but not data dict
                if f.default is not MISSING:
                    val = deepcopy(f.default)
                elif f.default_factory is not MISSING:
                    val = f.default_factory()
                else:
                    val = cls._empty_value(f)
            setattr(obj, f.name, val)

        return obj


    @classmethod
    def empty(cls, **overrides: Any) -> Self:
        """
        A class method to create an empty dataclass instance.

        Attributes:
            **overrides (dict): A dictionary of keyword arguments.

        Returns:
            Self: A new dataclass instance filled with empty values.
        """

        # Bypass __init__ and __post_init__
        obj = cls.__new__(cls)

        # Fill all the fields
        for f in fields(cls):
            if f.name == "_is_empty":
                setattr(obj, "_is_empty", True)
                continue
            if f.name in overrides:
                setattr(obj, f.name, overrides[f.name])
                continue

            if f.default is not MISSING:
                val = deepcopy(f.default)
            elif f.default_factory is not MISSING:
                val = f.default_factory()
            else:
                val = cls._empty_value(f)
            setattr(obj, f.name, val)

        return obj


    @classmethod
    def _empty_value(cls, f: Any) -> Any:
        """
        A class method to fill empty fields for a dataclass instance.

        Attributes:
            f (Any): A dataclass field.

        Returns:
            Any: An empty field.
        """

        # Typing constructs
        origin = get_origin(f.type) or f.type
        if type(None) in get_args(f.type):
            return None
        if origin in (list, set, dict, tuple):
            return origin()
        if origin is bool:
            return False
        if origin in (int, float):
            return 0
        if "ndarray" in str(f.type).lower():
            return empty(0, dtype=float)

        return None
