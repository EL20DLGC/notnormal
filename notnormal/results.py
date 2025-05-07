# cython: infer_types=True

"""
This module provides classes for representing events and iterations of event detection and baseline determination
in (nano)electrochemical time series data.
"""

from typing import Any, Optional
from dataclasses import dataclass
from numpy import ndarray
import cython

COMPILED = cython.compiled

@dataclass
class Events:
    """
    A class to represent a collection of events, used for the GUI.

    Attributes:
        label (str): The label for the events.
        events (list[dict[str, Any] | None): A list of event dictionaries. Default is None.
    """

    label: str
    events: Optional[list[dict[str, Any]]] = None

    def __post_init__(self):
        """
        Post initialisation check
        """

        if self.events is None:
            self.events = []

    def __iter__(self):
        """
        Iterate through the event dictionaries in the object.

        Yields:
            dict: An event dictionary.
        """

        return iter(self.events)

    def __len__(self):
        """
        Get the number of events.

        Returns:
            int: The number of events.
        """

        return len(self.events)

    def __getitem__(self, event_id: int) -> Optional[dict[str, Any]]:
        """
        Get an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to retrieve.

        Returns:
            dict[str, Any] | None: The event dictionary with the specified ID.
        """

        return self.get(event_id)

    def __setitem__(self, event_id: int, event: dict[str, Any]):
        """
        Set an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to set.
            event (dict[str, Any]): The event dictionary to set.
        """

        for i, e in enumerate(self.events):
            if e.get('ID') == event_id:
                self.events[i] = event
                return
        self.add(event)

    def add(self, event: dict[str, Any]):
        """
        Add an event dictionary to the events list.

        Args:
            event (dict[str, Any): The event dictionary to add.
        """

        if 'ID' not in event:
            raise ValueError("Each event must contain an 'ID' key.")
        self.events.append(event)

    def get(self, event_id: int) -> Optional[dict[str, Any]]:
        """
        Get an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to retrieve.

        Returns:
            dict[str, Any]: The event dictionary with the specified ID.
        """

        return next((event for event in self.events if event.get('ID') == event_id), None)

    def remove(self, event_id: int):
        """
        Remove an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to remove.
        """

        self.events = [event for event in self.events if event.get('ID') != event_id]

    def get_feature(self, key: str, event_ids: Optional[list[int]] = None) -> list[Any]:
        """
        Get a feature from all event dictionaries.

        Args:
            key (str): The key of the feature to retrieve.
            event_ids (list[int] | None): A list of event IDs to retrieve the feature from. Default is None (all events).

        Returns:
            list[Any]: A list of feature values.
        """

        if event_ids is None:
            return [i.get(key) for i in self.events]

        return [i.get(key) for i in self.events if i.get('ID') in event_ids]

    def add_feature(self, key: str, value: ndarray):
        """
        Add a feature to all event dictionaries.

        Args:
            key (str): The key of the feature to add.
            value (ndarray): The values of the feature to add.
        """

        if len(self.events) != len(value):
            raise ValueError("Length of feature values must match number of events.")

        for i, event in enumerate(self.events):
            event[key] = value[i]

    def add_features(self, features: dict[str, ndarray]):
        """
        Add multiple features to all event dictionaries.

        Args:
            features (dict[str, ndarray]): A dictionary of features to add.
        """

        for key, values in features.items():
            self.add_feature(key, values)

@dataclass
class Iteration:
    """
    A dataclass to represent an iteration of event detection and baseline determination.

    Attributes:
        label (str): The label for the iteration.
        args (dict): The arguments used in the iteration.
        trace (ndarray | None): The input signal trace. Default is None.
        filtered_trace (ndarray | None): The filtered version of the input trace. Default is None.
        baseline (ndarray | None): The baseline of the trace. Default is None.
        threshold (ndarray | None): The threshold for event detection. Default is None.
        initial_threshold (ndarray | None): The first computed threshold. Default is None.
        calculation_trace (ndarray | None): The trace used for calculations. Default is None.
        trace_stats (dict | None): Statistics for the trace. Default is None.
        event_coordinates (ndarray | None): The coordinates of detected events. Default is None.
        event_stats (dict | None): Statistics for the detected events. Default is None.
        events (Events | None): The events detected in the iteration. Default is None.
    """

    label: str
    args: dict
    trace: Optional[ndarray] = None
    filtered_trace: Optional[ndarray] = None
    baseline: Optional[ndarray] = None
    threshold: Optional[ndarray] = None
    initial_threshold: Optional[ndarray] = None
    calculation_trace: Optional[ndarray] = None
    trace_stats: Optional[dict] = None
    event_coordinates: Optional[ndarray] = None
    event_stats: Optional[dict] = None
    events: Optional[Events] = None
