# cython: infer_types=True

"""
This module provides classes for representing events and iterations of event detection and baseline determination
in (nano)electrochemical time series data.
"""

from typing import Optional
from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Events:
    """
    A class to represent a collection of events, used for the GUI.

    Attributes:
        label (str): The label for the events.
        events (list[dict], optional): A list of event dictionaries. Default is None.
    """

    label: str
    events: Optional[list[dict]] = None

    def __iter__(self):
        """
        Iterate through the event dictionaries in the object.

        Yields:
            dict: An event dictionary.
        """

        if not self.events:
            return
        for i in self.events:
            yield i

    def __len__(self):
        """
        Get the number of events.

        Returns:
            int: The number of events.
        """

        if self.events:
            return len(self.events)
        return 0

    def __getitem__(self, event_id: int):
        """
        Get an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to retrieve.

        Returns:
            dict: The event dictionary with the specified ID.
        """

        return self.get(event_id)

    def __setitem__(self, event_id: int, event: dict):
        """
        Set an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to set.
            event (dict): The event dictionary to set.
        """

        for i, e in enumerate(self.events):
            if e.get('ID') == event_id:
                self.events[i] = event
                return
        self.add(event)

    def add(self, event: dict):
        """
        Add an event dictionary to the events list.

        Args:
            event (dict): The event dictionary to add.
        """

        if self.events:
            self.events += [event]
        else:
            self.events = [event]

    def get(self, event_id: int):
        """
        Get an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to retrieve.

        Returns:
            dict: The event dictionary with the specified ID.
        """

        if not self.events:
            return

        event = [event for event in self if event.get('ID') == event_id]
        if event:
            return event[0]

    def remove(self, event_id):
        """
        Remove an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to remove.
        """

        if not self.events:
            return

        self.events = [event for event in self if event.get('ID') != event_id]

    def get_feature(self, key: str, event_id: list[int] = None):
        """
        Get a feature from all event dictionaries.

        Args:
            key (str): The key of the feature to retrieve.
            event_id (list[int], optional): A list of event IDs to retrieve the feature from. Default is None (all events).

        Returns:
            list: A list of feature values.
        """

        if not self.events:
            return

        if event_id is None:
            return [i.get(key) for i in self]

        return [i.get(key) for i in self if i.get('ID') in event_id]

    def add_feature(self, key: str, value: ndarray):
        """
        Add a feature to all event dictionaries.

        Args:
            key (str): The key of the feature to add.
            value (ndarray): The values of the feature to add.
        """

        if not self.events or len(self) != len(value):
            return

        for i, event in enumerate(self):
            event[key] = value[i]

    def add_features(self, features: dict):
        """
        Add multiple features to all event dictionaries.

        Args:
            features (dict): A dictionary of features to add.
        """

        for key, value in features.items():
            self.add_feature(key, value)


@dataclass
class Iteration:
    """
    A dataclass to represent an iteration of event detection and baseline determination.

    Attributes:
        label (str): The label for the iteration.
        args (dict): The arguments used in the iteration.
        trace (ndarray, optional): The input signal trace. Default is None.
        filtered_trace (ndarray, optional): The filtered version of the input trace. Default is None.
        baseline (ndarray, optional): The baseline of the trace. Default is None.
        threshold (ndarray, optional): The threshold for event detection. Default is None.
        initial_threshold (ndarray, optional): The first computed threshold. Default is None.
        calculation_trace (ndarray, optional): The trace used for calculations. Default is None.
        trace_stats (dict, optional): Statistics for the trace. Default is None.
        event_coordinates (ndarray, optional): The coordinates of detected events. Default is None.
        event_stats (dict, optional): Statistics for the detected events. Default is None.
        events (Events, optional): The events detected in the iteration. Default is None.
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
