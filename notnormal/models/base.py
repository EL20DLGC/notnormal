# Copyright (C) 2025-2026 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides data models for representing the various function outputs in the package including extraction,
reconstruction, clustering, and filtering.
"""

from __future__ import annotations
from notnormal.models.supers import BaseDataclass
from scipy.stats import norm
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from collections.abc import Iterator
from warnings import warn
from numpy import ndarray, arange, sum, asarray
import cython

_COMPILED = cython.compiled


"""
Representations of trace and events
"""

@dataclass(slots=True)
class Trace(BaseDataclass):
    """
    A dataclass to represent a trace.

    Attributes:
        label (str): The label for the trace.
        trace (ndarray): The trace data.
        sample_rate (int): The sample rate of the trace.
        units (str): The units of the trace. Default is 'pA'.
        path (str | None): The file path where the trace is stored. Default is None.
        time_step (float | None): The time step between samples in the trace. Default is None.
        samples (int | None): The number of samples in the trace. Default is None.
        duration (float | None): The duration of the trace in seconds. Default is None.
    """

    label: str
    trace: ndarray
    sample_rate: int
    units: str = field(default='pA')
    path: Optional[str] = None
    time_step: Optional[float] = None
    samples: Optional[int] = None
    duration: Optional[float] = None


    def _normalise(self):
        """
        Normalise input parameters.
        """

        # Ensure safety
        self.trace = asarray(self.trace, dtype=float)
        if (not self.trace.flags['C_CONTIGUOUS']) or (not self.trace.flags['OWNDATA']):
            self.trace = self.trace.copy(order='C')

        # Autofill time_step
        if self.time_step is None:
            self.time_step = 1 / self.sample_rate

        # Autofill samples
        if self.samples is None:
            self.samples = self.trace.shape[0]

        # Autofill duration
        if self.samples < 2:
            self.duration = 0
        if self.duration is None:
            self.duration = (self.samples - 1) * self.time_step


    def _validate(self):
        """
        Validate input parameters.
        """

        if len(self.trace) < 2:
            raise ValueError('Trace must have at least 2 samples.')

        if self.sample_rate <= 0:
            raise ValueError('Sample_rate must be a non-negative integer.')

        if self.units not in ['A', 'mA', 'uA', 'μA', 'nA', 'pA', 'fA', 'zA']:
            warn(f'Crazy units detected: {self.units}.')

        if not (0.99 < self.sample_rate * self.time_step < 1.01):
            raise ValueError('Sample_rate and time_step mismatch.')

        if self.samples != self.trace.shape[0]:
            raise ValueError('Samples must be the trace length.')

        if (self.samples - 1) * self.time_step != self.duration:
            raise ValueError('Duration and trace/sample_rate mismatch.')


    def get_time_vector(self) -> ndarray:
        """
        Get the time vector for the trace.

        Returns:
            ndarray: The time vector for the trace.
        """

        return self.time_step * arange(self.samples)


@dataclass(slots=True)
class Events(BaseDataclass):
    """
    A dataclass to represent a collection of events.

    Attributes:
        label (str): The label for the events.
        events (dict[int, dict[str, Any]]): A dictionary of event dictionaries keyed by event ID.
        feature_type (str): The type of features computed for the events. Default is 'Full'.
        _ids (set[int]): Internal attribute storing the used IDs.
        _schema (set[str] | None): Internal attribute storing the expected keys of the event schema.
        _req_keys (set[str]): Internal attribute storing the required keys for each event.
    """

    label: str
    events: dict[int, dict[str, Any]] = field(default_factory=dict)
    feature_type: str = field(default='full')
    _ids: set[int] = field(default_factory=set, init=False)
    _schema: Optional[set[str]] = field(default=None, init=False)
    _req_keys: set[str] = field(default_factory=lambda: {'ID', 'Coordinates', 'Vector', 'Direction'}, init=False)


    def _validate(self):
        """
        Validate input parameters.
        """

        if not self.events:
            return

        # Validate schema
        for event in self:
            self._check_schema(event)


    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate through the event dictionaries in the object.

        Yields:
            dict: An event dictionary.
        """

        return iter(self.events.values())


    def __len__(self) -> int:
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

        return self.events[event_id]


    def __setitem__(self, event_id: int, event: dict[str, Any]):
        """
        Set an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to set.
            event (dict[str, Any]): The event dictionary to set.
        """

        if event.get('ID') != event_id:
            raise ValueError("Replacement event must have the same ID as the target.")

        # Do not check ID if it is already in the events
        if event_id in self.events:
            self._check_schema(event, check_id=False)
            self.events[event_id] = event
        else:
            self.add(event)


    def _check_schema(self, event: dict[str, Any], check_id: bool = True):
        """
        Check the schema of an event dictionary and update the internal state.

        Args:
            event (dict[str, Any]): The event dictionary to check.
            check_id (bool): Whether to check for the 'ID' key. Default is True.
        """

        if not isinstance(event, dict):
            raise ValueError("Each event must be a dictionary.")
        keys = set(event.keys())
        if not self._req_keys.issubset(keys):
            raise ValueError(f"Key requirement mismatch.\nExpected: {sorted(self._req_keys)}\nReceived: {sorted(keys)}.")

        if self._schema is None:
            self._schema = keys
        elif keys != self._schema:
            raise ValueError(f"Event schema mismatch.\nExpected: {sorted(self._schema)}\nReceived: {sorted(keys)}.")

        if check_id:
            if event.get('ID') in self._ids:
                raise ValueError(f"Duplicate ID {event.get('ID')} not allowed.")
            self._ids.add(event.get('ID'))


    def add(self, event: dict[str, Any]):
        """
        Add an event dictionary to the events list.

        Args:
            event (dict[str, Any]): The event dictionary to add.
        """

        self._check_schema(event)
        self.events[event.get('ID')] = event


    def get(self, event_id: int) -> Optional[dict[str, Any]]:
        """
        Get an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to retrieve.

        Returns:
            dict[str, Any]: The event dictionary with the specified ID.
        """

        return self.events.get(event_id)


    def remove(self, event_id: int):
        """
        Remove an event dictionary by its ID.

        Args:
            event_id (int): The ID of the event dictionary to remove.
        """

        self.events.pop(event_id, None)
        self._ids.discard(event_id)


    def get_feature(
        self,
        key: str,
        event_ids: Optional[list[int]] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None
    ) -> list[Any]:
        """
        Get a feature from all event dictionaries, optionally filtering by event ID or a custom filter function.

        Args:
            key (str): The key of the feature to retrieve.
            event_ids (list[int] | None): A list of event IDs to retrieve the feature from. Default is None (all events).
            filter_fn (Callable[[dict], bool] | None): Optional callable that takes an event dict and returns True to include it. Default is None (no filtering).

        Returns:
            list[Any]: A list of feature values.
        """

        # Select events by ID if specified, else all
        if event_ids is not None:
            events = [self.events[idx] for idx in event_ids if idx in self.events]
        else:
            events = list(self.events.values())

        if filter_fn is not None:
            try:
                events = [e for e in events if filter_fn(e)]
            except Exception as e:
                raise ValueError(f"Filter function raised an error: {e}.")

        return [e.get(key) for e in events]


    def get_features(
        self,
        keys: list[str],
        event_ids: Optional[list[int]] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None
    ) -> dict[str, list[Any]]:
        """
        Get multiple features from all event dictionaries, optionally filtering by event ID or a custom filter function.

        Args:
            keys (list[str]): A list of keys for the features to retrieve.
            event_ids (list[int] | None): A list of event IDs to retrieve the features from. Default is None (all events).
            filter_fn (Callable[[dict], bool] | None): Optional callable that takes an event dict and returns True to include it. Default is None (no filtering).

        Returns:
            dict[str, list[Any]]: A dictionary where keys are feature names and values are lists of feature values.
        """

        return {key: self.get_feature(key, event_ids, filter_fn) for key in keys}


    def add_feature(self, key: str, value: ndarray):
        """
        Add a feature to all event dictionaries.

        Args:
            key (str): The key of the feature to add.
            value (ndarray): The values of the feature to add.
        """

        if key == 'ID':
            raise ValueError("Cannot add feature with key 'ID'.")
        if len(self.events) != len(value):
            raise ValueError("Length of feature values must match number of events.")
        if self._schema and key in self._schema:
            warn(f"Overwriting existing feature '{key}' across events.")

        for event, val in zip(self, value):
            event[key] = val

        if self.events:
            self._schema = set(next(iter(self)))



    def add_features(self, features: dict[str, ndarray]):
        """
        Add multiple features to all event dictionaries.

        Args:
            features (dict[str, ndarray]): A dictionary of features to add.
        """

        for key, values in features.items():
            self.add_feature(key, values)


"""
NotNormal results
"""

@dataclass(slots=True)
class InitialEstimateArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for initial estimation. See: notnormal.extract.methods.initial_estimate.
    """

    trace: ndarray | Trace = field(metadata={"serialise": False}, repr=False)
    filtered_trace: Optional[ndarray] = field(metadata={"serialise": False}, repr=False)
    sample_rate: Optional[int]
    cutoff: float
    replace_factor: float
    replace_gap: float
    threshold_window: float
    z_score: Optional[float]
    output_features: Optional[str]
    vector_results: bool
    parallel: bool
    segment_size: Optional[int]


    def _normalise(self):
        """
        Normalise input parameters.
        """

        # Trace or ndarray input
        if isinstance(self.trace, Trace):
            self.sample_rate, self.trace = self.trace.sample_rate, self.trace.trace
        else:
            # Ensure safety
            self.trace = asarray(self.trace, dtype=float)
            if (not self.trace.flags['C_CONTIGUOUS']) or (not self.trace.flags['OWNDATA']):
                self.trace = self.trace.copy(order='C')

        # Use the normal trace if no bounding trace is supplied
        if self.filtered_trace is None:
            self.filtered_trace = self.trace
        else:
            # Ensure safety
            self.filtered_trace = asarray(self.filtered_trace, dtype=float)
            if (not self.filtered_trace.flags['C_CONTIGUOUS']) or (not self.filtered_trace.flags['OWNDATA']):
                self.filtered_trace = self.filtered_trace.copy(order='C')

        # Default to 1 expected outlier per trace (computed on length, of course)
        if self.z_score is None:
            self.z_score = float(norm.ppf(1.0 - ((1.0 / len(self.trace)) / 2.0)))


    def _validate(self):
        """
        Validate input parameters.
        """

        if self.trace.shape[0] < 2:
            raise ValueError('Trace must have at least 2 samples.')

        if self.trace.shape != self.filtered_trace.shape:
            raise ValueError("Trace and filtered_trace must have the same shape.")

        if self.sample_rate is None:
            raise ValueError("Sample_rate must be provided if trace is a ndarray.")

        if self.sample_rate <= 0:
            raise ValueError('Sample_rate must be a non-negative integer.')

        if not (0 < self.cutoff <= self.sample_rate // 2):
            raise ValueError("Cutoff frequency must be between 0 and sample_rate // 2.")

        if self.replace_gap < 0 or self.replace_factor < 0:
            raise ValueError("Replace_factor and replace_gap must be non-negative.")

        if self.replace_factor == 0 and self.replace_gap > 0:
            raise ValueError("Replace-factor cannot be 0 if replace_gap is greater than 0.")

        if self.threshold_window <= 0:
            raise ValueError("Threshold window must be greater than 0.")

        if int(self.threshold_window * self.sample_rate) > self.trace.shape[0]:
            raise ValueError("Threshold window cannot be greater than the length of trace.")

        if self.z_score <= 0:
            raise ValueError("Z_score must be greater than 0.")

        if self.output_features is not None and self.output_features not in ['full', 'FWHM', 'FWQM']:
            raise ValueError("Output_features must be 'full', 'FWHM', or 'FWQM'.")

        if self.segment_size is not None and (self.segment_size <= 0 or self.segment_size > self.trace.shape[0]):
            raise ValueError("Segment_size must be greater than 0 and smaller than the length of trace.")


    def get_func_args(self) -> tuple[ndarray, ndarray, dict[str, any], dict[str, any], dict[str, any]]:
        """
        Package and return arguments for _baseline_threshold and _locate_replace
        """

        bl_args = {'cutoff': self.cutoff, 'sample_rate': self.sample_rate, 'z_score': self.z_score,
                   'threshold_window': int(self.threshold_window * self.sample_rate)}
        lr_args = {'replace_factor': self.replace_factor, 'replace_gap': self.replace_gap}
        gen_args = {'output_features': self.output_features, 'vector_results': self.vector_results,
                    'parallel': self.parallel, 'segment_size': self.segment_size}

        return self.trace, self.filtered_trace, bl_args, lr_args, gen_args


@dataclass(slots=True)
class IterateArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for initial estimation. See: notnormal.extract.methods.iterate.
    """

    trace : ndarray | Trace = field(metadata={"serialise": False}, repr=False)
    cutoff: float
    event_direction: str
    filtered_trace: Optional[ndarray] = field(metadata={"serialise": False}, repr=False)
    sample_rate: Optional[int]
    replace_factor: float
    replace_gap: float
    threshold_window: float
    z_score: Optional[float]
    output_features: Optional[str]
    vector_results: bool
    parallel: bool
    segment_size: Optional[int]


    def _normalise(self):
        """
        Normalise input parameters.
        """

        # Trace or ndarray input
        if isinstance(self.trace, Trace):
            self.sample_rate, self.trace = self.trace.sample_rate, self.trace.trace
        else:
            # Ensure safety
            self.trace = asarray(self.trace, dtype=float)
            if (not self.trace.flags['C_CONTIGUOUS']) or (not self.trace.flags['OWNDATA']):
                self.trace = self.trace.copy(order='C')

        # Use the normal trace if no bounding trace is supplied
        if self.filtered_trace is None:
            self.filtered_trace = self.trace
        else:
            # Ensure safety
            self.filtered_trace = asarray(self.filtered_trace, dtype=float)
            if (not self.filtered_trace.flags['C_CONTIGUOUS']) or (not self.filtered_trace.flags['OWNDATA']):
                self.filtered_trace = self.filtered_trace.copy(order='C')

        # Default to 1 expected outlier per trace (computed on length, of course)
        if self.z_score is None:
            self.z_score = float(norm.ppf(1.0 - ((1.0 / len(self.trace)) / 2.0)))


    def _validate(self):
        """
        Validate input parameters.
        """

        if self.trace.shape[0] < 2:
            raise ValueError('Trace must have at least 2 samples.')

        if self.trace.shape != self.filtered_trace.shape:
            raise ValueError("Trace and filtered_trace must have the same shape.")

        if self.sample_rate is None:
            raise ValueError("Sample_rate must be provided if trace is a ndarray.")

        if self.sample_rate <= 0:
            raise ValueError('Sample_rate must be a non-negative integer.')

        if not (0 < self.cutoff <= self.sample_rate // 2):
            raise ValueError("Cutoff frequency must be between 0 and sample_rate // 2.")

        if self.event_direction not in ['up', 'down', 'biphasic']:
            raise ValueError("Event_direction must be 'up', 'down', or 'biphasic'.")

        if self.replace_gap < 0 or self.replace_factor < 0:
            raise ValueError("Replace_factor and replace_gap must be non-negative.")

        if self.replace_factor == 0 and self.replace_gap > 0:
            raise ValueError("Replace-factor cannot be 0 if replace_gap is greater than 0.")

        if self.threshold_window <= 0:
            raise ValueError("Threshold window must be greater than 0.")

        if int(self.threshold_window * self.sample_rate) > self.trace.shape[0]:
            raise ValueError("Threshold window cannot be greater than the length of trace.")

        if self.z_score <= 0:
            raise ValueError("Z_score must be greater than 0.")

        if self.output_features is not None and self.output_features not in ['full', 'FWHM', 'FWQM']:
            raise ValueError("Output_features must be 'full', 'FWHM', or 'FWQM'.")

        if self.segment_size is not None and (self.segment_size <= 0 or self.segment_size > self.trace.shape[0]):
            raise ValueError("Segment_size must be greater than 0 and smaller than the length of trace.")


    def get_func_args(self) -> tuple[ndarray, ndarray, dict[str, any], dict[str, any], dict[str, any]]:
        """
        Package and return arguments for _baseline_threshold and _locate_replace
        """

        bl_args = {'cutoff': self.cutoff, 'sample_rate': self.sample_rate, 'z_score': self.z_score,
                   'threshold_window': int(self.threshold_window * self.sample_rate)}
        lr_args = {'event_direction': self.event_direction, 'replace_factor': self.replace_factor,
                   'replace_gap': self.replace_gap}
        gen_args = {'output_features': self.output_features, 'vector_results': self.vector_results,
                    'parallel': self.parallel, 'segment_size': self.segment_size}

        return self.trace, self.filtered_trace, bl_args, lr_args, gen_args


@dataclass
class Iteration(BaseDataclass):
    """
    A dataclass to represent an iteration of event detection and baseline determination.

    Attributes:
        calculation_trace (ndarray | None): The trace used for baseline and threshold calculations. Default is None.
        baseline (ndarray | None): The baseline of the trace. Default is None.
        threshold (ndarray | None): The threshold for event detection. Default is None.
        trace_stats (dict[str, float] | None): Statistics for the baseline adjusted calculation trace.
        event_coordinates (ndarray | None): The coordinates of detected events. Default is None.
        event_stats (dict[str, float | int] | None): Population statistics for the detected events.
    """

    calculation_trace: Optional[ndarray] = None
    baseline: Optional[ndarray] = None
    threshold: Optional[ndarray] = None
    trace_stats: Optional[dict[str, Any]] = None
    event_coordinates: Optional[ndarray] = None
    event_stats: Optional[dict[str, float | int]] = None


@dataclass(slots=True)
class InitialEstimateResults(BaseDataclass):
    """
    A dataclass that represents the results of initial estimation on the trace.

    Attributes:
        args (InitialEstimateArgs | None): The arguments used for initial estimation. Default is None.
        event_direction (str | None): The estimated event direction, either 'up' or 'down'. Default is None.
        max_cutoff (float | None): The estimated maximum cutoff that can be used for iteration. Default is None.
        initial_threshold (ndarray | None): The initial threshold from event detection. Default is None.
        events (Events | None): The final detected events. Default is None.
        iterations (list[Iteration] | None): A list of Iteration objects representing each iteration. Default is None.
    """

    args: Optional[InitialEstimateArgs] = None
    event_direction: Optional[str] = None
    max_cutoff: Optional[float] = None
    initial_threshold: Optional[ndarray] = None
    events: Optional[Events] = None
    iterations: Optional[list[Iteration]] = None


@dataclass(slots=True)
class IterateResults(BaseDataclass):
    """
    A dataclass that represents the results of iteration on the trace.

    Attributes:
        args (IterateArgs | None): The arguments used for iteration. Default is None.
        max_cutoff (float | None): The estimated maximum cutoff that can be used for iteration. Default is None.
        initial_threshold (ndarray | None): The initial threshold from event detection. Default is None.
        events (Events | None): The final detected events. Default is None.
        iterations (list[Iteration] | None): A list of Iteration objects representing each iteration. Default is None.
    """

    args: Optional[IterateArgs] = None
    max_cutoff: Optional[float] = None
    initial_threshold: Optional[ndarray] = None
    events: Optional[Events] = None
    iterations: Optional[list[Iteration]] = None


"""
Wavelet filter results
"""

@dataclass(slots=True)
class WaveletFilterArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for wavelet filtering. See: notnormal.filter.methods.wavelet_filter.
    """

    trace: ndarray | Trace = field(metadata={"serialise": False}, repr=False)
    events: Events = field(metadata={"serialise": False}, repr=False)
    wavelet: str
    u_length: int
    p_length: int
    q_pop: float
    q_thresh: float
    mode: str
    full_results: bool
    verbose: bool


    def _normalise(self):
        """
        Normalise input parameters.
        """

        # Trace or ndarray input
        if isinstance(self.trace, Trace):
            self.trace = self.trace.trace
        else:
            # Ensure safety
            self.trace = asarray(self.trace, dtype=float)
            if (not self.trace.flags['C_CONTIGUOUS']) or (not self.trace.flags['OWNDATA']):
                self.trace = self.trace.copy(order='C')


    def _validate(self):
        """
        Validate input parameters.
        """

        if self.trace.shape[0] < 2:
            raise ValueError('Trace must have at least 2 samples.')

        if len(self.events) < 2:
            raise ValueError('Events must have at least 2 events.')

        if self.wavelet not in ('haar', 'db2', 'db4', 'db6', 'db8', 'sym2', 'sym4', 'sym6', 'sym8', 'coif1',
                                'coif3', 'coif5', 'bior1.3', 'bior2.2', 'bior3.5', 'bior4.4', 'rbio1.3', 'rbio2.2',
                                'rbio3.5', 'rbio4.4'):
            raise ValueError('Wavelet must be supported by PyWavelets.')

        if self.u_length < 1 or (self.u_length % 2 and self.u_length != 1):
            raise ValueError("u_length must be a power of two (including 1).")

        if self.p_length < 1 or (self.p_length % 2 and self.p_length != 1):
            raise ValueError("p_length must be a power of two (including 1).")

        if not (0 <= self.q_pop <= 1):
            raise ValueError("q_pop must be in the range [0, 1].")

        if not (0 < self.q_thresh < 1):
            raise ValueError("q_thresh must be in the range (0, 1).")

        if self.mode not in ('soft', 'hard', 'garrote', 'greater', 'less'):
            raise ValueError("mode must be either 'soft', 'hard', 'garrote', 'greater', or 'less'.")


@dataclass(slots=True)
class WaveletFilterResults(BaseDataclass):
    """
    A dataclass that represents the results of a wavelet filter applied to a trace.

    Attributes:
        args (WaveletFilterArgs | None): The arguments used for filtering. Default is None.
        filtered_trace (ndarray | None): The filtered trace. Default is None.
        max_level (int | None): The calculated maximum level of decomposition. Default is None.
        lengths (ndarray | None): The event lengths used for maximum level calculation. Default is None.
        signal_vars (ndarray | None): The estimated signal variance for each band. Default is None.
        noise_vars (ndarray | None): The estimated noise variance for each band. Default is None.
        thresholds (list[float] | None): The calculated threshold for each band. Default is None.
        coeffs (list[ndarray] | None): The wavelet coefficients for the trace. Default is None.
        coeffs_mask (list[ndarray] | None): The wavelet coefficients for the event mask. Default is None.
        filtered_coeffs (list[ndarray] | None): The filtered wavelet coefficients for the trace. Default is None.
    """

    args: Optional[WaveletFilterArgs] = None
    filtered_trace: Optional[ndarray] = None
    max_level: Optional[int] = None
    lengths: Optional[ndarray] = None
    signal_vars: Optional[ndarray] = None
    noise_vars: Optional[ndarray] = None
    thresholds: Optional[list[float]] = None
    coeffs: Optional[list[ndarray]] = None
    coeffs_mask: Optional[list[ndarray]] = None
    filtered_coeffs: Optional[list[ndarray]] = None


"""
Event clustering/reconstruction/augmentation results
"""

@dataclass(slots=True)
class ShapeClusterArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for shape clustering. See: notnormal.reconstruct.events.shape_cluster.
    """

    vectors: list[ndarray] | ndarray | Events = field(metadata={"serialise": False}, repr=False)
    k: int | list[int]
    direction: Optional[str]
    n_components: int
    metric: str
    spread: float
    min_dist: float
    n_neighbors: int
    random_state: Optional[int]
    init_params: str
    max_iter: int
    n_init: int
    reconstruct: bool
    sigma: float
    align: bool
    dist_losses: bool
    full_results: bool
    verbose: bool


@dataclass(slots=True)
class EventAugmentationArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for event augmentation. See: notnormal.reconstruct.events.augment_clusters.
    """

    clusters: ShapeClusters = field(metadata={"serialise": False}, repr=False)
    n_vectors: Optional[int]
    max_k: int
    weight: float
    random_state: Optional[int]
    max_attempts: int
    sample_global: bool
    dist_losses: bool
    verbose: bool


@dataclass(slots=True)
class ShapeCluster(BaseDataclass):
    """
    A dataclass to represent a single cluster of event vectors.

    Attributes:
        label (int): The cluster label.
        confidence (float): Average assignment confidence.
        vectors (list[ndarray]): The original event vectors assigned to this cluster.
        weights (ndarray): The assignment probabilities for each event vector to this cluster.
        representative (ndarray | None): The representative vector for the cluster. Default is None.
        normalised (ndarray | None): The cluster specific normalised vectors. Default is None.
        reconstructed (list[ndarray] | None): Reconstructed event vectors. Default is None.
        loss (dict[str, float] | None): Reconstruction loss metrics.
        local_model (Any | None): The local scaling feature model. Default is None.
        sampled_durations (ndarray | list[int] | None): The sampled durations. Default is None.
        sampled_areas (ndarray | list[float] | None): The sampled areas. Default is None.
        augmented (list[ndarray] | None): Augmented event vectors. Default is None.
    """

    label: int
    confidence: float
    vectors: list[ndarray]
    weights: ndarray
    representative: Optional[ndarray] = None
    normalised: Optional[ndarray] = None
    reconstructed: Optional[list[ndarray]] = None
    loss: Optional[dict[str, list[float]]] = None
    local_model: Optional[Any] = None
    sampled_durations: Optional[ndarray | list[int]] = None
    sampled_areas: Optional[ndarray | list[float]] = None
    augmented: Optional[list[ndarray]] = None


@dataclass(slots=True)
class ShapeClusters(BaseDataclass):
    """
    A dataclass to represent the output of shape clustering, possibly including reconstruction and augmentation.

    Attributes:
        k (int): The number of clusters.
        model (any): The GaussianMixture instance.
        bic (float): The Bayesian Information Criterion value of the GMM.
        labels (ndarray): The cluster labels assigned to each event vector.
        probabilities (ndarray): The full GMM assignment probability matrix (N x K).
        clusters (list[ShapeCluster]): List of ShapeCluster objects holding detailed cluster data.
        mean_loss (dict[str, float] | None): Global mean reconstruction loss metrics. Default is None.
        distributional_loss (dict[str, float] | None): Distributional loss metrics across clusters. Default is None.
        global_model (Any | None): The global model scaling feature model. Default is None.
        args (EventAugmentationArgs): The arguments used for event augmentation.
        sampled_model (ndarray | None): The sampled transformed + normalised event vectors. Default is None.
        sample_labels (ndarray | None): The sampled labels. Default is None.
        aug_distributional_loss (dict[str, float] | None): Augmentation distributional loss metrics across clusters. Default is None.
    """

    k: int
    model: Any
    bic: float
    labels: ndarray
    probabilities: ndarray
    clusters: list[ShapeCluster]
    mean_loss: Optional[dict[str, float]] = None
    distributional_loss: Optional[dict[str, float]] = None
    global_model: Optional[Any] = None
    aug_args: Optional[EventAugmentationArgs] = None
    sampled_model: Optional[ndarray] = None
    sampled_labels: Optional[ndarray] = None
    aug_distributional_loss: Optional[dict[str, float]] = None


    def get_vectors(self, cluster_id: Optional[int] = None) -> list[ndarray] | ndarray:
        """
        Get the event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray] | ndarray: A list of event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [vectors for cluster in self.clusters for vectors in cluster.vectors]

        return self.clusters[cluster_id].vectors if 0 <= cluster_id < len(self.clusters) else []


    def get_representative(self, cluster_id: Optional[int] = None) -> list[ndarray] | ndarray:
        """
        Get representative vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray] | ndarray: All representative vectors or only for the specified cluster.
        """

        if cluster_id is None:
            return [cluster.representative for cluster in self.clusters if cluster.representative is not None]

        return self.clusters[cluster_id].representative if 0 <= cluster_id < len(self.clusters) else []


    def get_normalised(self, cluster_id: Optional[int] = None) -> list[ndarray] | ndarray:
        """
        Get the locally normalised event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray] | ndarray: An array of locally normalised event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [norm for cluster in self.clusters for norm in cluster.normalised if cluster.normalised is not None]

        return self.clusters[cluster_id].normalised if 0 <= cluster_id < len(self.clusters) else []


    def get_reconstructed(self, cluster_id: Optional[int] = None) -> list[ndarray] | ndarray:
        """
        Get the reconstructed event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray] | ndarray: A list of reconstructed event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [recon for cluster in self.clusters for recon in cluster.reconstructed if cluster.reconstructed is not None]

        return self.clusters[cluster_id].reconstructed if 0 <= cluster_id < len(self.clusters) else []


    def get_augmented(self, cluster_id: Optional[int] = None) -> list[ndarray] | ndarray:
        """
        Get the augmented event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray] | ndarray: A list of augmented event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [aug for cluster in self.clusters for aug in cluster.augmented if cluster.augmented is not None]

        return self.clusters[cluster_id].augmented if 0 <= cluster_id < len(self.clusters) else []


@dataclass(slots=True)
class ShapeClusterResults(BaseDataclass):
    """
    A dataclass to represent the results of shape clustering.

    Attributes:
        args (ShapeClusterArgs | None): The arguments used for clustering. Default is None.
        fits (dict[int, ShapeClusters] | None): All clustering results keyed by the number of clusters. Default is None.
        area_norm (ndarray | None): Globally area + length normalised event vectors. Default is None.
        length_norm (ndarray | None): Globally length normalised event vectors. Default is None.
        transform (ndarray | None): Globally transformed + normalised event vectors. Default is None.
        umap (Any | None): The UMAP object used for dimensionality reduction. Default is None.
        best_fit (ShapeClusters): The best clustering result. Default is None.
        loss_curves (dict[str, tuple[ndarray, ndarray]] | None): Loss curves for the clustering results. Default is None.
        knees (dict[str, ndarray] | None): The knee points for the clustering results. Default is None.
    """

    args: Optional[ShapeClusterArgs] = None
    fits: Optional[dict[int, ShapeClusters]] = None
    area_norm: Optional[ndarray] = None
    length_norm: Optional[ndarray] = None
    transform: Optional[ndarray] = None
    umap: Optional[Any] = None
    best_fit: Optional[ShapeClusters] = None
    loss_curves: Optional[dict[str, tuple[ndarray, ndarray]]] = None
    knees: Optional[dict[str, ndarray]] = None


"""
Noise reconstruction/augmentation results
"""

@dataclass(slots=True)
class NoiseReconstructionArgs(BaseDataclass):
    """
    A dataclass to represent the parameters for noise reconstruction. See: notnormal.reconstruct.noise.reconstruct_noise.
    """

    trace: ndarray | Trace = field(metadata={"serialise": False}, repr=False)
    event_mask: ndarray | Events = field(metadata={"serialise": False}, repr=False)
    aa_cutoff: int
    aa_order: int
    n_regimes: int | list[int]
    sample_rate: Optional[int]
    maxiter: tuple[int, int]
    popsize: int
    mutation: tuple[float, float]
    psd_period: Optional[float]
    nfft: Optional[int]
    generate: str | bool
    complex_gen: bool
    random_state: Optional[int]
    verbose: bool


@dataclass(slots=True)
class NoiseFitResults(BaseDataclass):
    """
    A dataclass to represent the results of noise fitting.

    Attributes:
        n_regimes (int): The number of regimes in the noise model.
        f (ndarray): The frequency array of the true noise used for fitting.
        pxx (ndarray): The power spectral density (PSD) array of the true noise used for fitting.
        p_filter (ndarray): The power response of the anti-aliasing filter used for fitting.
        cs (list[float]): The frequency domain numerators (directly related to variance) of the noise regimes.
        ms (list[float]): The frequency domain exponents (directly related to H) of the noise regimes.
        alphas (list[float]): The detrended fluctuation analysis (DFA) alpha exponents (directly related to H) of the noise regimes.
        SSLE (float): The SSLE (objective value) of the fit.
        success (bool): Whether the fitting was successful.
        global_opt (Any): The global optimisation result of the fit.
        local_opt (Any): The local optimisation result of the fit.
        SAM (float): The spectral angular map value of the fit.
        total (ndarray | None): The total generated noise. Default is None.
        regimes (dict[float, ndarray] | None): The individually generated noise regimes keyed by alpha value. Default is None.
    """

    n_regimes: int
    f: ndarray
    pxx: ndarray
    p_filter: ndarray
    cs: list[float]
    ms: list[float]
    alphas: list[float]
    SSLE: float
    success: bool
    global_opt: Any
    local_opt: Any
    SAM: Optional[float] = None
    total: Optional[ndarray] = None
    regimes: Optional[dict[float, ndarray]] = None


    def get_line_fit(self) -> tuple[ndarray, ndarray]:
        """
        Get the line fit for the noise model.

        Returns:
            tuple[ndarray, ndarray]: The line fit for the noise model and each individual component.
        """

        components = self.cs[:, None] / self.f[None, :] ** self.ms[:, None]
        return sum(components, axis=0) * self.p_filter, components


@dataclass(slots=True)
class NoiseReconstructResults(BaseDataclass):
    """
    A dataclass to represent the results of noise reconstruction.

    Attributes:
        args (NoiseReconstructionArgs): The arguments used for reconstruction.
        fits (dict[int, NoiseFitResults]): All noise fitting results keyed by the number of regimes.
        best_fit (NoiseFitResults | None): The best fitting results. Default is None.
        loss_curves (dict[str, tuple[ndarray, ndarray]] | None): Loss curves for the fitting results. Default is None.
    """

    args: NoiseReconstructionArgs
    fits: dict[int, NoiseFitResults]
    best_fit: Optional[NoiseFitResults] = None
    loss_curves: Optional[dict[str, tuple[ndarray, ndarray]]] = None
