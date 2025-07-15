"""
This module provides data models for representing the various function outputs in the package including extraction,
reconstruction, clustering, and filtering.
"""

from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections.abc import Iterator
from warnings import warn
from numpy import ndarray, arange, sum
import cython

_COMPILED = cython.compiled


"""
Representations of trace and events
"""

@dataclass(slots=True)
class Trace:
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
    units: str = 'pA'
    path: Optional[str] = None
    time_step: Optional[float] = None
    samples: Optional[int] = None
    duration: Optional[float] = None


    def __post_init__(self):
        """
        Populate the Trace object with values if not provided.
        """

        if self.time_step is None:
            self.time_step = 1 / self.sample_rate
        if self.samples is None:
            self.samples = len(self.trace)
        if self.duration is None:
            self.duration = (self.samples - 1) * self.time_step


    def get_time_vector(self) -> ndarray:
        """
        Get the time vector for the trace.

        Returns:
            ndarray: The time vector for the trace.
        """

        return self.time_step * arange(self.samples)


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Trace object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the Trace object.
        """

        return asdict(self)


@dataclass(slots=True)
class Events:
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
    feature_type: str = field(default='Full')
    _ids: set[int] = field(default_factory=set, init=False, repr=False)
    _schema: Optional[set[str]] = field(default=None, init=False, repr=False)
    _req_keys: set[str] = field(default_factory=lambda: {'ID', 'Coordinates', 'Vector', 'Direction'}, init=False, repr=False)


    def __post_init__(self):
        """
        Post initialisation check
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
            raise ValueError(f"Key requirement mismatch.\nExpected: {sorted(self._req_keys)}\nReceived: {sorted(keys)}")

        if self._schema is None:
            self._schema = keys
        elif keys != self._schema:
            raise ValueError(f"Event schema mismatch.\nExpected: {sorted(self._schema)}\nReceived: {sorted(keys)}")

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
                raise ValueError(f"Filter function raised an error: {e}")

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
            raise ValueError("Cannot add feature with key 'ID'")
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


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Events object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the Events object.
        """

        return asdict(self)


"""
NotNormal results
"""

@dataclass(slots=True)
class InitialEstimateArgs:
    """
    A dataclass to represent the parameters for initial estimation. See: notnormal.extract.methods.initial_estimate.
    """

    sample_rate: Optional[int]
    estimate_cutoff: float
    replace_factor: float
    replace_gap: float
    threshold_window: float
    z_score: Optional[float]
    output_features: Optional[str]
    vector_results: bool
    _validate: bool


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the InitialEstimateArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the InitialEstimateArgs object.
        """

        return asdict(self)


@dataclass(slots=True)
class IterateArgs:
    """
    A dataclass to represent the parameters for initial estimation. See: notnormal.extract.methods.iterate.
    """

    cutoff: float
    event_direction: str
    sample_rate: Optional[int]
    replace_factor: float
    replace_gap: float
    threshold_window: float
    z_score: Optional[float]
    output_features: Optional[str]
    vector_results: bool
    _validate: bool


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the IterateArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the IterateArgs object.
        """

        return asdict(self)


@dataclass
class Iteration:
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Iteration object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the Iteration object.
        """
        return asdict(self)


@dataclass(slots=True)
class InitialEstimateResults:
    """
    A dataclass that represents the results of initial estimation on the trace.

    Attributes:
        args (InitialEstimateArgs): The arguments used for initial estimation.
        event_direction (str | None): The estimated event direction, either 'up' or 'down'. Default is None.
        max_cutoff (float | None): The estimated maximum cutoff that can be used for iteration. Default is None.
        initial_threshold (ndarray | None): The initial threshold from event detection. Default is None.
        events (Events): The final detected events. Default is None.
        iterations (list[Iteration2]): A list of Iteration2 objects representing each iteration. Default is an empty list.
    """

    args: InitialEstimateArgs
    event_direction: Optional[str] = None
    max_cutoff: Optional[float] = None
    initial_threshold: Optional[ndarray] = None
    events: Optional[Events] = None
    iterations: list[Iteration] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the InitialEstimateResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the InitialEstimateResults object.
        """

        return asdict(self)


@dataclass(slots=True)
class IterateResults:
    """
    A dataclass that represents the results of iteration on the trace.

    Attributes:
        args (IterateArgs): The arguments used for iteration.
        initial_threshold (ndarray | None): The initial threshold from event detection. Default is None.
        events (Events): The final detected events. Default is None.
        iterations (list[Iteration2]): A list of Iteration2 objects representing each iteration. Default is an empty list.

    """

    args: IterateArgs
    initial_threshold: Optional[ndarray] = None
    events: Optional[Events] = None
    iterations: list[Iteration] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the IterateResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the IterateResults object.
        """

        return asdict(self)


"""
Wavelet filter results
"""

@dataclass(slots=True)
class WaveletFilterArgs:
    """
    A dataclass to represent the parameters for wavelet filtering. See: notnormal.filter.methods.wavelet_filter.
    """

    wavelet: str
    u_length: int
    p_length: int
    q_pop: float
    q_thresh: float
    mode: str
    full_results: bool
    verbose: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the WaveletFilterArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the WaveletFilterArgs object.
        """

        return asdict(self)


@dataclass(slots=True)
class WaveletFilterResults:
    """
    A dataclass that represents the results of a wavelet filter applied to a trace.

    Attributes:
        args (WaveletFilterArgs): The arguments used for filtering.
        filtered_trace (ndarray): The filtered trace.
        max_level (int): The calculated maximum level of decomposition.
        lengths (ndarray): The event lengths used for maximum level calculation.
        signal_vars (ndarray): The estimated signal variance for each band.
        noise_vars (ndarray): The estimated noise variance for each band.
        thresholds (list[float]): The calculated threshold for each band. Default is None.
        coeffs (list[ndarray] | None): The wavelet coefficients for the trace. Default is None.
        coeffs_mask (list[ndarray] | None): The wavelet coefficients for the event mask. Default is None.
        filtered_coeffs (list[ndarray] | None): The filtered wavelet coefficients for the trace. Default is None.
    """

    args: WaveletFilterArgs
    filtered_trace: ndarray
    max_level: int
    lengths: ndarray
    signal_vars: ndarray
    noise_vars: ndarray
    thresholds: list[float]
    coeffs: Optional[list[ndarray]] = None
    coeffs_mask: Optional[list[ndarray]] = None
    filtered_coeffs: Optional[list[ndarray]] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the WaveletFilterResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the WaveletFilterResults object.
        """

        return asdict(self)


"""
Event clustering/reconstruction/augmentation results
"""

@dataclass(slots=True)
class ShapeClusterArgs:
    """
    A dataclass to represent the parameters for shape clustering. See: notnormal.reconstruct.events.shape_cluster.
    """

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


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ShapeClusteringArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the ShapeClusteringArgs object.
        """

        return asdict(self)


@dataclass(slots=True)
class EventAugmentationArgs:
    """
    A dataclass to represent the parameters for event augmentation. See: notnormal.reconstruct.events.augment_clusters.
    """

    n_vectors: Optional[int]
    max_k: int
    weight: float
    random_state: Optional[int]
    max_attempts: int
    sample_global: bool
    dist_losses: bool
    verbose: bool


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the EventAugmentationArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the EventAugmentationArgs object.
        """

        return asdict(self)


@dataclass(slots=True)
class ShapeCluster:
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


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Cluster object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the Cluster object.
        """

        return asdict(self)


@dataclass(slots=True)
class ShapeClusters:
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


    def get_vectors(self, cluster_id: Optional[int] = None) -> list[ndarray]:
        """
        Get the event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray]: A list of event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [vectors for cluster in self.clusters for vectors in cluster.vectors]

        return self.clusters[cluster_id].vectors if 0 <= cluster_id < len(self.clusters) else []


    def get_representative(self, cluster_id: Optional[int] = None) -> list[ndarray]:
        """
        Get representative vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            ndarray: All representative vectors or only for the specified cluster.
        """

        if cluster_id is None:
            return [cluster.representative for cluster in self.clusters if cluster.representative is not None]

        return self.clusters[cluster_id].representative if 0 <= cluster_id < len(self.clusters) else []


    def get_normalised(self, cluster_id: Optional[int] = None) -> list[ndarray]:
        """
        Get the locally normalised event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            ndarray: An array of locally normalised event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [norm for cluster in self.clusters for norm in cluster.normalised if cluster.normalised is not None]

        return self.clusters[cluster_id].normalised if 0 <= cluster_id < len(self.clusters) else []


    def get_reconstructed(self, cluster_id: Optional[int] = None) -> list[ndarray]:
        """
        Get the reconstructed event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray]: A list of reconstructed event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [recon for cluster in self.clusters for recon in cluster.reconstructed if cluster.reconstructed is not None]

        return self.clusters[cluster_id].reconstructed if 0 <= cluster_id < len(self.clusters) else []


    def get_augmented(self, cluster_id: Optional[int] = None) -> list[ndarray]:
        """
        Get the augmented event vectors for all clusters or a specific cluster.

        Args:
            cluster_id (int | None): The ID of a specific cluster. Default is None.

        Returns:
            list[ndarray]: A list of augmented event vectors for all clusters or the specified cluster.
        """

        if cluster_id is None:
            return [aug for cluster in self.clusters for aug in cluster.augmented if cluster.augmented is not None]

        return self.clusters[cluster_id].augmented if 0 <= cluster_id < len(self.clusters) else []


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ShapeClusters object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the ShapeClusters object.
        """

        return asdict(self)


@dataclass(slots=True)
class ShapeClusterResults:
    """
    A dataclass to represent the results of shape clustering.

    Attributes:
        args (ShapeClusterArgs): The arguments used for clustering.
        fits (dict[int, ShapeClusters]): All clustering results keyed by the number of clusters.
        area_norm (ndarray | None): Globally area + length normalised event vectors. Default is None.
        length_norm (ndarray | None): Globally length normalised event vectors. Default is None.
        transform (ndarray | None): Globally transformed + normalised event vectors. Default is None.
        umap (Any | None): The UMAP object used for dimensionality reduction. Default is None.
        best_fit (ShapeClusters): The best clustering result. Default is None.
        loss_curves (dict[str, tuple[ndarray, ndarray]] | None): Loss curves for the clustering results. Default is None.
        knees (dict[str, ndarray] | None): The knee points for the clustering results. Default is None.
    """

    args: ShapeClusterArgs
    fits: dict[int, ShapeClusters]
    area_norm: Optional[ndarray] = None
    length_norm: Optional[ndarray] = None
    transform: Optional[ndarray] = None
    umap: Optional[Any] = None
    best_fit: Optional[ShapeClusters] = None
    loss_curves: Optional[dict[str, tuple[ndarray, ndarray]]] = None
    knees: Optional[dict[str, ndarray]] = None


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ShapeClusterResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the ShapeClusterResults object.
        """

        return asdict(self)


"""
Noise reconstruction/augmentation results
"""

@dataclass(slots=True)
class NoiseReconstructionArgs:
    """
    A dataclass to represent the parameters for noise reconstruction. See: notnormal.reconstruct.noise.reconstruct_noise.
    """

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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the NoiseReconstructionArgs object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the NoiseReconstructionArgs object.
        """

        return asdict(self)


@dataclass(slots=True)
class NoiseFitResults:
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


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the NoiseFitResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the NoiseFitResults object.
        """

        return asdict(self)


@dataclass(slots=True)
class NoiseReconstructResults:
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


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the NoiseReconstructionResults object to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the NoiseReconstructionResults object.
        """

        return asdict(self)
