"""
This module provides utility functions for (nano)electrochemical time series data.
"""

from typing import Optional, Any
from scipy.signal import welch
from numpy import zeros, ndarray, asarray, argmin, loadtxt, round
from numpy.random import default_rng
from notnormal.models.base import Trace, Events
from csv import Sniffer, reader
from pyabf import ABF
from pathlib import Path
from warnings import warn
import cython

_COMPILED = cython.compiled


"""
Public API
"""

def load_trace(path: str, label: Optional[str] = None) -> Trace:
    """
    Load a trace from its path. Currently, it supports Axon binary format (.abf) and separated values formats
    (.csv, .tsv, .txt, .dat). The expected format for separated values files is: | Time (s) | Current (pA) |, with or
    without a header. Support for FAST5/SLOW5 etc. will be added on request.

    Args:
        path (str): Path to the trace file.
        label (Optional[str]): Label for the trace. If not provided, the file name without extension is used. Default is None.

    Returns:
        Trace: A Trace object containing the trace and trace information.
    """

    # Check if the path exists and determine the file type
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    ext = path_obj.suffix.lower()

    # Axon binary format
    if ext == '.abf':
        data = ABF(path)
        time_vector = data.sweepX
        trace = data.sweepY
        units = getattr(data, "sweepUnitsY", "pA")
    # Separated format
    elif ext in {'.csv', '.tsv', '.txt', '.dat'}:
        # Determine the dialect and header of the SV file
        dialect, has_header = __detect_file_props(path)
        trace_data = loadtxt(path, delimiter=dialect.delimiter, skiprows=1 if has_header else 0)
        time_vector = trace_data[:, 0]
        trace = trace_data[:, 1]
        units = 'pA'
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for '{path}'.")

    if len(time_vector) < 2:
        raise ValueError("Trace must have at least two time points.")

    # Ensure safety
    trace = asarray(trace, dtype=float)
    if (not trace.flags['C_CONTIGUOUS']) or (not trace.flags['OWNDATA']):
        trace = trace.copy(order='C')

    # Convert to Trace obj
    trace = Trace(
        label=label if label else path_obj.stem,
        trace=trace,
        sample_rate=int(round(1.0 / (time_vector[1] - time_vector[0]))),
        units=units
    )

    return trace


def get_psd(
    trace: ndarray | Trace,
    sample_rate: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
) -> tuple[ndarray, ndarray]:
    """
    Compute the power spectral density (PSD) of a given trace using Welch's method.

    Args:
        trace (ndarray | Trace): The trace to be processed or a Trace object.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        fmin (float | None): The minimum frequency for the PSD. Default is None.
        fmax (float | None): The maximum frequency for the PSD. Default is None.
        psd_period (float | None): The period for the PSD calculation. Default is None.
        nfft (int | None): The length of FFT to use. Default is None.
    Returns:
        tuple[ndarray, ndarray]: The frequency array and PSD array for the trace.
    """

    if isinstance(trace, Trace):
        sample_rate, trace = trace.sample_rate, trace.trace
    if sample_rate is None:
        raise ValueError("Sample rate must be provided if trace is a ndarray.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    # Set the period if not provided
    if psd_period is None:
        length = len(trace)
        if length > (sample_rate * 10.0):
            psd_period = 2.0
        elif length < (sample_rate * 1.0):
            psd_period = length / sample_rate
        else:
            psd_period = 1.0

    if int(psd_period * sample_rate) > len(trace) or psd_period <= 0:
        raise ValueError("psd_period * sample_rate must be a positive value that does not exceed the length of the trace.")

    # Set the support if not provided
    if fmin is None:
        fmin = 1 / psd_period
    if fmax is None:
        fmax = sample_rate / 2
    if fmin < 0 or fmax <= fmin:
        raise ValueError("fmin must be non-negative and fmax must be greater than fmin.")

    # Set nfft
    if nfft is not None and nfft < int(psd_period * sample_rate):
        nfft = int(psd_period * sample_rate)

    # Calculate the PSD using Welch's method
    f, pxx = welch(trace, fs=sample_rate, nperseg=int(psd_period * sample_rate), nfft=nfft)

    # Truncate to support
    start_idx = argmin(abs(f - fmin))
    end_idx = argmin(abs(f - fmax))
    f = f[start_idx:end_idx]
    pxx = pxx[start_idx:end_idx]

    # Safety
    f[f == 1] = 1.001

    return f, pxx


def get_event_mask(trace: ndarray | Trace, coordinates: list[tuple[int, int]] | Events, float_mask: bool = False) -> ndarray:
    """
    Build a boolean or float (event locations inplace) event mask for the input trace. Note, the float mask is only
    baseline corrected if an Events object is provided, whereby the actual vector is inserted. Otherwise, the coordinates
    from the input trace are inserted.

    Args:
        trace (ndarray | Trace): The trace or a Trace object.
        coordinates (list[tuple[int, int]] | Events): List of tuple event coordinates or an Events object.
        float_mask (bool): If True, return a float mask instead of a boolean mask. Default is False.

    Returns:
        ndarray: Boolean or float event mask.
    """

    if isinstance(trace, Trace) and isinstance(coordinates, Events) and trace.label != coordinates.label:
        warn(f"Trace label '{trace.label}' does not match Events label '{coordinates.label}'.")

    if isinstance(trace, Trace):
        trace = trace.trace

    # Get vectors and coordinates from Events if provided
    if isinstance(coordinates, Events):
        vectors = coordinates.get_feature('Vector')
        coordinates = coordinates.get_feature('Coordinates')
    else:
        vectors = None

    # Build the event mask
    length = len(trace)
    event_mask = zeros(length, dtype=float if float_mask else bool)
    for i, (start, end) in enumerate(coordinates):
        if not (0 <= start <= end < length):
            raise ValueError(f"Event coordinates ({start}, {end}) are out of bounds for trace length {length}.")

        # Float mask or boolean mask
        if float_mask:
            event_mask[start:end + 1] = trace[start:end + 1] if vectors is None else vectors[i]
        else:
            event_mask[start:end + 1] = True

    return event_mask


def combine_noise_events(
    noise: ndarray,
    vectors: list[ndarray],
    coordinates: Optional[list[tuple[int, int]]  | Events] = None,
    max_attempts: int = 100,
    random_state: Optional[int] = None,
) -> tuple[ndarray, Optional[ndarray]]:
    """
    This function combines reconstructed/augmented noise (notnormal.reconstruct.reconstruct_noise) with reconstructed/
    augmented event vectors (notnormal.reconstruct.shape_cluster/augment_clusters) into a single trace. If coordinates
    are provided, the events are added at those specific locations. If no coordinates are provided, the function
    randomly places the events in the noise trace uniformly without replacement.

    Args:
        noise (ndarray): The reconstructed or augmented noise.
        vectors (list[ndarray]): The reconstructed or augmented event vectors.
        coordinates (list[tuple[int, int]] | Events): List of tuple event coordinates or an Events object. Default is None.
        max_attempts (int): If random placement, how many attempts to place the events before giving up. Default is 100.
        random_state (int | None): Random seed for reproducibility. Default is None.

    Returns:
        tuple[ndarray, Optional[ndarray]]: The noise with the events added and a boolean event mask.
    """

    if isinstance(coordinates, Events):
        coordinates = coordinates.get_feature('Coordinates')

    length = len(noise)
    trace = noise.copy()
    event_mask = zeros(length, dtype=bool)

    # Placing events at specific coordinates
    if coordinates:
        for i, (start, end) in enumerate(coordinates):
            if not (0 <= start <= end < length):
                raise ValueError(f"Event coordinates ({start}, {end}) are out of bounds for noise length {length}.")

            # Insert into trace and the event mask
            trace[start:end + 1] += vectors[i]
            event_mask[start:end + 1] = True

        return trace, event_mask

    # Randomly placing events in the noise trace
    rng = default_rng(random_state)
    attempts = 0
    i = 0

    while i < len(vectors):
        # Get random coordinates
        start = rng.integers(0, length - len(vectors[i]))
        end = (start + len(vectors[i])) - 1
        if not (0 <= start <= end < length):
            raise ValueError(f"Event coordinates ({start}, {end}) are out of bounds for noise length {length}.")

        # Check if the segment is already occupied
        if event_mask[start:end + 1].any():
            attempts += 1
            if attempts >= max_attempts:
                warn(f"Max attempts reached ({max_attempts}) for event {i}, skipping this event.")
                attempts = 0
                i += 1
            continue

        # Insert into trace and the event mask
        trace[start:end + 1] += vectors[i]
        event_mask[start:end + 1] = True
        attempts = 0
        i += 1

    return trace, event_mask


"""
Super Internal API
"""

def __detect_file_props(path: str) -> tuple[Any, bool]:
    """
    Detect the dialect and header presence of a given file separated values (.csv, .tsv, .txt, .dat) file.
    Note: This function is intended for internal use only.

    Args:
        path (str): Path to the trace file.

    Returns:
        tuple[Any, bool]: A tuple containing the dialect of the file and a boolean indicating if the file has a header.
    """

    with open(path, newline='') as csvfile:
        # Determine the dialect of the SV file
        dialect = Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        read = reader(csvfile, dialect)

        # Try to skip header (robust fallback)
        try:
            header = next(read)
        except StopIteration:
            raise ValueError("File is empty or malformed.")
        try:
            float(header[0])
            has_header = False
        except ValueError:
            has_header = True

    return dialect, has_header
