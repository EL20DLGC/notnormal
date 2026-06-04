# Copyright (C) 2025-2026 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides functions for anomaly detection and baseline determination in (nano)electrochemical time series
data.
"""

from numpy import quantile, sign, sort, concatenate, sqrt, round, ones, zeros, mean, ceil, cumsum, abs, atleast_1d, \
    percentile, ndarray, std, sum, max, median, asarray, array_split, double, int64, uint8, flatnonzero, empty, \
    bitwise_xor, greater_equal, less_equal, searchsorted, column_stack, full, minimum, maximum, roll, pad, argsort, int8
from scipy.signal import oaconvolve
from scipy.integrate import simpson
from scipy.ndimage import median_filter
from scipy.stats import norm, shapiro
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from notnormal.models.base import Iteration, Events, Trace, InitialEstimateArgs, InitialEstimateResults, IterateArgs, \
    IterateResults
from warnings import filterwarnings
from gc import collect
from psutil import virtual_memory, swap_memory
from typing import Optional
import cython

_COMPILED = cython.compiled
_DTYPE = double
__WINDOW_CACHE = {}

# Warnings from Shapiro p-value calculation (not needed)
filterwarnings("ignore", message=r".*scipy\.stats\.shapiro: For N > 5000.*", category=UserWarning)


"""
Public API
"""

def not_normal(
    trace: ndarray | Trace,
    sample_rate: Optional[int] = None,
    bounds_window: cython.int = 3,
    cutoff: cython.double = 10.0,
    replace_factor: cython.double = 8.0,
    replace_gap: cython.double = 2.0,
    threshold_window: cython.double = 2.0,
    z_score: Optional[float] = None,
    output_features: Optional[str] = 'full',
    vector_results: cython.bint = False,
    parallel: cython.bint = False,
    segment_size: Optional[int] = None,
) -> tuple[IterateResults, InitialEstimateResults]:
    """
    Anomaly detection and baseline determination for (nano)electrochemical time series data. This function combines
    initial_estimate and iterate steps into a single function. There are no required parameters other than the input
    trace and sample rate (or a Trace object). See: initial_estimate and iterate for further details.

    Args:
        trace (ndarray | Trace): The input trace or a Trace object. If ndarray, the sample rate must be provided.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        bounds_window (int): The window size for the median filter used in bounding. Default is 3.
        cutoff (float): The initial estimate for the cutoff frequency. Default is 10.0.
        replace_factor (float): Factor for replacing events in the calculation trace. Default is 8.0.
        replace_gap (float): Gap for replacing events in the calculation trace. Default is 2.0.
        threshold_window (float): The window size for threshold calculation. Default is 2.0.
        z_score (float | None): The z-score for event detection. If None, it is calculated based on the trace length.
            Default is None.
        output_features (str | None): The output event features. Can be 'full' for absolute features, 'FWHM', or 'FWQM'
            for full width at half maximum or quarter maximum, respectively. These apply to duration, area, and peak
            amplitude. None specifies no events on the output. Default is 'full'.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.
        parallel (bool): Whether to run the algorithm in parallel. Default is False.
        segment_size (int | None): Size of segments for parallel processing. Default is None and will calculate
            automatically.

    Returns:
        tuple[IterateResults, InitialEstimateResults]: The iteration results object and initial estimate results object.
    """

    # Default 3-point median filter for bounding
    filtered_trace = _bounds_filter(trace, bounds_window)

    # Initial estimate for the cutoff frequency and direction
    estimate_results = initial_estimate(trace, filtered_trace, sample_rate, cutoff, replace_factor, replace_gap,
                                        threshold_window, z_score, output_features, vector_results, parallel, segment_size)

    # Iterate based on the initial estimate
    iteration_results = iterate(trace, estimate_results.max_cutoff, estimate_results.event_direction, filtered_trace,
                                  sample_rate, replace_factor, replace_gap, threshold_window, z_score, output_features,
                                  vector_results, parallel, segment_size)

    return iteration_results, estimate_results


def initial_estimate(
    trace: ndarray | Trace,
    filtered_trace: Optional[ndarray] = None,
    sample_rate: Optional[int] = None,
    cutoff: cython.double = 10.0,
    replace_factor: cython.double = 8.0,
    replace_gap: cython.double = 2.0,
    threshold_window: cython.double = 2.0,
    z_score: Optional[float] = None,
    output_features: Optional[str] = 'full',
    vector_results: cython.bint = False,
    parallel: cython.bint = False,
    segment_size: Optional[int] = None,
) -> InitialEstimateResults:
    """
    Provides an initial estimate for the maximum cutoff frequency and event direction for the iterate function. The output
    events can also be used for wavelet filtering (notnormal.filter.methods.wavelet_filter). This is an improved version
    of the conventional event extraction method, using a robust sequence of iterations to make an initial estimate on
    the event population. From this, automatic iteration can be performed with the maximised cutoff frequency and event
    direction. The only required parameters are the input trace and sample_rate, with the cutoff frequency being the
    primary tuning parameter and ultimately decides how accurate the initial estimate is. Other parameters are provided
    for fine-tuning.

    Note: Both replace_factor and replace_gap as 0.0 indicates baseline replacement instead of individual replacement.

    Args:
        trace (ndarray | Trace): The input trace or a Trace object. If ndarray, the sample rate must be provided.
        filtered_trace (ndarray | None): The filtered version of the input trace, used for bounding events. If None,
            the trace is used. Default is None.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        cutoff (float): The initial estimate for the cutoff frequency. Default is 10.0.
        replace_factor (float): Factor for replacing events in the calculation trace. Default is 8.0.
        replace_gap (float): Gap for replacing events in the calculation trace. Default is 2.0.
        threshold_window (float): The window size for threshold calculation. Default is 2.0.
        z_score (float | None): The z-score for event detection. If None, it is calculated based on the trace length.
            Default is None.
        output_features (str | None): The output event features. Can be 'full' for absolute features, 'FWHM', or 'FWQM'
            for full width at half maximum or quarter maximum, respectively. These apply to duration, area, and peak
            amplitude. None specifies no events on the output. Default is 'full'.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.
        parallel (bool): Whether to run the initial estimate in parallel. Default is False.
        segment_size (int | None): Size of segments for parallel processing. Default is None and will calculate
            automatically.

    Returns:
        InitialEstimateResults: An object containing the initial estimate results.
    """

    # Init results and get args
    results = InitialEstimateResults(InitialEstimateArgs(**{k: v for k, v in locals().items() if k in InitialEstimateArgs.__annotations__}))
    args = results.args.get_func_args()

    # Run iterations
    if parallel:
        results.iterations, results.initial_threshold, results.event_direction = _parallel_estimate(*args)
    else:
        results.iterations, results.initial_threshold, results.event_direction = _serial_estimate(*args)

    # Finalise events
    results = _finalise_events(results)

    return results


def iterate(
    trace: ndarray | Trace,
    cutoff: cython.double,
    event_direction: str,
    filtered_trace: Optional[ndarray] = None,
    sample_rate: Optional[int] = None,
    replace_factor: cython.double = 8.0,
    replace_gap: cython.double = 2.0,
    threshold_window: cython.double = 2.0,
    z_score: Optional[float] = None,
    output_features: Optional[str] = 'full',
    vector_results: cython.bint = False,
    parallel: cython.bint = False,
    segment_size: Optional[int] = None,
) -> IterateResults:
    """
    Performs iterative anomaly detection and baseline determination based on the input (maximised) cutoff frequency and
    event direction deduced from initial_estimate. This function is an advanced version of the iterative method, built
    to be automatic when paired with the initial estimate. It uses various mechanisms to circumvent the downfalls
    of the maximised cutoff frequency, individual local replacement, automatic influence calculation, and an automatic
    stopping criterion, among other advancements. The only required parameters are the input trace, sample_rate, cutoff,
    and event_direction. The primary tuning parameters are the replace_factor and replace_gap, which determine how
    aggressive the individual replacement is and how much the event samples influence the replacement. When Both
    replace_factor and replace_gap are 0.0, baseline replacement is done instead of individual replacement. Other
    parameters are provided for fine-tuning.

    Note on the final extraction after iteration: When the stopping criterion is met, the (calc_trace - baseline) used
    in the previous iteration was the best for normality, so we use the event coordinates determined from that iteration
    (compared vs. best normality). The replacement of those coordinates then indicates that all events are replaced,
    so we use the calc_trace and baseline determined from this replacement. In other words, the calc_trace and baseline
    from the next iteration which triggered the stopping criterion. We then replace event locations in the baseline with
    the calc_trace, as this is a locally optimal baseline once we are sure all events have been replaced. Then, a final
    threshold is computed using the std method (more accurate without outliers) on the calc_trace - baseline, which is
    now a fully removed trace. Thus, the final extraction is computed on the locally optimal baseline and threshold.

    Args:
        trace (ndarray | Trace): The input trace or a Trace object. If ndarray, the sample rate must be provided.
        cutoff (float): The cutoff frequency for the baseline filter derived from initial_estimate.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic') derived from initial_estimate.
        filtered_trace (ndarray | None): The filtered version of the input trace, used for bounding events.
            If None, the trace is used. Default is None.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        replace_factor (float): Factor for replacing events in the calculation trace. Default is 8.0.
        replace_gap (float): Gap for replacing events in the calculation trace. Default is 2.0.
        threshold_window (float): The window size for threshold calculation. Default is 2.0.
        z_score (float | None): The z-score for event detection. If None, it is calculated based on the trace length.
            Default is None.
        output_features (str | None): The output event features. Can be 'full' for absolute features, 'FWHM', or 'FWQM'
            for full width at half maximum or quarter maximum, respectively. These apply to duration, area, and peak
            amplitude. None specifies no events on the output. Default is 'full'.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.
        parallel (bool): Whether to run the iteration in parallel. Default is False.
        segment_size (int | None): Size of segments for parallel processing. Default is None and will calculate
            automatically.

    Returns:
        IterateResults: An object containing the iterate results.
    """

    # Init results and get args
    results = IterateResults(IterateArgs(**{k: v for k, v in locals().items() if k in IterateArgs.__annotations__}))
    args = results.args.get_func_args()

    # Run iterations
    if parallel:
        results.iterations, results.initial_threshold = _parallel_iterate(*args)
    else:
        results.iterations, results.initial_threshold = _serial_iterate(*args)

    # Finalise events
    results = _finalise_events(results)

    return results


def simple_extractor(
    trace: ndarray | Trace,
    baseline: ndarray,
    event_coordinates: ndarray,
    sample_rate: Optional[int] = None,
    feature_type: str = 'full',
    label: str = ''
) -> Events:
    """
    Extract the events from the trace based on the event coordinates and the baseline. Then, simple features are
    calculated for each event. Duration is always in ms, while area and amplitude are in the same units as the trace.
    That is, if the trace is in pA, we have pC and pA for area and amplitude, respectively.

    Args:
        trace (ndarray): The input signal trace or a Trace object.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        feature_type (str): The output event features. Can be 'full' for absolute features, 'FWHM', or 'FWQM' for full
            width at half maximum or quarter maximum, respectively. These apply to duration, area, and peak amplitude.
            Default is 'full'.
        label (str): The label for the Events object, if a Trace object is provided, that label takes precedence.
            Default is ''.

    Returns:
        Events: An Events object containing the extracted events and their features.
    """

    if isinstance(trace, Trace):
        label, sample_rate, trace = trace.label, trace.sample_rate, trace.trace
    if sample_rate is None:
        raise ValueError("Sample rate must be provided if trace is a ndarray.")
    if trace.shape != baseline.shape:
        raise ValueError("Trace and baseline must have the same shape.")
    if feature_type not in ('full', 'FWHM', 'FWQM'):
        raise ValueError("Feature type must be 'full', 'FWHM', or 'FWQM'.")

    # Early return if no events
    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    if event_number == 0:
        return Events(label, {}, feature_type)

    # Get view
    coordinates: cython.longlong[:, ::1] = event_coordinates

    # Baseline adjust the trace
    adjusted: cython.double[::1] = trace - baseline

    events = {}
    event_id: cython.Py_ssize_t = 0
    for event_id in range(event_number):
        vector = adjusted[coordinates[event_id, 0]:coordinates[event_id, 1] + 1]

        # Different feature types
        if feature_type == 'FWHM':
            abs_vector = abs(vector)
            vec = asarray(vector)
            vec = vec[abs_vector >= max(abs_vector) / 2]
        elif feature_type == 'FWQM':
            abs_vector = abs(vector)
            vec = asarray(vector)
            vec = vec[abs_vector >= max(abs_vector) / 4]
        else:
            vec = vector

        # Calculate features
        abs_vector = abs(vec)
        events[event_id + 1] = {
            'ID': event_id + 1,
            'Coordinates': (coordinates[event_id, 0], coordinates[event_id, 1]),
            'Vector': asarray(vector),
            'Direction': 'up' if sum(sign(vector)) > 0 else 'down',
            'Amplitude': max(abs_vector),
            'Duration': len(vec) / sample_rate * 1e3,
            'Area': simpson(abs_vector) / sample_rate,
            'Outlier': False
        }

    return Events(label, events, feature_type)


"""
Internal API
"""


def _serial_estimate(
    trace: ndarray,
    filtered_trace: ndarray,
    bl_args: dict[str, any],
    lr_args: dict[str, any],
    gen_args: dict[str, any]
) -> tuple[list[Iteration], ndarray, str]:
    """
    Serial equivalent to the initial_estimate function, see: initial_estimate for details.
    """

    # Easter egg
    iterations, initial_threshold, current, coords, calc_trace = [], None, None, None, trace

    # Iterate 3 times
    i: cython.int = 1
    for i in range(3):
        if i == 0:
            # Baseline, threshold, and trace statistics on the calculation trace
            baseline, threshold, trace_stats = _baseline_threshold(trace, calc_trace, event_coordinates=coords, **bl_args)
        else:
            # Do not compute trace statistics after the first iteration
            baseline, threshold = _baseline_threshold(trace, calc_trace, event_coordinates=coords, compute_stats=False, **bl_args)

        # Store results if vector_results or the final iteration
        current = Iteration()
        if gen_args['vector_results'] or i == 2:
            current.calculation_trace, current.baseline, current.threshold = calc_trace, baseline, threshold

        if i == 0:
            # (1) Remove events from the side of most influence (save this initial threshold and direction)
            lr_args['event_direction'] = 'up' if trace_stats['Influence'] > 0 else 'down'
            initial_threshold, event_direction, current.trace_stats = threshold, lr_args['event_direction'], trace_stats
            coords, calc_trace, event_stats = _locate_replace(trace, filtered_trace, calc_trace, baseline, threshold,
                                                              **lr_args)
        elif i == 1:
            # (2) Remove events from the opposite side
            lr_args['event_direction'] = 'up' if event_direction == 'down' else 'down'
            coords, calc_trace, event_stats = _locate_replace(trace, filtered_trace, calc_trace, baseline, threshold,
                                                              **lr_args)
        else:
            # (3) Final extraction now both sides are removed
            lr_args['event_direction'] = 'biphasic'
            coords, event_stats = _locate_replace(trace, filtered_trace, calc_trace, baseline, threshold, replace=False,
                                                  **lr_args)

        # Store results
        current.event_stats, current.event_coordinates = event_stats, coords
        iterations.append(current)

    return iterations, initial_threshold, event_direction


def _parallel_estimate(
    trace: ndarray,
    filtered_trace: ndarray,
    bl_args: dict[str, any],
    lr_args: dict[str, any],
    gen_args: dict[str, any]
) -> tuple[list[Iteration], ndarray, str]:
    """
    Parallel equivalent to the initial_estimate function, see: initial_estimate for details.
    """

    # Calculate the parallel configuration
    splits, workers, chunksize = _calc_parallel(trace, bl_args['sample_rate'], bl_args['threshold_window'],
                                                gen_args['segment_size'])

    # Not worth it to parallelize if the trace is too small
    if splits < 10:
        result = _serial_estimate(trace, filtered_trace, bl_args, lr_args, gen_args)
        return result

    # Get the chunks for parallel processing
    chunk_pairs = list(zip(array_split(trace, splits), array_split(filtered_trace, splits)))
    # Get function (don't extract events, return vector results, or validate)
    args_list = [(trace_chunk, filtered_trace_chunk, bl_args.copy(), lr_args.copy(), gen_args.copy()) for
                 trace_chunk, filtered_trace_chunk in chunk_pairs]

    # Spawn thread pool
    with ThreadPool(processes=workers) as pool:
        result = pool.starmap(_serial_estimate, args_list, chunksize=chunksize)
    del chunk_pairs, args_list
    collect()

    # Reduce results
    iterations, initial_threshold = _reduce_parallel(result)

    # Get event direction
    event_direction = 'up' if iterations[0].trace_stats['Influence'] > 0 else 'down'

    return iterations, initial_threshold, event_direction


def _serial_iterate(
    trace: ndarray,
    filtered_trace: ndarray,
    bl_args: dict[str, any],
    lr_args: dict[str, any],
    gen_args: dict[str, any]
) -> tuple[list[Iteration], ndarray]:
    """
    Serial equivalent to the iterate function, see: iterate for details.
    """

    # Easter egg
    iterations, initial_threshold, current, coords, calc_trace = [], None, None, None, trace

    # Iterate until the stopping criterion is met
    i: cython.int = 1
    while True:
        # Baseline, threshold, and trace statistics on the calculation trace
        baseline, threshold, trace_stats = _baseline_threshold(trace, calc_trace, event_coordinates=coords, **bl_args)

        # For calculating maximum cutoffs
        if i == 1:
            initial_threshold = threshold

        # Stopping criterion
        if i > 2 and trace_stats['Overall'] <= current.trace_stats['Overall']:
            break

        # Direction criterion
        if i > 1 and (lr_args['event_direction'] == 'down' and trace_stats['Influence'] > 0 or
                      lr_args['event_direction'] == 'up' and trace_stats['Influence'] < 0):
            lr_args['event_direction'] = 'biphasic'

        # Store results
        current = Iteration(trace_stats=trace_stats)
        if gen_args['vector_results']:
            current.calculation_trace, current.baseline, current.threshold = calc_trace, baseline, threshold

        # Locate events from the raw trace, bounds from the filtered trace, and replace events in the calculation trace
        coords, calc_trace, event_stats = _locate_replace(trace, filtered_trace, calc_trace, baseline, threshold, **lr_args)

        # Store results
        current.event_stats, current.event_coordinates = event_stats, coords
        iterations.append(current)
        i += 1

    # Determine the final baseline and threshold (calc_trace IS local baseline after individual replacement)
    coordinate_view: cython.longlong[:, ::1] = coords
    baseline = _event_replacer(baseline, calc_trace, coordinate_view, replace_factor=0, replace_gap=0)
    threshold = _thresholder((trace - baseline), bl_args['z_score'], bl_args['threshold_window'], method='std',
                             event_mask=__event_mask(coords, len(trace)))

    # Final extraction is always biphasic
    lr_args['event_direction'] = 'biphasic'
    coords, event_stats = _locate_replace(trace, filtered_trace, calc_trace, baseline, threshold, replace=False, **lr_args)

    # Store the final vectors
    iterations.append(Iteration(calculation_trace=calc_trace, baseline=baseline, threshold=threshold,
                                trace_stats=current.trace_stats, event_coordinates=coords, event_stats=event_stats))

    return iterations, initial_threshold


def _parallel_iterate(
    trace: ndarray,
    filtered_trace: ndarray,
    bl_args: dict[str, any],
    lr_args: dict[str, any],
    gen_args: dict[str, any]
) -> tuple[list[Iteration], ndarray]:
    """
    Parallel equivalent to the iterate function, see: iterate for details.
    """

    # Calculate the parallel configuration
    splits, workers, chunksize = _calc_parallel(trace, bl_args['sample_rate'], bl_args['threshold_window'],
                                                gen_args['segment_size'])

    # Not worth it to parallelize if the trace is too small
    if splits < 10:
        result = _serial_iterate(trace, filtered_trace, bl_args, lr_args, gen_args)
        return result

    # Get the chunks for parallel processing
    chunk_pairs = list(zip(array_split(trace, splits), array_split(filtered_trace, splits)))
    # Get function (don't extract events, return vector results, or validate)
    args_list = [(trace_chunk, filtered_trace_chunk, bl_args.copy(), lr_args.copy(), gen_args.copy()) for
                 trace_chunk, filtered_trace_chunk in chunk_pairs]

    # Spawn thread pool
    with ThreadPool(processes=workers) as pool:
        result = pool.starmap(_serial_iterate, args_list, chunksize=chunksize)
    del chunk_pairs, args_list
    collect()

    # Reduce results
    iterations, initial_threshold = _reduce_parallel(result)

    return iterations, initial_threshold


def _calc_parallel(
    trace: ndarray,
    sample_rate: cython.int,
    threshold_window: cython.longlong,
    segment_size: Optional[int] = None
) -> tuple[int, int, int]:
    """
    Calculate the parallel processing configuration for the parallel_estimate and parallel_iterate functions.

    Args:
        trace (ndarray): The input trace.
        sample_rate (int): The sample rate of the trace.
        threshold_window (float): The window size for threshold calculation.
        segment_size (int | None): Size of segments for parallel processing. Default is None and will calculate
            automatically.

    Returns:
        tuple[int, int, int]: The number of trace splits, the number of workers, and the chunksize.
    """

    # Default to the same size as the confirmed stopping criterion accuracy or threshold accuracy, whichever is larger
    minimum_segment = max([1000000.0, (sample_rate * 2.0), threshold_window])
    segment_size = int(min([len(trace), max([segment_size or 0, minimum_segment])]))
    splits = len(trace) // segment_size

    # Memory per active thread from the maximum chunk size
    chunk_length = (len(trace) + splits - 1) // splits
    chunk_mem = chunk_length * trace.dtype.itemsize

    # Make an estimate of the total available memory and per worker (6 * trace overall, 8 * chunk per worker)
    final_mem = max([0, (virtual_memory().available + swap_memory().free) - (trace.nbytes * 6)])
    max_workers = max([1, final_mem // (chunk_mem * 8)])

    # Calculate workers (standard chunking)
    workers = int(min(cpu_count(), splits, max_workers))
    chunksize = int(max([1, splits // (4 * workers)]))

    return splits, workers, chunksize


def _reduce_parallel(items: list[tuple]) -> tuple[list[Iteration], ndarray]:
    """
    Reduce parallel chunk results into full trace iterations and initial threshold.

    Works for both:
        _serial_estimate -> (iterations, initial_threshold, event_direction)
        _serial_iterate  -> (iterations, initial_threshold)

    Args:
        items (list[tuple]): The chunk results.

    Returns:
        tuple[list[Iteration], ndarray]: Reduced iterations and concatenated initial threshold.
    """

    # Initial threshold first
    initial_threshold = concatenate([r[1] for r in items])

    # Iterations (use only common trailing iterations)
    n_iters = min([len(r[0]) for r in items])
    chunk_iterations = [r[0][len(r[0]) - n_iters:] for r in items]

    # Calculate segment offsets from the final iteration of each chunk
    lengths = [len(item.baseline) for item in [iters[len(iters) - 1] for iters in chunk_iterations]]
    offsets = cumsum([0] + lengths[:len(lengths) - 1])

    # All attr names (should be enum but old Dylan was risky and new Dylan wants to go home)
    attr_names = ["baseline", "threshold", "calculation_trace", "trace_stats", "event_coordinates", "event_stats"]

    # Now merge: arrays = concat; coords = concat (with offset); stats = recompute if fast else average
    iterations = []
    for i in range(n_iters):
        items = [iters[i] for iters in chunk_iterations]
        iteration = Iteration()

        # Reduce arrays
        for name in attr_names:
            arrays = [getattr(item, name) for item in items]

            # No arrays
            if all(a is None for a in arrays):
                setattr(iteration, name, None)
                continue

            # Trace statistic dictionaries
            if name == 'trace_stats':
                dicts = [d for d in arrays if d is not None]
                keys = dicts[0].keys()
                setattr(iteration, name, {k: mean([v for d in dicts if (v := d.get(k)) is not None]) for k in keys})
                continue

            # Offset and concatenate event coordinates
            if name == 'event_coordinates':
                coords_list = [item.event_coordinates + offset for item, offset in zip(items, offsets)
                               if item.event_coordinates is not None and item.event_coordinates.size]
                setattr(iteration, name, concatenate(coords_list) if coords_list else zeros((0, 2), dtype=int64))
                continue

            # Get event stats
            if name == 'event_stats':
                setattr(iteration, name, _event_statistics(iteration.event_coordinates))
                continue

            # Regular arrays
            arrays = [atleast_1d(a) for a in arrays if a is not None]
            setattr(iteration, name, concatenate(arrays))

        iterations.append(iteration)

    return iterations, initial_threshold


def _bounds_filter(trace: ndarray | Trace, window: cython.int) -> ndarray:
    """
    Apply a median filter to the trace to be used for bounding. This will be deprecated in the future.

    Args:
        trace (ndarray | Trace): The input trace or a Trace object.
        window (int): The window size for the median filter.

    Returns:
        ndarray: The filtered trace.
    """

    # Trace or ndarray input
    if isinstance(trace, Trace):
        trace = trace.trace

    return median_filter(trace, window) if window else asarray(trace).copy(order="C")


def _finalise_events(results: InitialEstimateResults | IterateResults) -> InitialEstimateResults | IterateResults:
    """
    Finalises the event results for initial estimate and iterate. This function always calculates the maximum cutoff,
    false positives, and significant events. If requested, it will also extract the events and calculate features.

    Args:
        results (InitialEstimateResults | IterateResults): The results from initial_estimate or iterate.

    Returns:
        InitialEstimateResults | IterateResults: The input results with event related results calculated.
    """

    # Args and final iteration
    args = results.args
    final = results.iterations[len(results.iterations) - 1]

    # Cutoff always calculated
    final.event_stats['Max Cutoff'], all_cutoffs = _calculate_cutoffs(
        args.trace,
        final.baseline,
        results.initial_threshold,
        final.event_coordinates,
        args.cutoff,
        args.sample_rate
    )
    results.max_cutoff = final.event_stats['Max Cutoff']

    # Expected number of extractions which are not events
    final.event_stats['False Positives'], final.event_stats['Significant Events'] = _expected_outliers(
        args.trace,
        final.baseline,
        final.threshold,
        final.event_coordinates,
        args.z_score
    )

    # If events are requested
    if args.output_features is not None:
        # Extract the events
        results.events = simple_extractor(
            args.trace,
            final.baseline,
            final.event_coordinates,
            args.sample_rate,
            args.output_features,
            "Initial Estimate" if isinstance(results, InitialEstimateResults) else "Iterate"
        )
        results.events.add_feature('Max Cutoff', all_cutoffs)

    return results


def _expected_outliers(
    trace: ndarray,
    baseline: ndarray,
    threshold: ndarray,
    event_coordinates: ndarray,
    z_score: cython.double
) -> tuple[cython.double, cython.Py_ssize_t]:
    """
    Calculate the expected number of false positive samples and significant events based on the chosen z-score and the
    event profiles. Essentially, we can remove all event samples from the calculation if they exceed the maximum expected
    value for the sample size, presuming a perfectly normal sample. This allows an estimate of the number of false
    positive extractions based on the chosen z-score, allowing capture of events which lie below the LOD if the false
    positives can be accounted for.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_coordinates (ndarray): The coordinates of detected events.
        z_score (float): The z-score for event detection.

    Returns:
        tuple[float, int]: The expected number of false positive samples and the number of significant events.
    """

    assert trace.shape == baseline.shape == threshold.shape

    # Cast to view
    coordinates: cython.longlong[:, ::1] = event_coordinates

    # Single outlier z_score expectation
    max_z: cython.double = norm.ppf(1.0 - ((1.0 / len(trace)) / 2.0))
    max_threshold: cython.double[::1] = (threshold / z_score) * max_z
    adjusted: cython.double[::1] = abs(trace - baseline)

    # Expected number of extractions which are part of the baseline
    total_length: cython.Py_ssize_t = 0
    significant_events: cython.Py_ssize_t = 0
    event_id: cython.Py_ssize_t = 0
    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    event_start: cython.Py_ssize_t
    event_end: cython.Py_ssize_t
    for event_id in range(event_number):
        event_start = coordinates[event_id, 0]
        event_end = coordinates[event_id, 1]

        if max(adjusted[event_start:event_end + 1]) > max_threshold[event_end]:
            total_length += event_end - event_start + 1
            significant_events += 1

    false_positives: cython.double = (2 * (1 - norm.cdf(z_score))) * (len(trace) - total_length)
    return false_positives, significant_events


def _calculate_cutoffs(
    trace: ndarray,
    baseline: ndarray,
    threshold: ndarray,
    event_coordinates: ndarray,
    cutoff: cython.double,
    sample_rate: cython.int
) -> tuple[cython.double, ndarray]:
    """
    Calculate the maximum cutoff frequency for each event presuming a boxcar event profile. This is used to determine
    the maximum cutoff frequency for the baseline filter during the iteration step.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_coordinates (ndarray): The coordinates of detected events.
        cutoff (float): The cutoff frequency used to determine the baseline.
        sample_rate (int): The sample rate of the trace.

    Returns:
        tuple[float, ndarray]: The determined maximum cutoff frequency and the maximum cutoff frequency for each event.
    """

    assert trace.shape == baseline.shape == threshold.shape

    # Baseline adjust the trace
    adjusted: cython.double[::1] = abs(trace - baseline)

    # Initialise the maximum cutoffs array and views
    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    max_cutoffs: cython.double[::1] = zeros(event_number, dtype=trace.dtype)
    thresh: cython.double[::1] = threshold
    coordinates: cython.longlong[:, ::1] = event_coordinates

    # Calculate the cutoff frequency for each event
    time_step: cython.double = 1.0 / sample_rate
    event_id: cython.Py_ssize_t = 0
    event_start: cython.Py_ssize_t
    event_end: cython.Py_ssize_t
    vector: cython.double[::1]
    damping_ratio: cython.double
    for event_id in range(event_number):
        event_start = coordinates[event_id, 0]
        event_end = coordinates[event_id, 1]
        vector = adjusted[event_start:event_end + 1]

        # Get the damping ratio for the event
        max_vec = max(vector)
        damping_ratio = (max_vec / (max_vec - thresh[event_end]))

        # -3dB of boxcar by length
        max_cutoffs[event_id] = 0.442947 / (sqrt((len(vector) * damping_ratio) ** 2 - 1) * time_step)

        # Clamp to the original cutoff (has to be larger than this)
        if max_cutoffs[event_id] < cutoff:
            max_cutoffs[event_id] = cutoff

    # Q1 of the required cutoffs
    calculated_cutoff: cython.double
    try:
        calculated_cutoff = percentile(max_cutoffs, 25)
    except IndexError:
        calculated_cutoff = cutoff

    # Do not want to be above recommended antialiasing cutoff
    aa_limit: cython.double = sample_rate / 5.0
    if not 0.0 < calculated_cutoff < aa_limit:
        calculated_cutoff = aa_limit

    return calculated_cutoff, asarray(max_cutoffs)


def _locate_replace(
    trace: ndarray,
    filtered_trace: ndarray,
    calculation_trace: ndarray,
    baseline: ndarray,
    threshold: ndarray,
    event_direction: str,
    replace_factor: cython.double,
    replace_gap: cython.double,
    replace: cython.bint = True,
) -> tuple[ndarray, dict[str, float | int]] | tuple[ndarray, ndarray, dict[str, float | int]]:
    """
    A wrapper for locating and replacing events.

    Args:
        trace (ndarray): The input signal trace.
        filtered_trace (ndarray): The filtered version of the input trace.
        calculation_trace (ndarray): The trace events are replaced in.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').
        replace_factor (float): Factor for replacing events in the calculation trace.
        replace_gap (float): Gap for replacing events in the calculation trace.
        replace (bool): Whether to replace events in the calculation trace. Default is True.

    Returns:
        tuple[ndarray, dict[str, float | int]] | tuple[ndarray, ndarray, dict[str, float | int]]: The coordinates of
            detected events, the event statistics, and optionally the replaced trace if `replace` is True.
    """

    assert trace.shape == filtered_trace.shape == calculation_trace.shape == baseline.shape == threshold.shape
    assert event_direction in ('up', 'down', 'biphasic')
    assert replace_factor >= 0
    assert replace_gap >= 0

    # Locates the events, trace for threshold crossings, filtered_trace for baseline crossings
    adjusted = trace - baseline
    adjusted_filtered = filtered_trace - baseline
    event_coordinates = _locate_events(adjusted, adjusted_filtered, threshold, event_direction)
    # Get the event statistics
    event_stats = _event_statistics(event_coordinates)

    if not replace:
        return event_coordinates, event_stats

    # Sort the coordinates by current residual influence
    adjusted_calc = calculation_trace - baseline
    coordinate_view: cython.longlong[:, ::1] = event_coordinates
    sorted_coordinates: cython.longlong[:, ::1] = _sort_coordinates(adjusted_calc, coordinate_view)
    # Replace events
    replaced_trace = _event_replacer(calculation_trace, baseline, sorted_coordinates, replace_factor, replace_gap)

    return event_coordinates, replaced_trace, event_stats


def _locate_events(
    adjusted: ndarray,
    adjusted_filtered: ndarray,
    threshold: ndarray,
    event_direction: str
) -> ndarray:
    """
    Locate events in the trace based on the filtered trace, baseline, and threshold.

    Args:
        adjusted (ndarray): The baseline adjusted input signal trace.
        adjusted_filtered (ndarray): The baseline adjusted input filtered trace.
        threshold (ndarray): The threshold for event detection.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').

    Returns:
        ndarray: The coordinates of detected events.
    """

    # Mask used everywhere
    n: cython.Py_ssize_t = adjusted.shape[0]
    mask = empty(n, dtype=uint8)

    # Filtered trace for baseline crossing indices
    baseline_idxs: cython.longlong[::1] = __find_crosses(greater_equal(adjusted_filtered, 0, out=mask), False)

    # Regular trace for threshold crossing indices
    if event_direction == 'up':
        # Positive deflection
        threshold_idxs: cython.longlong[::1] = __find_crosses(greater_equal(adjusted, threshold, out=mask))
    elif event_direction == 'down':
        # Negative deflection
        threshold_idxs: cython.longlong[::1] = __find_crosses(less_equal(adjusted, -threshold, out=mask))
    else:
        # Biphasic deflection
        threshold_idxs: cython.longlong[::1] = sort(concatenate((__find_crosses(greater_equal(adjusted, threshold,  out=mask)),
                                                                 __find_crosses(less_equal(adjusted, -threshold, out=mask)))))

    if threshold_idxs.shape[0]  == 0:
        return zeros((0, 2), dtype=int64)

    # Sort to find bounds
    event_coordinates = __event_bounds(threshold_idxs, baseline_idxs, n)

    return event_coordinates


def _event_statistics(event_coordinates: ndarray) -> dict[str, float | int]:
    """
    Calculate statistics for detected events.

    Args:
        event_coordinates (ndarray): The coordinates of detected events.

    Returns:
        dict[str, float | int]: A dictionary containing event statistics.
    """

    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    if event_number == 0:
        return {
            'Event Count': 0,
            'Total Event Samples': 0,
            'Average Event Samples': 0.0,
            'Median Event Samples': 0.0
        }

    event_lengths = (event_coordinates[:, 1] - event_coordinates[:, 0] + 1)
    total_length = int(sum(event_lengths))
    average_length = total_length / event_number
    median_length = median(event_lengths)

    return {
        'Event Count': event_number,
        'Total Event Samples': total_length,
        'Average Event Samples': average_length,
        'Median Event Samples': median_length
    }


def _sort_coordinates(adjusted: ndarray, event_coordinates: cython.longlong[:, ::1]) -> cython.longlong[:, ::1]:
    """
    Sort event coordinates based on event duration and amplitude. This is used to remove the largest events first,
    which mitigates influence problems during the replacement step.

    Args:
        adjusted (ndarray): The baseline adjusted input signal trace.
        event_coordinates (ndarray): The coordinates of detected events.

    Returns:
        ndarray: Sorted event coordinates.
    """

    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    if event_number == 0:
        sorted_coords: cython.longlong[:, ::1] = zeros((0, 2), dtype=int64)
        return sorted_coords

    # Baseline adjust the trace
    adj: cython.double[::1] = abs(adjusted)

    # Calculate the score for each event based on its duration and maximum amplitude
    scores: cython.double[::1] = empty(event_number, dtype=float)
    event_id: cython.Py_ssize_t = 0
    event_start: cython.Py_ssize_t
    event_end: cython.Py_ssize_t
    event_length: cython.Py_ssize_t
    j: cython.Py_ssize_t = 0
    max_amp: cython.double
    for event_id in range(event_number):
        event_start = event_coordinates[event_id, 0]
        event_end = event_coordinates[event_id, 1]
        event_length = event_end - event_start + 1

        # Find max abs event in the interval
        max_amp = 0.0
        for j in range(event_start, event_end + 1):
            if adj[j] > max_amp:
                max_amp = adj[j]
        scores[event_id] = max_amp * event_length

    # Sort the events by score (descending order)
    order = argsort(-asarray(scores), kind='stable')
    sorted_coords: cython.longlong[:, ::1] = asarray(event_coordinates)[order]

    return sorted_coords


def _event_replacer(
    trace: ndarray,
    baseline: ndarray,
    event_coordinates: cython.longlong[:, ::1],
    replace_factor: cython.double,
    replace_gap: cython.double
) -> ndarray:
    """
    The heart of the iterative process, this function replaces events in a copy of the trace (calculation trace)
    using an event specific filter. This trace is then used in the iterative process to calculate the baseline and
    threshold for the next iteration.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events.
        replace_factor (float): Factor for replacing events in the calculation trace.
        replace_gap (float): Gap for replacing events in the calculation trace.

    Returns:
        ndarray: The trace with events replaced (calculation trace).
    """

    event_number: cython.Py_ssize_t = event_coordinates.shape[0]
    if event_number == 0:
        return trace.copy()

    # Replacing with the baseline
    event_id: cython.Py_ssize_t = 0
    if abs(replace_gap) < 1e-12 and abs(replace_factor) < 1e-12:
        padded_trace: cython.double[::1] = trace.copy()
        baseline_view: cython.double[::1] = baseline
        for event_id in range(event_number):
            padded_trace[event_coordinates[event_id, 0]:event_coordinates[event_id, 1] + 1] = (
            baseline_view[event_coordinates[event_id, 0]:event_coordinates[event_id, 1] + 1])
        return asarray(padded_trace)

    # Pad now to avoid padding in the loop (largest event)
    event_starts = asarray(event_coordinates[:, 0])
    event_ends = asarray(event_coordinates[:, 1])
    event_lengths: cython.longlong[::1] = (event_ends - event_starts + 1)
    longest_event: cython.Py_ssize_t = max(event_lengths)
    longest_window: cython.Py_ssize_t = int(ceil(longest_event * (replace_factor + replace_gap)))

    # If the longest window is larger than the trace, we pad it extra after symmetric
    dtype = asarray(trace).dtype
    if longest_window <= trace.shape[0] - 1:
        padded_trace = pad(trace, longest_window, mode="symmetric").astype(dtype, copy=False)
    else:
        tmp = pad(trace, trace.shape[0], mode="symmetric")
        padded_trace = pad(tmp, longest_window - trace.shape[0], mode="edge").astype(dtype, copy=False)
    padded_trace: cython.double[::1] = padded_trace

    # We remove the events in order of largest to smallest (see: _sort_coordinates())
    event_id: cython.Py_ssize_t = 0
    event_length: cython.Py_ssize_t
    event_start: cython.Py_ssize_t
    event_end: cython.Py_ssize_t
    length: cython.Py_ssize_t
    filled: cython.Py_ssize_t
    half_length: cython.Py_ssize_t
    window: cython.double[::1]
    segment: cython.double[::1]
    filtered_segment: cython.double[::1]
    for event_id in range(event_number):
        # Adjust for padding, with extra sauce
        event_length = event_lengths[event_id]
        event_start = event_coordinates[event_id, 0] + longest_window
        event_end = event_coordinates[event_id, 1] + longest_window + 1

        # Make it odd and skip if it is 1
        length = int(round(event_length * (replace_factor + replace_gap)))
        if length & 1 == 0:
            length += 1
        if length == 1:
            continue

        # Try to reuse the window if it has been calculated before
        tupe = __WINDOW_CACHE.get((length, round(replace_factor, 6), round(replace_gap, 6)))
        if tupe is not None:
            window, half_length = tupe
        else:
            window = ones(length, dtype=dtype)
            if replace_gap > 0:
                filled = (length - int(round(event_length * replace_gap))) // 2
                window[filled:length - filled] = 0.0
            window = window / sum(window)

            half_length = length // 2
            __WINDOW_CACHE[(length, round(replace_factor, 6), round(replace_gap, 6))] = (window, half_length)

        # Filter and replace
        segment = padded_trace[event_start - half_length:event_end + half_length]
        filtered_segment = oaconvolve(segment, window, 'same')
        padded_trace[event_start:event_end] = filtered_segment[half_length:len(filtered_segment) - half_length]

    return asarray(padded_trace[longest_window:len(padded_trace) - longest_window])


def _baseline_threshold(
    trace: ndarray,
    calculation_trace: ndarray,
    cutoff: cython.double,
    sample_rate: cython.int,
    z_score: cython.double,
    threshold_window: cython.longlong,
    event_coordinates: Optional[ndarray] = None,
    method: str = 'iqr',
    compute_stats: bool = True
) -> tuple[ndarray, ndarray] | tuple[ndarray, ndarray, dict[str, cython.double]]:
    """
    A wrapper for calculating the baseline and threshold.

    Args:
        trace (ndarray): The input signal trace.
        calculation_trace (ndarray): The trace events are replaced in.
        cutoff (float): The cutoff frequency for the baseline filter.
        sample_rate (int): The sample rate of the trace.
        z_score (float): The z-score for event detection.
        threshold_window (int): The window size in samples for threshold calculation.
        event_coordinates (ndarray | None): The coordinates of detected events. Default is None.
        method (str): The method for threshold calculation ('iqr' or 'std'). Default is 'iqr'.
        compute_stats (bool): Whether to compute trace statistics. Default is True.

    Returns:
        tuple[ndarray, ndarray] | tuple[ndarray, ndarray, dict[str, float]]: The baseline, threshold, and optionally the
            trace statistics if `compute_stats` is True.
    """

    assert cutoff > 0
    assert sample_rate > 0
    assert z_score > 0
    assert threshold_window > 0
    assert method in ('iqr', 'std')

    # Get the baseline
    baseline = _baseline_filter(calculation_trace, cutoff, sample_rate)

    # Get event mask (view)
    if event_coordinates is None:
        event_mask: cython.uchar[::1] = zeros(trace.shape[0], dtype=uint8)
    else:
        event_mask: cython.uchar[::1] = __event_mask(event_coordinates, trace.shape[0])

    # Baseline-adjust the traces now to save repeats
    adjusted = trace - baseline

    # Get the threshold
    threshold = _thresholder(adjusted, z_score, threshold_window, event_mask, method)

    if not compute_stats:
        return baseline, threshold

    # Compute statistics if requested
    trace_stats = _trace_statistics(adjusted, event_mask)

    return baseline, threshold, trace_stats


def _baseline_filter(trace: ndarray, cutoff: cython.double, sample_rate: cython.int) -> ndarray:
    """
    Apply a baseline filter to the trace using a boxcar filter (moving average). This is a low-pass filter, the result
    is to be subtracted which is then a high-pass filter.

    Args:
        trace (ndarray): The input signal trace.
        cutoff (float): The cutoff frequency for the baseline filter.
        sample_rate (int): The sample rate of the trace.

    Returns:
        ndarray: The baseline.
    """

    # Cutoff to boxcar length conversion (odd length)
    length: cython.Py_ssize_t = int(round(sqrt(0.196202 + ((cutoff * (1.0 / sample_rate)) ** 2)) / (cutoff * (1.0 / sample_rate))))

    # Make it odd, set as 3 if it is 1, and ensure it is not longer than the trace
    if length & 1 == 0:
        length += 1
    if length == 1:
        length = 3
    if length >= trace.shape[0]:
        length = trace.shape[0] - 1

    # Odd length window
    dtype = asarray(trace).dtype
    half: cython.Py_ssize_t = length // 2
    # Add fit for the edge cases
    padded_trace = pad(trace, (half, half), mode="symmetric").astype(dtype, copy=False)

    # Cumulative sum
    baseline = empty(padded_trace.shape[0] + 1, dtype=dtype)
    baseline[0] = 0.0
    cumsum(padded_trace, out=baseline[1:])
    # Moving average
    baseline = (baseline[length:] - baseline[:len(baseline) - length]) / length

    return baseline


def _thresholder(
    adjusted: ndarray,
    z_score: cython.double,
    window: cython.longlong,
    event_mask: cython.uchar[::1],
    method: str = 'iqr'
) -> ndarray:
    """
    Calculate the threshold for event detection based on the trace and baseline. The baseline is subtracted from the
    trace (HPF) and the Wan et al. (doi:10.1186/1471-2288-14-135) method is used to estimate the standard deviation.

    Args:
        adjusted (ndarray): The baseline adjusted input signal trace.
        z_score (float): The z-score for event detection.
        window (int): The window size in samples for threshold calculation.
        event_mask (ndarray): The boolean event mask.
        method (str): The method for threshold calculation ('iqr' or 'std'). Default is 'iqr'.

    Returns:
        ndarray: The calculated threshold.
    """

    # Get view for slicing
    adj: cython.double[::1] = adjusted

    # Number of segments to divide the trace into
    length: cython.Py_ssize_t = adjusted.shape[0]
    segments: cython.Py_ssize_t = maximum(length // window, 1)

    # Calculate the threshold in segments
    i: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current_segment: ndarray
    q1: cython.double
    q3: cython.double
    n: cython.Py_ssize_t
    standard_deviation = empty(length, dtype=adjusted.dtype)
    standard_deviation_view: cython.double[::1] = standard_deviation
    # standard_deviation: cython.double[::1] = empty(length, dtype=adjusted.dtype)
    for i in range(segments):
        start = i * window
        end = (i + 1) * window if i != segments - 1 else length

        # Mask events
        data_segment = asarray(adj[start:end])
        mask_segment = asarray(event_mask[start:end], dtype=bool)
        current_segment = data_segment[~mask_segment]

        if current_segment.shape[0] < 2:
            current_segment = data_segment

        if method == 'iqr':
            # doi:10.1186/1471-2288-14-135
            q1, q3 = quantile(current_segment, [0.25, 0.75])
            n = current_segment.shape[0]
            standard_deviation_view[start:end] = (q3 - q1) / (2 * norm.ppf((0.75 * n - 0.125) / (n + 0.25)))
        else:
            standard_deviation_view[start:end] = std(current_segment)

    standard_deviation *= z_score
    return standard_deviation


def _trace_statistics(
    adjusted: ndarray,
    event_mask: cython.uchar[::1],
) -> dict[str, cython.double]:
    """
    Calculate statistics for the trace.

    Args:
        adjusted (ndarray): The baseline adjusted input signal trace.
        event_mask (ndarray | None): The boolean event mask. Default is None.

    Returns:
        dict[str, float]: A dictionary containing trace statistics.
    """

    # Get the masked adjusted
    adjusted = adjusted[~asarray(event_mask, dtype=bool)]

    results = {}
    # The overall Shapiro-Wilk (stopping criterion)
    results['Overall'] = shapiro(adjusted).statistic
    # The influence (difference in normality between sides)
    q1, q3 = quantile(adjusted, [0.25, 0.75])
    results['Positive Statistic'] = shapiro(adjusted[adjusted > q3]).statistic
    results['Negative Statistic'] = shapiro(adjusted[adjusted < q1]).statistic
    results['Influence'] = results['Negative Statistic'] - results['Positive Statistic']
    # Other useful statistics
    results['Standard Deviation'] = std(adjusted)
    results['Mean'] = mean(adjusted)

    return results


"""
Super Internal API
"""

def __find_crosses(sign_mask: cython.uchar[::1], pair_match: cython.bint = True) -> cython.longlong[::1]:
    """
    Internal function for finding crossing indices, using XOR of successive elements marking a flip.

    Args:
        sign_mask (ndarray): The boolean sign mask.
        pair_match (bool): Whether to pair match each crossing. Default is True.

    Returns:
        ndarray: Indices where the sign mask changes sign.
    """

    n: cython.Py_ssize_t = sign_mask.shape[0]
    if n < 2:
        out: cython.longlong[::1] = empty(0, dtype=int64)
        return out

    # XOR successive bools without an intermediate allocation
    zero_trace = empty(n - 1, dtype=uint8)
    bitwise_xor(sign_mask[:n - 1], sign_mask[1:], out=zero_trace)
    flips: cython.longlong[::1] = flatnonzero(zero_trace).astype(int64, copy=False)
    f_size: cython.Py_ssize_t = flips.shape[0]
    if f_size == 0:
        return flips

    # Pair matching
    pre: cython.Py_ssize_t
    post: cython.Py_ssize_t
    if pair_match:
        pre = 1 if sign_mask[0] else 0
        post = 1 if sign_mask[n - 1] else 0
    else:
        pre = 0
        post = 0

    out: cython.longlong[::1] = empty(f_size + pre + post, dtype=int64)
    if pre:
        out[0] = 0
    if post:
        out[len(out) - 1] = n - 1
    out[pre:pre + f_size] = flips

    return out


def __event_bounds(
    threshold_idxs: cython.longlong[::1],
    baseline_idxs: cython.longlong[::1],
    length: cython.Py_ssize_t
) -> ndarray:
    """
    Determine the bounds of events based on threshold and baseline crossing indices.

    Args:
        threshold_idxs (ndarray): Indices where the trace crosses the threshold.
        baseline_idxs (ndarray): Indices where the trace crosses the baseline.
        length (int): The length of the trace.

    Returns:
        ndarray: The coordinates of detected events.
    """

    assert threshold_idxs.shape[0] % 2 == 0

    # Get start indices and end indices of threshold crossings
    n: cython.Py_ssize_t = threshold_idxs.shape[0] // 2
    starts = asarray(threshold_idxs[::2])
    ends = asarray(threshold_idxs[1::2])
    baseline_id = asarray(baseline_idxs)

    # Degenerate case
    if baseline_id.shape[0] == 0:
        left = zeros(n, dtype=int64)
        right = full(n, length - 1, dtype=int64)
    else:
        # Nearest baseline cross left of each start
        left_idx = searchsorted(baseline_id, starts, side='right') - 1
        fell_off = (left_idx == -1)
        left_idx[fell_off] = 0
        left = baseline_id[left_idx]
        left[fell_off] = 0

        # Nearest baseline cross right of each end
        right_idx = searchsorted(baseline_id, ends, side='left')
        fell_off[:] = (right_idx == baseline_id.shape[0])
        right_idx[fell_off] = baseline_id.shape[0] - 1
        right = baseline_id[right_idx] + 1
        minimum(right, length - 1, out=right)
        right[fell_off] = length - 1

    # Remove events that overlap
    keep = (starts >= roll(maximum.accumulate(right - 1), 1))
    keep[0] = True

    return column_stack((left[keep], right[keep]))


def __event_mask(event_coordinates: ndarray, length: cython.Py_ssize_t) -> cython.uchar[::1]:
    """
    Function to create a boolean event mask from event coordinates and return a view

    Args:
         event_coordinates (ndarray): The coordinates of detected events.
         length (int): The length of the trace.

    Returns:
        ndarray: The boolean event mask.
    """

    # Get coordinate view
    coordinate_view: cython.longlong[:, ::1] = event_coordinates

    # Allocate output
    event_mask: cython.uchar[::1] = zeros(length, dtype=uint8)
    event_number: cython.Py_ssize_t = coordinate_view.shape[0]
    if event_number == 0:
        return event_mask

    # Fill array
    event_id: cython.Py_ssize_t = 0
    event_start: cython.Py_ssize_t
    event_end: cython.Py_ssize_t
    for event_id in range(event_number):
        event_start = coordinate_view[event_id, 0]
        event_end = coordinate_view[event_id, 1] + 1
        event_mask[event_start:event_end] = 1

    return event_mask
