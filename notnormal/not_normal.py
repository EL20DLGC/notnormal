# cython: infer_types=True

"""
This module provides functions for anomaly detection and baseline determination in (nano)electrochemical time series
data. It includes a method of estimating the cutoff frequency and event direction, an improved iterative method,
thresholding method and a simple extraction method.
"""

from typing import Optional
from numpy import quantile, diff, sign, where, sort, concatenate, sqrt, round, ones, zeros, mean, ceil
from numpy import percentile, ndarray, std, sum, max, median, asarray, array_split, double, int32, int64
from scipy.signal import oaconvolve
from scipy.ndimage import median_filter
from scipy.stats import norm, shapiro
from functools import partial
from multiprocessing import Pool, cpu_count
from notnormal.results import Iteration
from warnings import catch_warnings, simplefilter, filterwarnings
import cython

if cython.compiled:
    COMPILED = True
    print('not_normal compiled')
else:
    COMPILED = False
    print('not_normal not compiled')

DTYPE = double


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def not_normal(
    trace: ndarray,
    sample_rate: cython.int = 100000,
    bounds_window: cython.int = 3,
    estimate_cutoff: cython.double = 10.0,
    replace_factor: cython.int = 8,
    replace_gap: cython.int = 2,
    threshold_window: cython.double = 2.0,
    z_score: cython.double = None,
    vector_results: cython.bint = False,
    parallel: cython.bint = False
) -> tuple:
    """
    Anomaly detection and baseline determination for (nano)electrochemical time series data. This function combines the
    estimate and iteration steps into a single function. There are no required parameters other than the input trace and
    sample rate.

    Args:
        trace (ndarray): The input signal trace.
        sample_rate (int): The sample rate of the trace. Default is 100000.
        bounds_window (int): The window size for the median filter used in bounding. Default is 3.
        estimate_cutoff (float): Initial estimate for the cutoff frequency. Default is 10.0.
        replace_factor (int): Factor for replacing events in the calculation trace. Default is 8.
        replace_gap (int): Gap for replacing events in the calculation trace. Default is 2.
        threshold_window (float): Window size for threshold calculation. Default is 2.0.
        z_score (float): Z-score for event detection. If None, it is calculated based on the trace length.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.
        parallel (bool): Whether to run the iteration in parallel. Default is False.

    Returns:
        tuple: A tuple containing event coordinates, baseline, and a dictionary of results.
    """

    # Default to 1 expected outlier per trace (computed on length, of course)
    if z_score is None:
        z_score = norm.ppf(1 - ((1 / len(trace)) / 2))

    # Default 3-point median filter for bounding
    filtered_trace = bounds_filter(trace, bounds_window)

    # Initial estimate for the cutoff frequency and direction
    cutoff, event_direction, estimate_results = initial_estimate(
        trace,
        filtered_trace,
        sample_rate,
        estimate_cutoff,
        threshold_window,
        z_score
    )

    # Iterate based on the initial estimate
    args = dict(
        trace=trace,
        filtered_trace=filtered_trace,
        sample_rate=sample_rate,
        cutoff=cutoff,
        event_direction=event_direction,
        replace_factor=replace_factor,
        replace_gap=replace_gap,
        threshold_window=threshold_window,
        z_score=z_score,
        vector_results=vector_results
    )
    if parallel:
        event_coordinates, baseline, iteration_results = parallel_iterate(**args)
    else:
        event_coordinates, baseline, iteration_results = iterate(**args)

    return event_coordinates, baseline, dict(estimate=estimate_results, iteration=iteration_results)


@cython.boundscheck(False)
@cython.initializedcheck(False)
def parallel_iterate(
    trace: ndarray,
    filtered_trace: ndarray,
    sample_rate: cython.int,
    cutoff: cython.double,
    event_direction: str,
    replace_factor: cython.int,
    replace_gap: cython.int,
    threshold_window: cython.double,
    z_score: cython.double,
    vector_results: cython.bint = False,
    segment_size: Optional[int64] = None
):
    """
    Parallel equivalent to the iterate function.

    Args:
        trace (ndarray): The input signal trace.
        filtered_trace (ndarray): The filtered version of the input trace.
        sample_rate (int): The sample rate of the trace.
        cutoff (float): The cutoff frequency for the baseline filter.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').
        replace_factor (int): Factor for replacing events in the calculation trace.
        replace_gap (int): Gap for replacing events in the calculation trace.
        threshold_window (float): Window size for threshold calculation.
        z_score (float): Z-score for event detection.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.
        segment_size (int): Size of segments for parallel processing. Default is None and will calculate automatically.

    Returns:
        tuple: A tuple containing event coordinates, baseline, and iteration results.
    """

    assert trace.shape == filtered_trace.shape

    args = dict(trace=trace, filtered_trace=filtered_trace, sample_rate=sample_rate, cutoff=cutoff,
                event_direction=event_direction, replace_factor=replace_factor, replace_gap=replace_gap,
                threshold_window=threshold_window, z_score=z_score, vector_results=vector_results)

    # Default to the same size as the confirmed stopping criterion accuracy or threshold accuracy, whichever is larger
    minimum_segment: cython.longlong = int(max([200000, int(sample_rate * threshold_window)]))
    if segment_size is None or segment_size < minimum_segment:
        segment_size = minimum_segment
    if segment_size > len(trace):
        segment_size = len(trace)
    splits: cython.int = len(trace) // segment_size

    # Not worth it to parallelize if the trace is too small
    if splits < 5:
        return iterate(**args)

    pool = Pool(processes=min(cpu_count(), splits))
    results = pool.starmap(
        partial(
            iterate,
            sample_rate=sample_rate,
            cutoff=cutoff,
            event_direction=event_direction,
            replace_factor=replace_factor,
            replace_gap=replace_gap,
            threshold_window=threshold_window,
            z_score=z_score,
            vector_results=vector_results
        ),
        zip(array_split(trace, splits), array_split(filtered_trace, splits))
    )
    pool.close()
    pool.join()

    # Reduce the results
    iteration = Iteration(label=f'Final', args=args, trace=trace, filtered_trace=filtered_trace)
    baselines = []
    thresholds = []
    calculation_traces = []
    event_coordinates = []
    current_len: cython.size_t = 0
    for result in results:
        baselines.append(result[-1][-1].baseline)
        thresholds.append(result[-1][-1].threshold)
        calculation_traces.append(result[-1][-1].calculation_trace)
        if result[-1][-1].event_coordinates.size != 0:
            result[-1][-1].event_coordinates[:, 0] += current_len
            result[-1][-1].event_coordinates[:, 1] += current_len
            event_coordinates.append(result[-1][-1].event_coordinates)
        current_len += len(result[-1][-1].baseline)

    iteration.baseline = concatenate(baselines)
    iteration.threshold = concatenate(thresholds)
    iteration.calculation_trace = concatenate(calculation_traces)
    iteration.event_coordinates = concatenate(event_coordinates)
    coordinates_view: cython.longlong[:, ::1] = iteration.event_coordinates
    # recompute event stats, trace stats and over extractions
    iteration.trace_stats = trace_statistics(iteration.calculation_trace, iteration.baseline, coordinates_view)
    iteration.event_stats = event_statistics(coordinates_view)
    iteration.event_stats['Over Extractions'], iteration.event_stats['Significant Events'] = expected_outliers(
        trace,
        iteration.baseline,
        iteration.threshold,
        coordinates_view,
        z_score
    )

    return iteration.event_coordinates, iteration.baseline, iteration


@cython.boundscheck(False)
@cython.initializedcheck(False)
def iterate(
    trace: ndarray,
    filtered_trace: ndarray,
    sample_rate: cython.int,
    cutoff: cython.double,
    event_direction: str,
    replace_factor: cython.int,
    replace_gap: cython.int,
    threshold_window: cython.double,
    z_score: cython.double,
    vector_results: cython.bint = False
) -> ndarray:
    """
    Perform iterative anomaly detection and baseline determination on the trace.

    Args:
        trace (ndarray): The input signal trace.
        filtered_trace (ndarray): The filtered version of the input trace.
        sample_rate (int): The sample rate of the trace.
        cutoff (float): The cutoff frequency for the baseline filter.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').
        replace_factor (int): Factor for replacing events in the calculation trace.
        replace_gap (int): Gap for replacing events in the calculation trace.
        threshold_window (float): Window size for threshold calculation.
        z_score (float): Z-score for event detection.
        vector_results (bool): Whether to return vector results, this is very expensive. Default is False.

    Returns:
        ndarray: An array containing event coordinates, baseline, and iteration results.
    """

    assert trace.shape == filtered_trace.shape

    args = dict(sample_rate=sample_rate, cutoff=cutoff, event_direction=event_direction, replace_factor=replace_factor,
                replace_gap=replace_gap, threshold_window=threshold_window, z_score=z_score)
    results = []

    # Iterate until the stopping criterion is met
    calculation_trace = trace.copy()
    i: cython.int = 1
    while True:
        results.append(Iteration(label=f'Iteration {i}', args=args, trace=trace, filtered_trace=filtered_trace))
        # Baseline and threshold on the calculation trace
        baseline = baseline_filter(calculation_trace, cutoff, sample_rate)
        threshold = thresholder(calculation_trace, baseline, z_score, int(threshold_window * sample_rate),
                                results[-2].event_coordinates if i > 1 else None)

        # Calculate trace statistics
        results[-1].trace_stats = trace_statistics(calculation_trace, baseline,
                                                   results[-2].event_coordinates if i > 1 else None)

        if vector_results:
            results[-1].baseline = baseline
            results[-1].threshold = threshold
            results[-1].calculation_trace = calculation_trace

        # Direction criterion
        if i > 1 and (event_direction == 'down' and results[-1].trace_stats['Influence'] > 0 or
                      event_direction == 'up' and results[-1].trace_stats['Influence'] < 0):
            event_direction = 'biphasic'

        # Locate events from the raw trace
        results[-1].event_coordinates = locate_events(trace, filtered_trace, baseline, threshold, event_direction)
        results[-1].event_coordinates = sort_coordinates(trace, baseline, results[-1].event_coordinates)
        # Calculate event statistics
        results[-1].event_stats = event_statistics(results[-1].event_coordinates)

        # Stopping criterion
        if i > 1 and (results[-1].event_stats['Total Event Samples'] in
                      [r.event_stats['Total Event Samples'] for r in results[:-1]]):
            break

        # Replace events in the calculation trace
        calculation_trace = event_replacer(calculation_trace, baseline, results[-1].event_coordinates,
                                           replace_factor, replace_gap)

        i += 1

    # Final compute threshold using std as it is more accurate when outliers are no longer present
    threshold = thresholder(calculation_trace, baseline, z_score, int(threshold_window * sample_rate),
                            results[-1].event_coordinates, method='std')
    # Store the final vectors
    results.append(Iteration(label='Final', args=args, trace=trace, filtered_trace=filtered_trace, baseline=baseline,
                             threshold=threshold, calculation_trace=calculation_trace,
                             trace_stats=results[-1].trace_stats))
    # Final extraction is biphasic
    results[-1].event_coordinates = locate_events(trace, filtered_trace, baseline, threshold, 'biphasic')
    results[-1].event_stats = event_statistics(results[-1].event_coordinates)

    # Expected number of extractions which are not events
    results[-1].event_stats['False Positives'], results[-1].event_stats['Significant Events'] = expected_outliers(
        trace,
        baseline,
        threshold,
        results[-1].event_coordinates,
        z_score
    )

    return results[-1].event_coordinates, baseline, results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def initial_estimate(
        trace: ndarray,
        filtered_trace: ndarray,
        sample_rate: cython.int,
        cutoff: cython.double,
        threshold_window: cython.double,
        z_score: cython.double,
) -> tuple:
    """
    Provides an initial estimate for the cutoff frequency and event direction for the iterate function.
    This is an improved version of the conventional event extraction method.

    Args:
        trace (ndarray): The input signal trace.
        filtered_trace (ndarray): The filtered version of the input trace.
        sample_rate (int): The sample rate of the trace.
        cutoff (float): The initial estimate for the cutoff frequency.
        threshold_window (float): The window size for threshold calculation.
        z_score (float): The Z-score for event detection.

    Returns:
        tuple: A tuple containing the maximum cutoff frequency, event direction, and the results of the
                initial estimate.
    """

    assert trace.shape == filtered_trace.shape

    results = Iteration(
        label='Estimate',
        args=dict(sample_rate=sample_rate, cutoff=cutoff, threshold_window=threshold_window, z_score=z_score),
        trace=trace,
        filtered_trace=filtered_trace,
        calculation_trace=trace,
    )

    # Initial baseline and threshold
    trace_view: cython.double[::1] = trace
    results.baseline = baseline_filter(trace_view, cutoff, sample_rate)
    results.threshold = thresholder(trace, results.baseline, z_score, int(threshold_window * sample_rate))

    # Calculate trace statistics
    results.trace_stats = trace_statistics(trace, results.baseline)

    # Determine the starting direction
    results.event_direction = 'up' if results.trace_stats['Influence'] > 0 else 'down'

    # First extraction
    results.event_coordinates = locate_events(trace, filtered_trace, results.baseline, results.threshold,
                                              results.event_direction)
    results.event_stats = event_statistics(results.event_coordinates)

    # Calculate the required cutoffs
    threshold_view: cython.double[::1] = results.threshold
    coordinates_view: cython.longlong[:, ::1] = results.event_coordinates
    max_cutoffs: cython.double[::1] = calculate_cutoffs(trace, results.baseline, threshold_view, coordinates_view,
                                                        cutoff, sample_rate)

    # Q1 of the required cutoffs, original needs to be added on top
    calculated_cutoff: cython.double
    try:
        calculated_cutoff = percentile(max_cutoffs, 25)
    except IndexError:
        calculated_cutoff = cutoff
    # Do not want to be above recommended antialiasing cutoff
    results.event_stats['Max Cutoff'] = calculated_cutoff if calculated_cutoff < sample_rate // 5 else cutoff

    return results.event_stats['Max Cutoff'], results.event_direction, results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def expected_outliers(
        trace: ndarray,
        baseline: ndarray,
        threshold: ndarray,
        event_coordinates:  cython.longlong[:, ::1],
        z_score: cython.double
) -> tuple:
    """
    Calculate the expected number of false positives and significant events based on the chosen Z-score and the event
    profiles. Essentially, we can remove all event samples from the calculation if they exceed the maximium expected
    value for the sample size, presuming a perfectly normal sample. This allows an estimate of the number of false
    positive extractions based on the chosen Z-score, allowing capture of events which lie below the LOD if the false
    positives can be accounted for.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_coordinates (ndarray): The coordinates of detected events.
        z_score (float): The Z-score for event detection.

    Returns:
        int: The number of false positives and significant events.
    """

    assert trace.shape == baseline.shape == threshold.shape

    # Single outlier z_score expectation
    max_z: cython.double = norm.ppf(1 - ((1 / len(trace)) / 2))
    max_threshold: cython.double[::1] = (threshold / z_score) * max_z
    adjusted = trace - baseline

    # Expected number of extractions which are part of the baseline
    total_length: cython.size_t = 0
    significant_events: cython.long = 0
    event_id: cython.size_t
    event_number: cython.size_t = len(event_coordinates)
    for event_id in range(event_number):
        if (max(abs(adjusted[event_coordinates[event_id, 0]:event_coordinates[event_id, 1]])) >
                max_threshold[event_coordinates[event_id, 1]]):
            total_length += event_coordinates[event_id, 1] - event_coordinates[event_id, 0] + 1
            significant_events += 1

    false_positives: cython.double = (2 * (1 - norm.cdf(z_score))) * (len(trace) - total_length)
    return false_positives, significant_events


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def simple_extractor(
    trace: ndarray,
    baseline: ndarray,
    event_coordinates: cython.longlong[:, ::1],
    sample_rate: cython.int
) -> list:
    """
    Extract the events from the trace based on the event coordinates and the baseline. Then, simple features are
    calculated for each event.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events.
        sample_rate (int): The sample rate of the trace.

    Returns:
        list: A list of dictionaries containing event features.
    """

    assert trace.shape == baseline.shape

    # Baseline adjust the trace
    adjusted = trace - baseline
    event_id: cython.size_t
    event_number: cython.size_t = len(event_coordinates)
    events = []
    for event_id in range(event_number):
        vector = adjusted[event_coordinates[event_id, 0]:event_coordinates[event_id, 1] + 1]
        events.append({
            'ID': event_id + 1,
            'Coordinates': (event_coordinates[event_id, 0], event_coordinates[event_id, 1]),
            'Vector': vector,
            'Amplitude': max(abs(vector)),
            'Duration': len(vector) / sample_rate * 1e3,
            'Area': sum(abs(vector)) / sample_rate,
            'Direction': 'up' if sum(sign(vector)) > 0 else 'down',
            'Outlier': False
        })

    return events


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def event_statistics(event_coordinates: cython.longlong[:, ::1]) -> dict:
    """
    Calculate statistics for detected events.

    Args:
        event_coordinates (ndarray): The coordinates of detected events.

    Returns:
        dict: A dictionary containing event statistics.
    """

    event: cython.longlong[::1]
    results = dict()
    results['Event Count'] = len(event_coordinates)
    results['Total Event Samples'] = sum([event[1] - event[0] + 1 for event in event_coordinates]) if len(
        event_coordinates) else 0
    results['Average Event Samples'] = mean([event[1] - event[0] + 1 for event in event_coordinates]) if len(
        event_coordinates) else 0
    results['Median Event Samples'] = median([event[1] - event[0] + 1 for event in event_coordinates]) if len(
        event_coordinates) else 0

    return results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def trace_statistics(
    trace: ndarray,
    baseline: ndarray,
    event_coordinates: Optional[cython.longlong[:, ::1]] = None
) -> dict:
    """
    Calculate statistics for the trace.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events. Default is None.

    Returns:
        dict: A dictionary containing trace statistics.
    """

    assert trace.shape == baseline.shape

    results = dict()
    # Baseline adjust the trace
    adjusted = trace - baseline
    # Masking events from the stat calculations
    event_id: cython.size_t
    event_number: cython.size_t
    if event_coordinates is not None:
        event_number = len(event_coordinates)
    event_mask = zeros(trace.shape, dtype=int32)
    event_mask_view: cython.long[::1] = event_mask
    if event_coordinates is not None:
        for event_id in range(event_number):
            event_mask_view[event_coordinates[event_id, 0]:event_coordinates[event_id, 1] + 1] = 1

    adjusted: cython.double[::1] = adjusted[where(event_mask == 0)]
    # Warnings from Shapiro p-value calculation (not needed)
    with catch_warnings():
        filterwarnings('ignore')
        simplefilter('ignore')
        #results['Overall Statistic'] = shapiro(adjusted).statistic
        results['Positive Statistic'] = shapiro(adjusted[where(adjusted > percentile(adjusted, 75))]).statistic
        results['Negative Statistic'] = shapiro(adjusted[where(adjusted < percentile(adjusted, 25))]).statistic
        results['Influence'] = results['Negative Statistic'] - results['Positive Statistic']
    results['Standard Deviation'] = std(adjusted)
    results['Mean'] = mean(adjusted)

    return results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def sort_coordinates(trace: ndarray, baseline: ndarray, event_coordinates: cython.longlong[:, ::1]) -> ndarray:
    """
    Sort event coordinates based on event duration and amplitude. This is used to remove the largest events first,
    which mitigates influence problems during the replacement step.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events.

    Returns:
        ndarray: Sorted event coordinates.
    """

    assert trace.shape == baseline.shape

    if len(event_coordinates) == 0:
        return asarray([], dtype=int64)
    # Baseline adjust the trace
    adjusted = trace - baseline

    x: cython.longlong[::1]
    return asarray(sorted(event_coordinates, key=lambda x: (x[1] - x[0]) * max(abs(adjusted[x[0]:x[1]])),
                          reverse=True), dtype=int64)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def calculate_cutoffs(
        trace: ndarray,
        baseline: ndarray,
        threshold: cython.double[::1],
        event_coordinates: cython.longlong[:, ::1],
        cutoff: cython.double,
        sample_rate: cython.int
) -> ndarray:
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
        ndarray: The maximum cutoff frequency for each event.
    """

    assert trace.shape[0] == baseline.shape[0] == tuple(threshold.shape)[0]

    # Baseline adjust the trace
    adjusted = trace - baseline
    max_cutoffs = zeros(event_coordinates.shape[0], dtype=DTYPE)
    max_cutoffs_view: cython.double[::1] = max_cutoffs
    # Calculate the cutoff frequency for each event
    event_id: cython.size_t
    event_number: cython.size_t = len(event_coordinates)
    event_start: cython.longlong
    event_end: cython.longlong
    vector: cython.double[::1]
    damping_ratio: cython.double
    for event_id in range(event_number):
        event_start = event_coordinates[event_id, 0]
        event_end = event_coordinates[event_id, 1] + 1
        vector = abs(adjusted[event_start:event_end])
        damping_ratio = (max(vector) / (max(vector) - threshold[event_end]))
        # -3dB of boxcar by length
        max_cutoffs_view[event_id] = 0.442947 / (sqrt((len(vector) * damping_ratio) ** 2 - 1) * (1 / sample_rate))
        max_cutoffs_view[event_id] += cutoff

    return max_cutoffs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bounds_filter(trace: ndarray, window: cython.int) -> ndarray:
    """
    Apply a median filter to the trace to be used for bounding. This will be deprecated in the future.

    Args:
        trace (ndarray): The input signal trace.
        window (int): The window size for the median filter.

    Returns:
        ndarray: The filtered trace.
    """

    return median_filter(trace, window) if window else trace


@cython.boundscheck(False)
@cython.initializedcheck(False)
def event_replacer(
        trace: cython.double[::1],
        baseline: cython.double[::1],
        event_coordinates: cython.longlong[:, ::1],
        replace_factor: cython.int,
        replace_gap: cython.int
) -> ndarray:
    """
    The heart of the iterative process, this function replaces events in a copy of the trace (calculation trace)
    using an event specific filter. This trace is then used in the iterative process to calculate the baseline and
    threshold for the next iteration.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        event_coordinates (ndarray): The coordinates of detected events.
        replace_factor (int): Factor for replacing events in the calculation trace.
        replace_gap (int): Gap for replacing events in the calculation trace.

    Returns:
        ndarray: The trace with events replaced (calculation trace).
    """

    assert tuple(trace.shape) == tuple(baseline.shape)

    if len(event_coordinates) == 0:
        return asarray(trace, dtype=DTYPE)

    # Replacing with the baseline
    event_id: cython.size_t
    event_number: cython.size_t = len(event_coordinates)
    padded_trace: cython.double[::1] = zeros(trace.shape[0], dtype=DTYPE)
    padded_trace[:] = trace
    if replace_gap == 0 and replace_factor == 0:
        for event_id in range(event_number):
            padded_trace[event_coordinates[event_id, 0] + 1:event_coordinates[event_id, 1]] = (
                        baseline[event_coordinates[event_id, 0] + 1:event_coordinates[event_id, 1]])
        return asarray(padded_trace, dtype=DTYPE)

    # Pad now to avoid padding in the loop (largest event)
    longest: cython.int = int(ceil(max([event[1] - (event[0] + 1) for event in event_coordinates]) *
                              (replace_factor + replace_gap)))
    if longest > len(trace):
        padded_trace = concatenate((trace[0] * ones(longest - len(trace)), trace[::-1], trace,
                                    trace[::-1], trace[-1] * ones(longest - len(trace))))
    else:
        padded_trace = concatenate((trace[:longest][::-1], trace, trace[-longest:][::-1]))
    padded_trace: cython.double[::1] = padded_trace

    # We remove the events in order of largest to smallest (see: sort_coordinates)
    event_length: cython.longlong
    event_start: cython.longlong
    event_end: cython.longlong
    length: cython.long
    window: cython.double[::1]
    segment: cython.double[::1]
    filtered_segment: cython.double[::1]
    for event_id in range(event_number):
        # Adjust for padding, with extra sauce
        event_length = event_coordinates[event_id, 1] - (event_coordinates[event_id, 0] + 1)
        event_start = (event_coordinates[event_id, 0] + 1) + longest
        event_end = event_coordinates[event_id, 1] + longest

        # Create the filter window
        length = int(round(event_length * (replace_factor + replace_gap)))
        length = length if length % 2 == 1 else length + 1
        # Pointless
        if length == 1:
            continue
        window = ones(length, dtype=DTYPE)
        if replace_gap > 0:
            window[(event_length * replace_factor) // 2:-((event_length * replace_factor) // 2)] = 0
        window = window / sum(window)

        # Filter and replace
        segment = padded_trace[event_start - (len(window) // 2):event_end + (len(window) // 2)]
        filtered_segment = oaconvolve(segment, window, 'same')
        padded_trace[event_start:event_end] = filtered_segment[(len(window) // 2):-(len(window) // 2)]

    return asarray(padded_trace[longest:-longest], dtype=DTYPE)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def locate_events(
        trace: ndarray,
        filtered_trace: ndarray,
        baseline: ndarray,
        threshold: ndarray,
        event_direction: str
) -> ndarray:
    """
    Locate events in the trace based on the filtered trace, baseline, and threshold.

    Args:
        trace (ndarray): The input signal trace.
        filtered_trace (ndarray): The filtered version of the input trace.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').

    Returns:
        ndarray: The coordinates of detected events.
    """

    assert trace.shape == baseline.shape == threshold.shape == filtered_trace.shape

    # Filtered trace for baseline crossing indices
    baseline_idxs: cython.longlong[::1] = baseline_crosses(filtered_trace, baseline)
    # Regular trace for threshold crossing indices
    threshold_idxs: cython.longlong[::1] = threshold_crosses(trace, baseline, threshold, event_direction)
    if len(threshold_idxs) == 0:
        return asarray([], dtype=int64)
    # Iterate to find bounds
    event_coordinates = event_bounds(threshold_idxs, baseline_idxs)

    return event_coordinates


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def event_bounds(threshold_idxs: cython.longlong[::1], baseline_idxs: cython.longlong[::1]) -> ndarray:
    """
    Determine the bounds of events based on threshold and baseline crossing indices.

    Args:
        threshold_idxs (ndarray): Indices where the trace crosses the threshold.
        baseline_idxs (ndarray): Indices where the trace crosses the baseline, this is a masked version of the trace.

    Returns:
        ndarray: The coordinates of detected events.
    """

    spike_start: cython.size_t
    spike_end: cython.size_t = 0
    i: cython.size_t
    baseline_idxs_length: cython.size_t = len(baseline_idxs)
    event_coordinates = []
    # Iterate through the threshold crossing indices
    for i in range(len(threshold_idxs)):
        if threshold_idxs[i] < spike_end:
            continue

        # Iterate backwards to find the start of the event
        spike_start = threshold_idxs[i]
        while baseline_idxs[spike_start] == 0 and spike_start > 0:
            spike_start -= 1

        # Iterate forwards to find the end of the event
        spike_end = threshold_idxs[i] + 1
        while baseline_idxs[spike_end] == 0 and spike_end < baseline_idxs_length - 2:
            spike_end += 1

        # If the end was already passed the cutoff sample
        if spike_end > baseline_idxs_length - 2:
            spike_end = baseline_idxs_length - 2

        # This is the OTHER side of the baseline to the event, on both start/end
        event_coordinates.append((spike_start, spike_end + 1))

    return asarray(event_coordinates, dtype=int64)


@cython.boundscheck(False)
@cython.initializedcheck(False)
def baseline_filter(trace: cython.double[::1], cutoff: cython.double, sample_rate: cython.int) -> ndarray:
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
    length: cython.size_t = int(round(sqrt(0.196202 + ((cutoff * (1 / sample_rate)) ** 2)) /
                                (cutoff * (1 / sample_rate))))
    length = length if length % 2 == 1 else length + 1
    # Pointless
    if length == 1:
        length = 3
    # Ensure the window is not longer than the trace
    if length >= len(trace):
        length = len(trace) - 1
    # Odd length window
    window = ones(length, dtype=DTYPE) / length
    # Add fit for the edge cases
    padded_trace: cython.double[::1] = concatenate((trace[:int(length // 2)][::-1], trace,
                                                    trace[-int(length // 2):][::-1]))
    # Calculate the moving average
    baseline = oaconvolve(padded_trace, window, 'same')
    # Cut off the edges
    baseline = baseline[int(length // 2):-int(length // 2)]

    return baseline


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def baseline_crosses(trace: ndarray, baseline: ndarray) -> ndarray:
    """
    Identify the indices where the trace crosses the baseline.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.

    Returns:
        ndarray: Indices where the trace crosses the baseline, this is a masked version of the trace.
    """

    assert trace.shape == baseline.shape

    # Baseline adjust the trace
    adjusted: cython.double[::1] = trace - baseline
    # Sign the trace then difference to find zero crossings
    # Zero crossing indices are marked by 1, 0 otherwise
    zero_trace = zeros(trace.shape, dtype=int64)
    baseline_cross = where(diff(sign(adjusted)))[0]
    zero_trace[baseline_cross] = 1

    return zero_trace


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def thresholder(
        trace: ndarray,
        baseline: ndarray,
        z_score: cython.double,
        window: Optional[int64] = None,
        event_coordinates: Optional[cython.longlong[:, ::1]] = None,
        method: str = 'iqr'
) -> ndarray:
    """
    Calculate the threshold for event detection based on the trace and baseline. The baseline is subtracted from the
    trace (HPF) and the Wan et al. (doi:10.1186/1471-2288-14-135) method is used to estimate the standard deviation.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        z_score (float): The Z-score for event detection.
        window (int): The window size in samples for threshold calculation. Default is None.
        event_coordinates (ndarray): The coordinates of detected events. Default is None.
        method (str): The method for threshold calculation ('iqr' or 'std'). Default is 'iqr'.

    Returns:
        ndarray: The calculated threshold.
    """

    assert trace.shape == baseline.shape

    # Baseline adjust the trace
    adjusted = trace - baseline
    adjusted_view: cython.double[::1] = adjusted

    # Masking events from the std calculation
    event_id: cython.size_t
    event_number: cython.size_t
    if event_coordinates is not None:
        event_number = len(event_coordinates)
    event_mask = zeros(trace.shape, dtype=int32)
    event_mask_view: cython.long[::1] = event_mask
    if event_coordinates is not None:
        for event_id in range(event_number):
            event_mask_view[event_coordinates[event_id, 0]:event_coordinates[event_id, 1] + 1] = 1

    # Number of segments to divide the trace into
    window: cython.longlong = int(round(window)) if window is not None else len(trace)
    segments: cython.long = len(trace) // window if len(trace) // window else 1

    # Calculate the threshold in segments
    i: cython.long
    start: cython.longlong
    end: cython.longlong
    current_segment: cython.double[::1]
    q1: cython.double
    q3: cython.double
    n: cython.longlong
    standard_deviation = zeros(trace.shape, dtype=DTYPE)
    standard_deviation_view: cython.double[::1] = standard_deviation
    for i in range(segments):
        start = i * window
        end = (i + 1) * window if i != segments - 1 else len(trace)

        # Baseline adjusted segment where events are masked
        current_segment = adjusted[start:end][where(event_mask[start:end] == 0)]
        if len(current_segment) == 0:
            current_segment = adjusted_view[start:end]

        if method == 'iqr':
            # doi:10.1186/1471-2288-14-135
            q1 = quantile(current_segment, 0.25)
            q3 = quantile(current_segment, 0.75)
            n = len(current_segment)
            standard_deviation_view[start:end] = (q3 - q1) / (2 * norm.ppf((0.75 * n - 0.125) / (n + 0.25)))
        else:
            standard_deviation_view[start:end] = std(current_segment)

    return standard_deviation * z_score


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def threshold_crosses(
    trace: ndarray,
    baseline: ndarray,
    threshold: ndarray,
    event_direction: str
) -> ndarray:
    """
    Identify the indices where the trace crosses the threshold.

    Args:
        trace (ndarray): The input signal trace.
        baseline (ndarray): The baseline of the trace.
        threshold (ndarray): The threshold for event detection.
        event_direction (str): The direction of events ('up', 'down', or 'biphasic').

    Returns:
        ndarray: Indices where the trace crosses the threshold, this only returns the first index of each event opposed
        to both the first and last crossing index.
    """

    assert trace.shape == baseline.shape == threshold.shape

    # Baseline adjust the trace
    adjusted = trace - baseline

    crosses = zeros(trace.shape, dtype=int64)
    if event_direction == 'up':
        crosses = where(diff(sign(adjusted - threshold)) > 0)[0]
    elif event_direction == 'down':
        crosses = where(diff(sign(adjusted + threshold)) < 0)[0]
    else:
        crosses = sort(concatenate((where(diff(sign(adjusted + threshold)) < 0)[0],
                                    where(diff(sign(adjusted - threshold)) > 0)[0])))

    return crosses
