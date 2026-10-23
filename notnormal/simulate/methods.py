# Copyright (C) 2025-2026 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides functions to simulate traces with noise, baseline, and events for (nano)electrochemical time series
data. WARNING: This module will be deprecated on conclusion of my PhD, it is a rather basic implementation but offers
a quick and easy way to produce testing data.
"""

from typing import Optional, Any
from notnormal.reconstruct.noise import _generate_regime
from numpy import mean, std, zeros, arange, sqrt, sum, ndarray, max, asarray, maximum, argmax
from numpy.random import default_rng
from scipy.signal import get_window
from copy import deepcopy
import cython

_COMPILED = cython.compiled


"""
Public API
"""

def simulate_trace(
    length: int,
    sample_rate: int,
    noise_dict: Optional[dict[str, Any]] = None,
    baseline_dict: Optional[dict[str, Any]] = None,
    event_dicts: Optional[list[dict[str, Any]]] = None,
    wgn_base: bool = True,
    original_bounds: bool = False,
    random_state: Optional[int] = None
) -> tuple[ndarray, ndarray, list[dict[str, Any]], ndarray]:
    """
    Simulate a trace with noise, baseline, and events.

    Args:
        length (int): The length of the trace.
        sample_rate (int): The sample rate of the trace.
        noise_dict (dict[str, Any] | None): Dictionary containing noise parameters with 'regimes' in the format
            (alpha, crossover frequency) and 'sigma' specifying the overall sigma. See: simulate_regime().
            Default is None.
        baseline_dict (dict[str, Any] | None): Dictionary containing baseline parameters. See: simulate_baseline().
            Default is None.
        event_dicts (list[dict[str, Any]] | None): List of dictionaries containing event parameters. See: simulate_events().
            Default is None.
        wgn_base (bool): Whether to start the noise with a white Gaussian noise base for the noise_dict crossovers.
            Default is True.
        original_bounds (bool): Whether the event bounds are original or noise aware. Default is False.
        random_state (int | None): Seed for the random number generator for reproducibility. Default is None.

    Returns:
        tuple[ndarray, ndarray, list[dict[str, Any]], ndarray]: A tuple containing the trace, coordinate trace, events,
            and baseline.
    """

    if noise_dict is None:
        noise_dict = {
            'regimes': [],
            'sigma': 1
        }
    if baseline_dict is None:
        baseline_dict = {
            'voltage': 0,
        }
    if event_dicts is None:
        event_dicts = []

    # Random seeding
    rng = default_rng(random_state)
    def _next_seed() -> int:
        return int(rng.integers(0, 2 ** 32 - 1))

    # Simulate noise
    noise = zeros(length)
    noise_power = 1 / (sample_rate / 2)

    # Simulate wGn to start is requested
    if wgn_base is True:
        noise += simulate_regime(0.5, 1, length, sample_rate, _next_seed())

    # Now add the noise regimes
    for i, regime in enumerate(noise_dict['regimes']):
        # Get the c value from the crossover and alpha value
        gamma = (2 * regime[0]) - 1
        c = noise_power * (regime[1] ** gamma)
        # Integrate and get the standard deviation
        integral = lambda x: c * ((x ** (1 - gamma)) / (1 - gamma))
        regime_std = sqrt(integral(sample_rate / 2) - integral(sample_rate / length))
        # Sum the noise
        noise += simulate_regime(regime[0], regime_std, length, sample_rate, _next_seed())

    # Rescale to get the desired standard deviation
    noise -= mean(noise)
    noise *= noise_dict['sigma'] / std(noise)
    trace = noise

    # Create the events
    events = []
    for event_dict in event_dicts:
        events.append(simulate_events(**event_dict, sample_rate=sample_rate, random_state=_next_seed()))
    events = [event for sublist in events for event in sublist]

    # Add the events to the trace
    trace, events, coordinate_trace = _add_events(trace, events, length, sample_rate, original_bounds, _next_seed())

    # Add the baseline
    baseline = simulate_baseline(length, sample_rate, **baseline_dict)
    trace += baseline

    return trace, coordinate_trace, events, baseline


def simulate_regime(
    alpha: float,
    sigma: float,
    length: int,
    sample_rate: int,
    random_state: Optional[int] = None
) -> ndarray:
    """
    Simulate a single noise regime.

    Args:
        alpha (float): The alpha parameter for the noise.
        sigma (float): The standard deviation of the noise.
        length (int): The length of the noise regime.
        sample_rate (int): The sample rate of the trace.
        random_state (int | None): Seed for the random number generator for reproducibility. Default is None.

    Returns:
        ndarray: The simulated noise regime.
    """

    # Generate using the spectral method
    regime = _generate_regime(alpha, 1.0, length, sample_rate, random_state=random_state)

    # Rescale to get the desired standard deviation
    regime -= mean(regime)
    regime *= sigma / std(regime)

    return regime


def simulate_baseline(
    length: int,
    sample_rate: int,
    voltage: float,
    resistance: Optional[float] = None,
    change: Optional[float] = None,
    cottrell_numerator: Optional[float] = None,
) -> ndarray:
    """
    Simulate a baseline.

    Args:
        length (int): The length of the baseline.
        sample_rate (int): The sample rate of the trace.
        voltage (float): The initial voltage offset of the baseline.
        resistance (float | None): The initial resistance. Default is None.
        change (float | None): The change in the resistance over time. Default is None.
        cottrell_numerator (float | None): The numerator of the Cottrell equation. Default is None.
    Returns:
        ndarray: The simulated baseline.
    """

    time_vector = arange(1, length + 1) / sample_rate
    if resistance is None:
        baseline = [voltage] * length
    elif change is None:
        baseline = [voltage / resistance] * length
    elif cottrell_numerator is None:
        baseline = voltage / (resistance + (change * time_vector))
    else:
        baseline = voltage / (resistance + (change * time_vector)) + cottrell_numerator / sqrt(time_vector)

    return asarray(baseline)


def simulate_events(
    amplitude: float,
    amplitude_std: float,
    duration: float,
    duration_std: float,
    window: str,
    direction: str,
    event_number: int,
    sample_rate: int,
    amplitude_min: float = 1e-12,
    duration_min: int = 1,
    random_state: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Simulate events.

    Args:
        amplitude (float): The mean amplitude of the events.
        amplitude_std (float): The standard deviation of the event amplitudes.
        duration (float): The mean duration of the events in milliseconds.
        duration_std (float): The standard deviation of the event durations in milliseconds.
        window (str): The window function to use for the event shape, 'mixed' for different types.
        direction (str): The direction of the events ('up' or 'down').
        event_number (int): The number of events to simulate.
        sample_rate (int): The sample rate of the trace.
        amplitude_min (float): The minimum amplitude of the events. Default is 1e-12.
        duration_min (int): The minimum duration of the events in samples. Default is 1.
        random_state (int | None): Seed for the random number generator for reproducibility. Default is None.

    Returns:
        list[dict[str, Any]]: A list of simulated event dictionaries.
    """

    if event_number == 0:
        return []

    rng = default_rng(seed=random_state)

    # Convert duration from ms
    event_duration_mean = int(duration / (1e3 / sample_rate))
    event_duration_std = int(duration_std / (1e3 / sample_rate))

    # Sample amplitudes and durations from normal distributions, ensuring positive and clamped
    amplitudes = abs(rng.normal(amplitude, amplitude_std, event_number))
    amplitudes = maximum(amplitudes, amplitude_min)
    durations = abs(rng.normal(event_duration_mean, event_duration_std, event_number))
    durations = maximum(durations.astype(int), duration_min)

    # Generate the events
    events = []
    windows = ['boxcar', 'triang', 'hamming', 'hann', 'bartlett', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
               'barthann', 'cosine', 'tukey', 'taylor', 'lanczos']
    for i in range(event_number):
        current_window = windows[rng.integers(0, len(windows))] if window == 'mixed' else window
        event_vector = get_window(current_window, int(durations[i]))
        event_vector = amplitudes[i] * event_vector * (-1 if direction == 'down' else 1)
        events.append({
            'ID': i + 1,
            'Coordinates': [0, 0],
            'Vector': event_vector,
            'Amplitude': amplitudes[i],
            'Duration': len(event_vector) / sample_rate * 1e3,
            'Area': sum(abs(event_vector)) / sample_rate,
            'Direction': direction,
            'Outlier': False
        })

    return events


"""
Internal API
"""

def _add_events(
    trace: ndarray,
    events: list[dict[str, Any]],
    length: int,
    sample_rate: int,
    original_bounds: bool = False,
    random_state: Optional[int] = None
) -> tuple[ndarray, list[dict[str, Any]], ndarray]:
    """
    Add events to a trace and modify the event dictionaries accordingly.

    Args:
        trace (ndarray): The original trace to which events will be added.
        events (list[dict[str, Any]]): A list of event dictionaries to add to the trace.
        length (int): The length of the trace.
        sample_rate (int): The sample rate of the trace.
        original_bounds (bool): Whether the event bounds are original or noise aware. Default is False.
        random_state (int | None): Seed for the random number generator for reproducibility. Default is None.

    Returns:
        tuple[ndarray, list[dict[str, Any]], ndarray]: A tuple containing the modified trace, event dictionaries and
            the coordinate trace.
    """

    rng = default_rng(seed=random_state)
    coordinate_trace = zeros(length)
    i = 0
    attempts = 0
    max_attempts = 100
    while i < len(events):
        attempts += 1
        event = events[i]
        # Get random coordinates
        start_idx = rng.integers(0, length - len(event['Vector']) + 1)
        end_idx = (start_idx + len(event['Vector'])) - 1

        # Save the old segment and add to the trace
        idxs = (start_idx, end_idx)
        segment = deepcopy(trace[start_idx:end_idx + 1])
        trace[start_idx:end_idx + 1] += event['Vector']

        # Readjust coordinates for realism
        if original_bounds is False:
            start_idx += argmax(abs(event['Vector']))
            end_idx = start_idx
            if event['Direction'] == 'up':
                while trace[start_idx] > 0 and start_idx > 0:
                    start_idx -= 1

                while trace[end_idx] > 0 and end_idx < length - 1:
                    end_idx += 1
            else:
                while trace[start_idx] < 0 and start_idx > 0:
                    start_idx -= 1

                while trace[end_idx] < 0 and end_idx < length - 1:
                    end_idx += 1

        # Check if the coordinates are already occupied
        if coordinate_trace[start_idx:end_idx + 1].any() and attempts <= max_attempts:
            trace[idxs[0]:idxs[1] + 1] = segment
            continue
        else:
            attempts = 0
            coordinate_trace[start_idx:end_idx + 1] = 1

        # Original noise-free event information
        event['Original Coordinates'] = [idxs[0], idxs[1]]
        event['Original Vector'] = event['Vector']
        event['Original Amplitude'] = event['Amplitude']
        event['Original Duration'] = event['Duration']
        event['Original Area'] = event['Area']

        # Noise aware event information
        event['Coordinates'] = [start_idx, end_idx]
        event['Vector'] = trace[start_idx:end_idx + 1]
        event['Amplitude'] = max(abs(event['Vector']))
        event['Duration'] = len(event['Vector']) / sample_rate * 1e3
        event['Area'] = sum(abs(event['Vector'])) / sample_rate
        i += 1

    # Reorder the events
    events = sorted(events, key=lambda x: x['Coordinates'][0])
    events = [{'ID': i + 1, **event} for i, event in enumerate(events)]

    return trace, events, coordinate_trace
