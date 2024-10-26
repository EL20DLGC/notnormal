"""
This module provides functions to simulate traces with noise, baseline, and events for (nano)electrochemical time series
data.
"""

from stochastic.processes.noise import FractionalGaussianNoise as Fgn
from stochastic.processes.continuous.fractional_brownian_motion import FractionalBrownianMotion as Fbm
from numpy import mean, std, zeros, arange, sqrt
from numpy.random import default_rng
from scipy.signal import get_window


def simulate_trace(
    length: int,
    sample_rate: int,
    noise_dict: dict = None,
    baseline_dict: dict = None,
    event_dicts: list[dict] = None,
):
    """
    Simulate a trace with noise, baseline, and events.

    Args:
        length (int): The length of the trace.
        sample_rate (int): The sample rate of the trace.
        noise_dict (dict, optional): Dictionary containing noise parameters. Default is None.
        baseline_dict (dict, optional): Dictionary containing baseline parameters. Default is None.
        event_dicts (list[dict], optional): List of dictionaries containing event parameters. Default is None.

    Returns:
        tuple: A tuple containing the trace, coordinate trace, events, and baseline.
    """

    if noise_dict is None:
        noise_dict = {
            'regimes': [],
            'sigma': 1
        }
    if baseline_dict is None:
        baseline_dict = {
            'offset': 0,
            'change': 0,
            'type': 'linear'
        }

    # Simulate wGn to start
    noise = simulate_regime(0.5, 1, length)
    noise_power = 1 / (sample_rate / 2)

    # Now add the noise regimes
    for i, regime in enumerate(noise_dict['regimes']):
        # Get the c value from the crossover and alpha value
        gamma = (2 * regime[0]) - 1
        c = noise_power * (regime[1] ** gamma)
        # Then get the mean value
        f = arange(sample_rate / length, (sample_rate / 2) + 1, 0.1)
        regime_mean = mean(c / (f ** gamma))
        # Then convert the mean to the standard deviation
        regime_std = sqrt(regime_mean * (sample_rate / 2))
        # Sum the noise
        noise += simulate_regime(regime[0], regime_std, length)

    # Rescale to get the desired standard deviation
    current_mean = mean(noise)
    current_std = std(noise)
    for i in range(length):
        noise[i] += (noise_dict['sigma'] - current_std) * ((noise[i] - current_mean) / current_std)
    trace = noise

    # Add the events
    events = []
    for event_dict in event_dicts:
        events.append(simulate_events(
            event_dict['amplitude'],
            event_dict['amplitude_std'],
            event_dict['duration'],
            event_dict['duration_std'],
            event_dict['window'],
            event_dict['direction'],
            event_dict['event_number'],
            sample_rate
        ))

    # Combine the events and reorder
    events = [event for sublist in events for event in sublist]
    events = sorted(events, key=lambda x: x['Coordinates'][0])
    events = [{'ID': i + 1, **event} for i, event in enumerate(events)]
    # Add the events to the trace
    trace, events, coordinate_trace = add_events(trace, events, length, sample_rate)

    # Add the baseline
    baseline = simulate_baseline(baseline_dict['offset'], baseline_dict['change'], baseline_dict['type'], length)
    trace += baseline

    return trace, coordinate_trace, events, baseline


def simulate_regime(alpha: float, sigma: float, length: int):
    """
    Simulate a single noise regime.

    Args:
        alpha (float): The alpha parameter for the noise, H = alpha for fGn and H = alpha - 1 for fBm.
        sigma (float): The standard deviation of the noise.
        length (int): The length of the noise regime.

    Returns:
        ndarray: The simulated noise regime.
    """

    if alpha > 1:
        regime = Fbm(hurst=alpha - 1, t=length)
        regime = regime.sample(length - 1)
    else:
        regime = Fgn(hurst=alpha, t=length)
        regime = regime.sample(length)

    # Rescale to get the desired standard deviation
    regime_mean = mean(regime)
    regime_std = std(regime)
    for i in range(length):
        regime[i] += (sigma - regime_std) * ((regime[i] - regime_mean) / regime_std)

    return regime


def simulate_baseline(offset: float, change: float, baseline_type: str, length: int):
    """
    Simulate a baseline.

    Args:
        offset (float): The initial offset of the baseline.
        change (float): The change in the baseline over time.
        baseline_type (str): The type of baseline ('constant' or 'linear').
        length (int): The length of the baseline.

    Returns:
        list: The simulated baseline.
    """

    if baseline_type == 'constant':
        baseline = [offset] * length
    else:
        change = change / length
        baseline = [offset + i * change for i in range(length)]

    return baseline


def simulate_events(
    amplitude: float,
    amplitude_std: float,
    duration: float,
    duration_std: float,
    window: str,
    direction: str,
    event_number: int,
    sample_rate: int
):
    """
    Simulate events.

    Args:
        amplitude (float): The mean amplitude of the events.
        amplitude_std (float): The standard deviation of the event amplitudes.
        duration (float): The mean duration of the events in milliseconds.
        duration_std (float): The standard deviation of the event durations in milliseconds.
        window (str): The window function to use for the event shape.
        direction (str): The direction of the events ('up' or 'down').
        event_number (int): The number of events to simulate.
        sample_rate (int): The sample rate of the trace.

    Returns:
        list: A list of simulated event dictionaries.
    """

    rng = default_rng()
    # Calculate amplitudes of these events
    amplitudes = abs(rng.normal(amplitude, amplitude_std, event_number))
    # Calculate the durations of these events (convert from ms)
    event_duration_mean = int(duration / (1e3 / sample_rate))
    event_duration_std = int(duration_std / (1e3 / sample_rate))
    durations = abs(rng.normal(event_duration_mean, event_duration_std, event_number))

    # Generate the events
    events = []
    for i in range(event_number):
        event_vector = get_window(window, int(durations[i]))
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


def add_events(trace: list, events: list, length: int, sample_rate: int):
    """
    Add events to a trace and modify the event dictionaries accordingly.

    Args:
        trace (list): The original trace to which events will be added.
        events (list): A list of event dictionaries to add to the trace.
        length (int): The length of the trace.
        sample_rate (int): The sample rate of the trace.

    Returns:
        tuple: A tuple containing the modified trace, event dictionaries and the coordinate trace.
    """

    rng = default_rng()
    coordinate_trace = zeros(length)
    for event in events:
        # Get random start coordinates
        start_idx = rng.integers(0, length - len(event['Vector']))
        end_idx = (start_idx + len(event['Vector'])) - 1
        # Add the event to the trace
        trace[start_idx:end_idx + 1] += event['Vector']
        # Readjust coordinates for realism
        while trace[start_idx] > 0 and start_idx > 0:
            start_idx -= 1
        start_idx = start_idx + 1

        while trace[end_idx] > 0 and end_idx < length - 1:
            end_idx += 1
        end_idx = end_idx - 1
        # Add the event to the coordinate trace
        coordinate_trace[start_idx:end_idx + 1] = 1
        # Modify the event dictionary (this is what the true event would be after adding to the trace)
        event['Coordinates'] = [start_idx, end_idx]
        event['Vector'] = trace[start_idx:end_idx + 1]
        event['Amplitude'] = max(abs(event['Vector']))
        event['Duration'] = len(event['Vector']) / sample_rate * 1e3
        event['Area'] = sum(abs(event['Vector'])) / sample_rate

    return trace, events, coordinate_trace
