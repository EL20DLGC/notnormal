"""
This module provides functions to filter (nano)electrochemical time series data.
"""

from numpy import ceil, log2, pad, quantile, mean, sqrt, arange, roll, asarray, ndarray, max, maximum
from pywt._swt import Modes, _rescale_wavelet_filterbank, _as_wavelet, _check_dtype, idwt_single
from pywt import swt, iswt, threshold
from scipy.stats import norm
from copy import deepcopy
from notnormal.models.base import Trace, Events, WaveletFilterArgs, WaveletFilterResults
from notnormal.utils.methods import get_event_mask
from tqdm import tqdm
import cython

_COMPILED = cython.compiled
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"


"""
Public API
"""

def wavelet_filter(
    trace: ndarray | Trace,
    events: Events,
    wavelet: str = 'sym2',
    u_length: int = 2,
    p_length: int = 2,
    q_pop: float = 0.5,
    q_thresh: float = 0.05,
    mode: str = 'soft',
    full_results: bool = False,
    verbose: bool = False
) -> WaveletFilterResults:
    """
    Perform predictive wavelet filtering using the partial MOSWT (additional splitting of the final two levels) on the
    trace. First, the event lengths are used to determine the decomposition level such that the characteristic frequency
    of the events is captured, which is estimated via the q_pop parameter. The u_length and p_length parameters then
    control the number of splits for the final two levels, giving more resolution in the critical bands where prominent
    event power is located. Second, the events are used to build a float mask (event amplitude mask) to separate signal
    and noise in the wavelet coefficients, allowing highly accurate band specific SNR estimation and subsequent
    denoising. To account for phase shifts in the standard cascade decomposition, as well as additional contributions
    from the tree-like decomposition, the event mask is also decomposed identically to the trace. From this, a robust
    intra-band mask is constructed allowing generalisation of the method to arbitrary wavelets for which computing the
    shift is infeasible. The q_thresh parameter controls the quantile threshold for the mask gating, accounting for
    large spread of the mask in the higher wavelet levels, which would otherwise overestimate the POI support. Finally,
    the SNR thresholds are computed using BayesShrink with extreme value capping. The extreme value cap is analogous to
    the single expected outlier in the two-sided distribution used in NotNormal, exploiting the known distribution to
    prevent large thresholds eliminating signal in levels where they are underrepresented in the population statistics.

    Args:
        trace (ndarray | Trace): The trace to be filtered or a Trace object.
        events (Events): An Events object containing events from the trace.
        wavelet (str): The wavelet to use, see pywavelets documentation for available wavelets.
        u_length (int): Number of splits for the ultimate level (must be 2^N). Default is 2.
        p_length (int): Number of splits for the penultimate level (must be 2^N). Default is 2.
        q_pop (float): Quantile of the population to estimate the characteristic event frequency in [0, 1]. Default is 0.5.
        q_thresh (float): Quantile threshold for mask gating in (0, 1). Default is 0.05.
        mode (str): The thresholding mode, 'soft' or 'hard'. Default is 'soft'.
        full_results (bool): Whether to return the wavelet coefficients in the results. Default is False.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        WaveletFilterResults: An object containing the filtering results.
    """

    if isinstance(trace, Trace):
        trace = trace.trace

    # Get args
    args = WaveletFilterArgs(**{k: v for k, v in locals().items() if k in WaveletFilterArgs.__annotations__})

    # Configure progress bar
    with (tqdm(total=1, desc='Computing event mask and decomposition level', bar_format=_BAR_FORMAT, disable=not verbose)
          as progress):
        # Build event amplitude mask (signal + noise at POIs)
        event_mask = get_event_mask(trace, events, float_mask=True)
        # Get the maximum decomposition level based on the event support
        max_level, lengths = _compute_max_level(events.get_feature('Coordinates'), q_pop=q_pop)
        # Update progress bar
        progress.update(1)

    with tqdm(total=1, desc='Decomposing trace', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Decompose the trace using partial MOSWT
        coeffs = partial_moswt(trace, wavelet, max_level, u_length=u_length, p_length=p_length)
        progress.update(1)

    with tqdm(total=1, desc='Decomposing event mask', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Decompose the event mask using partial MOSWT
        coeffs_mask = partial_moswt(event_mask, wavelet, max_level, u_length=u_length, p_length=p_length)
        progress.update(1)

    with tqdm(total=1, desc='Computing SNR', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Compute signal variance on cDn[cDn_em > gate(q_thresh)] and noise variance [cDn_em < gate(q_thresh)]
        signal_var, noise_var = _get_wavelet_vars(coeffs, coeffs_mask, len(trace), q_thresh=q_thresh)
        progress.update(1)

    with tqdm(total=1, desc='Computing thresholds', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Compute threshold Tn = nv_n / sqrt(max(sv_n, 1e-15)) (BayesShrink) (with EVC two-sided outliers)
        thresholds = _get_snr_thresholds(signal_var, noise_var, len(trace))
        progress.update(1)

    with tqdm(total=1, desc='Applying thresholds', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Apply soft or hard thresholding
        filtered_coeffs = _apply_snr_thresholds(coeffs, thresholds, mode=mode)
        progress.update(1)

    with tqdm(total=1, desc='Reconstructing trace', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Inverse partial MOSWT to get the filtered trace
        filtered_trace = inverse_partial_moswt(filtered_coeffs, wavelet, max_level, u_length=u_length, p_length=p_length)
        progress.update(1)

    # Create results object
    results = WaveletFilterResults(args, filtered_trace, max_level, lengths, signal_var, noise_var, thresholds)

    # If full results are requested, return the coefficients
    if full_results:
        results.coeffs, results.coeffs_mask, results.filtered_coeffs = coeffs, coeffs_mask, filtered_coeffs

    return results


def partial_moswt(
    trace: ndarray | Trace,
    wavelet: str,
    max_level: int,
    u_length: int = 2,
    p_length: int = 2
) -> list[ndarray]:
    """
    Perform the partial MOSWT (additional splitting of the final two levels) on the trace.

    Args:
        trace (ndarray): The trace to be decomposed or a Trace object.
        wavelet (str): The wavelet to use for decomposition, see pywavelets documentation for available wavelets.
        max_level (int): The maximum level of decomposition.
        u_length (int): Number of splits for the ultimate level (must be 2^N). Default is 2.
        p_length (int): Number of splits for the penultimate level (must be 2^N). Default is 2.

    Returns:
        list[ndarray]: List of coefficients from the trace decomposition, in the form [cAn, cDn, ..., cD1].
    """

    if isinstance(trace, Trace):
        trace = trace.trace

    # Get the padding level (max decomposition level including further tree-like decompositions)
    pad_level = int(maximum(max_level + u_length, max_level + p_length - 1))

    # Pad the trace to allow decomposition
    padded_trace = _moswt_pad(trace, pad_level)

    # Compute MOSWT down to max level with splits, return cAn, cDn, ..., cD1
    coeffs = _partial_moswt(padded_trace, wavelet, max_level, u_length, p_length)

    # Return the coefficients
    return coeffs


def inverse_partial_moswt(
    coeffs: list[ndarray],
    wavelet: str,
    max_level: int,
    u_length: int = 2,
    p_length: int = 2
) -> ndarray:
    """
    Perform the inverse partial MOSWT (additional unsplitting of the final two levels) on the coefficients.

    Args:
        coeffs (list[ndarray]): The partial MOSWT coefficients, in the form [cAn, cDn, ..., cD1].
        wavelet (str): The wavelet used for decomposition.
        max_level (int): The maximum level used for decomposition.
        u_length (int): Number of decomposition splits used for the ultimate level (must be 2^N). Default is 2.
        p_length (int): Number of decomposition splits used for the penultimate level (must be 2^N). Default is 2.

    Returns:
        ndarray: The inverse partial MOSWT of the coefficients.
    """

    # Inverse MOSWT the coefficients
    inversed = _inverse_partial_moswt(coeffs, wavelet, max_level, u_length, p_length)

    # Get the padding level (max decomposition level including further tree-like decompositions)
    pad_level = int(maximum(max_level + u_length, max_level + p_length - 1))

    # Unpad the inversed trace
    inversed = _moswt_unpad(inversed, pad_level)

    # Return the inverse
    return inversed


"""
Internal API
"""

def _compute_max_level(coordinates: list[tuple[int, int]], q_pop: float = 0.5) -> tuple[int, ndarray]:
    """
    Compute the maximum level of decomposition based on the minimum decomposition level required to capture the
    characteristic frequency of the events given by their coordinates. The characteristic frequency is estimated
    from the quantile of event lengths, where the quantile is defined by q_thresh in [0, 1].

    Args:
        coordinates (list[tuple[int, int]]): List of tuple event coordinates.
        q_pop (float): Quantile of the population to estimate the characteristic event frequency in [0, 1]. Default is 0.5.

    Returns:
        tuple[int, ndarray]: The maximum decomposition level and the event lengths used for the computation.
    """

    if len(coordinates) < 2:
        raise ValueError("At least two sets of coordinates are required to compute the maximum level.")
    if not (0 <= q_pop <= 1):
        raise ValueError("Population quantile must be in the range [0, 1].")

    # Get the event lengths from the coordinates
    lengths = asarray([coord[1] - coord[0] + 1 for coord in coordinates])

    # Compute max level based on the event characteristic frequency: fs / 2 ** (max_level + 1) <= fs / length
    event_length = quantile(lengths, q_pop)
    max_level = int(ceil(log2(event_length) - 1))

    return max_level, lengths


def _moswt_pad(trace: ndarray, max_level: int) -> ndarray:
    """
    Pad the trace to the next power of two for partial MOSWT.

    Args:
        trace (ndarray): The trace to be padded.
        max_level (int): The maximum level of decomposition (including further tree-like decompositions).

    Returns:
        ndarray: The padded trace.
    """

    if len(trace) == 0:
        raise ValueError("Trace is empty. Cannot perform padding.")
    if max_level < 0:
        raise ValueError("Max level must be a non-negative integer.")

    # Required padding: N_pad = (2 ** j) * ceil[N / (2 ** j)] - N
    pad_len = int((2 ** max_level) * ceil(len(trace) / (2 ** max_level)) - len(trace))
    if pad_len == 0:
        return trace
    padded = pad(trace, (0, pad_len), mode='symmetric')
    return padded


def _moswt_unpad(trace: ndarray, max_level: int) -> ndarray:
    """
    Unpad the trace after inverse partial MOSWT (padding calculated implicitly from max level).

    Args:
        trace (ndarray): The trace after inverse partial MOSWT.
        max_level (int): The maximum level used in decomposition (including further tree-like decompositions).

    Returns:
        ndarray: The unpadded trace.
    """

    if len(trace) == 0:
        raise ValueError("Trace is empty. Cannot perform unpadding.")
    if max_level < 0:
        raise ValueError("Max level must be a non-negative integer.")

    # Required padding: N_pad = (2 ** j) * ceil[N / (2 ** j)] - N
    pad_len = int((2 ** max_level) * ceil(len(trace) / (2 ** max_level)) - len(trace))
    if pad_len == 0:
        return trace

    if pad_len > len(trace):
        raise ValueError(f"Padding length {pad_len} is greater than trace length {len(trace)}. Cannot unpad.")

    return trace[:len(trace) - pad_len]


def _partial_moswt(
    trace: ndarray,
    wavelet: str,
    max_level: int,
    u_length: int,
    p_length: int
) -> list[ndarray]:
    """
    Perform the partial MOSWT (additional splitting of the final two levels) on the trace.

    Args:
        trace (ndarray): The trace to be decomposed.
        wavelet (str): The wavelet to use for decomposition, see pywavelets documentation for available wavelets.
        max_level (int): The maximum level of decomposition.
        u_length (int): Number of splits for the ultimate level (must be 2^N).
        p_length (int): Number of splits for the penultimate level (must be 2^N).

    Returns:
        list[ndarray]: A list of coefficients from the trace decomposition, in the form [cAn, cDn, ..., cD1].
    """

    if max_level < 2:
        raise ValueError("Max level must be greater than 2 for partial MOSWT.")
    if u_length < 1 or p_length < 1 or (u_length % 2 and u_length != 1) or (p_length % 2 and p_length != 1):
        raise ValueError("Split length must be a power of two (including 1).")

    # Perform regular MOSWT first
    coeffs = swt(trace, wavelet, level=max_level, trim_approx=True, norm=True)

    # Split the last level detail into u_length
    ultimate = _split_band(coeffs[1], wavelet, u_length, max_level)

    # Split the penultimate level detail into p_length
    penultimate = _split_band(coeffs[2], wavelet, p_length, max_level - 1)

    # Reconstruct the coeffs
    return [coeffs[0]] + ultimate + penultimate + coeffs[3:]


def _inverse_partial_moswt(
    coeffs: list[ndarray],
    wavelet: str,
    max_level: int,
    u_length: int,
    p_length: int
) -> ndarray:
    """
    Perform the inverse partial MOSWT (additional unsplitting of the final two levels) on the coefficients.

    Args:
        coeffs (list[ndarray]): The partial MOSWT coefficients, in the form [cAn, cDn, ..., cD1].
        wavelet (str): The wavelet used for decomposition.
        max_level (int): The maximum level used for decomposition.
        u_length (int): Number of decomposition splits for the ultimate level (must be 2^N).
        p_length (int): Number of decomposition splits for the penultimate level (must be 2^N).

    Returns:
        ndarray: The inverse partial MOSWT of the coefficients.
    """

    if max_level < 2:
        raise ValueError("Max level must be greater than 2 for partial MOSWT.")
    if u_length < 1 or p_length < 1 or (u_length % 2 and u_length != 1) or (p_length % 2 and p_length != 1):
        raise ValueError("Split length must be a power of two (including 1).")
    if len(coeffs) != max_level + p_length + u_length - 1:
        raise ValueError(f"Expected {max_level + p_length + u_length - 1} coefficients, got {len(coeffs)}.")

    # Easier to work with a copy
    coeffs = deepcopy(coeffs)

    # Reconstruct the last level
    ultimate = _unsplit_band(coeffs[1:1 + u_length], wavelet, max_level)

    # Reconstruct the penultimate level
    penultimate = _unsplit_band(coeffs[1 + u_length:1 + u_length + p_length], wavelet, max_level - 1)

    # Reconstruct the coeffs
    coeffs = [coeffs[0], ultimate, penultimate] + coeffs[1 + u_length + p_length:]

    # Inverse MOSWT the reconstructed coefficients
    inversed = iswt(coeffs, wavelet, norm=True)

    return inversed


def _split_band(band: ndarray, wavelet: str, n_splits: int, start_level: int) -> list[ndarray]:
    """
    Split a single band into n_splits parts using SWT.

    Args:
        band (ndarray): The band to be split.
        wavelet (str): The wavelet to use for splitting, see pywavelets documentation for available wavelets.
        n_splits (int): The number of splits to perform (must be 2^N).
        start_level (int): The starting level for the split (which level is this band).
    Returns:
        list[ndarray]: A list of split bands.
    """

    assert n_splits > 0
    assert start_level > 0

    if n_splits == 1:
        return [band]

    assert n_splits % 2 == 0

    # Recursively split them with SWT
    ca, cd = swt(band, wavelet, level=1, start_level=start_level, trim_approx=True, norm=True)
    return _split_band(ca, wavelet, n_splits // 2, start_level + 1) + _split_band(cd, wavelet, n_splits // 2, start_level + 1)


def _unsplit_band(bands: list[ndarray], wavelet: str, start_level: int) -> ndarray:
    """
    Unsplit a list of bands into a single band using iSWT.

    Args:
        bands (list[ndarray]): The bands to be unsplit.
        wavelet (str): The wavelet used for splitting.
        start_level (int): The starting level for the original split (which level was the parent).
    Returns:
        ndarray: The original band.
    """

    assert len(bands) > 0
    assert start_level > 0

    if len(bands) == 1:
        return bands[0]

    assert len(bands) % 2 == 0

    # Recursively unsplit them with iSWT
    mid = len(bands) // 2
    low  = _unsplit_band(bands[:mid], wavelet, start_level + 1)
    high = _unsplit_band(bands[mid:], wavelet, start_level + 1)
    return _iswt_start_level([(low, high)], wavelet, start_level=start_level, norm=True)


def _iswt_start_level(
    coeffs: list[tuple[ndarray, ndarray]],
    wavelet: str,
    start_level: int = 0,
    norm: bool = False
) -> ndarray:
    """
    This function is a copy of pywavelets iswt function, but with an added start_level parameter to enable
    reconstruction of split bands from partial MOSWT, see: _split_band and _unsplit_band. Note, no error checking, use
    with care.

    Args:
        coeffs (list[tuple[ndarray, ndarray]]): A list of tuples containing coefficients to be reconstructed,
                                                in the form [(cAn, cDn), ..., (cA1, cD1)].
        wavelet (str): The wavelet to use for reconstruction.
        start_level (int): The starting level from the decomposition (which level is the parent).
        norm (bool): Whether to normalise the wavelet filterbank.

    Returns:
        ndarray: The reconstructed band from the coefficients.
    """

    # Copy for ease
    coeffs = deepcopy(coeffs)
    wavelet = deepcopy(wavelet)

    # Do the filter normalisation
    wave = _as_wavelet(wavelet)
    if norm:
        wave = _rescale_wavelet_filterbank(wave, sqrt(2.0))

    # Get correct mode
    mode = Modes.from_object('periodization')
    # Start from the *highest* approximation, exactly as pywt.iswt does
    output = asarray(coeffs[0][0])
    # Main reconstruction loop, identical to pywt.iswt, except that step_size is offset by start_level
    for j in range(len(coeffs), 0, -1):
        step_size = 2 ** (start_level + j - 1)
        cd = asarray(coeffs[len(coeffs) - j][1], dtype=_check_dtype(output))
        for first in range(step_size):
            idx = arange(first, output.shape[output.ndim - 1], step_size)
            even = idx[0::2]
            odd = idx[1::2]
            x_even = idwt_single(output[..., even], cd[..., even], wave, mode)
            x_odd = idwt_single(output[..., odd], cd[..., odd], wave, mode)
            x_odd = roll(x_odd, 1, axis=-1)
            output[..., idx] = (x_even + x_odd) / 2.0

    return output


def _get_wavelet_vars(
    coeffs: list[ndarray],
    coeffs_mask: list[ndarray],
    length: int,
    q_thresh: float = 0.05
) -> tuple[ndarray, ndarray]:
    """
    Compute the signal and noise variance per band from the trace and mask coefficients. The noise variance is
    estimated from the indices in the (absolute) mask coefficients that are below a quantile threshold
    (q_thresh, in [0, 1]). The signal + noise variance is estimated from the indices above this threshold, the signal
    variance is then uncovered by subtracting the noise variance.

    Args:
        coeffs (list[ndarray]): The trace coefficients, in the form [cAn, cDn, ..., cD1].
        coeffs_mask (list[ndarray]): The mask coefficients, in the form [cAn, cDn, ..., cD1].
        length (int): The length of the original trace.
        q_thresh (float): Quantile threshold for mask gating in (0, 1). Default is 0.05.

    Returns:
        tuple[ndarray, ndarray]: The signal variance and noise variance for each band.
    """

    if len(coeffs) != len(coeffs_mask):
        raise ValueError("Wavelet and mask coefficients must have the same length.")
    if not (0 < q_thresh < 1):
        raise ValueError("Threshold quantile must be in the range (0, 1).")

    signal_var = []
    noise_var = []
    for coeff, coeff_mask in zip(coeffs[1:], coeffs_mask[1:]):
        # Truncate to unpadded length
        u_coeff = coeff[:length]
        u_coeff_mask = abs(coeff_mask[:length])

        # Gate the variance estimates
        gate = quantile(u_coeff_mask[u_coeff_mask != 0], q_thresh)

        # Mean of squares as coefficients are 0 mean
        noise_var.append(mean(u_coeff[u_coeff_mask < gate] ** 2))
        total_var = mean(u_coeff[u_coeff_mask > gate] ** 2)

        # Signal variance is total variance minus noise variance
        signal_var.append(maximum(0, total_var - noise_var[len(noise_var) - 1]))

    return asarray(signal_var), asarray(noise_var)


def _get_snr_thresholds(
    signal_vars: ndarray,
    noise_vars: ndarray,
    length: int
) -> list[float]:
    """
    Compute SNR thresholds for each band using BayesShrink and extreme value capping, with signal and noise variances
    estimated from _get_wavelet_vars.

    Args:
        signal_vars (ndarray): The signal variance for each band.
        noise_vars (ndarray): The noise variance for each band.
        length (int): The length of the original trace.

    Returns:
        list[float]: The threshold for each band.
    """

    if len(signal_vars) != len(noise_vars):
        raise ValueError("Signal and noise variances must have the same length.")

    thresholds = []
    for sv, nv in zip(signal_vars, noise_vars):
        # Compute threshold Tn = nv_n / sqrt(max(sv_n, 1e-15)) (BayesShrink)
        bayes = nv / sqrt(maximum(sv, 1e-15))
        # Cap with the expected number of two-sided outliers above +-T is 1
        outlier_cap = sqrt(nv) * norm.ppf(1.0 - ((1.0 / length) / 2.0))
        # Append the minimum of the two thresholds
        thresholds.append(min(bayes, outlier_cap))

    return thresholds


def _apply_snr_thresholds(coeffs: list[ndarray], thresholds: list[float], mode: str = 'soft') -> list[ndarray]:
    """
    Apply the SNR thresholds derived from _get_snr_thresholds to the wavelet coefficients.

    Args:
        coeffs (list[ndarray]): The wavelet coefficients, in the form [cAn, cDn, ..., cD1].
        thresholds (list[float]): The thresholds for each band.
        mode: str: The thresholding mode, 'soft' or 'hard'. Default is 'soft'.

    Returns:
        list[ndarray]: The thresholded wavelet coefficients, in the form [cAn, cDn, ..., cD1].
    """

    if len(coeffs) != len(thresholds) + 1:
        raise ValueError("Coefficients and thresholds length mismatch.")

    # Get the last approximation coefficient
    filtered_coeffs = [coeffs[0]]
    for detail, thresh in zip(coeffs[1:], thresholds):
        # Soft thresholding by default: cD2n,k = sign(cDn,k) * max(|cDn,k| - Tn, 0)
        filtered_coeffs.append(threshold(detail, thresh, mode=mode))

    return filtered_coeffs
