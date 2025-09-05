# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides functions to reconstruct and generate noise for (nano)electrochemical time series data.
"""

from typing import Optional, Any
from numpy.random import default_rng
from tqdm import tqdm
from numpy import unique, logspace, log, e, pi, abs, sum, clip, max, min, argmin, concatenate, linspace, ndarray, ceil, \
    log2, sqrt, cumsum, diff, mean, exp, inf, arccos, asarray, interp, full, r_
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import simpson
from scipy.signal import freqs, bessel, firwin2, oaconvolve
from scipy.optimize import differential_evolution, minimize
from notnormal.models.base import NoiseReconstructionArgs, NoiseFitResults, NoiseReconstructResults, Trace, Events
from notnormal.utils.methods import get_event_mask, get_psd
import cython

_COMPILED = cython.compiled
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"


"""
Public API
"""

def reconstruct_noise(
    trace: ndarray | Trace,
    event_mask: ndarray | Events,
    aa_cutoff: int,
    aa_order: int,
    n_regimes: int | list[int] = (2, 3),
    sample_rate: Optional[int] = None,
    maxiter: tuple[int, int] = (2000, 10000),
    mutation: tuple[float, float] = (0.7, 1.2),
    popsize: Optional[int] = None,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
    generate: str | bool = 'best',
    complex_gen: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> NoiseReconstructResults:
    """
    This function uses global differential evolution optimisation and local L-BFGS-B optimisation to fit a composition
    of noise regimes to the power spectral density (PSD) of the noise, given by the trace without events. The fitting
    accounts for the anti-aliasing filter parameters, enabling an accurate fit which respects the condition of the noise
    post-sampling. Following, each individual noise regime is generated using a modified version of the spectral method
    that expands into the negative detrended fluctuation analysis (DFA) alpha and applies post-generation scaling to
    maximise the accuracy of each individual regime. Finally, the composition is computed, the anti-aliasing filter is
    applied by estimating the Bessel filter using the window design method, and the spectral angular map (SAM) against
    the true noise is computed to evaluate the quality of the reconstruction.

    Note: This function is computationally expensive. It is recommended to drop popsize to 50 or lower to get an estimate
    on the number of regimes, confirm solely the best fit with a conservative PSD period (e.g. 0.1 s) and then run a
    higher popsize (e.g. 500) with a longer PSD period (e.g. 1-2 s) to get the best fit for the noise regimes.

    Args:
        trace (ndarray | Trace): The trace to be processed or a Trace object. If ndarray, the sample rate must be provided.
        event_mask (ndarray | Events): A boolean mask indicating the events in the trace or an Events object.
        n_regimes (int | list[int]): The number of noise regimes to fit or a list of noise regime numbers to fit. Default is [2, 3].
        aa_order (int): The order of the anti-aliasing filter.
        aa_cutoff (int): The cutoff frequency for the anti-aliasing filter.
        sample_rate (int | None): The sample rate of the trace. Has to be provided if trace is a ndarray. Default is None.
        maxiter (tuple[int, int]): The maximum number evaluations for differential evolution and L-BFGS-B. Default is (2000, 10000).
        mutation (tuple[float, float]): The mutation constant for differential evolution. Default is (0.7, 1.2).
        popsize (int | None): The population size multiplier for differential evolution. If None, equates to number of
            parameters * 50. Default is None.
        psd_period (float | None): The period for the PSD calculation. Default is None.
        nfft (int | None): The length of FFT to use for PSD calculation. Default is None.
        generate (str | bool): Whether to generate noise regimes. If 'Best', only the best fit is generated. Default is 'best'.
        complex_gen (bool): Whether to generate noise regimes starting with complex noise. Default is False.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        NoiseReconstructResults: An object containing the reconstruction results.
    """

    if isinstance(trace, Trace):
        sample_rate, trace = trace.sample_rate, trace.trace
    if sample_rate is None:
        raise ValueError("Sample rate must be provided if trace is a ndarray.")
    if isinstance(event_mask, Events):
        event_mask = get_event_mask(trace, event_mask)
    if len(trace) != len(event_mask):
        raise ValueError("The length of the trace and the event mask must be the same.")
    if len(trace) == 0:
        raise ValueError("The trace must not be empty.")

    # Get args
    args = NoiseReconstructionArgs(**{k: v for k, v in locals().items() if k in NoiseReconstructionArgs.__annotations__})

    # Get the trace without the events
    removed_trace = trace[~event_mask]

    # Get the PSD, only need at most twice the aa cutoff for accurate fitting
    f, pxx = get_psd(removed_trace, sample_rate, fmax=aa_cutoff * 2.0, psd_period=psd_period, nfft=nfft)

    # Fit the noise regimes
    fits = fit_noise(
        f,
        pxx,
        aa_order,
        aa_cutoff,
        n_regimes=n_regimes,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        random_state=random_state,
        verbose=verbose
    )

    # Get the best result
    best_fit, loss_curves = _get_best_fit(fits)

    # No generation requested
    if not generate:
        return NoiseReconstructResults(args, fits, best_fit, loss_curves)

    # Generate the noise regimes
    fits = _generate_noises(
        fits,
        len(trace),
        sample_rate,
        complex_gen=complex_gen,
        psd_period=psd_period,
        nfft=nfft,
        aa_cutoff=aa_cutoff,
        aa_order=aa_order,
        n_regimes=n_regimes if generate != 'best' else best_fit.n_regimes,
        random_state=random_state,
        verbose=verbose
    )

    # Compute the SAM for the generated noise regimes
    fits = _compute_sam(fits, removed_trace, sample_rate, psd_period=psd_period, nfft=nfft)

    return NoiseReconstructResults(args, fits, best_fit, loss_curves)


def fit_noise(
    f: ndarray,
    pxx: ndarray,
    aa_order: int,
    aa_cutoff: int,
    n_regimes: int | list[int] = (2, 3),
    maxiter: tuple[int, int] = (2000, 10000),
    mutation: tuple[float, float] = (0.7, 1.2),
    popsize: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> dict[int, NoiseFitResults]:
    """
    Fit noise regimes to the provided power spectral density (PSD) using global differential evolution optimisation
    followed by local L-BFGS-B optimisation. The fitting is done by minimising the sum of squared log error between the
    predicted and actual PSDs, taking into account the true anti-aliasing filter.

    Args:
        f (ndarray): The frequency array of the true noise.
        pxx (ndarray): The PSD array of the true noise.
        aa_order (int): The order of the anti-aliasing filter.
        aa_cutoff (int): The cutoff frequency for the anti-aliasing filter.
        n_regimes (int | list[int]): The number of noise regimes to fit or a list of noise regime numbers to fit. Default is (2, 3).
        maxiter (tuple[int, int]): The maximum number evaluations for differential evolution and L-BFGS-B. Default is (2000, 10000).
        mutation (tuple[float, float]): The mutation constant for differential evolution. Default is (0.7, 1.2).
        popsize (int | None): The population size multiplier for differential evolution. If None, equates to number of
            parameters * 50. Default is None.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        dict[int, NoiseFitResults]: A dictionary of noise fitting results keyed by the number of regimes.
    """

    if len(f) != len(pxx):
        raise ValueError("Frequency and PSD arrays must have the same length.")

    n_list = [n_regimes] if isinstance(n_regimes, int) else list(n_regimes)
    if any(n < 1 for n in n_list):
        raise ValueError("Number of regimes must be at least 1.")

    # Configure progress bar
    with tqdm(total=len(n_list), desc='Fitting Noise', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        results = {}

        # Get logarithmic indices to reduce the number of points for fitting and bias towards lower frequencies
        log_indices = unique(logspace(0, log(len(f)), num=len(f), dtype=int, base=e) - 1)
        f = f[log_indices]
        pxx = pxx[log_indices]

        # Bessel power response for AA filter shaping
        _, h_filter = freqs(*bessel(aa_order, aa_cutoff * 2 * pi, analog=True, norm='mag', output='ba'),
                            worN=f * 2 * pi)
        p_filter = abs(h_filter) ** 2

        # Get the maximum possible C value 10 * (highest density, lowest frequency and highest exponent)
        c_max = log(100.0 * max(pxx) * f[0] ** 2.0)
        # Get the minimum possible C value 0.1 * (lowest density, highest frequency and lowest exponent)
        c_min = log(0.01 * min(pxx) * f[f.shape[0] - 1] ** -2.0)

        for n in n_list:
            # [-2, 2] bounds to cover expected range of exponents, log(c_i) to improve numerical stability
            bounds = [(c_min, c_max)] * n + [(-2.0, 2.0)] * n

            # Global optimization to get initial guesses
            global_opt = differential_evolution(
                _objective,
                bounds=bounds,
                args=(f, pxx, p_filter),
                strategy='randtobest1bin',
                maxiter=maxiter[0],
                popsize=popsize if popsize else len(bounds) * 50,
                mutation=mutation,
                tol=1e-4,
                seed=random_state,
                workers=-1,
                updating='deferred',
                polish=False
            )

            # Perform local optimization
            local_opt = minimize(
                _objective,
                global_opt.x,
                args=(f, pxx, p_filter),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': maxiter[1], 'ftol': 1e-12, 'gtol': 1e-10, 'eps': 1e-12}
            )

            # Store results
            results[n] = NoiseFitResults(
                f=f,
                pxx=pxx,
                p_filter=p_filter,
                n_regimes=n,
                cs=exp(local_opt.x[:len(local_opt.x) // 2]),
                ms=local_opt.x[len(local_opt.x) // 2:],
                alphas=[(m + 1.0) / 2.0 for m in local_opt.x[len(local_opt.x) // 2:]],
                SSLE=local_opt.fun,
                success=local_opt.success,
                global_opt=global_opt,
                local_opt=local_opt
            )

            # Update the progress bar
            progress.update(1)

    return results


def piecwise_fit_noise(f: ndarray, pxx: ndarray, n_regimes: int | list[int] = (2, 3)) -> dict[int, dict[str, Any]]:
    """
    Fit piecewise linear noise regimes to the provided power spectral density (PSD) using Nelder-Mead optimisation.
    The fitting is done by minimising the mean squared error between the predicted and actual PSDs. This function
    does not take into the account the anti-aliasing filter parameters, does not fit summed noise regimes, and does not
    generate. It is a simpler fitting method that can be used for quick estimates of the noise regimes, for diagnostic
    purposes, or for visualisation.

    Args:
        f (ndarray): The frequency array of the true noise.
        pxx (ndarray): The PSD array of the true noise.
        n_regimes (int | list[int]): The number of noise regimes to fit or a list of noise regime numbers to fit. Default is (2, 3).

    Returns:
        dict[int, dict[str, Any]]: A dictionary of noise fitting results keyed by the number of regimes.
    """

    if len(f) != len(pxx):
        raise ValueError("Frequency and PSD arrays must have the same length.")

    n_list = [n_regimes] if isinstance(n_regimes, int) else list(n_regimes)
    if any(n < 1 for n in n_list):
        raise ValueError("Number of regimes must be at least 1.")

    # Log transform for straight line fitting
    f = log(f)
    pxx = log(pxx)

    # Objective function for piecewise linear fit
    def objective(prediction, x, y, n):
        px = r_[x[0], prediction[:n - 1], x[len(x) - 1]]
        py = prediction[n - 1:]
        return mean((y - interp(x , px, py)) ** 2)

    fits = {}
    for n in n_list:
        # Create initial guess as n equally spaced points in the frequency domain (with first and last points fixed)
        init_f = min(f) + cumsum(full(n - 1, ((max(f) - min(f)) / n)))
        init_idx = [argmin(abs(f - val)) for val in init_f]
        initial_guess = r_[f[init_idx], r_[pxx[0], pxx[init_idx], pxx[len(pxx) - 1]]]

        # Nelder-Mead optimization
        result = minimize(objective, initial_guess, args=(f, pxx, n), method='Nelder-Mead')

        # Store results
        fits[n] = {}
        fits[n]['f'] = f
        fits[n]['pxx'] = pxx
        fits[n]['line fit'] = exp(interp(f, r_[f[0], result.x[:n - 1], f[len(f) - 1]], result.x[n - 1:]))
        fits[n]['n_regimes'] = n
        fits[n]['crossovers'] = exp(result.x[:n - 1])
        fits[n]['ms'] = -(diff(result.x[n - 1:]) / diff(r_[f[0], result.x[:n - 1], f[len(f) - 1]]))
        fits[n]['alphas'] = [(m + 1.0) / 2.0 for m in fits[n]['ms']],
        fits[n]['MSE'] = result.fun
        fits[n]['success'] = result.success
        fits[n]['opt'] = result

    return fits


def generate_noise(
    alphas: ndarray[float],
    cs: ndarray[float],
    length: int,
    sample_rate: int,
    complex_gen: bool = False,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
    aa_cutoff: Optional[int] = None,
    aa_order: Optional[int] = None,
    random_state: Optional[int] = None
) -> tuple[ndarray, dict[float, ndarray]]:
    """
    Generate a composition of noise regimes based on the provided alpha and c values using a modified version of the
    spectral method. Optionally applies an anti-aliasing filter to the generated noise to match the conditions of the
    trace the values were derived from.

    Args:
        alphas (ndarray[float]): The detrended fluctuation analysis (DFA) alpha exponents (directly related to H) of the noise regimes.
        cs (ndarray[float]): The frequency domain numerators (directly related to variance) of the noise regimes.
        length (int): The length of the noise regimes to generate.
        sample_rate (int): The sample rate of the noise regimes.
        complex_gen (bool): Whether to generate noise regimes starting with complex noise. Default is False.
        psd_period (float | None): The period of the power spectral density (PSD) for scale matching. Default is None.
        nfft (int | None): The number of PSD FFT points for scale matching. Default is None.
        aa_cutoff (int | None): The cutoff frequency for the anti-aliasing filter. Default is None.
        aa_order (int | None): The order of the anti-aliasing filter. Default is None.
        random_state (int | None): Random seed for reproducibility. Default is None.

    Returns:
        tuple[ndarray, dict[float, ndarray]]: A tuple containing the total generated noise and a dictionary of individual
        noise regimes keyed by their alpha values.
    """

    if len(alphas) != len(cs):
        raise ValueError("The length of alphas and cs must be the same.")

    # Simulate the regimes
    regimes = {}
    for alpha, c in zip(alphas, cs):
        regimes[alpha] = _generate_regime(
            alpha,
            c,
            length,
            sample_rate,
            complex_gen=complex_gen,
            psd_period=psd_period,
            nfft=nfft,
            random_state= random_state
        )

    # Sum the noises
    total = sum(list(regimes.values()), axis=0)

    # Apply the AA filter if specified
    if aa_cutoff and aa_order:
        total = _aa_filter(total, sample_rate, aa_cutoff, aa_order)

    return total, regimes


"""
Internal API
"""

def _objective(params: ndarray, f: ndarray, pxx: ndarray, p_filter: ndarray) -> float:
    """
    The objective function for fitting noise regimes. It calculates the sum of squared log error between a composition
    of noise regimes post-anti-aliasing filtering and the provided power spectral density (PSD). (No error handling
    is done here, don't flame me about assert efficiency with cython.)

    Args:
        params (ndarray): The parameters to fit, consisting of exponents (m_i) and log numerators (c_i) for each regime.
        f (ndarray): The frequency array of the true noise.
        pxx (ndarray): The PSD array of the true noise.
        p_filter (ndarray): The power response of the anti-aliasing filter.

    Returns:
        float: The sum of squared log error between the predicted and actual PSDs.
    """

    # Extract the exponents (m_i) and logarithm of the numerators (c_i) from the parameters
    n = params.shape[0] // 2
    ms = params[n:]
    cs = params[:n]

    # Define the model function: pxx = sum(c_i / f ^ m_i), e^(c_i) to reverse previous log, filter by p_filter
    pxx_pred = sum(exp(cs[:, None]) / f[None, :] ** ms[:, None], axis=0) * p_filter

    # Make sure 0s do not exist
    pxx_pred = clip(pxx_pred, 1e-15, None)

    # Return the objective function (sum of squared log error)
    return sum(log(pxx / pxx_pred) ** 2)


def _get_best_fit(noise_dict: dict[int, NoiseFitResults]) -> tuple[NoiseFitResults, dict[str, tuple[ndarray, ndarray]]]:
    """
    Get the best fit and loss curves from a dictionary of noise fitting results keyed by the number of regimes.

    Args:
        noise_dict (dict[int, NoiseFitResults]): A dictionary of noise fitting results keyed by the number of regimes.

    Returns:
        tuple[NoiseFitResults, dict[str, tuple[ndarray, ndarray]]]: The best fit based on the SSLE (objective value) and a dictionary of loss curves.
    """

    if not any(fit.success for fit in noise_dict.values()):
        raise ValueError("No successful fits found.")

    # Get the regimes and objectives
    n_regimes = asarray(list(noise_dict.keys()))
    objectives = asarray([fit.SSLE if fit.success else inf for fit in noise_dict.values()])

    # Get the loss curve and best fit based on the objective value
    loss_curves = {'SSLE': (n_regimes, objectives)}
    best_key = int(n_regimes[argmin(objectives)])

    return noise_dict[best_key], loss_curves


def _generate_noises(
    noise_dict: dict[int, NoiseFitResults],
    length: int,
    sample_rate: int,
    complex_gen: bool = False,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
    aa_cutoff: Optional[int] = None,
    aa_order: Optional[int] = None,
    n_regimes: Optional[int | list[int]] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> dict[int, NoiseFitResults]:
    """
    For each noise fit generate a composition of noise regimes based on the alpha and c values. Optionally applies an
    anti-aliasing filter to each generated noise to match the conditions of the trace the values were derived from.

    Args:
        noise_dict: (dict[int, NoiseFitResults]) A dictionary of noise fitting results keyed by the number of regimes.
        length (int): The length of the noise regimes to generate.
        sample_rate (int): The sample rate of the noise regimes.
        complex_gen (bool): Whether to generate noise regimes starting with complex noise. Default is False.
        psd_period (float | None): The period of the power spectral density (PSD) for scale matching. Default is None.
        nfft (int | None): The number of PSD FFT points for scale matching. Default is None.
        aa_cutoff (int | None): The cutoff frequency for the anti-aliasing filter. Default is None.
        aa_order (int | None): The order of the anti-aliasing filter. Default is None.
        n_regimes (int | list[int] | None): Which keys to generate for. Default is None, which means all keys in noise_dict will be generated.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        dict[int, NoiseFitResults]: The input dictionary with the requested noises generated.
    """

    # If not specified, generate all regimes
    if n_regimes is None:
        n_list = noise_dict.keys()
    else:
        n_list = [n_regimes] if isinstance(n_regimes, int) else list(n_regimes)
        if any(n not in noise_dict.keys() for n in n_list):
            raise ValueError(f"Some of the requested regime numbers {n_list} are not present in the noise dictionaries.")

    # Configure progress bar
    with tqdm(total=len(n_list), desc='Generating Noise', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        for n in n_list:
            # Generate the noise
            noise_dict[n].total, noise_dict[n].regimes = generate_noise(
                noise_dict[n].alphas,
                noise_dict[n].cs,
                length,
                sample_rate,
                complex_gen=complex_gen,
                psd_period=psd_period,
                nfft=nfft,
                aa_cutoff=aa_cutoff,
                aa_order=aa_order,
                random_state=random_state
            )

            # Update the progress bar
            progress.update(1)

    return noise_dict


def _generate_regime(
    alpha: float,
    c: float,
    length: int,
    sample_rate: int,
    complex_gen: bool = False,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
    random_state: Optional[int] = None
) -> ndarray:
    """
    Generate a single noise regime spanning from fractional Gaussian noise (fGn) to fractional Brownian motion (fBm) and
    fractional white noise (fWn). All regimes are generated first as fGn and then transformed to the desired regime.
    This is because fGn is the increment process of fBm and fWn is the increment process of fGn.

    Args:
        alpha (float): The detrended fluctuation analysis (DFA) alpha exponent (directly related to H) of the noise regime.
        c (float): The frequency domain numerator (directly related to variance) of the noise regime.
        length (int): The length of the noise regime to generate.
        sample_rate (int): The sample rate of the noise regime.
        complex_gen (bool): Whether to generate starting with complex noise. Default is False.
        psd_period (float | None): The period of the power spectral density (PSD) for scale matching. Default is None.
        nfft (int | None): The number of PSD FFT points for scale matching. Default is None.
        random_state (int | None): Random seed for reproducibility. Default is None.

    Returns:
        ndarray: The generated noise regime.
    """

    if length <= 0:
        raise ValueError("Length must be a positive integer.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if not (-0.5 <= alpha <= 1.5):
        raise ValueError("Alpha must be in the range [-0.5, 1.5].")
    if c <= 0:
        raise ValueError("C must be a positive value.")

    # We generate everything starting at fGn via differentiation or integration
    if alpha > 1.0:
        h = alpha - 1
    elif alpha < 0.0:
        h = alpha + 1
    else:
        h = alpha

    # Take next power of 2 for speed and long-range accuracy
    n = int(2 ** ceil(log2(length)))
    f = fftfreq(n, d=1.0 / sample_rate)

    # Start with complex white noise or transformed white noise
    rng = default_rng(random_state)
    if complex_gen:
        spectrum = rng.normal(loc=0, scale=1, size=n) + 1j * rng.normal(loc=0, scale=1, size=n)
    else:
        white_noise = rng.normal(scale=1, size=n)
        spectrum = fft(white_noise)

    # Spectral shape that fool
    s_f = concatenate(([0.0], 1.0 / (abs(f[1:]) ** (2.0 * h - 1.0))))
    spectrum = spectrum * sqrt(s_f)

    # Transform
    if alpha > 1.0:  # Cumulative sum of fGn is fBm
        spectrum = fft(cumsum(ifft(spectrum)))
    elif alpha < 0.0:  # First derivative of fGn is fWn
        if complex_gen:
            spectrum = fft(diff(ifft(spectrum)))
        else:
            spectrum *= (2j * pi * f)

    # Scale via difference of integrals
    f2, pxx2 = get_psd(ifft(spectrum).real, sample_rate, fmin=1.0, fmax=sample_rate // 4.0, psd_period=psd_period, nfft=nfft)
    mask = (f2 > 0)
    f2, pxx2 = f2[mask], pxx2[mask]
    current_integral = simpson(pxx2, x=f2)
    exponent = 1.0 - (2.0 * alpha - 1.0)
    # Special case when alpha == 1
    if exponent == 0:
        integral =  log(f2[len(f2) - 1] / f2[0])
    else:
        integral = ((f2[len(f2) - 1] ** exponent - f2[0] ** exponent) / exponent)
    d_squared = current_integral / integral
    spectrum *= sqrt(c / d_squared)

    # Inverse
    noise = ifft(spectrum)
    if complex_gen:
        noise -= mean(noise)

    return noise.real[:length]


def _aa_filter(noise: ndarray, sample_rate: int, cutoff: int, order: int, taps: int = 501) -> ndarray:
    """
    Apply an approximation of the Bessel anti-aliasing filter (window design method) to the noise profile.

    Args:
        noise (ndarray): The noise to filter.
        sample_rate (int): The sample rate of the trace.
        cutoff (int): The cutoff frequency for the Bessel filter.
        order (int): The order of the Bessel filter.
        taps (int): The number of taps for the FIR filter design. Default is 501.

    Returns:
        ndarray: The anti-aliasing filtered noise.
    """

    if not (0 <= cutoff <= sample_rate // 2):
        raise ValueError("Cutoff must be in the range [0, sample_rate // 2].")
    if not (1 <= order <= 10):
        raise ValueError("Order must be between 1 and 10.")
    if taps < 1 or taps % 2 == 0:
        raise ValueError("Taps must be a positive odd integer.")

    # Get the analog xth order Bessel
    b_a, a_a = bessel(order, cutoff * 2 * pi, analog=True, norm='mag', output='ba')

    # Get analog filter response
    w_analog, h_analog = freqs(b_a, a_a, worN=linspace(0, sample_rate / 2, 1024) * 2 * pi)
    f_analog = w_analog / (2 * pi)

    # Design a linear FIR filter using the window design method
    window = firwin2(taps, f_analog, abs(h_analog), fs=sample_rate)

    # Add sym pad for the edge cases
    padded_trace = concatenate((noise[:int(taps // 2)][::-1], noise, noise[len(noise) - int(taps // 2):][::-1]))
    # Calculate the filter
    filtered_trace = oaconvolve(padded_trace, window, 'same')
    # Cut off the edges
    filtered_trace = filtered_trace[int(taps // 2):len(filtered_trace) - int(taps // 2)]

    return filtered_trace


def _compute_sam(
    noise_dict: dict[int, NoiseFitResults],
    noise: ndarray,
    sample_rate: int,
    psd_period: Optional[float] = None,
    nfft: Optional[int] = None,
) -> dict[int, NoiseFitResults]:
    """
    Compute the spectral angular map (SAM) for each generated fit in the provided dictionary vs. the true noise.

    Args:
        noise_dict (dict[int, NoiseFitResults]): A dictionary of noise fitting results keyed by the number of regimes.
        noise (ndarray): The real noise to compute the SAM against.
        sample_rate (int): The sample rate of the noise.
        psd_period (float | None): The period for the power spectral density (PSD) calculation. Default is None.
        nfft (int | None): The length of FFT to use. Default is None.

    Returns:
        dict[int, NoiseFitResults]: The input dictionary with the SAM computed for each generated fit.
    """

    for result in noise_dict.values():
        if result.total is None:
            continue

        # Get the PSD of the true noise and the generated noise
        _, pxx = get_psd(noise, sample_rate, fmax=sample_rate // 2.0, psd_period=psd_period, nfft=nfft)
        _, pxx2 = get_psd(result.total[:len(noise)], sample_rate, fmax=sample_rate // 2.0, psd_period=psd_period, nfft=nfft)

        # Compute the spectral angle mapping (SAM)
        result.SAM = arccos(sum(log(pxx) * log(pxx2)) / (sqrt(sum(log(pxx) ** 2)) * sqrt(sum(log(pxx2) ** 2))))

    return noise_dict
