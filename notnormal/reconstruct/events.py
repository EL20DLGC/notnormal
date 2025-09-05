# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides functions to cluster, reconstruct, and augment events for (nano)electrochemical time series
data.
"""

from typing import Any, Optional
from warnings import catch_warnings
from tqdm import tqdm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from umap import UMAP
from numpy.linalg import norm
from numpy.random import default_rng
from numpy import mean, zeros, sum, linspace, ndarray, max, average, maximum, corrcoef, pad, array, cross, vstack, std, \
    argmax, correlate, log, exp, ceil, asarray, clip, int32
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance
from notnormal.models.base import Events, ShapeCluster, ShapeClusters, ShapeClusterResults, ShapeClusterArgs, \
    EventAugmentationArgs
import cython

_COMPILED = cython.compiled
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"


"""
Public API
"""

def shape_cluster(
    vectors: list[ndarray] | ndarray | Events,
    k: int | list[int] = tuple(range(2, 21)),
    direction: Optional[str] = None,
    n_components: int = 10,
    metric: str = 'cosine',
    spread: float = 1.0,
    min_dist: float = 0.0,
    n_neighbors: int = 5,
    random_state: Optional[int] = None,
    init_params: str = 'kmeans',
    max_iter: int = 100,
    n_init: int = 1,
    reconstruct: bool = True,
    sigma: float = 0.0,
    align: bool = True,
    dist_losses: bool = True,
    full_results: bool = False,
    verbose: bool = False
) -> ShapeClusterResults:
    """
    Shape cluster event vectors using Uniform Manifold Approximation and Projection, along with representative vector
    computation and subsequent reconstruction losses. The algorithm comes from the simple idea of shape being the
    principal encoder of analyte identity, with different scales representing different modes of transport and
    interaction. Each clustering operation produces a set of representative vectors from which the original event
    vectors are reconstructed to yield reconstruction losses. The optimal number of clusters is determined by
    the point of diminishing returns in reconstruction loss, with the associated representative vectors encoding the
    shapes in the population.

    Args:
        vectors (list[ndarray] | ndarray | Events): The event vectors to cluster. Should be 2D array-like or Events
            object. If Events object, direction must be specified. Otherwise, vectors must be monophasic, but not
            required to be strictly one-sided.
        k (int | list[int]): Number of clusters or list of cluster numbers. Default is range(2, 21).
        direction (int | None): The direction of the events. Has to be provided if vectors is an Events object, otherwise
            ignored. Default is None.
        n_components (int): Number of latent space dimensions (UMAP). Default is 10.
        metric (str): Metric to compute distances (UMAP). Default is 'cosine'.
        spread (float): The effective scale of embedded points (UMAP). Default is 1.0.
        min_dist (float): The effective minimum distance between embedded points (UMAP). Default is 0.0.
        n_neighbors (int): Size of the local neighbourhood (UMAP). Default is 5.
        random_state (int | None): Random seed for reproducibility (UMAP, GMM). Default is None.
        init_params (str): The method used to initialise the weights (GMM). Default is 'kmeans'.
        max_iter (int): Number of EM iterations (GMM). Default is 100.
        n_init (int): Number of initialisations (GMM). Default is 1.
        reconstruct (bool): Whether to reconstruct the event vectors from their representative vector. Default is True.
        sigma (float): The noise standard deviation for interpretability and stability. Default is 0.0.
        align (bool): Whether to align the vectors before calculating losses. Default is True.
        dist_losses (bool): Whether to compute the reconstruction distributional losses. Default is True.
        full_results (bool): Whether to return the globally normalised + transformed vectors in the results. Default is False.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        ShapeClusterResults: An object containing the clustering results.
    """

    if isinstance(vectors, Events):
        if direction is None:
            raise ValueError("Direction must be specified if vectors is an Events object.")
        vectors = vectors.get_feature('Vector', filter_fn=lambda x: x['Direction'] == direction)

    # Get args
    args = ShapeClusterArgs(**{k: v for k, v in locals().items() if k in ShapeClusterArgs.__annotations__})

    # Normalise events
    area_norm, length_norm = _normalise_events(vectors, verbose=verbose)

    # Reduce dimensionality
    transform, umap = _reduce_events(
        area_norm,
        n_components=n_components,
        metric=metric,
        spread=spread,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
        verbose=verbose
    )

    # Cluster the reduced vectors
    clusters = _cluster_latent(
        transform,
        vectors,
        k=k,
        init_params=init_params,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        verbose=verbose
    )

    # Calculate representative vectors and losses
    clusters = _representative_vectors(
        clusters,
        reconstruct=reconstruct,
        sigma=sigma,
        align=align,
        dist_losses=dist_losses,
        verbose=verbose
    )

    # Create results object
    results = ShapeClusterResults(args, clusters)

    # If full results are requested, add the normalised and transformed vectors
    if full_results:
        results.area_norm, results.length_norm, results.transform, results.umap = area_norm, length_norm, transform, umap

    # Get the knee of reconstruction losses
    if reconstruct:
        results.best_fit, results.loss_curves, results.knees = _get_knees(clusters)

    return results


def augment_clusters(
    clusters: ShapeClusters,
    n_vectors: Optional[int] = None,
    max_k: int = 50,
    weight: float = 1e-2,
    random_state: Optional[int] = None,
    max_attempts: int = 100,
    sample_global: bool = True,
    dist_losses: bool = True,
    verbose: bool = False,
) -> ShapeClusters:
    """
    Augment the event vectors using the representative vectors and scaling features derived from both local and global
    models. This function extends shape clustering by generating new event vectors based on the representative vectors
    (which encode the shapes in the population and their density) and the scales at which those shapes are observed.
    Distributional estimates of the global scaling features enables sampling of new scales that are consistent with the
    shape agnostic scaling features, providing an accurate representation of the overall population. Cluster specific
    (local) estimates enables sampling of scales consistent with each specific shape (representative vector), but
    provide a less accurate estimate under the sparser subset of events. By using density aware rejection sampling on
    both global and local models, scaling features can be sampled which respect both the overall and shape specific
    distribution of scales. In this way, augmented events can be generated which respect shape, the scales those shapes
    are observed at, and the distribution of scales in the entire population. Note, the sample_global parameter controls
    which model is sampled from, with samples being accepted if the log likelihood ratio between the not sampled and
    sampled models exceeds a threshold. When True, the local model has a higher weighting, when False, the global
    model has a higher weighting, see: _sample_log_method.

    Args:
        clusters (ShapeClusters): An object containing the clustering results.
        n_vectors (int | None): The number of vectors to generate. If None, one generation will occur per event in
            ShapeClusters. Default is None.
        max_k (int): The maximum number of components to fit (scaling features). Default is 50.
        weight (float): The weight concentration prior for the Dirichlet Process (scaling features). Default is 1e-2.
        random_state (int | None): Random seed for reproducibility (scaling features only). Default is None.
        max_attempts (int): The maximum number of attempts to sample a candidate. Default is 100.
        sample_global (bool): Whether to sample from the global model or the local model. Default is True.
        dist_losses (bool): Whether to compute the augmentation distributional losses. Default is True.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        ShapeClusters: The input clustering results with augmentation results added.
    """

    # Get args
    clusters.aug_args = EventAugmentationArgs(**{k: v for k, v in locals().items() if k in EventAugmentationArgs.__annotations__})

    # Fit scaling features to the clusters
    clusters = _fit_scaling_features(clusters, max_k=max_k, weight=weight, random_state=random_state, verbose=verbose)

    # Augment the event vectors
    clusters = _augment_vectors(
        clusters,
        n_vectors=n_vectors,
        max_attempts=max_attempts,
        sample_global=sample_global,
        random_state=random_state,
        verbose=verbose
    )

    # Calculate distributional losses if requested
    if dist_losses:
        clusters.aug_distributional_loss = _distributional_loss(clusters, recon=False)

    return clusters


"""
Internal API
"""

def _normalise_events(vectors: list[ndarray] | ndarray, clipped: bool = True, verbose: bool = False) -> tuple[ndarray, ndarray]:
    """
    Shape normalise event vectors using spline interpolation and matrix normalisation.

    Args:
        vectors (list[ndarray] | ndarray): The event vectors to normalise. Should be 2D array-like.
        clipped (bool): Whether to clip to assert monophasic. Reduce breaks if points lie on the opposite side of zero, beware.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        tuple[ndarray, ndarray]: A tuple containing the length + area normalised and length normalised event vectors.
    """

    if len(vectors) == 0:
        raise ValueError("No vectors to normalise.")

    # Assert monophasic for the benefit of reduce
    if clipped:
        vectors = [clip(vector, 0, None) if sum(vector) > 0 else
                   clip(vector, None, 0) for vector in vectors]
    else:
        vectors = [asarray(vector) for vector in vectors]

    # Init normalisation params
    lengths = asarray([len(vector) for vector in vectors])
    max_len = max(lengths)
    if any(length < 2 for length in lengths):
        raise ValueError("Vectors must have at least 2 points for normalisation.")

    # Init normalisation matrices
    linspace_max = linspace(0, 1, max_len)
    length_norm = zeros((len(vectors), max_len))
    area_norm = zeros((len(vectors), max_len))
    # Configure progress bar
    total = len(vectors)
    with tqdm(total=total, desc='Normalising', miniters=maximum(1, total // 100), bar_format=_BAR_FORMAT,
              disable=not verbose) as progress:
        # Normalise each vector
        for i, vector in enumerate(vectors):
            length_norm[i, :] = CubicSpline(linspace(0, 1, lengths[i]), vector)(linspace_max)
            area_norm[i, :] = length_norm[i, :] / norm(length_norm[i, :])

            # Update the progress bar
            progress.update(1)

    return area_norm, length_norm


def _reduce_events(
    norm_vectors: ndarray,
    n_components: int = 10,
    metric: str = 'cosine',
    spread: float = 1.0,
    min_dist: float = 0.0,
    n_neighbors: int = 5,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> tuple[ndarray, UMAP]:
    """
    Reduce dimensionality of normalised event vectors using Uniform Manifold Approximation and Projection.

    Args:
        norm_vectors (ndarray): The normalised event vectors to reduce. Should be 2D array-like.
        n_components (int): Number of latent space dimensions. Default is 10.
        metric (str): Metric to compute distances. Default is 'cosine'.
        spread (float): The effective scale of embedded points. Default is 1.0.
        min_dist (float): The effective minimum distance between embedded points. Default is 0.0.
        n_neighbors (int): Size of the local neighbourhood. Default is 5.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        tuple[ndarray, UMAP]: A tuple containing the transformed event vectors and the UMAP object.
    """

    if len(norm_vectors) == 0:
        raise ValueError("No vectors to reduce.")

    # Configure progress bar
    with catch_warnings(action="ignore"), tqdm(range(1), 'Reducing', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # Init UMAP params
        umap = UMAP(
            n_components=n_components,
            metric=metric,
            spread=spread,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        # Fit UMAP
        transform = umap.fit_transform(norm_vectors)

        # Update the progress bar
        progress.update(1)

    return transform, umap


def _cluster_latent(
    transform: ndarray,
    vectors: list[ndarray],
    k: int | list[int] = tuple(range(2, 21)),
    init_params: str = 'kmeans',
    max_iter: int = 100,
    n_init: int = 1,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> dict[int, ShapeClusters]:
    """
    Cluster the normalised and reduced event vectors using a Gaussian Mixture Model.

    Args:
        transform (ndarray): The normalised and reduced event vectors to cluster. Should be 2D array-like.
        vectors (list[ndarray]): The corresponding original event vectors (not normalised). Should be 2D array-like.
        k (int | list[int]): Number of clusters or list of cluster numbers. Default is range(2, 21).
        init_params (str): The method used to initialise the weights. Default is 'kmeans'.
        max_iter (int): Number of EM iterations. Default is 100.
        n_init (int): Number of initialisations. Default is 1.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        dict[int, ShapeClusters]: A dictionary of clustering results keyed by the number of clusters.
    """

    if len(transform) < 2 or len(vectors) < 2:
        raise ValueError("At least two vectors are required for clustering.")
    if len(transform) != len(vectors):
        raise ValueError("Transformed vectors and original vectors must have the same length.")

    k_list = [k] if isinstance(k, int) else list(k)
    if any(k < 2 for k in k_list):
        raise ValueError("Number of clusters 'k' must be at least 2.")

    # Configure progress bar
    with tqdm(total=len(k_list), desc='Clustering', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        results = {}
        for k in k_list:
            # Init GMM params
            gmm = GaussianMixture(n_components=k, random_state=random_state, init_params=init_params, max_iter=max_iter,
                                  n_init=n_init)

            # Fit GMM
            labels = gmm.fit_predict(transform)
            probabilities = gmm.predict_proba(transform)

            # Store results
            results[k] = ShapeClusters(
                k=k,
                model=gmm,
                bic=gmm.bic(transform),
                labels=labels,
                probabilities=probabilities,
                clusters=[
                    ShapeCluster(
                        label=cluster,
                        confidence=mean(probabilities[labels == cluster, cluster]),
                        vectors=[asarray(vector) for vector, label in zip(vectors, labels) if label == cluster],
                        weights=probabilities[labels == cluster, cluster]
                    ) for cluster in range(k)
                ]
            )

            # Update the progress bar
            progress.update(1)

    return results


def _representative_vectors(
    clusters_dict: dict[int, ShapeClusters],
    reconstruct: bool = True,
    sigma: float = 0.0,
    align: bool = True,
    dist_losses: bool = True,
    verbose: bool = False
) -> dict[int, ShapeClusters]:
    """
    Calculate the representative vectors for each clustering object. If reconstruct is True,
    reconstruct the original event vectors from their representative vectors and calculate losses.

    Args:
        clusters_dict (dict[int, ShapeClusters]): A dictionary of clustering results keyed by the number of clusters.
        reconstruct (bool): Whether to reconstruct the event vectors from their representative vector. Default is True.
        sigma (float): The noise standard deviation for interpretability and stability. Default is 0.0.
        align (bool): Whether to align the vectors before calculating losses. Default is True.
        dist_losses (bool): Whether to compute the reconstruction distributional losses. Default is True.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        dict[int, ShapeClusters]: The input dictionary with the representative vectors computed.
    """

    # Configure progress bar
    total = sum([len(clusters.clusters) for clusters in clusters_dict.values()])
    with tqdm(total=total, desc='Computing Representative Vectors', bar_format=_BAR_FORMAT, disable=not verbose) as progress:
        # For each of the clustering results
        for clusters in clusters_dict.values():
            # Calculate the representative vectors
            for cluster in clusters.clusters:
                # Representative vector is the weighted average of the normalised vectors
                cluster.representative, cluster.normalised = _representative_vector(cluster.vectors, cluster.weights)

                # Reconstruct the vectors if requested
                if reconstruct:
                    cluster.reconstructed, cluster.loss = _reconstruct(cluster.vectors, cluster.representative,
                                                                       sigma=sigma, align=align)

                # Update the progress bar
                progress.update(1)

            # Aggregate the cluster results
            if reconstruct:
                clusters.mean_loss = _aggregate_losses(clusters)
                if dist_losses:
                    clusters.distributional_loss = _distributional_loss(clusters, recon=True)

    return clusters_dict


def _representative_vector(vectors: list[ndarray], weights: ndarray) -> tuple[ndarray, ndarray]:
    """
    Calculate the representative vector for these event vectors.

    Args:
        vectors (list[ndarray]): The original event vectors for this cluster (not normalised).
        weights (ndarray): The weights which are the cluster assignment probabilities.

    Returns:
        tuple[ndarray, ndarray]: The representative vector and the locally normalised event vectors.
    """

    if len(vectors) == 0 or len(weights) == 0:
        raise ValueError("Vectors and weights must not be empty.")

    normalised, _ = _normalise_events(vectors, clipped=False)
    representative = average(normalised, axis=0, weights=weights)
    representative = (clip(representative, 0, None) if sum(representative) > 0 else
                      clip(representative, None, 0))

    return representative, normalised


def _reconstruct(
    vectors: list[ndarray],
    representative: ndarray,
    sigma: float = 0.0,
    align: bool = True,
) -> tuple[list[ndarray], dict[str, list]]:
    """
    Reconstruct clustered event vectors from their representative vector.

    Args:
        vectors (list[ndarray]): The original event vectors for this cluster (not normalised).
        representative (ndarray): The representative vector for this cluster.
        sigma (float): The noise standard deviation for interpretability and stability. Default is 0.0.
        align (bool): Whether to align the vectors before calculating losses. Default is True.

    Returns:
        tuple[list[ndarray], dict[str, list]]: The reconstructed event vectors and the reconstruction losses.
    """

    if len(vectors) == 0 or len(representative) == 0:
        raise ValueError("Vectors and representative vector must not be empty.")
    if sigma < 0:
        raise ValueError("Sigma must be non-negative.")

    results = {
        'Correlation': [],
        'Cosine': [],
        'MAE': [],
        'AmpRE': [],
        **({'AC': []} if align else {})
    }
    reconstructed = []
    # Rescale the representative vector to the original vectors
    for vector in vectors:
        # Inverse normalise the representative vector
        recon = _scale_representative(vector, representative)
        reconstructed.append(recon)

        # Amp relative error without alignment or masking
        results['AmpRE'].append(abs(max(abs(recon)) - max(abs(vector))) / (max(abs(vector)) + sigma))

        # Align (with abs(lag) / len(vector) as the alignment cost)
        if align:
            vector, recon, lag = _align_vectors(vector, recon)
            results['AC'].append(lag)

        # Mask to ignore zero values in both vectors
        mask = (vector != 0) & (recon != 0)
        vector = vector[mask]
        recon = recon[mask]

        # Calculate aligned losses
        results['Correlation'].append(corrcoef(recon, vector)[0, 1])
        results['Cosine'].append((recon @ vector) / (norm(recon) * norm(vector)))
        results['MAE'].append(mean(abs(recon - vector)) / (sigma if sigma > 0.0 else 1))

    return reconstructed, results


def _scale_representative(vector: ndarray, representative: ndarray) -> ndarray:
    """
    Scale the representative vector to the true vector.

    Args:
        vector (ndarray): The original event vector (not normalised).
        representative (ndarray): The associated representative vector.

    Returns:
        ndarray: The scaled representative vector.
    """

    assert len(representative) > 2 and len(vector) > 2

    # Inverse normalise the representative vector
    length_scale = (CubicSpline(linspace(0, 1, len(representative)), representative)
                   (linspace(0, 1, len(vector))))
    area_scale = length_scale * simpson(vector) / simpson(length_scale)

    return area_scale


def _align_vectors(vector: ndarray, reconstructed: ndarray) -> tuple[ndarray, ndarray, float]:
    """
    Align the reconstructed and true vector.

    Args:
        vector (ndarray): The original event vector (not normalised).
        reconstructed (ndarray): The reconstructed vector.

    Returns:
        tuple[ndarray, ndarray, float]: The aligned vector, reconstructed vector, and alignment cost.
    """

    assert len(vector) == len(reconstructed)
    assert len(vector) > 0

    vector, reconstructed = vector.copy(), reconstructed.copy()
    lag = argmax(correlate(reconstructed, vector, mode='full')) - (len(vector) - 1)
    if lag > 0:
        vector = pad(vector, (lag, 0))[:len(reconstructed)]
    elif lag < 0:
        reconstructed = pad(reconstructed, (-lag, 0))[:len(vector)]
    ac = abs(lag) / len(vector)

    return vector, reconstructed, ac


def _aggregate_losses(clusters: ShapeClusters) -> dict[str, Any]:
    """
    Aggregate losses from all clusters in the object.

    Args:
        clusters (ShapeClusters): An object containing the clustering results.

    Returns:
        dict[str, Any]: Average values of the losses across all clusters.
    """

    if clusters.clusters[0].loss is None:
        raise ValueError("ShapeClusters must contain 'loss' with loss metrics, see: _representative_vectors.")

    results = {}
    stats = clusters.clusters[0].loss.keys()
    for stat in stats:
        overall = []
        for cluster in clusters.clusters:
            overall.extend(cluster.loss[stat])
        results[stat] = mean(overall)

    return results


def _distributional_loss(clusters: ShapeClusters, recon: bool = True) -> dict[str, float]:
    """
    Calculate distributional losses between event vectors and their reconstructed/augmented counterparts.

    Args:
        clusters (ShapeClusters): An object containing the clustering results.
        recon (bool): Whether to use the reconstructed vectors for the losses (or augmented). Default is True.

    Returns:
        dict[str, float]: The distributional losses.
    """

    if recon and clusters.clusters[0].reconstructed is None:
        raise ValueError("ShapeClusters must contain 'reconstructed' vectors, see: _representative_vectors.")
    if not recon and clusters.clusters[0].augmented is None:
        raise ValueError("ShapeClusters must contain 'augmented' vectors, see: _augment_vectors.")

    # Get the entire population (static type checking sadness)
    vectors = [vec for cluster in clusters.clusters for vec in cluster.vectors]
    if recon:
        reconstructed = [rec for cluster in clusters.clusters for rec in cluster.reconstructed]
    else:
        reconstructed = [aug for cluster in clusters.clusters for aug in cluster.augmented]

    # Get discrete feature EMDs
    results = {
        'Amplitude EMD': _compute_emd(asarray([max(abs(vec)) for vec in vectors]),
                                      asarray([max(abs(rec)) for rec in reconstructed])),
        'Duration EMD': _compute_emd(asarray([len(vec) for vec in vectors]),
                                     asarray([len(rec) for rec in reconstructed])),
        'Area EMD': _compute_emd(asarray([simpson(vec) for vec in vectors]),
                                 asarray([simpson(rec) for rec in reconstructed])),
    }

    return results


def _compute_emd(x: ndarray, y: ndarray) -> float:
    """
    Calculate normalised Earth Mover's Distance for distributional losses.

    Args:
        x (ndarray): The array of original values.
        y (ndarray): The array of reconstructed values.

    Returns:
        float: The normalised EMD between the two distributions.
    """

    assert len(x) > 0 and len(y) > 0

    # Normalise
    x = (x - mean(x)) / std(x)
    y = (y - mean(y)) / std(y)

    # Return EMD
    return wasserstein_distance(x, y)


def _get_knees(clusters_dict: dict[int, ShapeClusters]) -> tuple[ShapeClusters, dict[str, tuple[ndarray, ndarray]], dict[str, ndarray]]:
    """
    Get the knees of reconstruction loss metrics and their change over time.

    Args:
        clusters_dict (dict[int, ShapeClusters]): A dictionary of clustering results keyed by the number of clusters (post-reconstruction).

    Returns:
        tuple[ShapeClusters, dict[str, ndarray], dict[str, tuple[ndarray, ndarray]]]: The best clustering object, a dictionary of loss curves and a dictionary of knees for each metric.
    """

    # Get the stats from the first cluster dict
    first = next(iter(clusters_dict.values()))
    if first.mean_loss is None:
        raise ValueError("ShapeClusters must contain 'mean_loss' with loss metrics, see: _representative_vectors.")
    else:
        stats = first.mean_loss.keys()

    # Calculate the knees and loss curves for each stat
    knees = {}
    loss_curves = {}
    ks = array([clusters.k for clusters in clusters_dict.values()])
    for stat in stats:
        vals = array([clusters.mean_loss[stat] for clusters in clusters_dict.values()])
        knees[stat] = _find_knee(ks, vals)
        loss_curves[stat] = (ks, vals)
    # Ceil mean 6
    best_k = int(ceil(mean(list(knees.values()))))

    return clusters_dict[best_k], loss_curves, knees


def _find_knee(x: ndarray, y: ndarray) -> ndarray:
    """
    Find the geometric knee between x and y.

    Args:
        x (ndarray): The array of monotonic x values.
        y (ndarray): The array of corresponding y values.

    Returns:
        ndarray: The x value of the knee point.
    """

    assert len(x) == len(y)
    if len(x) < 2:
        return x[0]

    p1, p2 = array([x[0], y[0]]), array([x[len(x) - 1], y[len(y) - 1]])
    distances = abs(cross(p2 - p1, p1 - vstack((x, y)).T)) / norm(p2 - p1)
    return x[argmax(distances)]


def _fit_scaling_features(
    clusters: ShapeClusters,
    max_k: int = 50,
    weight: float = 1e-2,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> ShapeClusters:
    """
    Fit density models to the (log) scaling features globally and locally for each cluster.

    Args:
        clusters (ShapeClusters): An object containing the clustering results.
        max_k (int): The maximum number of components to fit. Default is 50.
        weight (float): The weight concentration prior for the Dirichlet Process. Default is 1e-2.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        ShapeClusters: The input clustering results with models added.
    """

    all_durations = []
    all_areas = []
    # Configure progress bar
    with tqdm(total=len(clusters.clusters), desc='Fitting Scaling Features', bar_format=_BAR_FORMAT,
              disable=not verbose) as progress:
        # Fit locally to each cluster
        for cluster in  clusters.clusters:
            if len(cluster.vectors) < 3:
                raise ValueError("ShapeCluster must contain at least three vectors.")

            durations = [len(vector) for vector in cluster.vectors]
            areas = [abs(simpson(vector)) for vector in cluster.vectors]
            all_durations.extend(durations)
            all_areas.extend(areas)
            cluster.local_model = _fit_dp_gmm(log(durations), log(areas), max_k=max_k, weight=weight,
                                              random_state=random_state)

            # Update the progress bar
            progress.update(1)

    # Fit globally to all clusters
    clusters.global_model = _fit_dp_gmm(log(all_durations), log(all_areas), max_k=max_k, weight=weight,
                                        random_state=random_state)

    return clusters


def _fit_dp_gmm(
    durations: ndarray,
    areas: ndarray,
    max_k: int = 50,
    weight: float = 1e-2,
    random_state: Optional[int] = None
) -> BayesianGaussianMixture:
    """
    Fit a Dirichlet Process Gaussian Mixture Model to the given log transformed durations and areas.

    Args:
        durations (ndarray): The log transformed durations of the event vectors (samples).
        areas (ndarray): The log transformed areas of the event vectors (integral).
        max_k (int): The maximum number of components to fit. Default is 50.
        random_state (int | None): Random seed for reproducibility. Default is None.
        weight (float): The weight concentration prior for the Dirichlet Process. Default is 1e-2.

    Returns:
        BayesianGaussianMixture: The fitted Dirichlet Process Gaussian Mixture Model.
    """

    if len(durations) == 0 or len(areas) == 0:
        raise ValueError("Durations and areas must not be empty.")
    if len(durations) != len(areas):
        raise ValueError("Durations and areas must have the same length.")

    data = vstack([durations, areas]).T
    # Add rows in orthogonal directions until we have 3 rows total
    if data.shape[0] < 3:
        step = maximum(abs(data[0]) * 1e-4, 1e-12)
        if data.shape[0] == 1:
            extra = array([data[0] + [step[0], 0.0], data[0] + [0.0, step[1]]])
        else:
            extra = array([data[0] + [step[0], 0.0]])
        data = vstack([data, extra])
    max_k = min(max_k, len(durations))
    dp_gmm = BayesianGaussianMixture(n_components=max_k, covariance_type="full", weight_concentration_prior=weight,
                                    max_iter=500, init_params='k-means++', random_state=random_state)
    dp_gmm.fit(data)
    return dp_gmm


def _augment_vectors(
    clusters: ShapeClusters,
    n_vectors: Optional[int] = None,
    max_attempts: int = 100,
    sample_global: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> ShapeClusters:
    """
    Augment the event vectors using the representative vectors with scaling features derived from both local and global models.

    Args:
        clusters (ShapeClusters): An object containing the clustering results.
        n_vectors (int | None): The number of vectors to generate. If None, one generation will occur per event in
            ShapeClusters. Default is None.
        max_attempts (int): The maximum number of attempts to sample a candidate. Default is 100.
        sample_global (bool): Whether to sample from the global model or the local model. Default is True.
        random_state (int | None): Random seed for reproducibility. Default is None.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        ShapeClusters: The input clustering results with augmentation results added.
    """

    if clusters.clusters[0].representative is None:
        raise ValueError("ShapeClusters must contain 'representative' vector, see: _representative_vectors.")
    if clusters.clusters[0].local_model is None:
        raise ValueError("ShapeClusters must contain 'local_model', see: _fit_scaling_features.")
    if n_vectors is None:
        n_vectors = int(sum([len(cluster.vectors) for cluster in clusters.clusters]))

    # Sample the model to get the associated representative vectors
    clusters.sampled_model, clusters.sampled_labels = clusters.model.sample(n_vectors)

    # Initialise the augmentation results
    for cluster in clusters.clusters:
        cluster.sampled_durations = []
        cluster.sampled_areas = []
        cluster.augmented = []

    # Configure progress bar
    with tqdm(total=n_vectors, desc='Augmenting Clusters', miniters=maximum(1, n_vectors // 100), bar_format=_BAR_FORMAT,
              disable=not verbose) as progress:
        for label in clusters.sampled_labels:
            # Get a sample using the local and global models
            duration, area = _sample_log_method(
                clusters.clusters[label].local_model,
                clusters.global_model,
                max_attempts=max_attempts,
                sample_global=sample_global,
                random_state=random_state
            )

            # Inverse log the sampled duration and area
            duration = int(exp(duration))
            area = exp(area)

            # Augment the representative vector using the predicted duration and area (inv. log it)
            clusters.clusters[label].augmented.append(_augment_representative(
                duration,
                area,
                clusters.clusters[label].representative
            ))
            clusters.clusters[label].sampled_durations.append(duration)
            clusters.clusters[label].sampled_areas.append(area)

            # Update the progress bar
            progress.update(1)

    return clusters


def _sample_log_method(
    local_model: BayesianGaussianMixture,
    global_model: BayesianGaussianMixture,
    max_attempts: int = 100,
    sample_global: bool = True,
    random_state: Optional[int] = None
) -> ndarray:
    """
    Sample a log duration and log area candidate from the local or global model using the log method, similar to
    Metropolis-Hastings rejection sampling, where the candidate is accepted if the log likelihood ratio between
    the not sampled and sampled models exceeds the log of a random number between 0 and 1.

    Args:
        local_model (BayesianGaussianMixture): The local model to sample from.
        global_model (BayesianGaussianMixture): The global model to sample from.
        max_attempts (int): The maximum number of attempts to sample a candidate. Default is 100.
        sample_global (bool): Whether to sample from the global model or the local model. Default is True.
        random_state (int | None): Random seed for reproducibility. Default is None.

    Returns:
        ndarray: A sampled candidate (log duration and log area).
    """

    rng = default_rng(seed=random_state)
    attempts = 0
    while attempts < max_attempts:
        # Sample a candidate from the local or global model
        candidate = global_model.sample(1)[0] if sample_global else local_model.sample(1)[0]
        local_likelihood = local_model.score(candidate)
        global_likelihood = global_model.score(candidate)
        candidate = candidate[0]

        # Calculate the log likelihood ratio
        log_ratio = local_likelihood - global_likelihood if sample_global else global_likelihood - local_likelihood
        if log_ratio > log(rng.random()):
            return candidate
        attempts += 1

    # Fallback to sampling from the local model if no candidate was accepted
    return local_model.sample(1)[0][0]


def _augment_representative(duration: int32, area: float, representative: ndarray) -> ndarray:
    """
    Scale the representative vector to the duration and area.

    Args:
        duration (int): The duration to scale to.
        area (float): The area to scale to.
        representative (ndarray): The associated representative vector.

    Returns:
        ndarray: The scaled representative vector.
    """

    assert len(representative) > 2

    duration = maximum(duration, 3)  # Ensure minimum duration is 3
    area = maximum(area, 1e-12)  # Ensure minimum area is a small positive value

    # Scale the representative vector
    length_scale = (CubicSpline(linspace(0, 1, len(representative)), representative)
                   (linspace(0, 1, duration)))
    area_scale = length_scale * area / abs(simpson(length_scale))

    return area_scale
