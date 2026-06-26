import numpy as np
import mne
import warnings

from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier
from joblib import Parallel, delayed


def source_patches(src, k, d=None, exclude_medial=None):
    """
    Generate a searchlight ball of vertices within distance d from a starting vertex k.

    Parameters
    ----------
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    k : int
        Starting vertex index.
    d : int, optional
        Distance (number of edges) to define the searchlight ball. If None, defaults
        are set based on ico-sampling (see below).
    exclude_medial : array-like, optional
        Array of vertex indices (e.g., medial wall vertices) to exclude from the ball.

    Returns
    -------
    searchlight_ball : np.ndarray
        Array of vertex indices within distance d from vertex k, excluding medial wall
        if specified.

    Notes
    -----
    If d is not provided, default distances are based on ico-sampling:
    - ico=7 (~0.7mm dipole spacing) -> d=9 (~6mm radius)
    - ico=6 (~1.5mm dipole spacing) -> d=4 (~6mm radius)
    - ico=5 (~3mm dipole spacing) -> d=2 (~6mm radius)
    - ico=4 (~6mm dipole spacing) -> d=1 (~6mm radius)
    - ico<4 -> d=1, with a warning suggesting univariate analysis.

    """
    if isinstance(src, mne.SourceSpaces):
        adjacency, _ = mne.channels.find_ch_adjacency(src, ch_type='meg')
    elif issparse(src):
        adjacency = src
    else:
        raise ValueError("src must be either an mne.SourceSpaces or a scipy.sparse.csr_matrix")
    
    # Ensure k is valid
    if not (0 <= k < adjacency.shape[0]):
        raise ValueError(f"Starting vertex k={k} is out of bounds for adjacency matrix of size {adjacency.shape[0]}")

    # Determine default distance d based on ico-sampling if not provided
    if d is None:
        if isinstance(src, mne.SourceSpaces):
            # Estimate ico-sampling based on number of vertices
            n_vertices = sum(len(h['vertno']) for h in src)
            if n_vertices > 100000:  # ico=7, ~0.7mm spacing
                d = 9  # ~6mm radius
            elif n_vertices > 40000:  # ico=6, ~1.5mm spacing
                d = 4
            elif n_vertices > 20000:   # ico=5, ~3mm spacing
                d = 2
            else:                     # ico=4 or lower, ~6mm spacing
                d = 1
                warnings.warn("Low ico-sampling detected (ico<4). Default d=1 is used. "
                              "Consider univariate analysis for better results.")
        else:
            d = 1  # Default for sparse matrix if no src info
            warnings.warn("No SourceSpaces provided, using default d=1. "
                          "Consider specifying d explicitly or using univariate analysis.")
    
    # Validate d
    if not isinstance(d, int) or d < 1:
        raise ValueError("Distance d must be a positive integer")

    # Initialize the starting node as a sparse row vector
    current = csr_matrix(([1], ([0], [k])), shape=(1, adjacency.shape[0]))
    
    # Keep track of all visited nodes
    visited_nodes = set(current.indices.tolist())
    
    # Perform matrix multiplication up to distance d
    for _ in range(d):
        current = current @ adjacency
        visited_nodes.update(current.indices.tolist())

    # Convert visited nodes to array
    searchlight_ball = np.array(list(visited_nodes))

    # Exclude medial wall vertices if specified
    if exclude_medial is not None:
        exclude_medial = np.asarray(exclude_medial)
        if not np.all(np.isin(exclude_medial, np.arange(adjacency.shape[0]))):
            raise ValueError("exclude_medial contains invalid vertex indices")
        searchlight_ball = np.setdiff1d(searchlight_ball, exclude_medial)
    
    return searchlight_ball


def decode_vertex(idx, src, data, label, clf, cv, scoring_function, d=None, exclude_medial=None):
    """
    Perform decoding for a single vertex using its searchlight neighbors.

    Parameters
    ----------
    idx : int
        Index of the vertex to decode.
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    data : np.ndarray
        Source space data with shape (n_trials, n_vertices, n_timepoints).
    label : np.ndarray
        Labels for each trial with shape (n_trials,).
    clf : sklearn.base.BaseEstimator
        Scikit-learn classifier object (e.g., LogisticRegression).
    cv : sklearn.model_selection._BaseCrossValidator
        Cross-validation strategy.
    scoring_function : str or callable
        Scoring function for cross-validation (e.g., "roc_auc", "accuracy", or custom scorer).
    d : int, optional
        Distance for searchlight ball. If None, defaults are set in source_patches.
    exclude_medial : array-like, optional
        Vertex indices to exclude (e.g., medial wall vertices).

    Returns
    -------
    score : float
        Mean cross-validated decoding score for the vertex, or np.nan if decoding fails.

    """
    # Get neighboring vertices using source_patches
    if idx in exclude_medial:
        return np.nan
    
    neighbour_vertices = source_patches(src, k=idx, d=d, exclude_medial=exclude_medial)
    
    # Ensure neighbor vertices are valid
    neighbour_vertices = neighbour_vertices[np.isin(neighbour_vertices, np.arange(data.shape[1]))]
    if len(neighbour_vertices) == 0:
        return np.nan
    
    # Extract data for the searchlight ball
    X = data[:, neighbour_vertices, :]  # Shape: (n_trials, n_neighbors, n_timepoints)
    
    # Flatten spatio-temporal data into a single feature vector per trial
    X_flat = X.reshape(data.shape[0], -1)  # Shape: (n_trials, n_neighbors * n_timepoints)
    
    # Compute cross-validated decoding score
    try:
        score = cross_val_score(clf, X_flat, label, cv=cv, scoring=scoring_function, n_jobs=1)
        return np.mean(score)  # Average across folds
    except ValueError as e:
        print(f"Warning: Decoding failed at vertex {idx}: {e}")
        return np.nan


def searchlight_decoding(src, data, label, clf, scoring_function="roc_auc", d=None, exclude_medial=None, cv=None, n_jobs=1):
    """
    Perform searchlight decoding on source space data, using the entire (n_vertices, n_timepoints)
    as a single feature vector for each trial, with parallel processing over vertices.

    Parameters
    ----------
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    data : np.ndarray
        Source space data with shape (n_trials, n_vertices, n_timepoints).
    label : np.ndarray
        Labels for each trial with shape (n_trials,).
    clf : sklearn.base.BaseEstimator
        Scikit-learn classifier object (e.g., LogisticRegression).
    scoring_function : str or callable, optional
        Scoring function for cross-validation (e.g., "roc_auc", "accuracy", or custom scorer).
        Defaults to "roc_auc".
    d : int, optional
        Distance for searchlight ball. If None, defaults are set in source_patches.
    exclude_medial : array-like, optional
        Vertex indices to exclude (e.g., medial wall vertices).
    cv : sklearn.model_selection._BaseCrossValidator, optional
        Cross-validation strategy. If None, defaults to 5-fold StratifiedKFold.
    n_jobs : int, optional
        Number of parallel jobs for vertex processing. Defaults to 1 (no parallelization).

    Returns
    -------
    scores : np.ndarray, shape (n_vertices,)
        Decoding scores for each vertex.

    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data must be a 3D numpy array (n_trials, n_vertices, n_timepoints)")
    if not isinstance(label, np.ndarray) or label.shape[0] != data.shape[0]:
        raise ValueError("label must be a 1D array with length equal to n_trials")
    if not is_classifier(clf):
        raise ValueError("clf must be a scikit-learn classifier")
    if not (isinstance(scoring_function, str) or callable(scoring_function)):
        raise ValueError("scoring_function must be a string or callable")
    if not isinstance(n_jobs, int) or n_jobs < -1 or n_jobs == 0:
        raise ValueError("n_jobs must be a positive integer, -1, or None")

    _, n_vertices, _ = data.shape

    # Default cross-validation strategy
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define decoding vertices, excluding medial wall if specified
    decoding_vertices = np.arange(n_vertices)
    
    scores = Parallel(n_jobs=n_jobs)(
        delayed(decode_vertex)(idx, src, data, label, clf, cv, scoring_function, d, exclude_medial)
        for idx in decoding_vertices
    )

    # Convert results to numpy array
    scores = np.array(scores)

    # Ensure scores array has correct shape (n_vertices,)
    scores_full = np.full(n_vertices, np.nan)
    scores_full[decoding_vertices] = scores

    return scores_full