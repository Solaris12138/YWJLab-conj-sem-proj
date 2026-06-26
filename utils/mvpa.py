import numpy as np

from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, GeneralizingEstimator


def time_sliding_estimator_kfold(
    train_X,
    train_y,
    base_estimator,
    test_X=None,
    test_y=None,
    n_splits=10,
    n_jobs=None,
    average=False,
    scoring="accuracy",
    random_state=None
):
    """
    Time sliding estimator with K-fold

    Parameters
    ----------
    train_X : 3D array, shape (n_trials, n_vertices, n_times)
        Data for training.
    train_y : 1D array, shape (n_trials, )
        The labels of train_X.
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    test_X : 3D array, shape (n_trials, n_vertices, n_times) | None
        Extra data for testing. If None, model will not be tested on extra data.
    test_y : 1D array, shape (n_trials, )
        The labels of test_X.
    n_splits : int
        The number of folds.
    n_jobs : int
        The number of jobs to run in parallel. If -1, it is set to the number of CPU cores.
    average : bool
        Whether to average the metrics among resamplings.
    scoring : callable() | str | None
        Score function (or loss function) with signature score_func(y, y_pred, **kwargs).
    
    Returns
    -------
    scores : 1D or 2D array, shape ([n_splits], n_times)

    """
    
    if test_X is not None:
        assert train_X.shape[-1] == test_X.shape[-1], "Samples for training and testing should share the same timepoints."
    assert train_X.shape[0] == train_y.shape[0], "The number of labels should corresponds to training data."
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = np.zeros((n_splits, train_X.shape[-1]))
    
    cv_counter = 0
    for train_index, test_index in skf.split(train_X, train_y):
        train_X_, test_X_ = train_X[train_index], train_X[test_index]
        train_y_, test_y_ = train_y[train_index], train_y[test_index]
        
        time_slide = SlidingEstimator(base_estimator, n_jobs=n_jobs, scoring=scoring, verbose=True)
        
        time_slide.fit(train_X_, train_y_)
        if test_X is not None and test_y is not None:
            assert test_X.shape[0] == test_y.shape[0], "The number of labels should corresponds to testing data."
            scores[cv_counter, :] = time_slide.score(test_X, test_y)
        else:
            scores[cv_counter, :] = time_slide.score(test_X_, test_y_)
        
        cv_counter += 1

    if average:
        scores = np.mean(scores, axis=0)
    
    return scores


def time_generalizing_estimator_kfold(
    train_X,
    train_y,
    base_estimator,
    test_X=None,
    test_y=None,
    n_splits=10,
    n_jobs=None,
    average=False,
    scoring="accuracy",
    random_state=None
):
    """
    Time generalizing estimator with K-fold

    Parameters
    ----------
    train_X : 3D array, shape (n_trials, n_vertices, n_times)
        Data for training.
    train_y : 1D array, shape (n_trials, )
        The labels of train_X.
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    test_X : 3D array, shape (n_trials, n_vertices, n_times) | None
        Extra data for testing. If None, model will not be tested on extra data.
    test_y : 1D array, shape (n_trials, )
        The labels of test_X.
    n_splits : int
        The number of folds.
    n_jobs : int
        The number of jobs to run in parallel. If -1, it is set to the number of CPU cores.
    average : bool
        Whether to average the metrics among resamplings.
    scoring : callable() | str | None
        Score function (or loss function) with signature score_func(y, y_pred, **kwargs).
    
    Returns
    -------
    scores : 2D or 3D array, shape ([n_splits], n_times, n_times)

    """
    
    if test_X is not None:
        assert train_X.shape[-1] == test_X.shape[-1], "Samples for training and testing should share the same timepoints."
    assert train_X.shape[0] == train_y.shape[0], "The number of labels should corresponds to training data."
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = np.zeros((n_splits, train_X.shape[-1], train_X.shape[-1]))
    
    cv_counter = 0
    for train_index, test_index in skf.split(train_X, train_y):
        train_X_, test_X_ = train_X[train_index], train_X[test_index]
        train_y_, test_y_ = train_y[train_index], train_y[test_index]
        
        time_gen = GeneralizingEstimator(base_estimator, n_jobs=n_jobs, scoring=scoring, verbose=True)
        
        time_gen.fit(train_X_, train_y_)
        if test_X is not None and test_y is not None:
            assert test_X.shape[0] == test_y.shape[0], "The number of labels should corresponds to testing data."
            scores[cv_counter, :, :] = time_gen.score(test_X, test_y)
        else:
            scores[cv_counter, :, :] = time_gen.score(test_X_, test_y_)
        
        cv_counter += 1

    if average:
        scores = np.mean(scores, axis=0)
    
    return scores