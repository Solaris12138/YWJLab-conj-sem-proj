import mne
import numpy as np
import matplotlib.pyplot as plt

from operator import add
from functools import reduce
from collections import Counter
from collections.abc import Iterable


def extract_amplitude_marker_ROI(
    stcs, 
    labels, 
    src,
    tmin, 
    tmax,
    mode="max",
    scaling=1
):
    """
    Extract the amplitudes of time courses within given ROIs.

    Parameters
    ----------
    stcs : (Iterable of) mne.SourceEstimate
        The source estimates from which the time courses will be extracted.
    labels : (list of) mne.Label
        The given ROIs.
    src : mne.SourceSpaces
        The source spaces for the source time courses.
    tmin : float
        Minimum of time of interest.
    tmax : float
        Maximum of time of interest.
    mode : str
        The mode to extract amplitude marker. Either "max" or "mean". Default: "max"
    scaling : float or int
        Scale of numerical values.
    
    Returns
    -------
    marker : (list of) float

    """
    if mode not in ["mean", "max"]:
        raise ValueError(f"Invalid mode for extracting amplitude marker.")
    
    if isinstance(stcs, Iterable):
        marker = list()
        for stc in stcs:
            stc_ = stc.copy().crop(tmin=tmin, tmax=tmax)
            label_tc = mne.extract_label_time_course(stc_, labels, src, mode="pca_flip")
            
            if mode == "max":
                marker.append(np.max(label_tc, axis=-1).item() * scaling)
            else:
                marker.append(np.mean(label_tc, axis=-1).item() * scaling)
    else:
        stc_ = stcs.copy().crop(tmin=tmin, tmax=tmax)
        label_tc = mne.extract_label_time_course(stc_, labels, src, mode="pca_flip")
        if mode == "max":
            marker = np.max(label_tc, axis=-1).item() * scaling
        else:
            marker = np.mean(label_tc, axis=-1).item() * scaling
    
    return marker


def get_medial_vertices(atlas, specifier, ico_sample, subjects_dir, subject="fsaverage"):
    """
    Extract the vertices in medial wall, which is meaningless in MEG source sapce.

    Parameters
    ----------
    atlas : str
        The name of parcellation atlas.
    specifier : str
        The specific string denoting the medial wall. E.g., "???" or "unknown"
    ico_sample : str
        Recursively subdivided of an icosahedron. E.g., ico4 or ico5.
    subjects_dir : str | PathLike
        The path to freesurfer subjects directory.
    subject : str
        The example subject used for source space. Default: fsaverage
    
    Returns
    -------
    medial_vertices : np.ndarray, shape (n_vertices, )
        The indices of vertices within the medial wall.

    """
    if ico_sample not in ["ico4", "ico5"]:
        raise ValueError("Only support ico4 or ico5 until now.")

    cortex_label = mne.read_labels_from_annot(subject, parc=atlas, subjects_dir=subjects_dir)
    medial_wall = [label for label in cortex_label if specifier in label.name]

    num_vertex_dict = {
        "ico4" : 5124,
        "ico5" : 20484,
    }
    upper = num_vertex_dict[ico_sample] // 2

    lh_medial_vertices = medial_wall[0].get_vertices_used(np.arange(0, upper))
    rh_medial_vertices = medial_wall[1].get_vertices_used(np.arange(0, upper)) + upper
    medial_vertices = np.concatenate([lh_medial_vertices, rh_medial_vertices])
    
    return medial_vertices
    
    
def check_clu(clus, min_space, min_time):
    """
    Check whether a spatio-temporal cluster is too small or too short.
    
    Parameters
    ----------
    min_space : int
        The threshold of space size.
    min_time: int
        The threshold of time length.
    
    """
    _, vertices = clus
    counts = Counter(vertices)
    timepoints = np.array(list(counts.values()))
    if len(counts.keys()) < min_space:
        return False
    if timepoints.max() < min_time:
        return False
    return True


def crop_brain_and_make_transparent(image):
    """
    Crop the brain from an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image pixel values.
    
    Returns
    -------
    fig: matplotlib.figure.Figure

    """
    nonwhite_pix = (image != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = image[nonwhite_row][:, nonwhite_col]

    alpha = np.where(np.all(cropped_screenshot == [255, 255, 255], axis=-1), 0, 255)
    cropped_screenshot = np.dstack((cropped_screenshot, alpha))
    plt.ioff()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.imshow(cropped_screenshot)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def combine_labels(subject, subjects_dir, parc, label_names):
    """
    Combine the labels of selected regions.

    Parameters
    ----------
    subject : str
        The name of subject of whom the source space is.
    subjects_dir : str | path-like
        The path to the directory containing the FreeSurfer subjects reconstructions.
    parc : str
        The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
    label_names : list of str
        A list consists of the names of selected regions.
    
    Returns
    -------
    label : mne.Label

    """

    labels =  mne.read_labels_from_annot(
        subject=subject, 
        parc=parc, 
        subjects_dir=subjects_dir, 
        hemi="both", 
        regexp=None
    )
    labels = [label if label.name in label_names else None for label in labels] 
    labels = list(filter(None, labels))
    label = reduce(add, labels)

    return label


def mm_to_inches(width_mm, height_mm):
    """
    Convert dimensions from millimeters to inches.

    Parameters
    ----------
    width_mm : float
        Width in millimeters.
    height_mm : float 
        Height in millimeters.

    Returns
    -------
    tuple : A tuple containing (width_in_inches, height_in_inches).
    
    """
    return width_mm / 25.4, height_mm / 25.4