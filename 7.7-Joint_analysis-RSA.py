import os
import glob
import gc
import mne
import mne_rsa
import numpy as np
import pandas as pd
import scipy
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from mne.stats import permutation_cluster_1samp_test
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from utils.utils import mm_to_inches
from utils.sem_embed import extract_semantic_embeddings
from utils.smooth import gaussian_smooth
from utils.configs import (
    NUM_TOPICS,
    REJECT_SID,
    TOPIC_SEM_SETTINGS,
    BLOCK_MAPPING
)


# Global Parameters
SFREQ = 200
N_JOBS = 32
SEED = 1016
SNR = 3.0
INV_PARAMS = {
    "lambda2" : 1.0 / SNR ** 2,
    "method" : "eLORETA",
    "pick_ori" : "normal"
}
DS_ROOT = "./data/derivatives/meg_derivatives/preprocessing"
RESULTS_ROOT = "./results/7.7-RSA"
SUBJECTS_DIR = "./freesurfer"
SUBJECT_TO = "fsaverage"
SPACING = "ico4"
SIGMA = 2
WINDOW_LEN = 10 # 50 ms
P_THRESH = 0.05

RSA_PARAMS = {
    "data_rdm_metric" : "correlation",
    "rsa_metric" : "partial-spearman",
    "n_folds" : 1,
    "n_jobs" : N_JOBS,
    "verbose" : True
}

N_PERMS = 2000
TAIL=0
STATS_PARMS = {
    "n_permutations" : N_PERMS,
    "tail" : TAIL,
    "n_jobs" : N_JOBS,
    "seed" : SEED,
    "buffer_size" : None
}

COND_SETTINGS = {
    # Based on "TOPIC_SEM_SETTINGS": [context_idx, target_idx, local_expected_target_idx, Conn1Type, Conn2Type]
    # Conn1Type (1 for "因为", -1 for "虽然")
    # Conn2Type (1 for "所以", -1 for "但是")
    "Sta1" : [0, 0, 0, 1, 1],
    "Sta2" : [1, 1, 1, -1, -1],

    "B1Dev1" : [0, 1, 0, 1, 1],
    "B1Dev2" : [1, 0, 1, -1, -1],

    "B2Dev1" : [2, 1, 1, 1, -1],
    "B2Dev2" : [3, 0, 0, -1, 1],

    "B3Dev1" : [2, 0, 1, 1, -1],
    "B3Dev2" : [3, 1, 0, -1, 1]
}
EMBED_MODEL_PATH = "./utils/huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ROI_LABEL_PATH = "./results/7.1-Pretarget_Activation/MTG-lh.label"

TARGET_ONSET = 3.570
TMIN = TARGET_ONSET - 0.5
TMAX = TARGET_ONSET + 0.2


# Self-defined functions
def _summarize_stim_info():
    """
    Summarize the information of stimuli.

    Returns
    -------
    list of dict
    """
    stim_info = list()
    for topic_idx in range(1, NUM_TOPICS + 1):
        targets = TOPIC_SEM_SETTINGS[f"topic{topic_idx}"]["targets"]
        contexts = TOPIC_SEM_SETTINGS[f"topic{topic_idx}"]["contexts"]
        for cond, setting in COND_SETTINGS.items():
            stim_info.append(
                {
                    "context" : contexts[setting[0]],
                    "target" : targets[setting[1]],
                    "local_expected_target" : targets[setting[2]],
                    "conn1_type" : setting[3],
                    "conn2_type" : setting[4],
                    "condition" : cond,
                    "topic" : topic_idx
                }
            )
    return stim_info


def _build_rdm_mask(stim_info):
    """
    Within-topic RDM Mask
    
    Logic:
    - The same topics -> 1
    - Different topics -> nan
    """
    n_stims = len(stim_info)
    mask = np.zeros((n_stims, n_stims))

    for i in range(n_stims):
        for j in range(i + 1, n_stims):
            if stim_info[i]["topic"] != stim_info[j]["topic"]:
                mask[i, j] = np.nan  # No Cross-Topic Comparisons
                
    mask = mask + mask.T
    mask[mask == 0] = 1
    return mask


def _build_target_identity_rdm(stim_info):
    """
    M1: Target Identity RDM
    
    Logic:
    - The same target words -> dist = 0
    - Different target words -> dist = 1
    """
    n_stims = len(stim_info)
    rdm = np.zeros((n_stims, n_stims))
    
    for i in range(n_stims):
        for j in range(i + 1, n_stims):
            if stim_info[i]["target"] != stim_info[j]["target"]:
                rdm[i, j] = 1
                
    rdm = rdm + rdm.T
    return rdm


def _build_local_target_identity_rdm(stim_info):
    """
    M2: Local Predicted Target Identity RDM
    
    Logic:
    - The same target words -> dist = 0
    - Different target words -> dist = 1
    """
    n_stims = len(stim_info)
    rdm = np.zeros((n_stims, n_stims))
    
    for i in range(n_stims):
        for j in range(i + 1, n_stims):
            if stim_info[i]["local_expected_target"] != stim_info[j]["local_expected_target"]:
                rdm[i, j] = 1
                
    rdm = rdm + rdm.T
    return rdm


def _build_structure_rdm(stim_info):
    """
    M3: Structure RDM
    
    Structural Characteristics: (Conn1Type, Conn2Type)
    """
    n_stims = len(stim_info)
    rdm = np.zeros((n_stims, n_stims))
    
    for i in range(n_stims):
        for j in range(i + 1, n_stims):
            struct_i = (stim_info[i]["conn1_type"], stim_info[i]["conn2_type"])
            struct_j = (stim_info[j]["conn1_type"], stim_info[j]["conn2_type"])
            rdm[i, j] = 0 if struct_i == struct_j else 1
            
    rdm = rdm + rdm.T
    return rdm


def _build_stem_semantic_rdm(stim_info):
    """
    M4: Sentence Stem Semantic RDM
    """
    items = [s["context"] for s in stim_info]
    embeds = extract_semantic_embeddings(items, EMBED_MODEL_PATH)
    rdm = 1 - cosine_similarity(embeds)
    np.fill_diagonal(rdm, 0)

    return rdm


def prepare_model_rdm():
    save_dir = os.path.join(RESULTS_ROOT, "RDM")
    os.makedirs(save_dir, exist_ok=True)

    stim_info = _summarize_stim_info()
    df = pd.DataFrame(stim_info)
    df.to_csv(os.path.join(save_dir, "stim_info.csv"), index=False)

    mask = _build_rdm_mask(stim_info)
    np.save(os.path.join(save_dir, "mask.npy"), mask)

    # M1: Target Identity RDM
    rdm = _build_target_identity_rdm(stim_info)
    np.save(os.path.join(save_dir, "M1.npy"), rdm)

    # M2: Local Predicted Target Identity RDM
    rdm = _build_local_target_identity_rdm(stim_info)
    np.save(os.path.join(save_dir, "M2.npy"), rdm)

    # M3: Structure RDM
    rdm = _build_structure_rdm(stim_info)
    np.save(os.path.join(save_dir, "M3.npy"), rdm)

    # M4: Sentence Stem Semantic RDM
    rdm = _build_stem_semantic_rdm(stim_info)
    np.save(os.path.join(save_dir, "M4.npy"), rdm)

    print("Wrote:")
    for p in sorted(os.listdir(save_dir)):
        print(os.path.join(save_dir, p))


def prepare_neural_data():
    save_dir = os.path.join(RESULTS_ROOT, "neural_data")
    os.makedirs(save_dir, exist_ok=True)

    roi_label = mne.read_label(ROI_LABEL_PATH)

    neuro_data = list()
    time_vector = None
    for subject in os.listdir(SUBJECTS_DIR):
        if not subject.startswith("sub-") or subject in REJECT_SID: continue

        morph_fname = os.path.join(SUBJECTS_DIR, subject, "bem", f"{subject}2{SUBJECT_TO}-{SPACING}-morph.h5")
        morph = mne.read_source_morph(morph_fname)

        inv_root = os.path.join(DS_ROOT, f"{subject}/ses-01/meg/inv")

        fnames = glob.glob(
            os.path.join(
                DS_ROOT,
                f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-0[234]_meg-epo.fif"
            )
        )

        neuro_data_sub = [list() for _ in range(NUM_TOPICS * len(COND_SETTINGS))]

        for counter_f, fname in enumerate(fnames):
            basename = os.path.basename(fname)

            epochs = mne.read_epochs(
                fname=fname, 
                preload=True
            ).apply_baseline((None, 0)).resample(SFREQ).filter(l_freq=0.5, h_freq=30, n_jobs=N_JOBS).crop(TMIN, TMAX)

            if time_vector is None:
                time_vector = epochs.times

            inv_event_id = {v: k for k, v in epochs.event_id.items()}
            events = [inv_event_id[e] for e in epochs.events[:, -1]] # block-1/sent-3/topic-3/stimID-27
            sent_ids = [e[:e.rfind("/")] for e in events] # block-1/sent-3/topic-3

            inv = read_inverse_operator(
                os.path.join(inv_root, basename.replace("_meg-epo.fif", "-inv.fif"))
            )

            label_stcs = [morph.apply(stc).in_label(roi_label) for stc in apply_inverse_epochs(epochs, inv, **INV_PARAMS)]

            for i in range(len(label_stcs)):
                comps = sent_ids[i].split("/")

                topic_idx = int(comps[-1].split("-")[-1])
                cond_idx = list(COND_SETTINGS.keys()).index(BLOCK_MAPPING["/".join(comps[:2])])

                neuro_data_sub[(topic_idx - 1) * len(COND_SETTINGS) + cond_idx].append(label_stcs[i].data)

        try:
            neuro_data_sub = np.asarray([np.asarray(n).mean(axis=0) for n in neuro_data_sub])
            neuro_data.append(neuro_data_sub)
        except ValueError:
            print(f"Empty trial type exists for subject: {subject}")
            continue
    
    neuro_data = np.asarray(neuro_data)
    print(f"Saved data shape: {neuro_data.shape}")
    np.save(os.path.join(save_dir, "data.npy"), neuro_data)
    np.save(os.path.join(save_dir, "time_vector.npy"), time_vector)

    print("Wrote:")
    for p in sorted(os.listdir(save_dir)):
        print(os.path.join(save_dir, p))

    del epochs, inv, morph
    del label_stcs
    del neuro_data, neuro_data_sub
    gc.collect()


def run_RSA(if_mask):
    save_dir = os.path.join(RESULTS_ROOT, "RSA_values")
    os.makedirs(save_dir, exist_ok=True)

    # Load RDMs
    rdm_dir = os.path.join(RESULTS_ROOT, "RDM")

    model_names = ["M1", "M2", "M3", "M4"]
    full_rdms = [np.load(os.path.join(rdm_dir, f"{name}.npy")) for name in model_names]

    # Load neural data
    neural_data = np.load(os.path.join(RESULTS_ROOT, "neural_data/data.npy"))
    neural_data = gaussian_smooth(neural_data, sigma=SIGMA)

    n_subjects, n_items, n_vertices, n_times = neural_data.shape
    time_vector = np.load(os.path.join(RESULTS_ROOT, "neural_data/time_vector.npy"))

    # Time-resolved RSA
    rsa_values = None
    for subject_idx in range(n_subjects):
        if if_mask:
            topic_scores = None
            for topic_idx in range(NUM_TOPICS):
                start_idx = topic_idx * len(COND_SETTINGS)
                end_idx = start_idx + len(COND_SETTINGS)

                stcs_topic = neural_data[subject_idx, start_idx:end_idx, :, :]
                rdms_topic = [rdm[start_idx:end_idx, start_idx:end_idx] for rdm in full_rdms]

                patches = mne_rsa.searchlight(
                    shape=stcs_topic.shape, 
                    spatial_radius=None,
                    temporal_radius=WINDOW_LEN
                )

                rsa_values_topic = mne_rsa.rsa_array(
                    X=stcs_topic,
                    rdm_model=rdms_topic,
                    patches=patches,
                    **RSA_PARAMS
                )

                if topic_scores is None:
                    topic_scores = np.zeros(shape=(NUM_TOPICS, rsa_values_topic.shape[0], rsa_values_topic.shape[1]))
                
                topic_scores[topic_idx] = rsa_values_topic
            
            def _Fisher_z(corrs):
                """
                Integrating Correlation Coefficients Using Fisher's z-Transform
                """
                cliped_corrs = np.clip(corrs, -0.999999, 0.999999)
                z_values = np.arctanh(cliped_corrs)
                z_mean = np.mean(z_values, axis=0)
                return np.tanh(z_mean)
            
            rsa_values_sub = _Fisher_z(topic_scores)
        else:
            patches = mne_rsa.searchlight(
                shape=neural_data[subject_idx].shape, 
                spatial_radius=None,
                temporal_radius=WINDOW_LEN
            )
            
            rsa_values_sub = mne_rsa.rsa_array(
                X=neural_data[subject_idx],
                rdm_model=full_rdms,
                patches=patches,
                **RSA_PARAMS
            )
        
        if rsa_values is None:
            rsa_values = np.zeros(shape=(n_subjects, rsa_values_sub.shape[0], rsa_values_sub.shape[1]))
        
        rsa_values[subject_idx] = rsa_values_sub
    
    # Save the results
    print(f"RSA values shape: {rsa_values.shape}")
    np.save(os.path.join(save_dir, "rsa_values.npy"), rsa_values)
    np.save(os.path.join(save_dir, "time_vector.npy"), time_vector[WINDOW_LEN:-WINDOW_LEN])
    np.save(os.path.join(save_dir, "model_names.npy"), np.array(model_names))

    print("Wrote:")
    for p in sorted(os.listdir(save_dir)):
        print(os.path.join(save_dir, p))


def run_stats():
    plt.rc("font", family="Arial")

    save_dir = os.path.join(RESULTS_ROOT, "stats_results")
    os.makedirs(save_dir, exist_ok=True)

    rsa_values = np.load(os.path.join(RESULTS_ROOT, "RSA_values/rsa_values.npy"))
    time_vector = np.load(os.path.join(RESULTS_ROOT, "RSA_values/time_vector.npy"))
    model_names = np.load(os.path.join(RESULTS_ROOT, "RSA_values/model_names.npy")).tolist()

    n_subjects, n_models, n_times = rsa_values.shape

    # Okabe-Ito inspired palette
    okabe_ito = np.array([
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ])

    if n_models <= len(okabe_ito):
        colors = okabe_ito[:n_models]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(1, 1, figsize=(mm_to_inches(100, 45)))
    
    sig_counter = 0
    for idx, name in enumerate(model_names):

        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(rsa_values[:, idx, :], **STATS_PARMS)

        mean = rsa_values[:, idx, :].mean(axis=0)
        se = rsa_values[:, idx, :].std(axis=0) / np.sqrt(n_subjects)

        mean_smooth = scipy.ndimage.gaussian_filter1d(mean, sigma=2)
        lower_smooth = scipy.ndimage.gaussian_filter1d(mean - se, sigma=2)
        upper_smooth = scipy.ndimage.gaussian_filter1d(mean + se, sigma=2)
        
        ax.plot(time_vector, mean_smooth, color=colors[idx], linewidth=1, label=name, zorder=3)
        ax.fill_between(time_vector, lower_smooth, upper_smooth, color=colors[idx], alpha=0.14, linewidth=0, zorder=2)

        sig_y = -0.07 - sig_counter * 0.005
        for clu_idx, clu in enumerate(clusters):
            if p_values[clu_idx] < P_THRESH:
                t_inds = clu[0]
                start_time = time_vector[t_inds[0]]
                end_time = time_vector[t_inds[-1]]
                if end_time - start_time < 0.1:
                    continue
                sig_counter += 1
                print(f"Suggested significant times: {start_time:.3f}s-{end_time:.3f}s")
                ax.hlines(y=sig_y, xmin=start_time, xmax=end_time, colors=colors[idx], linewidth=2, alpha=0.95, zorder=4)
    
    ax.axvline(
        TARGET_ONSET,
        linestyle="--",
        color="0.55",
        linewidth=0.8,
        zorder=1
    )

    ax.axhline(
        0,
        linestyle="dashed",
        color="black",
        linewidth=0.8,
        zorder=1
    )
    
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xlabel("Time [s]", fontsize=7, fontweight="bold")
    if "spearman" in RSA_PARAMS["rsa_metric"]:
        rho = "\u03C1"
        ax.set_ylabel(f"Spearman's {rho}", fontsize=7, fontweight="bold")
    if "pearson" in RSA_PARAMS["rsa_metric"]:
        ax.set_ylabel("Pearson's R", fontsize=7, fontweight="bold")
    
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.axvline(TARGET_ONSET, linestyle="--", color="gray")
    leg = ax.legend(
        title="Model RDM",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0
    )

    plt.tight_layout(pad=0.4)

    save_fig_path = os.path.join(save_dir, "rsa_result.svg")
    plt.savefig(save_fig_path, transparent=True, dpi=600, format="svg", bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved to {save_fig_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to perform RSA before target-onset.")
    parser.add_argument("--prepare_model_rdm", help="Prepare model RDM for further analysis.", action="store_true")
    parser.add_argument("--prepare_neural_data", help="Prepare neural data for further analysis.", action="store_true")
    parser.add_argument("--run", help="Conduct RSA.", action="store_true")
    parser.add_argument("--stats", help="Conduct the statistics analysis.", action="store_true")
    parser.add_argument("--mask", help="Whether to perform a mask for within-topic RSA.", default=True)

    args = parser.parse_args()

    if args.prepare_model_rdm:
        prepare_model_rdm()
    
    if args.prepare_neural_data:
        prepare_neural_data()

    if args.run:
        run_RSA(args.mask)

    if args.stats:
        run_stats()

    if not args.prepare_model_rdm and not args.prepare_neural_data and not args.run and not args.stats:
        parser.print_help()