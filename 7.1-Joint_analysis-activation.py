import os
import glob
import gc
import mne
import numpy as np
import json
import argparse

from scipy import stats
from collections import Counter
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

from utils.utils import check_clu
from utils.configs import (
    REJECT_SID,
    NUM_VERTEX_DICT
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
RESULTS_ROOT = "./results/7.1-Pretarget_Activation"
SUBJECTS_DIR = "./freesurfer"
SUBJECT_TO = "fsaverage"
SPACING = "ico4"
TAIL = 0
ALPHA_CLUSTER_FORMING = 0.001
N_PERMS = 2000
STATS_PARMS = {
    "n_permutations" : N_PERMS,
    "tail" : TAIL,
    "n_jobs" : N_JOBS,
    "seed" : SEED,
    "buffer_size" : None
}
P_THRESH = 0.05
MIN_CLUSTER_SPACE = 0
MIN_CLUSTER_TIMES = 20 # 100 ms (200 Hz)

TARGET_ONSET = 3.570
TMIN = TARGET_ONSET - 0.4
TMAX = TARGET_ONSET + 0.2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to perform source activation analysis before target-onset.")
    parser.add_argument("--prepare", help="Prepare data for further analysis.", action="store_true")
    parser.add_argument("--stats", help="Conduct the statistics analysis.", action="store_true")
    args = parser.parse_args()

    results_npy_dir = os.path.join(RESULTS_ROOT, "npy")
    results_stc_dir = os.path.join(RESULTS_ROOT, "stc")

    # Prepare data
    if args.prepare:
        os.makedirs(results_npy_dir, exist_ok=True)
        os.makedirs(results_stc_dir, exist_ok=True)

        stc_diff, stc_sta, stc_dev = list(), list(), list()
        for subject in os.listdir(SUBJECTS_DIR):
            if not subject.startswith("sub-") or subject in REJECT_SID: continue

            morph_fname = os.path.join(SUBJECTS_DIR, subject, "bem",
                                    f"{subject}2{SUBJECT_TO}-{SPACING}-morph.h5")
            morph = mne.read_source_morph(morph_fname)

            inv_root = os.path.join(DS_ROOT, f"{subject}/ses-01/meg/inv")

            fnames = glob.glob(
                os.path.join(
                    DS_ROOT,
                    f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-*_meg-epo.fif"
                )
            )
            
            stc_diff_sub, stc_sta_sub, stc_dev_sub = None, None, None
            for counter_f, fname in enumerate(fnames):
                basename = os.path.basename(fname)

                epochs = mne.read_epochs(
                    fname=fname, 
                    preload=True
                ).apply_baseline((None, 0)).resample(SFREQ).filter(l_freq=0.5, h_freq=30, n_jobs=N_JOBS)

                inv = read_inverse_operator(
                    os.path.join(inv_root, basename.replace("_meg-epo.fif", "-inv.fif"))
                )

                stc_sta_f, stc_dev_f = list(), list()
                for topic_idx in range(1, 8):
                    evoked_sta_t = epochs[f"block-2/sent-1/topic-{topic_idx}", 
                                        f"block-2/sent-4/topic-{topic_idx}",
                                        f"block-3/sent-1/topic-{topic_idx}", 
                                        f"block-3/sent-4/topic-{topic_idx}"].average().crop(TMIN - 0.2, TMAX)
                    evoked_dev_t = epochs[f"block-2/sent-2/topic-{topic_idx}", 
                                        f"block-2/sent-3/topic-{topic_idx}",
                                        f"block-3/sent-2/topic-{topic_idx}", 
                                        f"block-3/sent-3/topic-{topic_idx}"].average().crop(TMIN - 0.2, TMAX)
                    
                    stc_sta_f.append(morph.apply(apply_inverse(evoked_sta_t, inv, **INV_PARAMS)).data)
                    stc_dev_f.append(morph.apply(apply_inverse(evoked_dev_t, inv, **INV_PARAMS)).data)
                
                stc_sta_f = np.asarray(stc_sta_f)
                stc_dev_f = np.asarray(stc_dev_f)

                stc_diff_f = stc_dev_f - stc_sta_f
                stc_diff_f = np.mean(stc_diff_f, axis=0)

                if stc_diff_sub is None:
                    stc_diff_sub = np.zeros(shape=(len(fnames), stc_diff_f.shape[0], stc_diff_f.shape[1]))
                    stc_sta_sub = np.zeros(shape=(len(fnames), stc_diff_f.shape[0], stc_diff_f.shape[1]))
                    stc_dev_sub = np.zeros(shape=(len(fnames), stc_diff_f.shape[0], stc_diff_f.shape[1]))
                
                stc_diff_sub[counter_f] = stc_diff_f
                stc_sta_sub[counter_f] = stc_sta_f.mean(axis=0)
                stc_dev_sub[counter_f] = stc_dev_f.mean(axis=0)
            
            stc_diff_sub = stc_diff_sub.mean(axis=0)
            stc_sta_sub = stc_sta_sub.mean(axis=0)
            stc_dev_sub = stc_dev_sub.mean(axis=0)
            
            stc_diff.append(stc_diff_sub)
            stc_sta.append(stc_sta_sub)
            stc_dev.append(stc_dev_sub)
        
        time_vector = evoked_sta_t.times
        np.save(os.path.join(results_npy_dir, "time_vector.npy"), time_vector)

        stc_diff = np.asarray(stc_diff)
        np.save(os.path.join(results_npy_dir, "stc_diff.npy"), stc_diff)

        stc_data = np.asarray([stc_sta, stc_dev]) # n_conds, n_subs, n_vertices, n_times
        stc_data = np.transpose(stc_data, (1, 0, 2, 3)) # n_subs, n_conds, n_vertices, n_times
        np.save(os.path.join(results_npy_dir, "subject_evoked.npy"), stc_data)

        del stc_sta_f, stc_dev_f, stc_data
        del stc_sta_sub, stc_dev_sub, stc_diff_sub
        del stc_diff_f, epochs, morph, inv
        del evoked_sta_t, evoked_dev_t
        gc.collect()

    # Perform statistics analysis
    if args.stats:
        stc_diff = np.load(os.path.join(results_npy_dir, "stc_diff.npy"))
        time_vector = np.load(os.path.join(results_npy_dir, "time_vector.npy"))

        src = mne.read_source_spaces(
            os.path.join(SUBJECTS_DIR,
                        f"{SUBJECT_TO}/bem/{SUBJECT_TO}-{SPACING}-src.fif")
        )
        fs_vertices = [s["vertno"] for s in src]

        # Exclude vertices in medial walls
        num_vertex = NUM_VERTEX_DICT[SPACING]
        cortex_label = mne.read_labels_from_annot(SUBJECT_TO, 
                                                parc="aparc.a2009s", 
                                                hemi="both",
                                                subjects_dir=SUBJECTS_DIR)
        medial_wall = [label for label in cortex_label if "Unknown" in label.name]
        lh_medial_vertices = medial_wall[0].get_vertices_used(np.arange(0, num_vertex//2))
        rh_medial_vertices = medial_wall[1].get_vertices_used(np.arange(0, num_vertex//2)) + num_vertex//2  
        medial_mask = np.zeros(num_vertex, dtype=bool)
        medial_mask[lh_medial_vertices] = True
        medial_mask[rh_medial_vertices] = True
        exclude = medial_mask

        time_mask = (time_vector >= TMIN) & (time_vector <= TMAX)
        time_vector = time_vector[time_mask]
        stc_diff = stc_diff[:, :, time_mask]
    
        adjacency = mne.spatio_temporal_src_adjacency(
            src=src,
            n_times=len(time_vector)
        )

        n_observations = stc_diff.shape[0]
        df = n_observations - 1
        t_thresh = stats.distributions.t.ppf(1 - ALPHA_CLUSTER_FORMING / 2, df=df)

        T_obs, clusters, p_values, _ = cluster_stats = spatio_temporal_cluster_1samp_test(
            np.transpose(stc_diff, (0, 2, 1)),
            threshold=t_thresh,
            adjacency=adjacency,
            spatial_exclude=exclude,
            **STATS_PARMS
        )

        good_clusters, cluster_vis  = list(), list()
        for idx in np.where(p_values < P_THRESH)[0]:
            if check_clu(clusters[idx], MIN_CLUSTER_SPACE, MIN_CLUSTER_TIMES):
                mask = np.zeros(num_vertex)
                _, vertices = clusters[idx]
                counts = Counter(vertices)
                for key, val in counts.items():
                    mask[key] = val
                cluster_vis.append(mask)
                good_clusters.append(clusters[idx])
        cluster_vis = np.sum(cluster_vis, axis=0)

        # Save results
        if len(good_clusters) == 0:
            print("No good significant cluster was found.")
            print(np.min(p_values))
        else:
            stc_all_clusters = summarize_clusters_stc(
                cluster_stats, 
                p_thresh=P_THRESH, 
                vertices=fs_vertices, 
                subject=SUBJECT_TO
            )
            stc_good_clusters = mne.SourceEstimate(
                cluster_vis * (1000 / SFREQ),
                vertices=fs_vertices,
                tmin=0, tstep=1,
                subject=SUBJECT_TO
            )

            ## Make a T_obs stc for all/good clusters
            T_obs_all_data = np.zeros_like(T_obs)
            for idx in np.where(p_values < P_THRESH)[0]:
                t_idx, v_idx = clusters[idx]
                T_obs_all_data[t_idx, v_idx] = T_obs[t_idx, v_idx]
            
            masked_T_all = np.ma.masked_equal(T_obs_all_data, 0)
            T_obs_all_static = masked_T_all.mean(axis=0).filled(0)
            stc_T_obs_all = mne.SourceEstimate(
                data=T_obs_all_static[:, np.newaxis],
                vertices=fs_vertices,
                tmin=0, tstep=1,
                subject=SUBJECT_TO
            )

            T_obs_good_data = np.zeros_like(T_obs)
            for good_cluster in good_clusters:
                t_idx, v_idx = good_cluster
                T_obs_good_data[t_idx, v_idx] = T_obs[t_idx, v_idx]
            
            masked_T_good = np.ma.masked_equal(T_obs_good_data, 0)
            T_obs_good_static = masked_T_good.mean(axis=0).filled(0)
            stc_T_obs_good = mne.SourceEstimate(
                data=T_obs_good_static[:, np.newaxis],
                vertices=fs_vertices,
                tmin=0, tstep=1,
                subject=SUBJECT_TO
            )

            stc_all_clusters.save(
                os.path.join(results_stc_dir, f"Summary-all_alpha-{ALPHA_CLUSTER_FORMING}"), 
                overwrite=True
            )
            stc_good_clusters.save(
                os.path.join(results_stc_dir, f"Summary-good_alpha-{ALPHA_CLUSTER_FORMING}"), 
                overwrite=True
            )
            stc_T_obs_all.save(
                os.path.join(results_stc_dir, f"T-obs-all_alpha-{ALPHA_CLUSTER_FORMING}"),
                overwrite=True
            )
            stc_T_obs_good.save(
                os.path.join(results_stc_dir, f"T-obs-good_alpha-{ALPHA_CLUSTER_FORMING}"),
                overwrite=True
            )
            np.save(
                os.path.join(results_npy_dir, f"T-obs_alpha-{ALPHA_CLUSTER_FORMING}.npy"),
                T_obs
            )

            sig_times_info = dict()
            for counter, good_cluster in enumerate(good_clusters):
                time_idx, vertice_idx = good_cluster
                time_idx = np.unique(time_idx)
                sig_times = time_vector[time_idx]
                sig_times_info[f"Cluster-{counter}"] = dict(start=sig_times[0], end=sig_times[-1])
                
                ## Make a T_obs stc for this cluster
                mask_summary = np.zeros_like(T_obs, dtype=bool)
                time_idx, vertice_idx = good_cluster
                mask_summary[time_idx, vertice_idx] = True   
                masked_data_summary = np.ma.masked_equal(T_obs * mask_summary, 0)
                stc_cluster_t_summary = np.mean(masked_data_summary, axis=0).data
                
                stc_cluster = mne.SourceEstimate(
                    data=stc_cluster_t_summary,
                    vertices=fs_vertices,
                    tmin=0, tstep=1,
                    subject=SUBJECT_TO
                )            
                stc_cluster.save(
                    os.path.join(results_stc_dir, f"T-obs_alpha-{ALPHA_CLUSTER_FORMING}_cluster-{counter}"),
                    overwrite=True
                )
            
            ## Save significant time duration info as a json file
            with open(os.path.join(results_stc_dir, f"sig_times_info_alpha-{ALPHA_CLUSTER_FORMING}.json"), "w") as f:
                json.dump(sig_times_info, f, indent=4)
    
    if not args.prepare and not args.stats:
        parser.print_help()