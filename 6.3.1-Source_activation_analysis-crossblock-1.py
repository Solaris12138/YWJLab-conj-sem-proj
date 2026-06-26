import os
import glob
import gc
import mne
import numpy as np
import argparse

from mne.minimum_norm import apply_inverse, read_inverse_operator

from utils.configs import REJECT_SID


# Global Parameters
SFREQ = 200
N_JOBS = 32
SNR = 3.0
INV_PARAMS = {
    "lambda2" : 1.0 / SNR ** 2,
    "method" : "eLORETA",
    "pick_ori" : "normal"
}
DS_ROOT = "./data/derivatives/meg_derivatives/preprocessing"
RESULTS_ROOT = "./results/6.3-Source_Activation_CrossBlock-1"
SUBJECTS_DIR = "./freesurfer"
SUBJECT_TO = "fsaverage"
SPACING = "ico4"

TMIN = 3.570
TMAX = 4.270


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to perform cross-block contrast of source activations.")
    parser.add_argument("--prepare", help="Prepare data for further analysis.", action="store_true")
    args = parser.parse_args()

    # Prepare data
    if args.prepare:
        os.makedirs(RESULTS_ROOT, exist_ok=True)

        stc_b1, stc_b2, stc_b3 = list(), list(), list()
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

            stc_b1_sub, stc_b2_sub, stc_b3_sub = None, None, None
            for counter_f, fname in enumerate(fnames):
                basename = os.path.basename(fname)

                epochs = mne.read_epochs(
                    fname=fname, 
                    preload=True
                ).apply_baseline((None, 0)).resample(SFREQ).filter(l_freq=0.5, h_freq=30, n_jobs=N_JOBS)

                evoked_b1_f = epochs["block-1/sent-1", "block-1/sent-4"].average().crop(TMIN, TMAX)
                evoked_b2_f = epochs["block-2/sent-2", "block-2/sent-3"].average().crop(TMIN, TMAX)
                evoked_b3_f = epochs["block-3/sent-2", "block-3/sent-3"].average().crop(TMIN, TMAX)

                inv = read_inverse_operator(
                    os.path.join(inv_root, basename.replace("_meg-epo.fif", "-inv.fif"))
                )

                stc_b1_f = morph.apply(apply_inverse(evoked_b1_f, inv, **INV_PARAMS))
                stc_b2_f = morph.apply(apply_inverse(evoked_b2_f, inv, **INV_PARAMS))
                stc_b3_f = morph.apply(apply_inverse(evoked_b3_f, inv, **INV_PARAMS))

                if stc_b1_sub is None:
                    stc_b1_sub = np.zeros(shape=(len(fnames), stc_b1_f.shape[0], stc_b1_f.shape[1]))
                    stc_b2_sub = np.zeros(shape=(len(fnames), stc_b2_f.shape[0], stc_b2_f.shape[1]))
                    stc_b3_sub = np.zeros(shape=(len(fnames), stc_b3_f.shape[0], stc_b3_f.shape[1]))
                
                stc_b1_sub[counter_f] = stc_b1_f.data
                stc_b2_sub[counter_f] = stc_b2_f.data
                stc_b3_sub[counter_f] = stc_b3_f.data
                
            stc_b1_sub = stc_b1_sub.mean(axis=0)
            stc_b2_sub = stc_b2_sub.mean(axis=0)
            stc_b3_sub = stc_b3_sub.mean(axis=0)

            stc_b1.append(stc_b1_sub)
            stc_b2.append(stc_b2_sub)
            stc_b3.append(stc_b3_sub)
        
        time_vector = stc_b1_f.times
        np.save(os.path.join(RESULTS_ROOT, "time_vector.npy"), time_vector)

        stc_b1 = np.asarray(stc_b1)
        stc_b2 = np.asarray(stc_b2)
        stc_b3 = np.asarray(stc_b3)

        stc_data = np.asarray([stc_b1, stc_b2, stc_b3]) # n_conds, n_subs, n_vertices, n_times
        stc_data = np.transpose(stc_data, (1, 0, 2, 3)) # n_subs, n_conds, n_vertices, n_times
        np.save(os.path.join(RESULTS_ROOT, "subject_evoked.npy"), stc_data)

        del stc_b1_sub, stc_b2_sub, stc_b3_sub
        del stc_b1_f, stc_b2_f, stc_b3_f, epochs, morph, inv
        del evoked_b1_f, evoked_b2_f, evoked_b3_f
        gc.collect()
    
    if not args.prepare:
        parser.print_help()