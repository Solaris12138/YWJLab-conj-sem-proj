import os
import re
import glob
import mne
import pandas as pd
import argparse

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from utils.utils import extract_amplitude_marker_ROI, combine_labels
from utils.configs import (
    REJECT_SID, 
    ATLAS,
    MidTL_LH, 
    ATL_LH
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
BEH_ROOT = "./data/derivatives/behaviors"
RESULTS_ROOT = "./results/7.5-Neuro_Behavior_Extraction"
SUBJECTS_DIR = "./freesurfer"
SUBJECT_TO = "fsaverage"
SPACING = "ico4"

NEURO_MARKERS = {
    "Post_400_MidTL_LH_amp" : {
        "label" : combine_labels(SUBJECT_TO, SUBJECTS_DIR, ATLAS, MidTL_LH),
        "tmin" : 3.905,
        "tmax" : 4.020
    },
    "Post_400_ATL_LH_amp" : {
        "label" : combine_labels(SUBJECT_TO, SUBJECTS_DIR, ATLAS, ATL_LH),
        "tmin" : 3.860,
        "tmax" : 4.070
    }
}

TMIN = 3.0
TMAX = 4.5


# Self-defined functions
def _reformat_strings(strings):
    """
    e.g., "topic-6_block-3_sent-1" -> "block-3/sent-1/topic-6"
    """
    return ["/".join([parts[1], parts[2], parts[0]]) 
            for parts in (s.split("_") for s in strings)]


def run_extraction():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    src = mne.read_source_spaces(
        os.path.join(SUBJECTS_DIR,
                    f"{SUBJECT_TO}/bem/{SUBJECT_TO}-{SPACING}-src.fif")
    )

    MTG_label_lh = mne.read_label("./results/7.1-Pretarget_Activation/MTG-lh.label")
    NEURO_MARKERS["Pre_MTG_LH_amp"] = {
        "label" : MTG_label_lh,
        "tmin" : 3.125,
        "tmax" : 3.390
    }

    ############################################################
    # Create a new DataFrame object in the below form
    ############################################################
    # |   sid  |         SentID         |  RT | amp |
    # | sub-xx | block-1/sent-1/topic-1 | ... | ... |
    # | sub-xx | block-1/sent-1/topic-2 | ... | ... |
    ############################################################

    data_dict = {
        "sid" : list(),
        "SentID" : list(),
        "RT" : list(),
        "Run" : list(),
        "Correctness": list(),
        "Pre_MTG_LH_amp" : list(),
        "Post_400_MidTL_LH_amp" : list(),
        "Post_400_ATL_LH_amp" : list(),
    }

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

        for counter_f, fname in enumerate(fnames):
            basename = os.path.basename(fname)

            # Neuro data
            epochs = mne.read_epochs(
                fname=fname, 
                preload=True
            ).apply_baseline((None, 0)).resample(SFREQ).filter(l_freq=0.5, h_freq=30, n_jobs=N_JOBS).crop(TMIN, TMAX)

            inv_event_id = {v: k for k, v in epochs.event_id.items()}
            events = [inv_event_id[e] for e in epochs.events[:, -1]]
            sent_ids = [e[:e.rfind("/")] for e in events]

            inv = read_inverse_operator(
                os.path.join(inv_root, basename.replace("_meg-epo.fif", "-inv.fif"))
            )

            stcs = [morph.apply(stc) for stc in apply_inverse_epochs(epochs, inv, **INV_PARAMS)]

            data_dict["SentID"] += sent_ids
            data_dict["sid"] += [subject] * len(epochs)

            for key in NEURO_MARKERS.keys():
                data_dict[key] += extract_amplitude_marker_ROI(stcs, NEURO_MARKERS[key]["label"], src,
                                                               NEURO_MARKERS[key]["tmin"], NEURO_MARKERS[key]["tmax"],
                                                               mode="mean", scaling=1e12)
            
            # Behavior data
            match = re.search(r'_run-(\d+)', basename)
            if match: 
                run = int(match.group(1))
            else:
                run = None
            data_dict["Run"] += [run] * len(epochs)

            beh_re = re.sub(r'^sub-(\d+)([a-z]+)$', lambda m: m.group(1) + m.group(2).upper(), subject) + f"_run{run}_ConjSemantic_*.csv"
            beh_fname = glob.glob(os.path.join(BEH_ROOT, beh_re))[0]
            df_beh = pd.read_csv(beh_fname, encoding="utf-8", sep=",")

            stim_fnames = df_beh["Stim_fname"][1:-1].values
            stim_ids = [f.replace(".wav", "") for f in stim_fnames]
            stim_ids = _reformat_strings(stim_ids)
            resp_correctness = df_beh["key_resp.corr"][1:-1].values
            resp_times = df_beh["key_resp.rt"][1:-1].values * 1000 # sec -> ms

            for sent_id in sent_ids:
                index = stim_ids.index(sent_id)
                data_dict["RT"].append(resp_times[index])
                data_dict["Correctness"].append(resp_correctness[index])
    
    df = pd.DataFrame(data_dict)
    df.drop(df[df["Correctness"] == 0].index, inplace=True)
    df.drop(columns=["Correctness"], inplace=True)
    df.to_csv(path_or_buf=os.path.join(RESULTS_ROOT, "ROISourceAmpRT_PrePostTarget.csv"), 
              index=False, float_format="%.9f", lineterminator="\n")

    print("Wrote: ", os.path.join(RESULTS_ROOT, "ROISourceAmpRT_PrePostTarget.csv"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to extract ROI-based brain activities and behavior data for each trial.")
    parser.add_argument("--prepare", help="Prepare the dataframe for model comparation and store it as a .csv file.", action="store_true")
    args = parser.parse_args()

    # Amplitude marker extraction
    if args.prepare:
        run_extraction()

    if not args.prepare:
        parser.print_help()