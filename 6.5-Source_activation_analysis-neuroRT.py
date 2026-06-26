import os
import glob
import mne
import numpy as np
import pandas as pd
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from mne.minimum_norm import apply_inverse, read_inverse_operator

from utils.utils import extract_amplitude_marker_ROI, combine_labels, mm_to_inches
from utils.configs import (
    REJECT_SID,
    ATLAS,
    MidTL_LH
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
RESULTS_ROOT = "./results/6.5-Source_Activation_NeuroRT"
SUBJECTS_DIR = "./freesurfer"
SUBJECT_TO = "fsaverage"
SPACING = "ico4"

TMIN = 3.905
TMAX = 4.020


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to perform correlation analysis for source activations and behaviors.")
    parser.add_argument("--prepare", help="Prepare the dataframe for correlation analysis and store it as a .csv file.", action="store_true")
    parser.add_argument("--plot", help="Perform the correlation analyses and display the result.", action="store_true")
    args = parser.parse_args()

    # Prepare data
    if args.prepare:
        os.makedirs(RESULTS_ROOT, exist_ok=True)

        src = mne.read_source_spaces(
            os.path.join(SUBJECTS_DIR,
                        f"{SUBJECT_TO}/bem/{SUBJECT_TO}-{SPACING}-src.fif")
        )
        fs_vertices = [s["vertno"] for s in src]

        label = combine_labels(SUBJECT_TO, SUBJECTS_DIR, ATLAS, MidTL_LH)
        label.name = "MidTL-lh"

        df_data = list()
        for subject in os.listdir(SUBJECTS_DIR):
            if not subject.startswith("sub-") or subject in REJECT_SID: continue
        
            morph_fname = os.path.join(SUBJECTS_DIR, subject, "bem",
                                    f"{subject}2{SUBJECT_TO}-{SPACING}-morph.h5")
            morph = mne.read_source_morph(morph_fname)

            inv_root = os.path.join(DS_ROOT, f"{subject}/ses-01/meg/inv")

            fif_fnames = glob.glob(
                os.path.join(
                    DS_ROOT,
                    f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-*_meg-epo.fif"
                )
            )

            delta_amp = list()
            for counter_f, fname in enumerate(fif_fnames):
                basename = os.path.basename(fname)

                epochs = mne.read_epochs(
                    fname=fname, 
                    preload=True
                ).apply_baseline((None, 0)).resample(SFREQ).filter(l_freq=0.5, h_freq=30, n_jobs=N_JOBS)

                evoked_b2_f = epochs["block-2/sent-2", "block-2/sent-3"].average().crop(TMIN, TMAX)
                evoked_b3_f = epochs["block-3/sent-2", "block-3/sent-3"].average().crop(TMIN, TMAX)

                inv = read_inverse_operator(
                    os.path.join(inv_root, basename.replace("_meg-epo.fif", "-inv.fif"))
                )

                stc_b2_f = morph.apply(apply_inverse(evoked_b2_f, inv, **INV_PARAMS))
                stc_b3_f = morph.apply(apply_inverse(evoked_b3_f, inv, **INV_PARAMS))

                b2_amp_f = extract_amplitude_marker_ROI(stc_b2_f, label, src, TMIN, TMAX, mode="mean", scaling=1e12)
                b3_amp_f = extract_amplitude_marker_ROI(stc_b3_f, label, src, TMIN, TMAX, mode="mean", scaling=1e12)

                delta_amp_f = b3_amp_f - b2_amp_f
                delta_amp.append(delta_amp_f)
            delta_amp = np.asarray(delta_amp).mean()

            csv_fnames = glob.glob(
                os.path.join(
                    BEH_ROOT,
                    f"{subject.split('-')[-1].upper()}_run?_ConjSemantic_*.csv"
                )
            )

            rt_b2, rt_b3 = list(), list()
            for counter_f, fname in enumerate(csv_fnames):
                df_beh = pd.read_csv(fname, encoding="utf-8", sep=",")

                stim_fnames = df_beh["Stim_fname"][1:-1].values
                resp_correctness = df_beh["key_resp.corr"][1:-1].values
                resp_times = df_beh["key_resp.rt"][1:-1].values

                for index, (stim_fname, correctness, rt) in enumerate(zip(stim_fnames, resp_correctness, resp_times)):
                    topic_id_, block_id_, sent_id_ = stim_fname.replace(".wav", "").split("_")
                    block_id_ = int(block_id_.split("-")[-1])

                    if block_id_ == 1: continue

                    if correctness == 1:
                        if block_id_ == 2:
                            rt_b2.append(float(rt) * 1000)
                        else:
                            rt_b3.append(float(rt) * 1000)
            rt_b2 = np.asarray(rt_b2).mean()
            rt_b3 = np.asarray(rt_b3).mean()
            delta_rt = rt_b3 - rt_b2

            df_data.append({"sid": subject, "delta_RT": delta_rt, "delta_Amp": delta_amp})

        df = pd.DataFrame(df_data)
        df.to_csv(path_or_buf=os.path.join(RESULTS_ROOT, "MidTL_SourceAmpRT.csv"), index=False)

    # Correlation Analysis
    if args.plot:
        
        plt.rcParams["font.family"] = "Arial"

        df = pd.read_csv(os.path.join(RESULTS_ROOT, "MidTL_SourceAmpRT.csv"))

        r_val, p_pearson = stats.pearsonr(df["delta_RT"], df["delta_Amp"])
        print(f"Pearson's R: r = {r_val:.3f}, p = {p_pearson:.3f}")
        tau, p_kendall = stats.kendalltau(df["delta_RT"], df["delta_Amp"])
        print(f"Kendall's tau-B: tau = {tau:.3f}, p = {p_kendall:.3f}")
        rho, p_spearman = stats.spearmanr(df["delta_RT"], df["delta_Amp"])
        print(f"Spearman's rho: rho = {rho:.3f}, p = {p_spearman:.3f}")

        sns.set_style("whitegrid")

        plt.figure(figsize=mm_to_inches(55, 55))

        ax = sns.regplot(
            data=df,
            x="delta_RT",
            y="delta_Amp",

            # Scatter style
            scatter_kws={
                "s": 45,
                "alpha": 0.75,
                "color": "#4C78A8",
                "edgecolor": "white",
                "linewidths": 0.8
            },

            # Regression line style
            line_kws={
                "color": "#2F5597",
                "linewidth": 2
            },

            ci=95
        )
        sns.despine()

        for collection in ax.collections:
            try:
                collection.set_alpha(0.18)
            except:
                pass

        plt.xlabel(r'$\Delta$ RT [ms]', fontsize=7, fontweight="bold")
        plt.ylabel(r'$\Delta$ Amplitude [pA]', fontsize=7, fontweight="bold")

        plt.text(
            0.97, 0.05,
            f"Pearson's $r$ = {r_val:.2f}\n$p$ = {p_pearson:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.9
            )
        )

        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(axis="both", labelsize=6)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_ROOT, f"RT-Amp.svg"),
                 transparent=True, dpi=600, format="svg", bbox_inches="tight")
        plt.close()
    
    if not args.prepare and not args.plot:
        parser.print_help()