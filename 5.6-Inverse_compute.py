import os
import glob
import mne
from mne.io import read_info, read_raw_fif
from mne.preprocessing import maxwell_filter_prepare_emptyroom, find_bad_channels_maxwell
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne_bids import BIDSPath, read_raw_bids

from utils.configs import REJECT_SID


if __name__ == "__main__":

    ds_root = "./data"
    derivatives_root = "./data/derivatives/meg_derivatives/preprocessing"
    subjects_dir = "./freesurfer"

    for subject in os.listdir(subjects_dir):
        if not subject.startswith("sub-") or subject in REJECT_SID:
            continue

        fwd_root = os.path.join(derivatives_root, f"{subject}/ses-01/meg/fwd")
        cov_root = os.path.join(derivatives_root, f"{subject}/ses-01/meg/noise_cov")
        inv_root = os.path.join(derivatives_root, f"{subject}/ses-01/meg/inv")
        if not os.path.exists(cov_root):
            os.mkdir(cov_root)
        if not os.path.exists(inv_root):
            os.mkdir(inv_root)

        cal_file = os.path.join(ds_root, subject, "ses-01", "meg", f"{subject}_ses-01_acq-calibration_meg.dat")
        ct_file = os.path.join(ds_root, subject, "ses-01", "meg", f"{subject}_ses-01_acq-crosstalk_meg.fif")

        bids_path = BIDSPath(subject=subject.split("-")[1], session="01", run="01", 
                             task="ConjSemProj", datatype="meg", root=ds_root)
        
        raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True))
        
        ## Detect bad channels
        raw.info["bads"] = []
        raw_check = raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check,
            cross_talk=ct_file,
            calibration=cal_file,
            return_scores=True,
            verbose=True,
        )

        bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
        bads = bads + ["MEG0113", "MEG0142", "MEG0111"]
        bads = list(set(bads))
        raw.info["bads"] = bads
        
        ## Find corresponding emptyroom recordings
        # er_bids_path = bids_path.find_empty_room(use_sidecar_only=True) ## It does not work.
        date = subject.split("-")[1][:8]
        er_fname = os.path.join(
            ds_root,
            f"sub-emptyroom/ses-{date}",
            "meg",
            f"sub-emptyroom_ses-{date}_task-noise_meg.fif"
        )
        raw_er = read_raw_fif(er_fname, allow_maxshield=True)
        
        ## Perform MaxFilter on emptyroom recordings
        raw_er_prepared = maxwell_filter_prepare_emptyroom(
            raw_er=raw_er,
            raw=raw,
            bads="from_raw",
            emit_warning=False
        )
        raw_er_prepared = mne.preprocessing.maxwell_filter(
            raw_er_prepared, cross_talk=ct_file, calibration=cal_file,
            st_correlation=0.98, st_duration=10
        )

        reject = dict(grad=4000e-13, # T / m (gradiometers)
                      mag=4e-12, # T (magnetometers)
                     )

        ## Compute noise covariance
        noise_cov = mne.compute_raw_covariance(raw_er_prepared, 
                                               tmin=0, tmax=None,
                                               picks="meg", reject=reject,
                                               method="auto", cv=5)
        
        mne.write_cov(os.path.join(cov_root, "noise-cov.fif"),
                      noise_cov, overwrite=True)
        
        ## Compute inverse operator
        fnames = glob.glob(
            os.path.join(
                derivatives_root,
                f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-*_meg.fif"
            )
        )

        for fname in fnames:
            info = read_info(fname)

            basename = os.path.basename(fname)

            fwd = mne.read_forward_solution(
                os.path.join(
                    fwd_root, 
                    basename.replace("_meg.fif", "-fwd.fif")
                )
            )

            inv = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed="auto")

            write_inverse_operator(os.path.join(inv_root, basename.replace("_meg.fif", "-inv.fif")),
                                   inv, overwrite=True)