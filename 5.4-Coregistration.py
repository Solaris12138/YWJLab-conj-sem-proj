import os
import numpy as np
import glob

import mne
from mne.coreg import Coregistration
from mne.io import read_info

from utils.configs import REJECT_SID


if __name__ == "__main__":

    root = "./data/derivatives/meg_derivatives/preprocessing"
    subjects_dir = "./freesurfer"

    for subject in os.listdir(subjects_dir):
        if not subject.startswith("sub-") or subject in REJECT_SID:
            continue
        
        trans_root = os.path.join(root, f"{subject}/ses-01/meg/trans")
        if not os.path.exists(trans_root):
            os.mkdir(trans_root)

        fnames = glob.glob(
            os.path.join(
                root,
                f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-*_meg.fif"
            )
        )
        
        fiducials = "auto"
        
        for fname in fnames:
            info = read_info(fname)

            coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=10, nasion_weight=5., verbose=True)
            coreg.omit_head_shape_points(distance=5. / 1000)
            coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)
            dists = coreg.compute_dig_mri_distances() * 1e3
            print(
                f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
                f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            )
            
            basename = os.path.basename(fname)
            mne.write_trans(os.path.join(trans_root, basename.replace("_meg.fif", "-trans.fif")), 
                            coreg.trans, overwrite=True)