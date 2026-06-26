import os
import glob
import mne
from mne.io import read_info

from utils.configs import REJECT_SID


if __name__ == "__main__":
    
    spacing = "oct6"
    bem_ico = 5
    
    root = "./data/derivatives/meg_derivatives/preprocessing"
    subjects_dir = "./freesurfer"
    
    for subject in os.listdir(subjects_dir):
        if not subject.startswith("sub-") or subject in REJECT_SID:
            continue
        
        trans_root = os.path.join(root, f"{subject}/ses-01/meg/trans")
        fwd_root = os.path.join(root, f"{subject}/ses-01/meg/fwd")
        if not os.path.exists(fwd_root):
            os.mkdir(fwd_root)
        
        bem_sol_fname = os.path.join(subjects_dir, subject, "bem",
                                     f"{subject}-{spacing}-ico{bem_ico}-bem-sol.fif")
        src_fname = os.path.join(subjects_dir, subject, "bem",
                                 f"{subject}-{spacing}-ico{bem_ico}-src.fif")
        
        bem = mne.read_bem_solution(bem_sol_fname)
        src = mne.read_source_spaces(src_fname)

        fnames = glob.glob(
            os.path.join(
                root,
                f"{subject}/ses-01/meg/{subject}_ses-01_task-ConjSemProj_run-*_meg.fif"
            )
        )

        for fname in fnames:
            info = read_info(fname)
            
            basename = os.path.basename(fname)

            trans = mne.read_trans(
                os.path.join(
                    trans_root, 
                    basename.replace("_meg.fif", "-trans.fif")
                )
            )

            fwd = mne.make_forward_solution(info, trans, src, bem,
                                            meg=True, eeg=False, mindist=5.0, 
                                            n_jobs=None, verbose=True,)

            mne.write_forward_solution(os.path.join(fwd_root, basename.replace("_meg.fif", "-fwd.fif")), 
                                       fwd, overwrite=True)