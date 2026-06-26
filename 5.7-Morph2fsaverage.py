import os
import glob
import mne

from utils.configs import REJECT_SID


if __name__ == "__main__":

    # We set spacing as ico4 and ico5.
    # Cause Nilearn only provides ico# meshes for fsaverage.
    # See "nilearn.datasets.load_fsaverage" for more details.
    # https://nilearn.github.io/dev/modules/generated/nilearn.datasets.load_fsaverage.html#nilearn.datasets.load_fsaverage
    for ico in [4, 5]:
        spacing = f"ico{ico}"

        derivatives_root = "./data/derivatives/meg_derivatives/preprocessing"
        subjects_dir = "./freesurfer"
        subject_to = "fsaverage"

        src_to = mne.read_source_spaces(
            os.path.join(subjects_dir,
                         f"{subject_to}/bem/{subject_to}-{spacing}-src.fif")
        )

        for subject in os.listdir(subjects_dir):
            if not subject.startswith("sub-") or subject in REJECT_SID:
                continue

            fwd_root = os.path.join(derivatives_root, f"{subject}/ses-01/meg/fwd")
            fwd_fname = glob.glob(os.path.join(
                derivatives_root,
                f"{subject}/ses-01/meg/fwd/{subject}_ses-01_task-ConjSemProj_run-*-fwd.fif"
            ))[0]
            fwd = mne.read_forward_solution(fwd_fname)
            
            src = fwd["src"]

            morph = mne.compute_source_morph(
                src=src,
                src_to=src_to,
                subject_from=subject,
                subject_to=subject_to,
                subjects_dir=subjects_dir,
                spacing=ico,
                smooth=10
            )

            morph_fname = os.path.join(subjects_dir, subject, "bem",
                                       f"{subject}2{subject_to}-{spacing}")
            morph.save(morph_fname, overwrite=True)