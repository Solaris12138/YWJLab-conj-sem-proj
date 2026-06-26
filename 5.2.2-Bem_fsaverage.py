import os
import mne

# Make sure that you have the permission to run this script.
# Asking the root before running this script is recommended.

if __name__ == "__main__":

    # We initially set spacing as ico4.
    # Cause Nilearn only provides ico# meshes for fsaverage.
    # See "nilearn.datasets.load_fsaverage" for more details.
    # https://nilearn.github.io/dev/modules/generated/nilearn.datasets.load_fsaverage.html#nilearn.datasets.load_fsaverage
    # You could also change this into oct#.
    # We also set spacing as ico5 and oct6 for potential application possibilities.
    for spacing in ["ico3", "ico4", "ico5", "oct6"]:

        subject = "fsaverage"
        subjects_dir = os.environ.get("SUBJECTS_DIR")

        src_fname = os.path.join(subjects_dir, subject, "bem",
                                 f"{subject}-{spacing}-src.fif")

        # Create the surface source space
        src = mne.setup_source_space(subject, spacing, subjects_dir=subjects_dir)
        mne.write_source_spaces(src_fname, src, overwrite=True)