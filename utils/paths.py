# LA5_study_general
la5_data = ['../../../../datasets/LA5study/BIDS_LA5study/']
la5_temp_npy_folder_path = '../../../../datasets/LA5study//LA5study_temp_npy/' # for numpy files
la5_target_path = '../LA5study_targets.csv'
#  LA5_study_smri
la5_smri_file_suffix = 'T1w_space-MNI152NLin2009cAsym_preproc.nii'
la5_smri_brain_mask_suffix = 'T1w_space-MNI152NLin2009cAsym_brainmask.nii'
#  LA5_study_fmri (train on rest, test on task, participant ids must not intersect in train and test)
la5_rest_fmri_file_suffix = 'task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii'
la5_task_fmri_file_suffix = 'task-bart_bold_space-MNI152NLin2009cAsym_preproc.nii'
la5_rest_fmri_brain_mask_suffix = 'task-rest_bold_space-MNI152NLin2009cAsym_brainmask.nii'
la5_task_fmri_brain_mask_suffix = 'task-bart_bold_space-MNI152NLin2009cAsym_brainmask.nii'

# siblings general
sibl_data = ['../../fmriprep_ds00115/fmriprep/']
sibl_temp_npy_folder_path = './data/siblings_temp_npy/' # for numpy files
sibl_target_path = './data/ds00115_targets.csv'
# siblings smri
sibl_smri_file_suffix = 'T1w_space-MNI152NLin2009cAsym_preproc.nii'
sibl_smri_brain_mask_suffix = 'T1w_space-MNI152NLin2009cAsym_brainmask.nii'
#  siblings fmri (train on rest, test on task, participant ids must not intersect in train and test)
sibl_fmri_file_suffix = 'task-letter2backtask_bold_space-MNI152NLin2009cAsym_preproc.nii'
sibl_fmri_brain_mask_suffix = 'task-letter2backtask_bold_space-MNI152NLin2009cAsym_brainmask.nii'

# siblings converted to LA5 style using CycleGAN on slices
sibl2la5_data = ['../data/final_siblings/train/', '../data/final_siblings/val/']
sibl2la5_smri_file_suffix = 'scan.npy'
