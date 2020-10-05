import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()


class LA5_Siblings_MRI(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path, load_online=False, mri_type="sMRI",
                 mri_file_suffix="", brain_mask_suffix=None, coord_min=(20, 20, 20,),
                 img_shape=(152, 188, 152,), fixed_start_pos=None, seq_len=None, problems=None, transform=None,
                 temp_storage_path=None):

        self.load_online = load_online
        self.mri_type = mri_type
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.fixed_start_pos = fixed_start_pos
        self.seq_len = seq_len
        self.problems = problems
        self.transform = transform
        self.temp_storage_path = temp_storage_path
        
        if self.problems != None:
            assert len(self.problems) == 1 # more is not supported now

        self.mri_paths = {
            "participant_id": [],
            "path": [],
        }
        self.brain_mask_paths = {
            "participant_id": [],
            "mask_path": [],
        }

        for path_to_folder in paths:
            for patient_folder_name in os.listdir(path_to_folder):
                if 'sub-' in patient_folder_name:
                    path_to_patient_folder = path_to_folder + patient_folder_name
                    mri_type_folder_name = "anat" if self.mri_type == "sMRI" else "func"
                    if os.path.isdir(path_to_patient_folder) and mri_type_folder_name in os.listdir(
                            path_to_patient_folder):
                        temp_path = path_to_patient_folder + "/" + mri_type_folder_name + "/"
                        for filename in os.listdir(temp_path):
                            full_path = temp_path + filename
                            if mri_file_suffix in filename:
                                self.mri_paths["participant_id"].append(patient_folder_name)
                                self.mri_paths["path"].append(full_path)
                            if brain_mask_suffix is not None and brain_mask_suffix in filename:
                                self.brain_mask_paths["participant_id"].append(patient_folder_name)
                                self.brain_mask_paths["mask_path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
        if brain_mask_suffix is not None:
            self.brain_mask_paths = pd.DataFrame(self.brain_mask_paths)
            self.mri_paths = self.mri_paths.merge(self.brain_mask_paths, on="participant_id")

        if target_path is None:
            if brain_mask_suffix is not None:
                self.brain_mask_paths = self.mri_paths["mask_path"].tolist()
            self.mri_paths = self.mri_paths["path"].tolist()
        else:
            target_df = pd.read_csv(target_path)
            target_df = target_df.merge(self.mri_paths, on="participant_id")
            assert self.problems != None
            #             target_df.dropna(subset=problems, how='any', inplace=True)
            target_df.dropna(subset=problems, how='all', inplace=True)
            target_df.fillna(value=-100,
                             inplace=True)  # -100 default value for ignore_index in cross-entropy loss in PyTorch
            target_df.reset_index(drop=True, inplace=True)
            self.labels = target_df[problems].astype('long').values
            if self.labels.shape[1] == 1:
                self.labels = self.labels.squeeze()
            self.mri_paths = target_df["path"].tolist()
            self.pids = target_df["participant_id"].values
            assert len(set(self.pids)) == len(self.pids)
            if brain_mask_suffix is not None:
                self.brain_mask_paths = target_df["mask_path"].tolist()
            del target_df

        if not self.load_online:
            self.mri_files = [self.get_image(i) for i in tqdm(range(len(self.mri_paths)))]

    def reshape_image(self, img, coord_min, img_shape):
        img = img[
              coord_min[0]:coord_min[0] + img_shape[0],
              coord_min[1]:coord_min[1] + img_shape[1],
              coord_min[2]:coord_min[2] + img_shape[2],
              ]
        if img.shape[:3] != img_shape:
            print("Current image shape: {}".format(img.shape[:3]))
            print("Desired image shape: {}".format(img_shape))
            raise AssertionError
        if self.mri_type == "sMRI":
            img = img.reshape((1,) + img_shape)
        elif self.mri_type == "fMRI":
            seq_len = img.shape[-1]
            img = img.reshape((1,) + img_shape + (seq_len,))
        return img

    def get_image(self, index):
        def load_mri(mri_path):
            if "nii" in mri_path:
                if self.temp_storage_path is not None:
                    if not os.path.exists(self.temp_storage_path):
                        os.makedirs(self.temp_storage_path)
                    temp_file_path = self.temp_storage_path + mri_path.split('/')[-1].split('.')[0] + '.npy'
                    try:
                        img = np.load(temp_file_path)  # 1s
                    except FileNotFoundError:
                        img = load_nii_to_array(mri_path)  # 2.5s
                        np.save(temp_file_path, img)
                else:
                    img = load_nii_to_array(mri_path)  # 2.5s
            else:
                img = np.load(mri_path)  # 1s
            return img

        img = load_mri(self.mri_paths[index])

        try:
            brain_mask = load_mri(self.brain_mask_paths[index])
            if self.mri_type == "fMRI":
                brain_mask = brain_mask[..., np.newaxis]
            img *= brain_mask
            del brain_mask
        except KeyError:
            pass

        img = self.reshape_image(img, self.coord_min, self.img_shape)

        if self.mri_type == "fMRI":
            assert self.seq_len != None and self.seq_len > 0
            start_pos = np.random.choice(
                img.shape[-1] - self.seq_len) if self.fixed_start_pos is None else self.fixed_start_pos
            img = img[:, :, :, :, start_pos:start_pos + self.seq_len]
            assert img.shape[-1] == self.seq_len

        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)

        if self.mri_type == "fMRI":
            item = np.moveaxis(item, -1, 0)
        if self.transform is not None:
            item = self.transform(item)
        if self.problems is not None:
            return (item, self.labels[index])
        return (item, None)

    def __len__(self):
        return len(self.mri_paths)
    
def targets_complete(sample, 
                     prefix=False, 
                     mask_path=False,
                     image_path='/gpfs/gpfs0/sbi/data/fcd_classification_bank',
                     targets_path='../targets/targets_fcd_bank.csv', 
                     ignore_missing=True, data_type = False):
    """
    Custom function to complete dataset composition in the local environement.
    Walks through directories and completes fils list, according to targets.
    
    """
    targets = pd.read_csv(targets_path)
    files = pd.DataFrame(columns = ['patient','scan','fcd','img_file','img_seg'])
    clause = (targets['sample'] == sample)
        
    if prefix:
        clause = (targets['sample'] == sample)&(targets['patient'].str.startswith(prefix))
     
    files['patient']= targets['patient'][clause].copy()
    files['fcd'] = targets['fcd'][clause].copy()
    files['scan'] = targets['scan'][clause].copy()
    files['detection'] = targets['detection'][clause].copy()
    files['comments'] = targets['comments'][clause].copy()
    
    if mask_path:
        files['img_mask'] = ''
        
    elif sample == 'all':
        files['patient']= targets['patient'].copy()
        files['fcd'] = targets['fcd'].copy()
        files['scan'] = targets['scan'].copy()   
        files['detection'] = targets['detection'].copy()
        files['comments'] = targets['comments'].copy()
                
    for i in tqdm(range(len(files))):
        for file_in_folder in glob.glob(os.path.join(image_path,'*norm*')):
                if sample == 'pirogov':
                    if ((files['patient'].iloc[i] +'_norm.nii.gz') == file_in_folder.split('/')[-1]):
                        files['img_file'].iloc[i] = file_in_folder
                else:
                    if (files['patient'].iloc[i] in file_in_folder):
                        files['img_file'].iloc[i] = file_in_folder
        
        for file_in_folder in glob.glob(os.path.join(image_path,'*aseg*')):
                if sample == 'pirogov':
#                     print((files['patient'].iloc[i] +'_aparc+aseg.nii.gz'), file_in_folder.split('/')[-1])
                    if ((files['patient'].iloc[i] +'_aparc+aseg.nii.gz') == file_in_folder.split('/')[-1]) or\
    ((files['patient'].iloc[i] +'_aparc+aseg.nii') == file_in_folder.split('/')[-1]):
                        files['img_seg'].iloc[i] = file_in_folder 
                else:    
                    if (files['patient'].iloc[i] in file_in_folder):
                        files['img_seg'].iloc[i] = file_in_folder       
        if mask_path:
            for file_in_folder in glob.glob(os.path.join(mask_path,'*.nii*')):
                if ((files['patient'].iloc[i] +'.nii.gz') == file_in_folder.split('/')[-1]):
                    files['img_mask'].iloc[i] = file_in_folder
        
    # treating missing objects
    if ignore_missing:
        # if only 'img' is needed for classification
        if data_type =='img':
            files.dropna(subset = ['img_file'], inplace= True)
         # if only 'seg' is needed for classification
        elif data_type =='seg':
            files.dropna(subset = ['img_seg'], inplace= True)
        # saving only full pairs of data    
        else: 
            files.dropna(subset = ['img_seg','img_file'], inplace= True)
        
    # reindexing an array
    files = files.reset_index(drop=True)
    le = LabelEncoder() 
    files['scan'] = le.fit_transform(files['scan'])
    
    return files, le

    
class MriClassification(data.Dataset):
    """
    Arguments:
        image_path (str): paths to data folders 
        prefix (str): patient name prefix (optional)
        sample (str): subset of the data, 'all' for whole sample
        targets_path (str): targets file path
        if ignore_missing (bool): delete subject if the data partially missing
        data_type (str): ['img', 'seg'] 
                         'img' - for T1 normalised image
                         'seg' - for image Freesurfer aseg+aparc.nii.gz 
    """
    def __init__(self, sample, prefix=False, mask_path=False,
                 image_path='/gpfs/gpfs0/sbi/data/fcd_classification_bank',
                 targets_path='../targets/targets_fcd_bank.csv', ignore_missing=True,
                 coord_min=(30,30,30,), img_shape=(192, 192, 192,),
                 data_type ='seg'):
        
        super(MriClassification, self).__init__()
        print('Assembling data for: ', sample, ' sample.')

        files,le = targets_complete(sample, prefix, mask_path, image_path,
                                 targets_path, ignore_missing, data_type)
        
        self.img_files = files['img_file']
        self.img_seg = files['img_seg']
        self.scan = files['scan']
        self.scan_keys = le.classes_
        self.target = files['fcd'] 
        self.detection = files['detection']
        self.misc = files['comments']
           
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.data_type = data_type
        
        assert data_type in ['seg','img'], "Invalid file format!"
            
    def __getitem__(self, index):
            img_path = self.img_files[index]
            img_array = load_nii_to_array(img_path)                       
            img = reshape_image(img_array, self.coord_min, self.img_shape)
            
            if self.data_type == 'img':
                return torch.from_numpy(img).float(), self.target[index], self.scan[index]
            
            elif self.data_type == 'seg':
                # not binarising cortical structures
                seg_path = self.img_seg[index]
                seg_array = load_nii_to_array(seg_path)
                seg = reshape_image(seg_array, self.coord_min, self.img_shape)
                return torch.from_numpy(seg).float(), self.target[index], self.scan[index]

    def __len__(self):
        return len(self.img_files)