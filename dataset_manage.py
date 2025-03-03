import os
import numpy as np
import shutil
import tqdm
import pandas as pd
import nibabel
import requests
import random
import SimpleITK as sitk
import pydicom
from torch.utils.data import Dataset
from scipy import ndimage
import matplotlib.pyplot as plt

eps = 1e-5 ### use for finding zero area



'''
Structured data: 
[Gender, Years of Education, Marriage, APOE4, MMSE, ADNI_EF, ADNI_MEM, Age]
'''

key_name = ['Gender', 'Years of Education', 'Marriage', 'APOE4', 'MMSE', 'Age']

def to_GPT_Q(x:list) -> dict :
    x = x[:5] + x[-1:]
    ret = dict()
    for i in range(6) :
        ret[key_name[i]] = x[i]
    return ret




'''
    root_dir like: 
    root
    |-Y0
    ||-xxx_s_xxxx
    ||....
    |-Y2
    ||....
    |-ADNI_TABLE.csv
'''

def Img_valid_patient(root_dir, time, modality:str) :
    if type(time) == int :
        time = [time]
    assert type(time) == list
    dirs = []
    for t in time :
        dirs += [os.path.basename(i) for i in os.listdir(os.path.join(root_dir, str(t), modality + '.nii'))]
    return dirs

def Label_valid_patient(root_dir, time, dropnan=False) :
    tab = pd.read_csv(os.path.join(root_dir, 'ADNI_TABLE.csv'))
    tab = tab[tab['VISM_IN'] == time]
    if dropnan :
        tab = tab.dropna(axis=0, how='any')
    ret = []
    for index, row in tab.iterrows() :
        if not pd.isna(row['DX']) :
            ret.append(row['PTID'])
    return ret




def verify_MRI(time : str) :
    tab = pd.read_csv('./processed/' + time + '/' + time + '_MRI.csv')
    dirs = MRI_valid_patient('./processed/', time)
    print('Missing Patients: ')
    miss_list = []
    for index, row in tab.iterrows() :
        if row['Subject'] not in dirs :
            miss_list.append(row['Subject'])
    print(f"total: {len(miss_list)}")
    print(miss_list)

def vis_nii(path:str, pic_name:str='./vis.png') :
    tmp = nibabel.load(path)
    tmp = tmp.get_fdata().transpose(2, 1, 0)
    plt.matshow(tmp[tmp.shape[0] // 2])
    plt.show()
    plt.savefig(pic_name)

def visualize(ds, fig_save_path, idx, z:int=32, window:bool=False) :
    if type(idx) == str :
        for i in range(len(ds.PTID_list)) :
            if ds.PTID_list[i] == idx :
                idx = i
                break
        assert type(idx) == int
    print(ds.PTID_list[idx])
    plt.matshow(ds[idx][0][0, z, :, :])
    print(f"Structured Data: {ds[idx][1]}   Diagnosis: {ds[idx][2]}")
    if window :
        plt.show()
    if fig_save_path is not None :
        plt.savefig(os.path.join(fig_save_path, ds.PTID_list[idx] + '.png'))
    if not window :
        plt.close()

def vis_dataset(ds, fig_save_path, z:int=32) :
    if not os.path.exists(fig_save_path) :
        os.makedirs(fig_save_path)
    for i in range(len(ds)) :
        visualize(ds, fig_save_path, i, z)

def dataset_GPT_Q(ds, z:int=32) :
    for i in range(len(ds)) :
        print('Q:' + str(to_GPT_Q(ds[i][1])))

def dataset_table(ds, z:int=32) :
    for i in range(len(ds)) :
        print('Q:' + str(ds[i][1]))

def dataset_label(ds, z:int=32) :
    for i in range(len(ds)) :
        print('A:' + str(ds[i][2]))

def login_ida_loni(email, password) :
    data = {'userEmail': email, 'userPassword': password, 'project':'ADNI'}
    ret = requests.post('https://ida.loni.usc.edu/login.jsp', data=data)
    web = open('ret.html', 'w')
    web.write(ret.text)

def download(url:str, local_path:str='.') :
    command = ''
    if not os.path.exists(local_path) :
        print('Path doesn\'t exist, creating folders...')
        print(os.path.abspath(local_path))
        os.makedirs(local_path)
        print('Done!')
    local_path = os.path.abspath(local_path)
    command = f'wget -P {local_path} {url}'
    os.system(command)

def download_adni2(raw:dict) :
    for mod, url in raw.items() :
        download(url, os.path.join('.', 'raw', mod[1]))

def unzip(fpath:str, path:str) :
    os.system(f'unzip -o -d {path} -x {fpath} > /dev/null')

def dcm2nii(path_read, path_save):
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    names_str = series_file_names[0]#' '.join(series_file_names)
    os.system(f'mri_convert -i {names_str} -o {path_save} > /dev/null')


'''
go_down_k_times: head into the first subfolder for 3 times and return the path
------------------------------------------------------------
k: an integer
'''
def go_down_k_times(path, k:int) :
    for i in range(k) :
        path = os.path.join(path, os.listdir(path)[0])
    return path

'''
convert_to_nii_and_move: 
Convert all the DICOM series into .nii files and move them to ./process
------------------------------------------------------------
path: the root directory path of ADNI, for finding ./process correctly
m: which month of visiting is the data from
modality: the directory path of specific modality (PET/T1/T2...)
'''
def convert_to_nii_and_move(path:str, m:str, modality:str) :
    modality_dir = os.path.join(path, 'raw', m, modality)
    if not os.path.exists(modality_dir) :
        print(f'Warning: The folder of {modality} is not found when converting month {m} data!')
        return 
    subj = os.listdir(modality_dir)
    for sname in tqdm.tqdm(subj) :
        now_dir = os.path.join(modality_dir, sname)
        now_dir = go_down_k_times(now_dir, 3)
        des_dir = os.path.join(path, 'processed', m, sname)
        if not os.path.exists(des_dir) :
            os.makedirs(des_dir)
        ls_now = os.listdir(now_dir)
        if len(ls_now) == 1 and ls_now[0][-4:] == '.nii' :
            shutil.copy(os.path.join(now_dir, ls_now[0]), os.path.join(des_dir, modality + '.nii'))
        else :
            dcm2nii(now_dir, os.path.join(des_dir, modality + '.nii'))

'''
unpack_data: automatically unpack downloaded ADNI data at ./raw to ./processed folder.
path: the root dir of ADNI dataset.
modality_list: the list of all the modalities needed to be unpack.
month_list: the list of all the months needed to be unpack.
(since dataset manager should be put in the root directory, 
the default value is '.')

The structure of ./processed is: 
processed
|--0(months)
| |--aaa_S_xxxx
| | |-T1.nii
| | |-T2.nii
| | |-PET.nii
| |--aaa_S_yyyy
| | |-T1.nii
| | |-T2.nii
| | |-PET.nii
| | ...
|--3
|...
|--24

You should prepare ./raw folder like: 
raw
|--0(months)
| |--PET.zip
| |--T1.zip
| |--T2.zip
|--3
| |--PET.zip
| |--T1.zip
| |--T2.zip
|...
|--24
  |--PET.zip
  |--T1.zip
  |--T2.zip

'''
def unpack_data(modality_list:list=['PET', 'T1', 'T2'], month_list:list=None, path:str='.') :
    raw_dir = os.path.join(path, 'raw')
    month_dirs = os.listdir(raw_dir) if month_list is None else [str(m) for m in month_list]
    print(f'Unzipping data ...')
    for m in month_dirs :
        print(f'Unzipping month {m} data.')

        now_dir = os.path.join(raw_dir, m)
        
        for mod in modality_list :
            mod_path = os.path.join(now_dir, f'{mod}.zip')
            if not os.path.exists(os.path.join(now_dir, mod)) :
                if os.path.exists(mod_path) :  
                    print(f'Unzipping {mod} ...') 
                    unzip(mod_path, now_dir)
                    os.rename(os.path.join(now_dir, 'ADNI'), os.path.join(now_dir, mod))
                else :
                    print(f'Warning: {mod} data not found, thus not unzipped!')
            else :
                print(f'{mod} data have already been unzipped.')
    
    print(f'Converting & Moving data ...')
    
    for m in month_dirs :
        print(f'Converting and moving month {m} data.')
        for mod in modality_list :
            convert_to_nii_and_move(path, m, mod)

'''
Image_Filter: return a list of image paths, filter images by 
specific visiting months and modalities.
**Notes that the returning value can be either
    - A list of strings denoting the paths of images, or
    - A list of tuples of strings denoting the paths of image pairs.
------------------------------------------------------------
month_list: the list of months
modality_list: the list of modalities

    - Tips: If the elements in modality_list are TUPLES,
            the returning list will comprise TUPLES of strings 
            which are paired image paths. This can be used to
            filter images for Img_Reg function.

            HOWEVER, IF YOU DO NOT WANT TO PAIR MODALITIES, 
            SIMPLY USE LIST FOR modality_list AND DON'T USE TUPLE!

path: path of the root directory of ADNI dataset
return_PTID: If is True, return ID of patients whose data contain certain modalities
    instead of returning the paths or pairs of paths of images.
'''

def Image_Filter(month_list:list, modality_list, return_PTID=False, path:str='.') :
    ret_list = []
    path = os.path.abspath(path)
    if os.path.basename(path) != 'processed' :
        rt = os.path.join(path, 'processed')
    else :
        rt = path
    if type(modality_list) == str :
        modality_list = [modality_list]
    for m in month_list :
        m_rt = os.path.join(rt, str(m))
        if not os.path.exists(m_rt) :
            print(f'Warning: data of month {m} not found')
            continue
        p_list = os.listdir(m_rt)
        for p in p_list :
            p_rt = os.path.join(m_rt, p)
            if type(modality_list) != tuple :
                for mod in modality_list :
                    i_path = os.path.join(p_rt, str(mod) + '.nii')
                    if os.path.exists(i_path) :
                        ret_list.append((p, int(m)) if return_PTID else i_path)
            else :
                pair = []
                for submod in modality_list :
                    i_path = os.path.join(p_rt, str(submod) + '.nii')
                    if os.path.exists(i_path) :
                        pair.append(i_path)
                    else : 
                        #if len(pair) > 0 and 'PET' in pair[0] :
                        #    print('#')
                        pair = []
                        break
                if len(pair) > 0 :
                    ret_list.append((p, int(m)) if return_PTID else pair)
    ret_list.sort()
    return ret_list

'''
Skull_Strip: Skull-strip all images in the img_list.
Based on SynthStrip, FreeSurfer. Please cite:
-------------------------------------------------------
SynthStrip: Skull-Stripping for Any Brain Image
A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
NeuroImage 206 (2022), 119474
https://doi.org/10.1016/j.neuroimage.2022.119474 

'''

def Skull_Strip(img_list : list, verbose=False) :
    for img in tqdm.tqdm(img_list) :
        dst = img[:-4] + '_SS' + img[-4:]
        if os.path.exists(dst) :   ### If already exists, skip this image
            if verbose :
                print(f'Image {img} has already been skull stripped')
            continue
        command = f'mri_synthstrip -i {img} -o {dst} > /dev/null'
        os.system(command)

'''
Img_Reg: Register A to B for each tuple (A, B) in img_list.
Each element in this img_list should be a tuple of two strings.
The first element is the path of A and the second
element is the path of B.
-------------------------------------------------------
This is based on mri_robust_register in FreeSurfer.
Please cite:
-------------------------------------------------------
Highly Accurate Inverse Consistent Registration: A Robust Approach, M.
Reuter, H.D. Rosas, B. Fischl.  NeuroImage 53(4):1181-1196, 2010.
http://dx.doi.org/10.1016/j.neuroimage.2010.07.020
http://reuter.mit.edu/papers/reuter-robreg10.pdf

'''

def Img_Reg(img_list : list, verbose=False) :
    for imgA, imgB in tqdm.tqdm(img_list) :
        res = imgA[:-4] + '_coregto_' + os.path.basename(imgB[:-4]) + '.nii'
        if os.path.exists(res) :   ### If already exists, skip this image
            if verbose :
                print(f'Image {imgA} has already been registered to {imgB}')
            continue
        command = f'mri_robust_register --mov {imgA} --dst {imgB} --lta tmp.lta --mapmov {res} --iscale --satit > /dev/null'
        os.system(command)


# def Img_Make_Avg(img_list : list, out_file_name : str, verbose=False) :
#     # for imgA, imgB in tqdm.tqdm(img_list) :
#     #     res = imgA[:-4] + '_coregto_' + os.path.basename(imgB[:-4]) + '.nii'
#     #     if os.path.exists(res) :   ### If already exists, skip this image
#     #         if verbose :
#     #             print(f'Image {imgA} has already been registered to {imgB}')
#     #         continue
#     #     command = f'mri_robust_register --mov {imgA} --dst {imgB} --lta tmp.lta --mapmov {res} --iscale --satit > /dev/null'
#     #     os.system(command)
#     vols = ""
#     for name in img_list :
#         vols += " " + name
#     command = f'mri_robust_template --mov{vols} --template {out_file_name} --satit'
#     # print(command)
#     # os.system(command)




def Img_Reg_Template(img_list : list, modality : str, template_path : str = './tmp/template', verbose=False) :
    imgB = os.path.join(template_path, modality + '.nii')
    for imgA in tqdm.tqdm(img_list) :
        res = imgA[:-4] + '_Treg' + modality + '.nii'
        if os.path.exists(res) :   ### If already exists, skip this image
            if verbose :
                print(f'Image {imgA} has already been registered to {imgB}')
            continue
        command = f'mri_robust_register --mov {imgA} --dst {imgB} --lta tmp.lta --mapmov {res} --iscale --satit > /dev/null'
        os.system(command)

def Img_Delete(img_list) :
    for path in img_list :
        shutil.os.remove(path)

'''
filter_patients: Return tuples as (PTID, time (in month)).
Each denotes that the patient of ID 'PTID' has a piece of data in month 'time'
with all the modalities in 'modality' list
-------------------------------------------------------
time: a list of int, each denotes a possible month of visit
modality: a list of str, each denotes a possible name of modality
    - e.g. 'structured', 'PET_TregPET_SS', 'T2_TregT2'
    - *Notes that if you are to fetch processed modality,
       don't forget the following process suffix '_proc1_proc2...'
'''

def filter_patients(root_dir, time:list, modality:list) :
    ## modality: structured
    tab = pd.read_csv(os.path.join(root_dir, 'ADNI_TABLE.csv'))
    tab['PTID_VISM_IN'] = list(zip(tab['PTID'], tab['VISM_IN']))
    # filtered = tab[tab['PTID_VISM_IN'].isin(PTID_m_list)]
    # remove temporal column PTID_VISM_IN
    # filtered = filtered.drop('PTID_VISM_IN', axis=1)
    assert tab.index.is_unique
    table = tab[['PTID_VISM_IN', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'APOE4', 'MMSE', 'ADNI_EF', 'ADNI_MEM', 'VISM_IN', 'AGE']]
    label = tab[['PTID_VISM_IN', 'DX']] ## with PTID, VISM_IN

    ## filter valid patients
    PTID_m_list = None
    PTID_m_list = set(table[table['VISM_IN'].isin(time) & (~tab['DX'].isna())]['PTID_VISM_IN'])
    PTID_m_list &= set(Image_Filter(time, modality, True, root_dir))
    PTID_m_list = list(PTID_m_list)
    PTID_m_list.sort()
    return PTID_m_list

'''
ADNI2: The multimodal dataset, ADNI2.
-------------------------------------------------------
root_dir: the root directory of ADNI2, it should contain folder 'processed' and 'raw'
input_D/H/W: the shape of input tensor
phase: 'train'/'test'
time: the list of all possible visiting months
modality: the list of all modalities desired
    -* Each call of __getitem__ will receive a return tuple like (mod1, mod2, ..., modk, label)
        where mod1 ... modk are multimodal inputs ordered exactly same as the list 'modality'
        (e.g. you will receive mod1 as PET image and mod2 as structured data given modality=['PET_Treg_SS', 'structured'])
        and label is the classification label
PTID_m_list: a list of tuples, each of which are (PTID, time (in month)). 
    - For convenient training/test set split and direct datapoints appointment

-------------------------------------------------------
A brief tutorial for building training/test sets:

## filtering patients
PTID_List = filter_patients(root_dir, [0, 24], ['PET_TregPET_SS', 'structured'])
train_list = PTID_List[:int(len(PTID_List) * 0.85)]
test_list = PTID_List[len(train_list):]

## building datasets
trainset = ADNI2(root_dir=root_dir, input_D=args.input_D, input_H=args.input_H, input_W=args.input_W, phase='train',
                    time=[0, 24], modality=['PET_TregPET_SS', 'structured'], PTID_m_list=train_list)
testset = ADNI2(root_dir=root_dir, input_D=args.input_D, input_H=args.input_H, input_W=args.input_W, phase='test',
                    time=[0, 24], modality=['PET_TregPET_SS', 'structured'], PTID_m_list=test_list)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=0)


'''
class ADNI2(Dataset): ### Modified from Tecent MedicalNet Pipeline
    def __init__(self, root_dir, input_D, input_H, input_W, phase, time:list, modality:list, PTID_m_list:list=None, image_transform=None, file=None):
        super().__init__()
        self.root_dir = root_dir
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase
        self.root_dir = root_dir
        self.modality = modality
        self.image_transform = image_transform
        
        ## modality: structured
        tab = pd.read_csv('/data/Data2/gqyang//dataset/ADNI2/ADNI_TABLE.csv')
        tab['PTID_VISM_IN'] = list(zip(tab['PTID'], tab['VISM_IN']))
        if PTID_m_list != None :
            tab = tab[tab['PTID_VISM_IN'].isin(PTID_m_list)]
        # remove temporal column PTID_VISM_IN
        # filtered = filtered.drop('PTID_VISM_IN', axis=1)
        tab = tab.set_index('PTID_VISM_IN')
        assert tab.index.is_unique
        self.table = tab[['PTGENDER', 'PTEDUCAT', 'PTMARRY', 'APOE4', 'MMSE', 'ADNI_EF', 'ADNI_MEM', 'VISM_IN', 'AGE']]
        self.label = tab[['DX']] ## with PTID, VISM_IN

        ## filter valid patients
        if PTID_m_list == None :
            self.PTID_m_list = None
            self.PTID_m_list = set(self.table[self.table['VISM_IN'].isin(time) & (~self.table['DX'].isna())].index)
            self.PTID_m_list &= set(Image_Filter(time, modality, True, root_dir))
            self.PTID_m_list = list(self.PTID_m_list)
        else :
            self.PTID_m_list = PTID_m_list

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.PTID_m_list)

    def __getitem__(self, idx):
        
        # read image and labels
        ith_info = self.PTID_m_list[idx]
        ret = []
        for mod in self.modality :
            if mod != 'structured' :
                Image_name = os.path.join(self.root_dir, str(ith_info[1]), ith_info[0], mod + '.nii')
                #assert os.path.isfile(Image_name)
                if not os.path.isfile(Image_name):
                    print('Image not found: ', Image_name)
                assert os.path.isfile(Image_name)
                Image = nibabel.load(Image_name)  # We have transposed the data from WHD format to DHW
                Image = Image.get_fdata()
                Image = np.reshape(Image, [1, 128, 128, 128])
                Image = Image.astype("float32")
                # Add random Gaussian noise to the image
                noise_mean = 0.0  # Mean of the Gaussian noise
                noise_std = 0.1   # Standard deviation of the Gaussian noise
                gaussian_noise = np.random.normal(noise_mean, noise_std, Image.shape)  # Generate random Gaussian noise
                Image += gaussian_noise  # Add noise to the image
                if self.image_transform != None :
                    Image = self.image_transform(Image)
                ret.append(Image)

            else :
                tab = dict(self.table.loc[[ith_info], :].iloc[0])
                table = f"This patient is {tab['PTGENDER']} gender, " \
                    f"{tab['PTEDUCAT']} years of education, " \
                    f"{tab['AGE']} years old, " \
                    f"{tab['PTMARRY']} and has an APOE4 score of " \
                    f"{tab['APOE4']}, " \
                    f"an MMSE score of {tab['MMSE']}, " \
                    f"an ADNI-EF score of {tab['ADNI_EF']}," \
                    f" and an ADNI-MEM score of {tab['ADNI_MEM']}." \
                    f" After {tab['VISM_IN']} months of observation, "
                ret.append(table)
        lb = (self.label.loc[[ith_info], ['DX']]).iloc[0].item()
        if lb == 'MCI':
            lb = 1
        elif lb == 'Dementia':
            lb = 2
        elif lb == 'CN':
            lb = 0
        ret.append(lb)

        return tuple(ret)

            

    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area (i.e. zero area)
        """
        #org_z = volume.shape[0] // 2
        zero_value = volume.min()
        non_zeros_idx = np.where((volume - zero_value) > eps)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        ret = volume[min_z:max_z, min_h:max_h, min_w:max_w]
        #plt.matshow(ret[(org_z-min_z)])
        #plt.show()

        return ret


    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[(volume - volume.min()) > eps]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape) ## strange for padding zero voxels with Gaussian
        out[(volume - volume.min()) <= eps] = out_random[(volume - volume.min()) <= eps]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.zoom(data, scale, order=0)

        return data

    def __training_img_process__(self, data): 

        # crop data according net input size
        data = data.get_fdata().transpose(2, 1, 0)

        
        # drop out the invalid range
        data = self.__drop_invalid_range__(data)

        # resize data
        data = self.__resize_data__(data)
        # label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


    def __testing_img_process__(self, data): 

        # crop data according net input size
        data = data.get_fdata().transpose(2, 1, 0)

        # drop out the invalid range
        data = self.__drop_invalid_range__(data)

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


class ADNI2_pre(Dataset): ### Modified from Tecent MedicalNet Pipeline
    def __init__(self, root_dir, input_D, input_H, input_W, phase, time:list, modality:list, PTID_m_list:list=None, image_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase
        self.modality = modality
        self.image_transform = image_transform
        self.PTID_m_list = PTID_m_list

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.PTID_m_list)

    def __getitem__(self, idx):
        
        # read image and labels
        ith_info = self.PTID_m_list[idx]
        #print(ith_info)
        ret = []
        for mod in self.modality :
            if mod != 'structured' :
                Image_name = os.path.join(self.root_dir, ith_info[0], 'processed',str(ith_info[1]),ith_info[2], mod + '.nii')
                if not os.path.isfile(Image_name):
                    print('Image not found: ', Image_name)
                assert os.path.isfile(Image_name)
                Image = nibabel.load(Image_name)  # We have transposed the data from WHD format to DHW
                Image = Image.get_fdata()
                Image = np.reshape(Image, [1, 128, 128, 128])
                Image = Image.astype("float32")
                if self.image_transform != None :
                    Image = self.image_transform(Image)
                ret.append(Image)

            else :
                tab = dict(self.table.loc[[ith_info], :].iloc[0])
                table = f"This patient is {tab['PTGENDER']} gender, " \
                    f"{tab['PTEDUCAT']} years of education, " \
                    f"{tab['AGE']} years old, " \
                    f"{tab['PTMARRY']} and has an APOE4 score of " \
                    f"{tab['APOE4']}, " \
                    f"an MMSE score of {tab['MMSE']}, " \
                    f"an ADNI-EF score of {tab['ADNI_EF']}," \
                    f" and an ADNI-MEM score of {tab['ADNI_MEM']}." \
                    f" After {tab['VISM_IN']} months of observation, "
                ret.append(table)

        return tuple(ret)

            

    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area (i.e. zero area)
        """
        #org_z = volume.shape[0] // 2
        zero_value = volume.min()
        non_zeros_idx = np.where((volume - zero_value) > eps)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        ret = volume[min_z:max_z, min_h:max_h, min_w:max_w]
        #plt.matshow(ret[(org_z-min_z)])
        #plt.show()

        return ret


    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[(volume - volume.min()) > eps]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape) ## strange for padding zero voxels with Gaussian
        out[(volume - volume.min()) <= eps] = out_random[(volume - volume.min()) <= eps]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.zoom(data, scale, order=0)

        return data

    def __training_img_process__(self, data): 

        # crop data according net input size
        data = data.get_fdata().transpose(2, 1, 0)

        
        # drop out the invalid range
        data = self.__drop_invalid_range__(data)

        # resize data
        data = self.__resize_data__(data)
        # label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


    def __testing_img_process__(self, data): 

        # crop data according net input size
        data = data.get_fdata().transpose(2, 1, 0)

        # drop out the invalid range
        data = self.__drop_invalid_range__(data)

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

if __name__ == "__main__" :
    # unpack_data(['T1'], [24])
    imgs = Image_Filter([0, 3, 6, 12, 24, 36, 48, 60], "PET-FDG")
    # print(imgs)
    # pts = filter_patients('./processed', [24], ["PET", "PET-FDG", "T1", "T2"])
    # T1s = [x for x, _, _, _ in imgs]
    # Img_Avg(T1s, './test_T1_avg.nii')
    Img_Reg_Template(imgs, "PET-FDG")
    imgs = Image_Filter([0, 3, 6, 12, 24, 36, 48, 60], "PET-FDG_TregPET-FDG")
    Skull_Strip(imgs)