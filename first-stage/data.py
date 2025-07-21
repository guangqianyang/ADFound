import os
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import distributed
from torchvision.datasets import CIFAR10, CIFAR100
from dataset_manage import ADNI2, ADNI2_pre
import psutil
def use_cpus(gpus: list):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu*24, (gpu+1)*24)))
    p = psutil.Process()
    p.cpu_affinity(cpus)
    print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))


def build_data_cifar(batch_size=128, cutout=False, workers=4, use_cifar10=False, dpath='/data/krdu/dataset', resize224=False, DDP=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tr_test = []
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    if resize224:
        aug.append(transforms.Resize((224, 224)))
        tr_test.append(transforms.Resize((224, 224)))
    aug.append(transforms.ToTensor())
    tr_test.append(transforms.ToTensor())


    if use_cifar10:
        aug.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        tr_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose(tr_test)
        train_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),)
        tr_test.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose(tr_test)
        train_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                               train=False, download=True, transform=transform_test)

    train_sampler = distributed.DistributedSampler(train_dataset,shuffle=True) if DDP else None
    val_sampler = distributed.DistributedSampler(val_dataset,shuffle=False) if DDP else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True if not DDP else None,
                            num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False if not DDP else None, num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


########### ADNI ############


def data_augmentation(x:np.ndarray) :
    ### x: (C, D, H, W)
    ### random roll
    # roll_max_shift = 10
    # x = np.roll(x, shift=(0, 0, random.randint(-roll_max_shift, roll_max_shift), random.randint(-roll_max_shift, roll_max_shift)))

    ### random rotate-shake
    rotate_max_deg = 10
    x_list = []
    RotMat = cv2.getRotationMatrix2D((x.shape[3] / 2, x.shape[2] / 2), rotate_max_deg * (random.random() * 2 - 1), 1.0)
    for i in range(x.shape[1]) :
        x_list.append(cv2.warpAffine(x[0, i], RotMat, (x.shape[3], x.shape[2])))
    x = np.stack(x_list, axis=0)
    x = np.expand_dims(x, axis=0)

    ### random rotate-nod
    x_list = []
    RotMat = cv2.getRotationMatrix2D((x.shape[2] / 2, x.shape[1] / 2), rotate_max_deg * (random.random() * 2 - 1), 1.0)
    for i in range(x.shape[3]) :
        x_list.append(cv2.warpAffine(x[0, ..., i], RotMat, (x.shape[2], x.shape[1])))
    x = np.stack(x_list, axis=2)
    x = np.expand_dims(x, axis=0)

    # if random.random() > 0.5 : ### upside down
    #     x = np.flip(x, axis=2)
    if random.random() > 0.5 : ### mirror
        x = np.flip(x, axis=3)
    # cv2.imwrite('./tmp.png', (x[0, x.shape[1] // 2] + 0.5) * 255)


    return x.copy()


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
            #print(f'Warning: data of month {m} not found')
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



def build_data_adni2_paired_data(batch_size, input_D, input_H, input_W, workers=4) :
    root_dir = r'/data/Data2/dataset/'
    csv_file = r'/data/zhyang/Alzheimer Prognosis/ADNI2.csv'

    #img_mod = 'PET_TregPET_SS'
    
    img_mod = ['T1_TregT1_SS','PET-FDG_TregPET-FDG_SS']
    time = [0,3,6,12,24,36,48,60]

    division_root = "/data/Data2/gqyang/AD_algorithm/Data_division/subjects/T1_FDG/three-class"

    train_id_list = pd.read_excel(os.path.join(division_root, 'train_data.xlsx'))['PTID_list'].values.tolist()
    val_id_list = pd.read_excel(os.path.join(division_root, 'val_data.xlsx'))['PTID_list'].values.tolist()
    test_id_list = pd.read_excel(os.path.join(division_root, 'test_data.xlsx'))['PTID_list'].values.tolist()

    train_m_list = pd.read_excel(os.path.join(division_root, 'train_data.xlsx'))['month'].values.tolist()
    val_m_list = pd.read_excel(os.path.join(division_root, 'val_data.xlsx'))['month'].values.tolist()
    test_m_list = pd.read_excel(os.path.join(division_root, 'test_data.xlsx'))['month'].values.tolist()

    train_list = list(zip(train_id_list, train_m_list))
    val_list = list(zip(val_id_list, val_m_list))
    test_list = list(zip(test_id_list, test_m_list))

    PTID_List = train_list + val_list + test_list
    print('A total of {} patients, {} for training, {} for validation, {} for testing'.format(len(PTID_List), len(train_list), len(val_list), len(test_list)))


    trainset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='train',
                     time=time, modality=img_mod, PTID_m_list=train_list, image_transform=data_augmentation)
    testset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='test',
                     time=time, modality=img_mod, PTID_m_list=test_list)
    valset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='test',
                     time=time, modality=img_mod, PTID_m_list=val_list)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)
    
    return train_loader, val_loader, test_loader



def build_data_adni2_paired_data_v2(batch_size, input_D, input_H, input_W, workers=4) :
    root_dir = r'/data/Data1/gqyang/dataset/ADFound/finetune/ADNI2/'
    csv_file = r'/data/zhyang/Alzheimer Prognosis/ADNI2.csv'

    #img_mod = 'PET_TregPET_SS'
    
    img_mod = ['T1_TregT1_SS','PET-FDG_TregPET-FDG_SS']
    time = [0,3,6,12,24,36,48,60]

    division_root = "/data/Data2/gqyang/AD_algorithm/Data_division/subjects/T1_FDG/three-class"

    train_file = os.path.join(division_root, 'train_data.xlsx')
    val_file = os.path.join(division_root, 'val_data.xlsx')
    test_file = os.path.join(division_root, 'test_data.xlsx')

    train_id_list = pd.read_excel(os.path.join(division_root, 'train_data.xlsx'))['PTID_list'].values.tolist()
    val_id_list = pd.read_excel(os.path.join(division_root, 'val_data.xlsx'))['PTID_list'].values.tolist()
    test_id_list = pd.read_excel(os.path.join(division_root, 'test_data.xlsx'))['PTID_list'].values.tolist()

    train_m_list = pd.read_excel(os.path.join(division_root, 'train_data.xlsx'))['month'].values.tolist()
    val_m_list = pd.read_excel(os.path.join(division_root, 'val_data.xlsx'))['month'].values.tolist()
    test_m_list = pd.read_excel(os.path.join(division_root, 'test_data.xlsx'))['month'].values.tolist()

    train_list = list(zip(train_id_list, train_m_list))
    val_list = list(zip(val_id_list, val_m_list))
    test_list = list(zip(test_id_list, test_m_list))

    PTID_List = train_list + val_list + test_list
    print('A total of {} patients, {} for training, {} for validation, {} for testing'.format(len(PTID_List), len(train_list), len(val_list), len(test_list)))

    testset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='test',
                     time=time, modality=img_mod, PTID_m_list=test_list,file=test_file)

    trainset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='train',
                     time=time, modality=img_mod, PTID_m_list=train_list, image_transform=data_augmentation,file=train_file)
                     
    valset = ADNI2(root_dir=root_dir, input_D=input_D, input_H=input_H, input_W=input_W, phase='test',
                     time=time, modality=img_mod, PTID_m_list=val_list,file=val_file)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)
    
    return train_loader, val_loader, test_loader


def build_data_adni2_paired_data_pretrain(batch_size, input_D, input_H, input_W, workers=4) :
    
    root= '/data/Data2/gqyang/dataset/ADFound/pretrain'
    img_mod = ['T1_TregT1_SS','PET-FDG_TregPET-FDG_SS']
    time = [0,3,6,12,24,36,48,60]
    division_root = "/data/Data2/gqyang/AD_algorithm/Data_division/Unpaired_data"
    paired_cohort = pd.read_excel(os.path.join(division_root, 'paired_data.xlsx'))['cohort'].values.tolist()
    T1_cohort = pd.read_excel(os.path.join(division_root, 'individual_T1_data.xlsx'))['cohort'].values.tolist()
    PET_cohort = pd.read_excel(os.path.join(division_root, 'individual_PET-FDG_data.xlsx'))['cohort'].values.tolist()

    paired_month = pd.read_excel(os.path.join(division_root, 'paired_data.xlsx'))['month'].values.tolist()
    T1_month = pd.read_excel(os.path.join(division_root, 'individual_T1_data.xlsx'))['month'].values.tolist()
    PET_month = pd.read_excel(os.path.join(division_root, 'individual_PET-FDG_data.xlsx'))['month'].values.tolist()

    paired_id = pd.read_excel(os.path.join(division_root, 'paired_data.xlsx'))['PTID'].values.tolist()
    T1_id = pd.read_excel(os.path.join(division_root, 'individual_T1_data.xlsx'))['PTID'].values.tolist()
    PET_id = pd.read_excel(os.path.join(division_root, 'individual_PET-FDG_data.xlsx'))['PTID'].values.tolist()

    paired_data = list(zip(paired_cohort, paired_month, paired_id))
    T1_data = list(zip(T1_cohort, T1_month, T1_id))
    PET_data = list(zip(PET_cohort, PET_month, PET_id))

    modalities = ['T1_TregT1_SS','PET-FDG_TregPET-FDG_SS']

    paired_conditions = [{modalities[0]:1, modalities[1]:1} for i in range(len(paired_data))]
    T1_conditions = [{modalities[0]:1, modalities[1]:0} for i in range(len(T1_data))]
    PET_conditions = [{modalities[0]:0, modalities[1]:1} for i in range(len(PET_data))]


    all_data  = paired_data + T1_data + PET_data
    all_conditions = paired_conditions + T1_conditions + PET_conditions

    #shuffle data
    order = np.arange(len(all_data))
    np.random.shuffle(order)
    all_data = [all_data[i] for i in order]
    all_conditions = [all_conditions[i] for i in order]



    #all_PET_data = paired_data + PET_data

    #random.shuffle(all_PET_data)
    
    trainset = ADNI2_pre(root_dir=root, input_D=input_D, input_H=input_H, input_W=input_W, phase='train',
                     time=time, modality=img_mod, PTID_m_list=all_data, conditions=all_conditions,image_transform=data_augmentation)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=workers, drop_last=True)

    
    return train_loader
from tqdm import tqdm

if __name__ == '__main__':
    use_cpus(gpus=[int(6)])

    train_loader = build_data_adni2_paired_data_pretrain(48, 128, 128, 128)
    for image_data, condition in tqdm(train_loader):
        a = 0
