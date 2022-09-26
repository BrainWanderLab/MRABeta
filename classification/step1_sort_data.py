##### convert  the nii data to npy for train data
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt 
from PIL import Image
import utilities as UT
import glob

##############################################################################################
path = "./data/gen/data_217/nifti/ad_117"
out_file = "./data/gen/data_217/npy/ad_117"
extname = '.nii.gz'
##############################################################################################
file = os.listdir(path)
for f in file:
    name = os.path.basename(f).replace(extname,'')
    img = nib.load(os.path.join(path,f))
    imgdata = img.get_fdata()
    data = []
    ## resize the shape
    for slices in range(imgdata.shape[-1]):
        slices = Image.fromarray(imgdata[:,:,slices])
        slices = slices.resize((110,110),Image.ANTIALIAS)
        slices = np.asarray(slices)
        data.append(slices)
    data = np.asarray(data)
    ### augment the data
    trsf_ler = np.fliplr(data)
    trsf_upd = np.flipud(data)
    trsf_rot = np.rot90(data,axes=(2,1))
    trsf_org = data

    if np.isnan(data).any():
        print(f)
    data_dic = {'ler':trsf_ler,
                'upd':trsf_upd,
                'rot':trsf_rot,
                'org':trsf_org}
    for key,value in data_dic.items():
        save_path = os.path.join(out_file,''.join([key,'_',name,'.npy']))
        np.save(save_path,value)
######################################################################################################
# split the file for 5 fold 
file_dir="./data/gen/data_217/npy"
train_path = "./fold/gen/train"
test_path = "./fold/gen/test"


fold_0 = []
fold_1 = []
fold_2 = []
fold_3 = []
fold_4 = []
####
fold_0_train = []
fold_1_train = []
fold_2_train = []
fold_3_train = []
fold_4_train = []

np.random.seed(999)
folders = os.listdir(file_dir)
folders.sort()
for folder in  folders:
    cur_folder = os.path.join(file_dir, folder)
    files = glob.glob(os.path.join(cur_folder,'org*'))
    for file in files:  
        cur_file = os.path.join(cur_folder, file)
        ler_file = cur_file.replace('/org_','/ler_')
        rot_file = cur_file.replace('/org_','/rot_')
        upd_file = cur_file.replace('/org_','/upd_')

        ran = np.random.randint(5)
        
        if(ran==0):
            fold_0.append([cur_file])
            fold_0_train.append([[cur_file],[ler_file],[rot_file],[upd_file]])
        elif(ran==1):
            fold_1.append([cur_file])
            fold_1_train.append([[cur_file],[ler_file],[rot_file],[upd_file]])
        elif(ran==2):
            fold_2.append([cur_file])
            fold_2_train.append([[cur_file],[ler_file],[rot_file],[upd_file]])
        elif(ran==3):
            fold_3.append([cur_file])
            fold_3_train.append([[cur_file],[ler_file],[rot_file],[upd_file]])
        elif(ran==4):
            fold_4.append([cur_file]) 
            fold_4_train.append([[cur_file],[ler_file],[rot_file],[upd_file]]) 

### write the npy path of every fold to csv
UT.write_csv(os.path.join(test_path, 'fold_0.csv'), fold_0)
UT.write_csv(os.path.join(test_path, 'fold_1.csv'), fold_1)
UT.write_csv(os.path.join(test_path, 'fold_2.csv'), fold_2)
UT.write_csv(os.path.join(test_path, 'fold_3.csv'), fold_3)
UT.write_csv(os.path.join(test_path, 'fold_4.csv'), fold_4)
###for train
fold_0 = []
fold_1 = []
fold_2 = []
fold_3 = []
fold_4 = []
[fold_0.extend(i) for i in fold_0_train]
[fold_1.extend(i) for i in fold_1_train]
[fold_2.extend(i) for i in fold_2_train]
[fold_3.extend(i) for i in fold_3_train]
[fold_4.extend(i) for i in fold_4_train]

UT.write_csv(os.path.join(train_path, 'fold_0.csv'), fold_0)
UT.write_csv(os.path.join(train_path, 'fold_1.csv'), fold_1)
UT.write_csv(os.path.join(train_path, 'fold_2.csv'), fold_2)
UT.write_csv(os.path.join(train_path, 'fold_3.csv'), fold_3)
UT.write_csv(os.path.join(train_path, 'fold_4.csv'), fold_4)
##show the number of each fold and all
print(len(fold_0),len(fold_1),len(fold_2),len(fold_3),len(fold_4),len(fold_0)+len(fold_1)+len(fold_2)+len(fold_3)+len(fold_4))