#### for the five cross validation
import os
import numpy as np
#import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt 
from PIL import Image
import csv
import shutil

img_root = "data_prepro_tmp"
sub_dir = ["data_AD/AV45","data_HC/AV45"]
csv_save = "./five_cross_valid/"
split_char1 = 'brain_AV45_'
split_char2 = '.nii.gz'
split_char3 = '_'

def write_csv(file_name, data):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
def data_list(csv_list):
    sub_list = []
    flist = np.loadtxt(csv_list,dtype='<U13',delimiter=',',usecols=1)
    print(flist.shape[0])
    for i in range(flist.shape[0]):
        sub = flist[i]
        for k in range(0,100):
            sub_slice = '_'.join([sub,str(k).zfill(2)])
            sub_list.append(sub_slice)
    #sub_list.sort()
    return sub_list
             
def prep_data(csv_save,test_num):
    # This function is used to prepare train/test labels for 5-fold cross-validation
    test_label = os.path.join(csv_save ,''.join(['fold_',str(test_num),'.csv']))

    # combine train labels
    filenames = [os.path.join(csv_save, 'fold_0.csv'), 
                os.path.join(csv_save, 'fold_1.csv'), 
                os.path.join(csv_save, 'fold_2.csv'), 
                os.path.join(csv_save, 'fold_3.csv'), 
                os.path.join(csv_save, 'fold_4.csv')]
    fold_dir = test_label.replace('.csv','')
    fold_exit = os.path.exists(fold_dir)
    if not fold_exit:
        os.makedirs(fold_dir) 
    filenames.remove(test_label)
    test_new = os.path.join(fold_dir,''.join(['fold_',str(test_num),'.csv']))
    shutil.copyfile(test_label,test_new)
    train_new = os.path.join(fold_dir,'combined_train_list.csv')
    
    with open(train_new,'w') as combined_train_list:
        for fold in filenames:
            for line in open(fold,'r'):                
                combined_train_list.write(line)
    train_flist_main2 = data_list(train_new)
    test_flist_main2 = data_list(test_new) 
    np.save(os.path.join(fold_dir,'train_flist_main2.npy'),np.asarray(train_flist_main2))
    np.save(os.path.join(fold_dir,'test_flist_main2.npy'),np.asarray(test_flist_main2))
                
################
### main code
################
fold_0 = []
fold_1 = []
fold_2 = []
fold_3 = []
fold_4 = []
np.random.seed(999)
                
for folder in sub_dir:
    cur_folder = os.path.join(img_root, folder)
    files = os.listdir(cur_folder)
    for file in files:  
        cur_file = os.path.join(cur_folder, file)
        cur_id = cur_file.split(sep=split_char1)[-1].replace(split_char2,'')
        cur_type = cur_file.split(sep=split_char1)[0].split(sep=split_char3)[-2]
        cur_file = [cur_type,cur_id]
        ran = np.random.randint(5)
        
        if(ran==0):
            fold_0.append(cur_file)
        elif(ran==1):
            fold_1.append(cur_file)
        elif(ran==2):
            fold_2.append(cur_file)
        elif(ran==3):
            fold_3.append(cur_file)
        elif(ran==4):
            fold_4.append(cur_file) 

# ### write the npy path of every fold to csv
write_csv(os.path.join(csv_save, 'fold_0.csv'), fold_0)
write_csv(os.path.join(csv_save, 'fold_1.csv'), fold_1)
write_csv(os.path.join(csv_save, 'fold_2.csv'), fold_2)
write_csv(os.path.join(csv_save, 'fold_3.csv'), fold_3)
write_csv(os.path.join(csv_save, 'fold_4.csv'), fold_4)                    

for test_num in range(0, 5):
    prep_data(csv_save,test_num)