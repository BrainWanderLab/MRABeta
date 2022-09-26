### list of npy file
import glob
import random
import numpy  as np
import os 
train_flist_main1 = []
valid_flist_main1 = []
test_flist_main1 = []

random.seed(259)
save_path = "./five_cross_valid"
nii_path = "data_prepro"
npy_path = "./five_cross_valid"



subj_ad = glob.glob(os.path.join(nii_path,"FLAIR/*_AD_*"))
subj_ad = [ '_'.join(subj_ad[i].split(sep='_')[-3:]).replace('.nii.gz','') for i in range(len(subj_ad))]
subj_cn = glob.glob(os.path.join(nii_path,"FLAIR/*_CN_*"))
subj_cn = [ '_'.join(subj_cn[i].split(sep='_')[-3:]).replace('.nii.gz','') for i in range(len(subj_cn))]
subj_set = subj_ad + subj_cn 
np.save(os.path.join(save_path,'subject.npy'),np.asarray(subj_set))

##train_set
################ must change ######################
train_ad = random.sample(subj_ad,100)  
train_cn = random.sample(subj_cn,85)  
####################################################
train_set = train_ad + train_cn

train_npy = [ ''.join(['*',i,'*']) for i in train_set ]
train_flist_main = [ glob.glob(os.path.join(npy_path,'flair',t)) for t in train_npy]
train_flist_main1 = []
[ train_flist_main1.extend(k) for k in train_flist_main ]
train_flist_main2 = [ p.split(sep='/')[-1].replace('_flair.npy','') for p in train_flist_main1]
train_flist_main2.sort()
np.save(os.path.join(save_path,'train_flist_main2.npy'),np.asarray(train_flist_main2))

test_ad = list(set(subj_ad) - set(train_ad))
test_cn = list(set(subj_cn) - set(train_cn))
test_set = test_ad + test_cn

test_npy = [ ''.join(['*',i,'*']) for i in test_set ]
test_flist_main = [ glob.glob(os.path.join(npy_path,'flair',t)) for t in test_npy]
test_flist_main1 = []
[ test_flist_main1.extend(k) for k in test_flist_main ]
test_flist_main2 = [ p.split(sep='/')[-1].replace('_flair.npy','') for p in test_flist_main1]
test_flist_main2.sort()
np.save(os.path.join(save_path,'test_flist_main2.npy'),np.asarray(test_flist_main2))
print(len(train_cn),len(test_cn))
print(len(train_ad),len(test_ad))
print(len(train_set),len(test_set),len(train_flist_main2),len(test_flist_main2))