######nii2npy
import numpy as np
import os
np.set_printoptions(threshold=np.inf)
from dipy.io.image import load_nifti, save_nifti
from PIL import Image

path_src = "data_prepro"
path_tgr = "data_prepro_tmp"

### each group was converted to npy file independently; The directory tree can be see in "data_origial_demo"

mod_dic = dict([('T2STAR','t2star'),('AV45','av45'),('wmoT1','t1'),('mwp2oT1','wm'),('mwp1oT1','gm'),('FLAIR','flair')])
for i in mod_dic.keys():
    path_mod = os.path.join(path_src,i)
    subj = os.listdir(path_mod)
    for t in subj:
        path_subj = os.path.join(path_mod,t)
        data,affine=load_nifti(path_subj)
        data[ np.isnan(data)] = 0
        data[ np.isinf(data)] = 0
        for k in range(0,100):
            slices = data[:,:,k] 
            slices = Image.fromarray(slices)
            slices = slices.resize((240,240),Image.ANTIALIAS)
            slices = np.asarray(slices)
            slices[np.isnan(slices)] = 0
            slices[np.isinf(slices)] = 0
            name_tgr = ''.join([t.split(sep="_",maxsplit=6)[-1][:-7],'_',str(k).zfill(2),'_',mod_dic[i],'.npy'])
            print(nan_num,inf_num) 
            path_save = os.path.join(path_tgr,mod_dic[i],name_tgr)
            np.save(path_save,slices)