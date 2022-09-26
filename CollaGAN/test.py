import os
import numpy as np
import tensorflow as tf
from util.util import myNumExt
import time
from sort.rafd8 import pet as myDB
from ipdb import set_trace as st
from math import ceil
import random
from options.AV45 import BaseOptions
from tqdm import tqdm
import logging 

opt = BaseOptions().parse()

# parameter setting
nB          = opt.nB
ckpt_dir    = opt.savepath+'/'+opt.name+'/ckpt_dir'
########init DB##############
DB_test    = myDB()
DB_test.initialize(opt,'test') ##valid,train
l_test     = len(DB_test)
print(l_test)
opt = DB_test.get_info(opt)
nY  = opt.nY 
nX  = opt.nX
nCh_in      = opt.nCh_in
nCh_out     = opt.nCh_out


nStep_test     = ceil(l_test/nB)

## model initialize
str_ = "/device:GPU:"+str(opt.gpu_ids[0])
print(str_)
from model.CollaGAN_mri import CollaGAN as myModel
with tf.device(str_):
    Colla = myModel(opt)

saver = tf.train.Saver()

##
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    print("Start from saved model -"+latest_ckpt)
    saver.restore(sess, latest_ckpt)
    epoch_start=myNumExt(latest_ckpt)+1

    if  opt.test_mode:
         ssimvalue=[]
         msevalue=[]
         psnrvalue=[]
         for step in tqdm(range(nStep_test)):
                _tar_class_idx, _a,_b,_c,_d,_e,_f, _am, _bm, _cm, _dm,_em,_fm, _tar_class_bools, _tar_img,pname = DB_test.getBatch_RGB_varInp(step*nB,(step+1)*nB)
                feed_dict = {Colla.is_Training:False,Colla.tar_class_idx:_tar_class_idx, Colla.a_img:_a, Colla.b_img:_b, Colla.c_img:_c, Colla.d_img:_d, Colla.e_img:_e, Colla.f_img:_f, Colla.targets:_tar_img,
                        Colla.a_mask:_am, Colla.b_mask:_bm, Colla.c_mask:_cm, Colla.d_mask:_dm, Colla.e_mask:_em, Colla.f_mask:_fm, 
                        Colla.bool1:_tar_class_bools[0], Colla.bool2:_tar_class_bools[1], Colla.bool3:_tar_class_bools[2], Colla.bool4:_tar_class_bools[3],Colla.bool5:_tar_class_bools[4], Colla.bool6:_tar_class_bools[5] }                
                results = sess.run([Colla.reconpic,Colla.ssim,Colla.mse,Colla.psnr], feed_dict = feed_dict)
                reconvalue=results[0]
                np.save(os.path.join(opt.test_fig,''.join([pname+'.npy'])),reconvalue)
                
                ssimvalue.append(results[1])
                msevalue.append(results[2])
                psnrvalue.append(results[3])
         np.save(os.path.join(opt.test_ssim,'ssim.npy'),ssimvalue)
         np.save(os.path.join(opt.test_ssim,'mse.npy'),msevalue)
         np.save(os.path.join(opt.test_ssim,'psnr.npy'),psnrvalue)