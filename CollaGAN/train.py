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
log_dir     = opt.savepath+'/'+opt.name+'/log_dir/train'
ckpt_dir    = opt.savepath+'/'+opt.name+'/ckpt_dir'

########init DB##############
DB_train    = myDB()
DB_train.initialize(opt,'train')
l_train     = len(DB_train)

opt = DB_train.get_info(opt)
nY  = opt.nY 
nX  = opt.nX
nCh_in      = opt.nCh_in
nCh_out     = opt.nCh_out

nStep_train     = ceil(l_train/nB)
disp_step_train = ceil(nStep_train/opt.disp_div_N)

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
    if latest_ckpt==None:
        print("start! from  initialization!")
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        print("Start from saved model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExt(latest_ckpt) + 1

    train_writer = tf.summary.FileWriter(log_dir, sess.graph)   
    disp_t = 0+epoch_start*opt.disp_div_N

    if not opt.test_mode:
        for iEpoch in range(epoch_start, opt.nEpoch + 1):
            DB_train.shuffle(seed=iEpoch)   
            print('============================EPOCH # %d # =============' % (iEpoch) )
            s_epoch = time.time()
            if (iEpoch<opt.nEpochDclsf):
                out_arg  = [Colla.C_optm, Colla.G_loss, Colla.D_loss]
                out_argm = [Colla.C_optm, Colla.G_loss, Colla.D_loss, Colla.summary_op]
            else:
                if iEpoch%opt.nEpochD==0:
                    out_arg  = [Colla.reconpic,Colla.C_optm, Colla.G_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss]
                    out_argm = [Colla.reconpic,Colla.C_optm, Colla.G_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss, Colla.summary_op]
                else:
                    out_arg  = [Colla.reconpic,Colla.C_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss]
                    out_argm = [Colla.reconpic,Colla.C_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss, Colla.summary_op]
    
            loss_G = 0.
            loss_D = 0.
            cnt=0
            for step in tqdm(range(nStep_train)):
                _tar_class_idx, _a,_b,_c,_d,_e,_f, _am, _bm, _cm, _dm,_em,_fm, _tar_class_bools, _tar_img ,pname= DB_train.getBatch_RGB_varInp(step*nB,(step+1) * nB)
                feed_dict = {Colla.is_Training:True, Colla.tar_class_idx:_tar_class_idx, Colla.a_img:_a, Colla.b_img:_b, Colla.c_img:_c, Colla.d_img:_d, Colla.e_img:_e, Colla.f_img:_f,  Colla.targets:_tar_img,
                        Colla.a_mask:_am, Colla.b_mask:_bm, Colla.c_mask:_cm, Colla.d_mask:_dm, Colla.e_mask:_em, Colla.f_mask:_fm,
                        Colla.bool1:_tar_class_bools[0], Colla.bool2:_tar_class_bools[1], Colla.bool3:_tar_class_bools[2], Colla.bool4:_tar_class_bools[3],Colla.bool5:_tar_class_bools[4], Colla.bool6:_tar_class_bools[5] }
                # train
                if step % disp_step_train == 0:
                    results = sess.run(out_argm, feed_dict=feed_dict)
                    train_writer.add_summary(results[-1],disp_t)
                    disp_t+=1
                    train_writer.flush()
                    loss_G = loss_G + results[-3]
                    loss_D = loss_D + results[-2]
                else:
                    results = sess.run(out_arg, feed_dict = feed_dict)
                    loss_G = loss_G + results[-2]
                    loss_D = loss_D + results[-1]
                if iEpoch ==opt.nEpoch:
                    reconvalue=results[0]
                    np.save(os.path.join(opt.train_fig,''.join([pname+'.npy'])),reconvalue)   #smscale80
            str_train = (' %d epoch -- train loss (G / D) : %.4f /  %.4f' %(iEpoch, loss_G/nStep_train, loss_D/nStep_train))            
            print(str_train)
            if iEpoch %5 ==0:
                path_saved = saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=iEpoch)
                logging.info("Model saved in file: %s" % path_saved)            
          
