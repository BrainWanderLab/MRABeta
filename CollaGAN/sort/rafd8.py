from os import listdir
from os.path import join, isfile
from scipy import misc
import numpy as np
import random
from ipdb import set_trace as st
from math import ceil
import copy
import time
from PIL import Image

class pet():
    def __init__(self):
        super(pet, self).__init__()
    
    def name(self):
        return 'rafd-pet8'

    def initialize(self, opt, phase):
        t_start = time.time()
        random.seed(0)
        self.root   = opt.dataroot
        self.flist = np.load(join(opt.dataroot, phase+'_flist_main2.npy'))  
        self.N = 6 ### total of modality
        self.nCh_out = 1 #opt.nCh_out
        self.nCh_in  = self.N*self.nCh_out + self.N #opt.nCh_in 
        self.nY      = 240     ###128
        self.nX      = 240    
        self.len     = len(self.flist) 
        self.fExp=['_t1','_flair','_t2star','_gm','_wm','_av45']         
        self.use_aug = (phase=='train') and opt.AUG
        self.use_norm_std =  opt.wo_norm_std
        self.N_null  = opt.N_null

        # Here, for dropout input
        self.null_N_set = [x+1 for x in range(opt.N_null)] #[1,2,3,4,5,6]
        self.list_for_null = []
        
        for i in range(self.N):
            self.list_for_null.append( self.get_null_list_for_idx(i) )

    ''' Here, initialize the null vectors for random input dropout selection''' 
    def get_null_list_for_idx(self, idx):
        a_list = []
        for i_null in self.null_N_set:
            tmp_a = []
            if i_null == 1:
                tmp = [ bX==idx for bX in range(self.N) ]
                tmp_a.append(tmp)

            elif i_null ==2:
                for i_in in range(self.N):
                    if not i_in==idx:
                        tmp = [ bX in [i_in, idx] for bX in range(self.N) ]
                        tmp_a.append(tmp)
            
            elif i_null ==3:
                for i_in in range(self.N):
                    for ii_in in range(self.N):
                        if not (i_in==ii_in or (i_in==idx or ii_in==idx)):
                            tmp = [ ( bX in [i_in, ii_in, idx]) for bX in range(self.N) ]
                            tmp_a.append(tmp)
            
            elif i_null ==4:
                for i_in in range(self.N):
                    for ii_in in range(self.N):
                        for iii_in in range(self.N):
                            if not ( (i_in==ii_in or i_in==iii_in or ii_in==iii_in)  or (i_in==idx or ii_in==idx or iii_in==idx)):
                                tmp = [ (bX in [i_in, ii_in, iii_in, idx]) for bX in range(self.N) ]
                                tmp_a.append(tmp)
                
           
            else:
                st()
            
            a_list.append(tmp_a)

        return a_list 


    def get_info(self,opt):
        opt.nCh_in = self.nCh_in
        opt.nCh_out= self.nCh_out
        opt.nY     = self.nY
        opt.nX     = self.nX
        return opt

    def getBatch_RGB_varInp(self, start, end):
        nB = end - start
        end = min([end,self.len])
        start = end - nB
        batch = self.flist[start:end]
        # channel First 
        sz_a   = [nB,self.nCh_out,  self.nY, self.nX] 
        sz_M   = [nB,  1,  self.nY, self.nX] 

        target_class_idx = np.empty([nB,1],dtype=np.uint8)
        a_img = np.empty(sz_a, dtype=np.float32)
        b_img = np.empty(sz_a, dtype=np.float32)
        c_img = np.empty(sz_a, dtype=np.float32)
        d_img = np.empty(sz_a, dtype=np.float32)
        e_img = np.empty(sz_a, dtype=np.float32)
        f_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)
   
        a_mask = np.zeros(sz_M, dtype=np.float32)
        b_mask = np.zeros(sz_M, dtype=np.float32)
        c_mask = np.zeros(sz_M, dtype=np.float32)
        d_mask = np.zeros(sz_M, dtype=np.float32)
        e_mask = np.zeros(sz_M, dtype=np.float32)
        f_mask = np.zeros(sz_M, dtype=np.float32)

        if  opt.test_mode:
            targ_idx=5   
            tar_class_bools=[False,False,False,False,False,True]
        else
            targ_idx = random.randint(0,self.N-1)
            tar_class_bools = [ x==targ_idx for x in range(self.N) ]

        for iB, aFname in enumerate(batch):
            a_img = self.read_png( join(self.root, 't1',aFname+self.fExp[0]+'.npy'))
            b_img = self.read_png( join(self.root, 'flair',aFname+self.fExp[1]+'.npy')) 
            c_img = self.read_png( join(self.root, 't2star',aFname+self.fExp[2]+'.npy'))  
            d_img = self.read_png( join(self.root, 'gm',aFname+self.fExp[3]+'.npy')) 
            e_img = self.read_png( join(self.root, 'wm',aFname+self.fExp[4]+'.npy'))  
            f_img = self.read_png( join(self.root, 'av45',aFname+self.fExp[5]+'.npy'))  

            name=aFname                    
            if targ_idx ==0:
                target_img[iB,:,:,:] = a_img[iB,:,:,:]
                a_mask[iB,0,:,:] = 1.
            elif targ_idx ==1:
                target_img[iB,:,:,:] = b_img[iB,:,:,:]
                b_mask[iB,0,:,:] = 1.
            elif targ_idx ==2:
                target_img[iB,:,:,:] = c_img[iB,:,:,:]
                c_mask[iB,0,:,:] = 1.
            elif targ_idx ==3:
                target_img[iB,:,:,:] = d_img[iB,:,:,:]
                d_mask[iB,0,:,:] = 1.
            elif targ_idx ==4:
                target_img[iB,:,:,:] = e_img[iB,:,:,:]
                e_mask[iB,0,:,:] = 1.
            elif targ_idx ==5:
                target_img[iB,:,:,:] = f_img[iB,:,:,:]
                f_mask[iB,0,:,:] = 1.                            
            else:
                st()
            target_class_idx[iB] = targ_idx
        return target_class_idx, a_img, b_img, c_img, d_img,e_img,  f_img, a_mask, b_mask, c_mask, d_mask, e_mask, f_mask, tar_class_bools, target_img,name 

    '''this function is made for the rebuttal '''

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    def __len__(self):
        return self.len
    @staticmethod
    def read_png(filename):
        png=np.load(filename)  #1*128*128*1
        png=png[np.newaxis,np.newaxis,:,:]
        return png    



