import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
from torchsummary import summary

import os
import numpy as np
from sklearn import metrics
from tqdm import trange,tqdm ### for py file

import matplotlib.pyplot as plt
import utilities as UT
import pandas as pd
import warnings
from scipy.stats import percentileofscore
warnings.filterwarnings("ignore", category=Warning)

############################################## function area
def prep_data(TRAIN_PATH,TEST_PATH,TEST_NUM):
    # This function is used to prepare train/test labels for 5-fold cross-validation
    TEST_LABEL = os.path.join(TEST_PATH,''.join(['fold_',str(TEST_NUM),'.csv']))
    REMOVE_LABEL = os.path.join(TRAIN_PATH,''.join(['fold_',str(TEST_NUM),'.csv']))
    # combine train labels
    filenames = [os.path.join(TRAIN_PATH, 'fold_0.csv'), 
                os.path.join(TRAIN_PATH, 'fold_1.csv'), 
                os.path.join(TRAIN_PATH, 'fold_2.csv'), 
                os.path.join(TRAIN_PATH, 'fold_3.csv'), 
                os.path.join(TRAIN_PATH, 'fold_4.csv')]
    filenames.remove(REMOVE_LABEL)

    with open(os.path.join(TRAIN_PATH, ''.join(['combined_train_list','_',str(TEST_NUM),'.csv'])), 'w') as combined_train_list:
        for fold in filenames:
            for line in open(fold, 'r'):                
                combined_train_list.write(line)
    TRAIN_LABEL = os.path.join(TRAIN_PATH,''.join(['combined_train_list','_',str(TEST_NUM),'.csv']))
    
    return TRAIN_LABEL, TEST_LABEL

class Dataset_Early_Fusion(Dataset):
    def __init__(self, LABEL_PATH):
        self.files = UT.read_csv(LABEL_PATH)
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        full_path = self.files[idx]        
        full_path = full_path[0]        
        label = full_path.split('/')[-1]
        if 'CN' in label:
            label=0
        elif 'MCI' in label:
            label=1
        elif 'AD' in label:
            label=1
        else:
            print('Label Error',label)
        
        im = np.load(full_path) 
        im = np.expand_dims(im, axis=0)
        return im, int(label)   

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2, input_shape=(1,100,110,110)): #input_shape=(1,110,110,110)# input: input_shape:[num_of_filters, kernel_size] (e.g. [256, 25])
        super(ResNet3D, self).__init__()
        #stage 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=input_shape[0],out_channels=32,kernel_size=(3,3,3),padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
            in_channels=32,       
            out_channels=32,      
            kernel_size=(3,3,3),          
            padding=1              
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),                  
            nn.Conv3d(
            in_channels=32,       
            out_channels=64,       
            kernel_size=(3,3,3), 
            stride=2,
            padding=1            
            )
        )
        self.bot2=Bottleneck(64,64,1)
        self.bot3=Bottleneck(64,64,1)
        self.conv4=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,       
            out_channels=64,       
            kernel_size=(3,3,3),      
            padding=1,
            stride=2
            )
        )
        self.bot5=Bottleneck(64,64,1)
        self.bot6=Bottleneck(64,64,1)
        self.conv7=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,        
            out_channels=128,       
            kernel_size=(3,3,3),          
            padding=1,
            stride=2
            )
        )
        self.bot8=Bottleneck(128,128,1)
        self.bot9=Bottleneck(128,128,1)
        self.conv10=nn.Sequential(
            nn.MaxPool3d(kernel_size=(7,7,7)))
        
        fc1_output_features=128     
        self.fc1 = nn.Sequential(
             nn.Linear(512, 128),
             nn.ReLU()
        )

        fc2_output_features=2           
        self.fc2 = nn.Sequential(
        nn.Linear(fc1_output_features, fc2_output_features),
        nn.Sigmoid()
        )

    def forward(self, x, drop_prob=0.8):
        x = self.conv1(x)
        x = self.bot2(x)
        x = self.bot3(x)
        x = self.conv4(x)
        x = self.bot5(x)
        x = self.bot6(x)
        x = self.conv7(x)     
        x = self.bot8(x)
        x = self.bot9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1) 
        x = self.fc2(x)
        return x
        
def quant_result(y_true, y_pred):
    acc = float(torch.sum(torch.max(y_pred, 1)[1]==y_true))/ float(len(y_pred)) 
    auc = metrics.roc_auc_score(y_true, y_pred[:,1])
    f1 = metrics.f1_score(y_true, torch.max(y_pred, 1)[1])
    precision = metrics.precision_score(y_true, torch.max(y_pred, 1)[1])
    recall = metrics.recall_score(y_true, torch.max(y_pred, 1)[1])
    confs_mat = metrics.confusion_matrix(y_true, torch.max(y_pred, 1)[1])
    spec = confs_mat[0, 0] / float(confs_mat[0, 0] + confs_mat[0, 1])     
    return acc,auc,f1,precision,recall,spec        

# def test(val_dataloader, epoch, best_epoch, test_loss, old_acc, old_auc, net, loss_fcn):
def val(val_dataloader, net, loss_fcn):
    val_y_true = []
    val_y_pred = []
    val_loss = 0
    for step, (img, label) in enumerate(val_dataloader):
        img = img.float().to(device)
        label = label.long().to(device)
        out = net(img)
        loss = loss_fcn(out, label)
        val_loss += loss.item()
        label = label.cpu().detach()
        out = out.cpu().detach()
        val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)                

    val_loss = val_loss/(step+1)                               
    val_quant = quant_result(val_y_true,val_y_pred)
    val_set = [val_loss, val_quant[0], val_quant[1], val_quant[2],
                     val_quant[3], val_quant[4], val_quant[5]]                       
    return val_set, val_y_true, val_y_pred

############################################## train detail
def train(train_dataloader, val_dataloader_all):
    net = ResNet3D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))
    times = trange(EPOCHS, desc=' ', leave=True) 
    train_hist = []
    val_y_true = [[],[]] 
    val_y_pred = [[],[]]     
    val_hist = [[],[]]  
    test_y_true = [[],[]] 
    test_y_pred = [[],[]] 
    test_performance = [[],[]] 
    for epoch in times:    
        y_true = []
        y_pred = []        
        train_loss = 0
        # training
        net.train()
        for step, (img, label) in enumerate(train_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            opt.zero_grad()
            out = net(img)
            loss = loss_fcn(out, label)
            loss.backward()
            opt.step()            
            label = label.cpu().detach()
            out = out.cpu().detach()              
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)        
            train_loss += loss.item()
        train_loss = train_loss/(step+1)
        train_quant = quant_result(y_true,y_pred)
        train_hist.append([train_loss, train_quant[0], train_quant[1], train_quant[2],
                           train_quant[3], train_quant[4], train_quant[5]]) 
        # validation 
        net.eval()
        with torch.no_grad():
            for seq in range(len(val_dataloader_all)):                
                val_dataloader = val_dataloader_all[seq]
                if epoch == 0:
                    val_result = val(val_dataloader, net, loss_fcn)
                    val_hist[seq].append(val_result[0])
                    test_performance[seq] = [epoch] + val_result[0]
                    test_y_true[seq] = val_result[1]
                    test_y_pred[seq] = val_result[2]
                else:
                    val_result = val(val_dataloader, net, loss_fcn)
                    val_hist[seq].append(val_result[0])
                    if test_performance[seq][2] < val_result[0][1] \
                        or (test_performance[seq][2] == val_result[0][1] and test_performance[seq][3] == val_result[0][2]):
                        test_performance[seq] = [epoch] + val_result[0]
                        test_y_true[seq] = val_result[1]
                        test_y_pred[seq] = val_result[2]

                ### save the lastest epoch 
                if (epoch + 1) == EPOCHS:
                    val_y_true[seq]  = val_result[1] 
                    val_y_pred[seq] = val_result[2]
  
        times.set_description("Epoch: %i,\
                               train loss: %.4f, train acc: %.4f,\
                               val0 loss: %.4f, val0 acc: %.4f,\
                               val1 loss: %.4f, val1 acc: %.4f,\
                               test0 acc: %.4f, test1 acc: %.4f" 
                          %(epoch,
                            train_loss, train_quant[0],
                            val_hist[0][epoch][0], val_hist[0][epoch][1],
                            val_hist[1][epoch][0], val_hist[1][epoch][1],
                            test_performance[0][2], test_performance[1][2])) 

    return train_hist, val_hist, test_performance, test_y_true, test_y_pred, val_y_true, val_y_pred 


##FLAIR mwp1oT1 mwp2oT1 T2STAR wmoT1
############################################## main code
TRAIN_PATH = "./fold/gen/train"
TEST_PATH = "./fold/gen/test"

save_path = "./result/gen"
excel_name = 'all_result'
roc_name = 'roc'
loss_name = 'loss'
org_result_name = 'test_org_result'
perm_model_name = 'perm_model'
list_spc = ['data_217','data_32']
stat_index = ['test_acc','test_auc','test_f1','test_precision','test_recall','test_spec']
fold = 5 

print('TRAIN_PATH:',TRAIN_PATH)
GPU = 4
BATCH_SIZE = 8
EPOCHS = 50

LR = 0.00001
LOSS_WEIGHTS = torch.tensor([1.17, 1.]) 

device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')

train_hist = []
val_hist = []
test_performance = []
test_y_true = []
test_y_pred = []
val_y_true = []
val_y_pred = []


for i in range(0, fold):
    print('Test Fold', i)    
    TEST_NUM = i
    TRAIN_LABEL, TEST_LABEL0 = prep_data(TRAIN_PATH, TEST_PATH,TEST_NUM)
    
    train_dataset = Dataset_Early_Fusion(TRAIN_LABEL)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=False)
    ####################################################################################
    ### input parameters
    TEST_LABEL2 = os.path.join(TEST_PATH,''.join(['data_32_fold_',str(i),'.csv']))
    ####################################################################################
    val_dataset0 = Dataset_Early_Fusion(TEST_LABEL0)
    val_dataset2 = Dataset_Early_Fusion(TEST_LABEL2) 
    
    val_dataloader0 = torch.utils.data.DataLoader(val_dataset0, num_workers=4, 
                                                  batch_size=1, 
                                                  shuffle=False, drop_last=False)
    val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, num_workers=4, 
                                                  batch_size=1, 
                                                  shuffle=False, drop_last=False)
    val_dataloader_all = [val_dataloader0,val_dataloader2]
    cur_result = train(train_dataloader, val_dataloader_all)

    
    ###for loss curve
    train_hist.append(cur_result[0])
    val_hist.append(cur_result[1]) 
    ## for best result
    test_performance.append(cur_result[2]) 
    ### for roc curve
    test_y_true.append(cur_result[3])
    test_y_pred.append(cur_result[4]) 
    ### for save the all result 
    val_y_true.append(cur_result[5])
    val_y_pred.append(cur_result[6]) 
print('finish')

################
def sort_val(val_y,num):
    val_y = np.array(val_y).transpose(1,0)
    mat_y  = []
    for seq in range(num):
        mat_y.append(np.concatenate(val_y[seq]))
    return mat_y

list_true = sort_val(val_y_true,len(list_spc))
list_pred = sort_val(val_y_pred,len(list_spc))
roc_save = os.path.join(save_path,''.join([roc_name,'.pdf']))
print('save the roc fig')
for seq in range(len(list_true)): 
    fpr_fold,tpr_fold,_ = metrics.roc_curve(list_true[seq],list_pred[seq][:,1],)
    auc_fold = metrics.auc(fpr_fold,tpr_fold)
    plt.plot(fpr_fold,tpr_fold,label=str(''.join([list_spc[seq], ' AUC = %0.4f' % auc_fold])))
    plt.legend()
    plt.savefig(roc_save)     
### plot the loss figure
print('save the loss fig')
loss_save = os.path.join(save_path,''.join([loss_name,'.pdf']))    
# fig,axes = plt.subplots(15,2,figsize=(20,90))
fig,axes = plt.subplots(10,2,figsize=(20,90))
for seq in range(len(val_hist[0])):
    for i in range(len(train_hist)):
        train_pd = pd.DataFrame(train_hist[i],columns=['train_loss','train_acc','train_auc',
                                         'train_f1','train_precision','train_recall','train_spec'])
        val_pd = pd.DataFrame(val_hist[i][seq],columns=['val_loss','val_acc','val_auc','val_f1',
                                                   'val_precision','val_recall','val_spec'])
        ax1 = train_pd.plot(ax=axes[i+seq*5,0],title=''.join([list_spc[seq],' fold_',str(i)]))
        ax1.legend(loc='center right')
        ax2 = val_pd.plot(ax=axes[i+seq*5,1],title=''.join([list_spc[seq],' fold_',str(i)]))
        ax2.legend(loc='center right')
        plt.savefig(loss_save)  
#### for the last validation for report
pd_val = []
for seq in range(len(list_spc)):
    pd_val.append(quant_result(torch.from_numpy(list_true[seq]),torch.from_numpy(list_pred[seq])))
pd_val = pd.DataFrame(pd_val,index=list_spc,columns=['test_acc','test_auc','test_f1',
                                                        'test_precision','test_recall','test_spec'])        
### for the test ferform
mat_perfm = np.array(test_performance).transpose(1,0,2)
pd_perfm = []
for seq in range(len(list_spc)):
    pd_result = pd.DataFrame(mat_perfm[seq],index=['fold_0','fold_1','fold_2','fold_3','fold_4'],
                             columns=['best_epoch','test_loss','test_acc','test_auc',
                                     'test_f1','test_precision','test_recall','test_spec'])

    pd_perfm.append(pd_result)
pd_perfm = pd.concat(pd_perfm,keys=list_spc)
### write to excel
#### save the test_org_result as excel file
pd_org_result = []
for i in range(len(list_spc)):
    pd_org_result.append(pd.DataFrame([list_true[i],list_pred[i]],index=['y_true','y_pred']).T)
pd_org_result = pd.concat(pd_org_result,keys=list_spc,axis=1)

excel_save = os.path.join(save_path,''.join([excel_name,'.xlsx'])) 
xlsx_set = {'last_test_result':pd_val,'best_epoch_result':pd_perfm, 'test_org_result':pd_org_result}
writer = pd.ExcelWriter(excel_save)
for sheet_name in xlsx_set.keys():
    xlsx_set[sheet_name].to_excel(writer, sheet_name=sheet_name)    
writer.save()
#### save the test_org_result as npy file
npy_save = os.path.join(save_path,''.join([org_result_name,'.npy'])) 
test_org_result = np.array([list_true,list_pred])
np.save(npy_save,test_org_result)

npy_save_fold = os.path.join(save_path,''.join([org_result_name,'_fold','.npy'])) 
test_org_result_fold = np.array([val_y_true,val_y_pred])
np.save(npy_save_fold,test_org_result)
print('done')


