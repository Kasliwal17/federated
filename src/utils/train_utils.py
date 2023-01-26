import numpy as np
import torch
import torch.nn as nn
from .model import First_Conv, Fully_Connected_Layer, GNN_Network
from .dataloader import construct_dataset
from .loss import Custom_Loss
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import cohen_kappa_score
import copy
from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN
from .inference_utils import inference
from .transformations import train_transform, test_transform

def compute_edge_attr(A):
    edge=[]
    edge_attr=[]
    for j in range(0,14):
        for k in range(0,14):
            if j==k:
                continue
            edge.append(np.array([j,k]))
            edge_attr.append(A[j,k])
    
    edge=np.array(edge)
    edge_attr=np.array(edge_attr)
    
    edge=torch.from_numpy(np.transpose(edge))
    edge=edge.long()
    
    edge_attr=torch.from_numpy(edge_attr)
    edge_attr=torch.unsqueeze(edge_attr, dim=1)
    edge_attr=edge_attr.float()
       
    return edge, edge_attr


def compute_adjacency_matrix(adj_type, site, split_npz='/storage/aneesh/split.npz'):
    
    # load the npz file
    a=np.load(split_npz, allow_pickle=True)    
    gt=a['gt']
    clstr_assgn=a['clstr_assgn']
    trn_val_tst=a['trn_val_tst']
    del a
    
    if site==-999:
        idx=np.where(trn_val_tst==0)[0]
    else:
        idx=np.where((clstr_assgn==site) & (trn_val_tst==0))[0]

        ###################################### EDITED HERE ####################################33    
#     gt=gt[idx,:]

    gt=gt[idx]
    
    #################################################3
    
    kappa=np.zeros((14,14))
    TP=np.zeros((14,14))
    TN=np.zeros((14,14))
    FP=np.zeros((14,14))
    FN=np.zeros((14,14))
    kappa=np.zeros((14,14))
    agree=np.zeros((14,14))
    P1=np.zeros((14,14))
    
    
    for j in range(0,14):
#         gt_j=gt[:,j]
        gt_j=gt[j]
        for k in range(0, 14):
#             gt_k=gt[:,k]
            gt_k=gt[k]
            
            ## Kappa and agree are symmetric ie., A(i,j)=A(j,i)
            kappa[j,k]=cohen_kappa_score(gt_j, gt_k)
            agree[j,k]=(np.where(gt_j==gt_k)[0].shape[0])/gt.shape[0]
            
            # How many times are both j and k =1---> This will be symmetric
            TP[j,k]=(np.where((gt_j==1) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many times are both j and k=0 ---> This will be symmetric
            TN[j,k]=(np.where((gt_j==0) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
            ####### FP and FN will get reversed for A(i,j) and A(j,i)
            # How many time k is 1 but j is 0
            FP[j,k]=(np.where((gt_j==0) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many time k is 0 but j is 1
            FN[j,k]=(np.where((gt_j==1) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
    if adj_type=='kappa':
        A=kappa
    elif adj_type=='fraction_agreement':
        A=agree
    elif adj_type=='confusion_matrix':
        A=np.concatenate((np.expand_dims(TP, axis=2), np.expand_dims(TN, axis=2),np.expand_dims(FP, axis=2), np.expand_dims(FN, axis=2)), axis=2)
                    
    if A.ndim==2:
        tmp_edge, edge_attr=compute_edge_attr(A)
    else:
        edge_lst=[]
        edge_attr_lst=[]
        for x in range(A.shape[2]):
            tmp_edge, tmp_edge_attr=compute_edge_attr(np.squeeze(A[:,:,x]))
            edge_lst.append(tmp_edge)
            edge_attr_lst.append(tmp_edge_attr)
            
            
        edge_attr=torch.cat(edge_attr_lst, dim=1)
      
    return tmp_edge, edge_attr

def compute_lcl_wt(epoch, cmb_wts, glbl_wt, prev_lcl_wt, device):
    
    cmb_wt=cmb_wts[epoch]
    lcl_wt=1-cmb_wt
    
    wt=average_weights(device, [prev_lcl_wt, glbl_wt], [lcl_wt, cmb_wt])
    
    return wt
        
################ Compute weighted average of model weights ##################
def average_weights(w, cmb_wt, device):
    
    cmb_wt=np.array(cmb_wt)
    cmb_wt=cmb_wt.astype(np.float)
    cmb_wt=cmb_wt/np.sum(cmb_wt)
    #print(cmb_wt)

    wts = torch.tensor(cmb_wt).to(device)
    wts=wts.float()
    
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys(): # for each layer
        layer = key.split('.')[-1]
    
        if layer == 'num_batches_tracked':
            for i in range(1,len(w)): # for each model
                w_avg[key] += w[i][key].to(device)
            w_avg[key] = torch.div(w_avg[key].float(), torch.tensor(len(w)).float()).to(torch.int64)
        else:
            w_avg[key]=torch.mul(w_avg[key].to(device), wts[0].to(float))
            for i in range(1,len(w)):
                w_avg[key] += torch.mul(w[i][key].to(device), wts[i].to(float))
            
            
    return w_avg


def aggregate_local_weights(wt0, wt1, wt2, wt3, wt4, device):
    
    wt=average_weights([wt0, wt1, wt2, wt3, wt4], 
                       [1.0, 2030.0/1997, 2093.0/1997, 1978.0/1997, 2122.0/1997])
    return wt



# Train 1 batch update
def train_one_batch(sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4, 
                    trn_typ, criterion, device, edge_index=None, edge_attr=None):
    ##Keep gnn_model and optim4 as None if training is to be done without GNN
    img=sample['img']
    gt=sample['gt']
    
    img=img.to(device)
    gt=gt.to(device)
    
    ########Forward Pass ##############
    img_3chnl=cnv_lyr(img)
    gap_ftr=backbone_model(img_3chnl)
    ftr_lst, prd=fc_layers(gap_ftr)
    if gnn_model is not None:
        ftr_lst=torch.cat(ftr_lst, dim=1)
        
        data_lst=[]
        for k in range(0, ftr_lst.shape[0]):
            data_lst.append(Data_GNN(x=ftr_lst[k,:,:], edge_index=edge_index, edge_attr=edge_attr, y=torch.unsqueeze(gt[k,:], dim=1))) 
        loader = DataLoader_GNN(data_lst, batch_size=ftr_lst.shape[0])
        loader=next(iter(loader)).to(device)
        gt=loader.y
        
        prd_final=gnn_model(loader)
    else:
        prd_final=prd
    
    loss=criterion(prd_final, gt)
    
    ####### Backward Pass ##########
    ### Remove previous gradients
    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()
    if optim4 is not None:
        optim4.zero_grad()
    
    
    ### Compute Gradients
    loss.backward()
    
    ### Optimizer Gradients
    #update weights through backprop using Adam 

    #if training is without gnn 
    if gnn_model==None:
        optim1.step()
        optim2.step()
        optim3.step()
        return cnv_lyr, backbone_model, fc_layers, loss, optim1, optim2, optim3

    #if training is with gnn
    if trn_typ=='full':
        optim1.step()     
        optim2.step()
    
    optim3.step()
    optim4.step()
    
    
    return cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4


#### Train main
def train_end_to_end(cnv_lyr, backbone_model, fc_layers, gnn_model, train_loader, trn_typ, n_batches, criterion, device, 
                     edge_index=None, edge_attr=None):
    
    cnv_lyr.train() 
    backbone_model.train()
    fc_layers.train()
    
    
    ########## Optimizers and Schedulers #############
    total_batches=len(train_loader)
    #print(total_batches)
    lr=10**(-5)
    
    # optimizer
    optim1 = torch.optim.Adam(cnv_lyr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim2 = torch.optim.Adam(backbone_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim3 = torch.optim.Adam(fc_layers.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    if gnn_model is not None:
        gnn_model.train()
        optim4 = torch.optim.Adam(gnn_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    
    cnt=0
    trn_run_loss=0
    for i, sample in enumerate(train_loader):
        cnt=cnt+1
        if gnn_model is not None:
            cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4=train_one_batch(
                            sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4, 
                            trn_typ, criterion, device, edge_index, edge_attr)
        else:
            #Set gnn_model and optim4 as None if training is to be done without GNN
            cnv_lyr, backbone_model, fc_layers, loss, optim1, optim2, optim3=train_one_batch(
                            sample, cnv_lyr, backbone_model, fc_layers, None, optim1, optim2, optim3, None, 
                            trn_typ, criterion, device)  
        trn_run_loss=trn_run_loss+loss
            
        if (i+1) % 20== 0: # displays after every 20 batch updates
            print ("cnt {}, Train Loss: {:.4f}".format(cnt,(trn_run_loss/(cnt))), end ="\r")
                
        ############# Monitor Validation Acc and Early Stopping ############
        if cnt>=n_batches:
            break
    if gnn_model is None:
        return cnv_lyr, backbone_model, fc_layers

    return cnv_lyr, backbone_model, fc_layers, gnn_model


####For training with gnn
def lcl_train_gnn(trn_loader, val_loader, criterion, cnv_lyr, backbone_model,fc_layers, gnn_model, edge_index, edge_attr, device):
    
    n_batches=1500
    ####### Freeze and train the part which is specific to each site only
    print('Freeze global CNN, fine-tune GNN ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(cnv_lyr, backbone_model, 
                                                fc_layers, gnn_model, trn_loader,'gnn', n_batches, criterion, device,
                                                edge_index, edge_attr)
    
    ###### Compute the Validation accuracy #######
    print('Computing Validation Performance ...')
    prev_val=inference(cnv_lyr, backbone_model, fc_layers, gnn_model, val_loader, criterion, device,
                        edge_index, edge_attr)
    
    ######## Train the entire network in an end-to-end manner ###
    print('Train end-to-end for Local Site ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(cnv_lyr, backbone_model, fc_layers, 
                                                        gnn_model, trn_loader,'full', 2*n_batches, criterion, device,
                                                        edge_index, edge_attr)
    
    
    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())
    
    return prev_val, cnv_wt,backbone_wt, fc_wt, gnn_wt

####For training without gnn
def lcl_train(trn_loader, val_loader, criterion, cnv_lyr1, backbone_model,fc_layers, device):
    n_batches = 4000
    ###### Compute the Validation accuracy #######
    prev_val=inference(cnv_lyr1, backbone_model, fc_layers, None, val_loader, criterion, device)
    
    ######## Train the entire network in an end-to-end manner ###
    train_end_to_end(cnv_lyr1, backbone_model, fc_layers, None, trn_loader, None, n_batches, criterion, device)
    
    
    cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    return prev_val, cnv_lyr1_wt,backbone_wt, fc_wt

def initialize_training(site, img_pth, split_npz, train_transform, test_transform, b_sz):
    
    data_trn=construct_dataset(img_pth, split_npz, site, train_transform, tn_vl_idx=0)
    trn_loader=DataLoader(data_trn,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    
    data_val=construct_dataset(img_pth, split_npz, site, test_transform, tn_vl_idx=1)
    val_loader=DataLoader(data_val, 1, shuffle=False, num_workers=1, pin_memory=False)
    
    criterion=Custom_Loss(site)
    edge_index, edge_attr= compute_adjacency_matrix('confusion_matrix', site, '/storage/aneesh/split.npz')
    
    
    return trn_loader, val_loader, criterion, edge_index, edge_attr



def initialize_model_weights(cnv_lyr, backbone_model, fc_layers, gnn_model):
    
    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())
    
    return cnv_wt, backbone_wt, fc_wt, gnn_wt




def save_model_weights(mx_nm, glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, sit0_gnn_wt=None, sit1_gnn_wt=None, 
                       sit2_gnn_wt=None,sit3_gnn_wt=None, sit4_gnn_wt=None):
    torch.save({
                'cnv_lyr_state_dict': glbl_cnv_wt,
                'backbone_model_state_dict': glbl_backbone_wt,
                'fc_layers_state_dict': glbl_fc_wt,
                'sit0_gnn_model': sit0_gnn_wt,
                'sit1_gnn_model': sit1_gnn_wt,
                'sit2_gnn_model': sit2_gnn_wt,
                'sit3_gnn_model': sit3_gnn_wt,
                'sit4_gnn_model': sit4_gnn_wt,
                }, mx_nm)
    
    
def instantiate_architecture(ftr_dim, model_name, gnn=False):
    # If gnn=True, then instantiate the GNN architecture
  
    if model_name=='densenet':
        inp_dim=1024
        backbone_model=models.densenet121(pretrained=True)
        backbone_model.classifier=nn.Identity()
    elif model_name=='resnet':
        inp_dim=512
        backbone_model=models.resnet18(pretrained=True)
        backbone_model.fc=nn.Identity()
    
    elif model_name=='xception':
        inp_dim=2048
        backbone_model=xception.xception(pretrained=True)
        backbone_model.fc=nn.Identity()
    

    cnv_lyr=First_Conv()
    fc_layers=Fully_Connected_Layer(inp_dim, ftr_dim)
    if gnn:
        gnn_model=GNN_Network(in_chnls=512, base_chnls=1, grwth_rate=1, depth=1, aggr_md='mean', ftr_dim=4)
        return cnv_lyr, backbone_model, fc_layers, gnn_model
    
    return cnv_lyr, backbone_model, fc_layers


#Main function for training                
def trainer_with_GNN(b_sz, img_pth, split_npz, train_transform, test_transform, max_epochs, backbone, device, restart_checkpoint=''):
    
    ###### Instantiate the CNN-GNN Architecture ##############
    cnv_lyr, backbone_model, fc_layers, gnn_model=instantiate_architecture(ftr_dim=512, model_name=backbone, gnn=True)
    cnv_lyr = cnv_lyr.to(device)
    fc_layers = fc_layers.to(device)
    backbone_model = backbone_model.to(device)
    gnn_model = gnn_model.to(device)

    #####################################################################################
    ############## Initialize Data Loaders #################
    
    trn_loader0, val_loader0, criterion0, edge_index0, edge_attr0=initialize_training(0, img_pth, split_npz, 
                                                              train_transform, test_transform, b_sz)
    
    trn_loader1, val_loader1, criterion1, edge_index1, edge_attr1=initialize_training(1, img_pth, split_npz, 
                                                              train_transform, test_transform)
    
    trn_loader2, val_loader2, criterion2, edge_index2, edge_attr2=initialize_training(2, img_pth, split_npz, 
                                                              train_transform, test_transform)
    
    trn_loader3, val_loader3, criterion3, edge_index3, edge_attr3=initialize_training(3, img_pth, split_npz, 
                                                              train_transform, test_transform)
    
    trn_loader4, val_loader4, criterion4, edge_index4, edge_attr4=initialize_training(4, img_pth, split_npz, 
                                                              train_transform, test_transform)
    
    #########################################################################################
    ### Initialize local and global model weights with the Imagenet pre-trained weights for backbone 
    #and identical model weights for the other layers.

    
    glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, gnn_wt=initialize_model_weights(cnv_lyr, backbone_model, 
                                                                                    fc_layers, gnn_model)
    sit0_gnn_wt=copy.deepcopy(gnn_wt)
    sit1_gnn_wt=copy.deepcopy(gnn_wt)
    sit2_gnn_wt=copy.deepcopy(gnn_wt)
    sit3_gnn_wt=copy.deepcopy(gnn_wt)
    sit4_gnn_wt=copy.deepcopy(gnn_wt)
    
    del gnn_wt
    
    # Load previous checkpoint if resuming the  training else comment out
    if (restart_checkpoint!=''):
        checkpoint=torch.load(restart_checkpoint)
        glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
        glbl_backbone_wt=checkpoint['backbone_model_state_dict']
        glbl_fc_wt=checkpoint['fc_layers_state_dict']
        sit0_gnn_wt=checkpoint['sit0_gnn_model']
        sit1_gnn_wt=checkpoint['sit1_gnn_model']
        sit2_gnn_wt=checkpoint['sit2_gnn_model']
        sit3_gnn_wt=checkpoint['sit3_gnn_model']
        sit4_gnn_wt=checkpoint['sit4_gnn_model']
    
    ##########################################################################################
    ################ Begin Actual Training ############
    max_val=0
    for epoch in range(0, max_epochs):
        print('############ Epoch: '+str(epoch)+'   #################')
        
        ###### Load the global model weights ########
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit0_gnn_wt)
        
        print('\n \n SITE 0 \n')
        prv_val0, sit0_cnv_wt,sit0_backbone_wt, sit0_fc_wt, sit0_gnn_wt=lcl_train_gnn(trn_loader0, val_loader0,
                                                criterion0, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index0, edge_attr0, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit1_gnn_wt)
        
        print('\n \n SITE 1 \n')
        prv_val1, sit1_cnv_wt,sit1_backbone_wt, sit1_fc_wt, sit1_gnn_wt=lcl_train_gnn(trn_loader1, val_loader1, 
                                                criterion1, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index1, edge_attr1, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit2_gnn_wt)
        
        print('\n \n SITE 2 \n')
        prv_val2, sit2_cnv_wt,sit2_backbone_wt, sit2_fc_wt, sit2_gnn_wt=lcl_train_gnn(trn_loader2, val_loader2, 
                                                criterion2, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index2, edge_attr2, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit3_gnn_wt)
        
        print('\n \n SITE 3 \n')
        prv_val3, sit3_cnv_wt,sit3_backbone_wt, sit3_fc_wt, sit3_gnn_wt=lcl_train_gnn(trn_loader3, val_loader3, 
                                                criterion3, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index3, edge_attr3, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit4_gnn_wt)
        
        print('\n \n SITE 4 \n')
        prv_val4, sit4_cnv_wt,sit4_backbone_wt, sit4_fc_wt, sit4_gnn_wt=lcl_train_gnn(trn_loader4, val_loader4, 
                                                criterion4, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index4, edge_attr4, device)

        
        avg_val=(prv_val0+prv_val1+prv_val2+prv_val3+prv_val4)/5
        print('Avg Val AUC: '+str(avg_val))
        
        if avg_val>max_val:
            max_val=avg_val
            mx_nm='best_weight_'+str(max_val)+'_'+str(epoch)+'.pt'
            save_model_weights(mx_nm, glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, sit0_gnn_wt, sit1_gnn_wt, sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt)
            print('Validation Performance Improved !')
            
            
        ############### Compute the global model weights #############
        
        glbl_cnv_wt=aggregate_local_weights(sit0_cnv_wt, sit1_cnv_wt, sit2_cnv_wt,
                                                sit3_cnv_wt, sit4_cnv_wt, device)
        
        glbl_backbone_wt=aggregate_local_weights(sit0_backbone_wt, sit1_backbone_wt, sit2_backbone_wt,
                                                sit3_backbone_wt, sit4_backbone_wt, device)
        
        glbl_fc_wt=aggregate_local_weights(sit0_fc_wt, sit1_fc_wt, sit2_fc_wt, sit3_fc_wt, sit4_fc_wt, device)
        
        glbl_gnn_wt=aggregate_local_weights(sit0_gnn_wt, sit1_gnn_wt, sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt, device)
        
    
    
def trainer_without_GNN( avg_schedule, b_sz, img_pth, split_npz, train_transform, test_transform, max_epochs, backbone, device):

    cnv_lyr1, backbone_model, fc_layers = instantiate_architecture(ftr_dim=512, model_name=backbone)
    cnv_lyr1 = cnv_lyr1.to(device)
    backbone_model = backbone_model.to(device)
    fc_layers = fc_layers.to(device)

    ### Dataloaders and model weights for each site
    # Site-0
    data_trn0=construct_dataset(img_pth, split_npz, site=0, transforms=train_transform, tn_vl_idx=0)
    trn_loader0=DataLoader(data_trn0,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val0=construct_dataset(img_pth, split_npz, site=0, transforms=test_transform, tn_vl_idx=1)
    val_loader0=DataLoader(data_val0, b_sz, shuffle=False, num_workers=1, pin_memory=False)
    
    # Site-1
    data_trn1=construct_dataset(img_pth, split_npz, site=1, transforms=train_transform, tn_vl_idx=0)
    trn_loader1=DataLoader(data_trn1,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val1=construct_dataset(img_pth, split_npz, site=1, transforms=test_transform, tn_vl_idx=1)
    val_loader1=DataLoader(data_val1, b_sz, shuffle=False, num_workers=1, pin_memory=False)
    
    # Site-2
    data_trn2=construct_dataset(img_pth, split_npz, site=2, transforms=train_transform, tn_vl_idx=0)
    trn_loader2=DataLoader(data_trn2,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val2=construct_dataset(img_pth, split_npz, site=2, transforms=test_transform, tn_vl_idx=1)
    val_loader2=DataLoader(data_val2, b_sz, shuffle=False, num_workers=1, pin_memory=False)
    
    # Site-3
    data_trn3=construct_dataset(img_pth, split_npz, site=3, transforms=train_transform, tn_vl_idx=0)
    trn_loader3=DataLoader(data_trn3,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val3=construct_dataset(img_pth, split_npz, site=3, transforms=test_transform, tn_vl_idx=1)
    val_loader3=DataLoader(data_val3, b_sz, shuffle=False, num_workers=1, pin_memory=False)
    
    
    # Site-4
    data_trn4=construct_dataset(img_pth, split_npz, site=4, transforms=train_transform, tn_vl_idx=0)
    trn_loader4=DataLoader(data_trn4,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val4=construct_dataset(img_pth, split_npz, site=4, transforms=test_transform, tn_vl_idx=1)
    val_loader4=DataLoader(data_val4, b_sz, shuffle=False, num_workers=1, pin_memory=False)
    
    
    criterion = Custom_Loss(site=-999)
    
    
    ###### Initialize model weights with pre-trained weights
    ## Global
    glbl_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    glbl_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    glbl_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
    ## Site0
    sit0_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit0_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit0_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
    ## Site 1
    sit1_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit1_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit1_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
    ## Site 2
    sit2_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit2_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit2_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
    ## Site 3
    sit3_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit3_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit3_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
    ## Site 4
    sit4_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit4_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit4_fc_wt=copy.deepcopy(fc_layers.state_dict())
    
        
        
    ###### Now begin training
    max_val=0
    for epoch in range(0, max_epochs):
        
    
        
        ############ Perform the local trainings for each site #####
        ## Site 0
        print('\n \n SITE 0 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit0_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit0_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit0_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)
        
        prev_val0, sit0_cnv_lyr1_wt,sit0_backbone_wt, sit0_fc_wt=lcl_train(trn_loader0, val_loader0, criterion, 
                                                               cnv_lyr1, backbone_model,fc_layers, device )
        
        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt
        
        
        ## Site 1
        print('\n \n SITE 1 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit1_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit1_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit1_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)
        
        prev_val1, sit1_cnv_lyr1_wt,sit1_backbone_wt, sit1_fc_wt=lcl_train(trn_loader1, val_loader1, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers, device )
        
        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt
        
        
        ## Site 2
        print('\n \n SITE 2 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit2_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit2_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit2_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)
        
        prev_val2, sit2_cnv_lyr1_wt,sit2_backbone_wt, sit2_fc_wt=lcl_train(trn_loader2, val_loader2, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers, device )
        
        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt
        
        
        ## Site 3
        print('\n \n SITE 3 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit3_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit3_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit3_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)
        
        prev_val3, sit3_cnv_lyr1_wt,sit3_backbone_wt, sit3_fc_wt=lcl_train(trn_loader3, val_loader3, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers , device)
        
        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt
        
        
        ## Site 4
        print('\n \n SITE 4 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit4_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit4_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit4_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)
        
        prev_val4, sit4_cnv_lyr1_wt,sit4_backbone_wt, sit4_fc_wt=lcl_train(trn_loader4, val_loader4, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers , device)
        
        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt
        
        
        avg_val=(prev_val0+prev_val1+prev_val2+prev_val3+prev_val4)/5
        
        if avg_val>max_val:
            max_val=avg_val
            # save model weight, local weights
            torch.save({
                    'cnv_lyr1_state_dict': glbl_cnv_lyr1_wt,
                    'backbone_model_state_dict': glbl_backbone_wt,
                    'fc_layers_state_dict': glbl_fc_wt,
                    }, 'best_glbl_weights.pth')
            
            torch.save({
                    'cnv_lyr1_state_dict': sit0_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit0_backbone_wt,
                    'fc_layers_state_dict': sit0_fc_wt,
                    }, 'best_site0_weights.pth')
            
            torch.save({
                    'cnv_lyr1_state_dict': sit1_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit1_backbone_wt,
                    'fc_layers_state_dict': sit1_fc_wt,
                    }, 'best_site1_weights.pth')
            
            torch.save({
                    'cnv_lyr1_state_dict': sit2_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit2_backbone_wt,
                    'fc_layers_state_dict': sit2_fc_wt,
                    }, 'best_site2_weights.pth')
            
            torch.save({
                    'cnv_lyr1_state_dict': sit3_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit3_backbone_wt,
                    'fc_layers_state_dict': sit3_fc_wt,
                    }, 'best_site3_weights.pth')
            
            torch.save({
                    'cnv_lyr1_state_dict': sit4_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit4_backbone_wt,
                    'fc_layers_state_dict': sit4_fc_wt,
                    }, 'best_site4_weights.pth')
    
        ######### aggregate to compute global weight ###############
                        
        glbl_cnv_lyr1_wt=aggregate_local_weights(sit0_cnv_lyr1_wt, sit1_cnv_lyr1_wt, sit2_cnv_lyr1_wt,
                                                sit3_cnv_lyr1_wt, sit4_cnv_lyr1_wt, device)
        
        glbl_backbone_wt=aggregate_local_weights(sit0_backbone_wt, sit1_backbone_wt, sit2_backbone_wt,
                                                sit3_backbone_wt, sit4_backbone_wt, device)
        
        glbl_fc_wt=aggregate_local_weights(sit0_fc_wt, sit1_fc_wt, sit2_fc_wt, sit3_fc_wt, sit4_fc_wt, device)
        
        
    ###### Just before returning, save the final weights
    
    # save model weight, local weights
    torch.save({
            'cnv_lyr1_state_dict': glbl_cnv_lyr1_wt,
            'backbone_model_state_dict': glbl_backbone_wt,
            'fc_layers_state_dict': glbl_fc_wt,
            }, 'final_glbl_weights.pth')
            
    torch.save({
            'cnv_lyr1_state_dict': sit0_cnv_lyr1_wt,
            'backbone_model_state_dict': sit0_backbone_wt,
            'fc_layers_state_dict': sit0_fc_wt,
            }, 'final_site0_weights.pth')
            
    torch.save({
            'cnv_lyr1_state_dict': sit1_cnv_lyr1_wt,
            'backbone_model_state_dict': sit1_backbone_wt,
            'fc_layers_state_dict': sit1_fc_wt,
            }, 'final_site1_weights.pth')
            
    torch.save({
            'cnv_lyr1_state_dict': sit2_cnv_lyr1_wt,
            'backbone_model_state_dict': sit2_backbone_wt,
            'fc_layers_state_dict': sit2_fc_wt,
            }, 'final_site2_weights.pth')
            
    torch.save({
            'cnv_lyr1_state_dict': sit3_cnv_lyr1_wt,
            'backbone_model_state_dict': sit3_backbone_wt,
            'fc_layers_state_dict': sit3_fc_wt,
            }, 'final_site3_weights.pth')
            
    torch.save({
            'cnv_lyr1_state_dict': sit4_cnv_lyr1_wt,
            'backbone_model_state_dict': sit4_backbone_wt,
            'fc_layers_state_dict': sit4_fc_wt,
            }, 'final_site4_weights.pth')
        
    return 

def train_model(config):
    if torch.cuda.is_available() and config['gpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config['GNN']==True:
        trainer_with_GNN(config['batch_size'], config['data'], config['split'], train_transform, test_transform, config['epochs'], config['backbone'], device, config['checkpoint'] )
    else:
        avg_schedule = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        trainer_without_GNN(avg_schedule, config['batch_size'], config['data'], config['split'], train_transform, test_transform, config['max_epochs'], config['backbone'], device)


