import torch
import torch.nn.functional as F
from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN
import numpy as np
from metric import compute_performance

# Validation code
################# Inference ####################
def inference(cnv_lyr, backbone_model, fc_layers, gnn_model, val_loader, criterion, device,edge_index=None, edge_attr=None):
    
    tot_loss=0
    tot_auc=0
    
    gt_lst=[]
    pred_lst=[]
    
    cnv_lyr.eval()
    backbone_model.eval() 
    fc_layers.eval()
    gnn_model.eval()
    
    with torch.no_grad():
        for count, sample in enumerate(val_loader):
            
            img=sample['img']
            gt=sample['gt']
    
            img=img.to(device)
            gt=gt.to(device)
            
            ##############################################################
            
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
            ########Forward Pass #############################################
            loss=criterion(prd_final, gt)
            
            # Apply the sigmoid
            prd_final=F.sigmoid(prd_final)
            
            gt_lst.append(gt.cpu().numpy())
            pred_lst.append(prd_final.cpu().numpy())
            
            
            
            tot_loss=tot_loss+loss.cpu().numpy()
           
            del loss, gt, prd_final, prd
            
    
    gt_lst=np.concatenate(gt_lst, axis=1)
    pred_lst=np.concatenate(pred_lst, axis=1)
    
    gt_lst=np.transpose(gt_lst)
    pred_lst=np.transpose(pred_lst)
    
    # Now compute and display the average
    count=count+1 # since it began from 0
    avg_loss=tot_loss/count
    
    sens_lst, spec_lst, acc_lst, auc_lst=compute_performance(pred_lst, gt_lst)
    avg_auc=np.mean(auc_lst)
    
    
    print ("\n Val_Loss:  {:.4f},  Avg. AUC: {:.4f}".format(avg_loss, avg_auc))
    
          
    metric=avg_auc # this will be monitored for Early Stopping
            
    cnv_lyr.train()
    backbone_model.train() 
    fc_layers.train()
    if gnn_model is not None:
        gnn_model.train()
    
    return metric