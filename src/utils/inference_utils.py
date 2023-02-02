import torch
import torch.nn.functional as F
from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN
import numpy as np
from .metric import compute_performance
from .model import Infer_model
from .loss import Custom_Loss
from .dataloader import construct_dataset
from torch.utils.data import DataLoader
from .transformations import test_transform
from openvino.inference_engine import IECore
import torchvision.transforms as transforms
from .misc import aggregate_local_weights
import os


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Validation code
################# To be used for inference during training ####################
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

#####To be used for inference##########
def load_inference_model(config, run_type):
    if config['gnn']=='True':
        gnn=True
    else:
        gnn=False
    if run_type == 'pytorch':
        model = Infer_model(config['backbone'],config['split_npz'],gnn)
        checkpoint = torch.load(config['model_file'], map_location=torch.device('cpu'))
        glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
        glbl_backbone_wt=checkpoint['backbone_model_state_dict']
        glbl_fc_wt=checkpoint['fc_layers_state_dict']

        model.cnv_lyr.load_state_dict(glbl_cnv_wt)
        model.backbone_model.load_state_dict(glbl_backbone_wt)
        model.fc_layers.load_state_dict(glbl_fc_wt)
        if gnn:
            sit0_gnn_wt=checkpoint['sit0_gnn_model']
            sit1_gnn_wt=checkpoint['sit1_gnn_model']
            sit2_gnn_wt=checkpoint['sit2_gnn_model']
            sit3_gnn_wt=checkpoint['sit3_gnn_model']
            sit4_gnn_wt=checkpoint['sit4_gnn_model']
            glbl_gnn_wt=aggregate_local_weights(sit0_gnn_wt, sit1_gnn_wt, sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt, torch.device('cpu'))
            model.gnn_model.load_state_dict(glbl_gnn_wt)
        model.eval()

    elif run_type == 'onnx':
        model = onnxruntime.InferenceSession(os.path.splitext(config['checkpoint'])[0] + ".onnx")

    else:
        ie = IECore()
        split_text = os.path.splitext(config['checkpoint'])[0]
        model_xml =  split_text + ".xml"
        model_bin = split_text + ".bin"
        model_temp = ie.read_network(model_xml, model_bin)
        model = ie.load_network(network=model_temp, device_name='CPU')

    return model
def inference_model(model, config, run_type):
    # GPU transfer - Only pytorch models needs to be transfered.
    if run_type == 'pytorch':
        if torch.cuda.is_available() and config['gpu'] == 'True':
            model = model.cuda()
    data_test=construct_dataset(config['dataset'], config['split_npz'], -999, test_transform, tn_vl_idx=2)
    test_loader=DataLoader(data_test,batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    tot_loss=0
    tot_auc=0
    
    gt_lst=[]
    pred_lst=[]
    criterion = Custom_Loss(-999)
    
    with torch.no_grad():
        for count, sample in enumerate(test_loader):
            
            img=sample['img']
            gt=sample['gt']
            if torch.cuda.is_available() and config['gpu'] == 'True':
                img = img.cuda()
                gt = gt.cuda()
            if run_type == 'pytorch':
                prd_final, gt = model(img,gt)  # forward through encoder
                prd_final = prd_final.cpu().numpy()
                gt = gt.cpu().numpy()
            elif run_type == 'onnx':
                resize_tensor = transforms.Resize([1024,1024])
            prd_final ,gt= model(img,gt)
            ########Forward Pass #############################################
            loss=criterion(prd_final, gt)
            
            # Apply the sigmoid
            prd_final=F.sigmoid(prd_final)
            
            gt_lst.append(gt.cpu().numpy())
            pred_lst.append(prd_final.cpu().numpy())
            
            
            tot_loss=tot_loss+loss.cpu().numpy()
           
            del loss, gt, prd_final
            
    
    gt_lst=np.concatenate(gt_lst, axis=1)
    pred_lst=np.concatenate(pred_lst, axis=1)
    
    gt_lst=np.transpose(gt_lst)
    pred_lst=np.transpose(pred_lst)
    
    # Now compute and display the average
    count=count+1 # since it began from 0
    avg_loss=tot_loss/count
    
    sens_lst, spec_lst, acc_lst, auc_lst=compute_performance(pred_lst, gt_lst)
    avg_auc=np.mean(auc_lst)
    
    
    print ("\n Test_Loss:  {:.4f},  Avg. AUC: {:.4f}".format(avg_loss, avg_auc))