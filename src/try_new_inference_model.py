#### Trying out the new inference model


from utils.model import Infer_model
import torch
from numpy import load
from utils.misc import aggregate_local_weights

# model = Infer_model('resnet',split_path='/storage/adityak/split.npz',gnn=False)
model = Infer_model('resnet',split_path='/storage/adityak/split.npz',gnn=True)
checkpoint=torch.load('/storage/adityak/with_GNN_wt/best_weight_0.65908386103622_19.pt')
glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
glbl_backbone_wt=checkpoint['backbone_model_state_dict']
glbl_fc_wt=checkpoint['fc_layers_state_dict']
sit0_gnn_wt=checkpoint['sit0_gnn_model']
sit1_gnn_wt=checkpoint['sit1_gnn_model']
sit2_gnn_wt=checkpoint['sit2_gnn_model']
sit3_gnn_wt=checkpoint['sit3_gnn_model']
sit4_gnn_wt=checkpoint['sit4_gnn_model']
glbl_gnn_wt=aggregate_local_weights(sit0_gnn_wt, sit1_gnn_wt, sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt, torch.device('cpu'))
model.gnn_model.load_state_dict(glbl_gnn_wt)
model.cnv_lyr.load_state_dict(glbl_cnv_wt)
model.backbone_model.load_state_dict(glbl_backbone_wt)
model.fc_layers.load_state_dict(glbl_fc_wt)
model.eval()
prd = model(torch.randn(1,1,320,320))
print(prd,prd.shape)
#### Result the new inference model works perfectly
