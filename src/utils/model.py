import torch.nn as nn
import torch
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm as GNN_BatchNorm
import torch.nn.functional as F
# Define Architecture
''' Instead of creating an instance of the models within the constructor, 
 We will pass the initial layer for 1 to 3 channel, the backbone model 
 and the FC layers part separately as input arguments. 
 
 This will allow us to simply load the weights for each CNN model separately, 
 may be useful when updating only part of the network
 
 '''
class Fully_Connected_Layer(nn.Module):
    def __init__(self, inp_dim, ftr_dim):
        super(Fully_Connected_Layer, self).__init__()
        
        ftr_lyr=nn.ModuleList()
        cls_lyr=nn.ModuleList()
        
        for cls in range(0,14):
            ftr_lyr.append(
                            nn.Sequential(
                                            nn.Linear(inp_dim, ftr_dim, bias=False), 
                                            nn.BatchNorm1d(ftr_dim),
                                            nn.ReLU(),
                                            nn.Linear(ftr_dim, ftr_dim, bias=False), 
                                            nn.BatchNorm1d(ftr_dim),    
                                            nn.ReLU()
                                        )
                          )
            
            cls_lyr.append(
                            nn.Sequential(
                                            nn.Linear(ftr_dim, 1, bias=False), 
                                            nn.BatchNorm1d(1)
                                        )
                            )
            self.ftr_lyr=ftr_lyr
            self.cls_lyr=cls_lyr

    def forward(self, x):
        prd_lst=[]
        ftr_lst=[]
        for cls in range(0,14):
            
            ftr=self.ftr_lyr[cls](x)
            ftr_lst.append(torch.unsqueeze(ftr, dim=1))
            prd=self.cls_lyr[cls](ftr)
            prd_lst.append(prd)
            
        
        prd=torch.cat(prd_lst, axis=1)
        return ftr_lst, prd
    
    
############## Conv 1st layer #######################

class First_Conv(nn.Module):
    def __init__(self):
        super(First_Conv, self).__init__()
        
        # Convert 1 channel to 3 channel also can be made unique for each site
        self.convert_channels=nn.Sequential(nn.Conv2d(1,3,1,1, bias=False), 
                                            nn.BatchNorm2d(3),
                                            nn.ReLU() )
    def forward(self, x):
        x=self.convert_channels(x)
        return x




################## GNN Architecture Classes #############

# This MLP will map the edge weight to the weights used to avg. the features from the neighbors
class create_mlp(nn.Module):
    def __init__(self, in_chnl, out):
        super(create_mlp, self).__init__() 
        
        self.lyr=nn.Sequential(
                                nn.Linear(in_chnl, out, bias=True),
                                #nn.BatchNorm1d(out),
                                nn.Tanh()
                                    )
        
    def forward(self, x):
        out=self.lyr(x)
        return out

#######################################################################################

# The Resdiual Block for the GNN
class Res_Graph_Conv_Lyr(nn.Module):
    def __init__(self, in_chnls, base_chnls, mlp_model, aggr_md):
        super(Res_Graph_Conv_Lyr, self).__init__() 
        
        self.GNN_lyr=NNConv(in_chnls, base_chnls, mlp_model, aggr=aggr_md)
        self.bn=GNN_BatchNorm(base_chnls)
        
    def forward(self, x, edge_index, edge_attr):
        h=self.GNN_lyr(x, edge_index, edge_attr)
        h=self.bn(h)
        h=F.relu(h)
        
        return (x+h)
        #return h
    
########################################################################################

############### The Graph Convolution Network ############################
class GNN_Network(nn.Module):
    def __init__(self, in_chnls, base_chnls, grwth_rate, depth, aggr_md, ftr_dim):
        super(GNN_Network, self).__init__()
        
        my_gcn=nn.ModuleList()
        
        # Base channels is actually the fraction of inp.
        in_chnls=int(in_chnls)
        base_chnls=int(base_chnls*in_chnls)
        
        # A GCN to map input channels to base channels dimensions
        my_gcn.append(Res_Graph_Conv_Lyr(in_chnls, base_chnls, create_mlp(ftr_dim, in_chnls*base_chnls), aggr_md))
        
        in_chnls=base_chnls
        for k in range(0, depth):
            out_chnls=int(in_chnls*grwth_rate)
            # Get a GCN
            in_chnls=max(in_chnls,1)
            out_chnls=max(out_chnls,1)
            my_gcn.append(Res_Graph_Conv_Lyr(in_chnls, out_chnls, create_mlp(ftr_dim,in_chnls*out_chnls), aggr_md))
            in_chnls=out_chnls
        
        #### Add the final classification layer that will convert output to 1D 
        my_gcn.append(NNConv(in_chnls, 1, create_mlp(ftr_dim, 1*in_chnls), aggr='mean'))
        
        self.my_gcn=my_gcn
        self.dpth=depth
        
    
    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cnt=0
        x=self.my_gcn[cnt](x, edge_index, edge_attr)
        #x=F.relu(x)
        
        #print('entering loop ...')
        for k in range(0, self.dpth):
            cnt=cnt+1
            #print(cnt)
            x=self.my_gcn[cnt](x, edge_index, edge_attr)
            #x=F.relu(x)
        
        #print('out of loop...')
        cnt=cnt+1
        out=self.my_gcn[cnt](x, edge_index, edge_attr) # num_nodes by 1
        #out=F.sigmoid(out)
        return out
