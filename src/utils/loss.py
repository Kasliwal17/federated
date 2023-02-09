import torch.nn as nn
import torch
import numpy as np

############# Define the Weighted Loss. The weights are different for each class ########

class Custom_Loss(nn.Module):
    def __init__(self, site, device=torch.device('cpu')):
        super(Custom_Loss, self).__init__()
        
        if site==-999:
            wts_pos = np.array([ 6.07201409, 12.57545272,  5.07639982,  1.29352719, 14.83679525,  
                                2.61834939, 9.25154963, 22.75312856, 4.12082252,  7.02592567,  
                                1.58836049, 38.86513797, 15.04438092,  1.17096019])

            wts_neg = np.array([0.68019345, 0.64294624, 0.69547317, 1.16038896, 0.63797481, 0.79812282,
                                0.6549775,  0.62857107, 0.71829276, 0.67000328, 0.99474773, 0.62145383,
                                0.63759652, 1.2806393 ])
        elif site==0:
            wts_pos=np.array([636.94267516,  13.93728223, 5.28038864, 5.26537489, 87.87346221, 8.61623298,
                    67.56756757, 228.83295195, 20.40816327,  79.42811755, 5.70450656, 276.24309392,
                    87.79631255,   5.6840789 ])

            wts_neg=np.array([3.14594016, 4.0373047,  7.68875903, 7.72081532, 3.24612089, 4.91690432, 
                    3.28256303, 3.17389786, 3.69767786, 3.2589213,  6.93769946, 3.16636059, 
                    3.24622626, 6.96815553])
            
        elif site==1:
            wts_pos=np.array([ 31.82686187, 649.35064935, 568.18181818,  11.06439478,  75.75757576, 
                    16.73920321,  11.19319454,  27.94076558,  25.4517689,  158.73015873, 
                    11.25999324, 387.59689922,  88.73114463,   7.74653343])

            wts_neg= np.array([3.40901343, 3.09386795, 3.09597523, 4.26657565, 3.20965464, 3.77330013,
                     4.24772747, 3.46056684, 3.50299506, 3.14011179, 4.23818606, 3.10385499, 
                     3.18989441, 5.11064547])
        elif site==2:
            wts_pos= np.array([653.59477124, 662.25165563, 584.79532164, 4.56350112, 45.12635379, 11.55401502, 
                     675.67567568, 746.26865672, 14.69723692, 29.20560748, 5.70418116, 159.23566879,
                     87.03220191,   5.50721445])

            wts_neg= np.array([3.00057011, 3.00039005, 3.0021916, 8.645284, 3.19856704, 4.02819738, 3.00012, 
                     2.99886043, 3.74868796, 3.3271227,  6.26998558, 3.04395471, 3.09300671, 6.52656311])
        elif site==3:
            wts_pos=np.array([359.71223022, 675.67567568, 515.46391753,   6.02772755,  65.40222368, 16.94053871, 
                    800.0, 740.74074074,19.9960008, 11.29433025, 10.53962901, 78.49293564, 87.2600349, 
                    7.8486775 ])

            wts_neg=np.array([3.18775901, 3.17460317, 3.17924588, 6.64098818, 3.32016335, 3.88424937, 3.1722869, 
                    3.17329356, 3.75276767, 4.38711942, 4.51263538, 3.29228946, 3.27847354, 5.28904638])
        elif site==4:
            wts_pos=np.array([7.84990973, 308.64197531, 454.54545455, 9.28074246, 186.21973929, 16.51800463, 
                    819.67213115, 909.09090909, 27.52546105, 1515.15151515, 10.49538203, 1960.78431373, 
                    47.93863854, 4.16684029])

            wts_neg=np.array([ 4.71720364, 2.97495091, 2.96577496, 4.31723007, 2.99392234, 3.58628604, 
                    2.95718003,  2.95613102, 3.29978551,  2.95229098,  4.09668169,  2.95098415, 
                    3.13952028, 10.06137438])

            
        wts_pos = torch.from_numpy(wts_pos)
        wts_pos = wts_pos.type(torch.Tensor)
        wts_pos=wts_pos.to(device) # size 1 by cls
        
        wts_neg = torch.from_numpy(wts_neg)
        wts_neg = wts_neg.type(torch.Tensor)
        wts_neg=wts_neg.to(device) # size 1 by cls
        
        
        self.wts_pos=wts_pos
        self.wts_neg=wts_neg
        
        #self.bce=nn.BCELoss(reduction='none')
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, ypred, ytrue):
        
        msk = ((1-ytrue)*self.wts_neg) + (ytrue*self.wts_pos) #1 if ytrue is 0
        #print(msk.shape)
        
        loss=self.bce(ypred,ytrue) # bsz, cls
        loss=loss*msk
        
        loss=loss.view(-1) # flatten all batches and class losses
        loss=torch.mean(loss) 
        
        return loss