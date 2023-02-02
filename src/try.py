from utils.model import Infer_model
import torch
from numpy import load
data = load('/home/aditya/Downloads/split.npz',allow_pickle=True)
gt=data['gt']

# gt1 = gt[1]
# gt1 = torch.FloatTensor(gt1)
# gt1 = gt1.unsqueeze(0)
gt1 = torch.FloatTensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
print(gt1.shape)
model = Infer_model('resnet',split_path='/home/aditya/Downloads/split.npz',gnn=True)
model.eval()
z=model(torch.randn(1,1,320,320))
print(z)