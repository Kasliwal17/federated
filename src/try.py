from utils.model import Infer_model
import torch
from numpy import load
# data = load('/home/aditya/Downloads/split.npz',allow_pickle=True)
# gt=data['gt']
model = Infer_model('resnet',split_path='/home/aditya/Downloads/split.npz',gnn=True)
model.eval()
z=model(torch.randn(1,1,512,512),[list([1,0,0,0,0,0,0,0,0,0,0,0,0,0])])
print(z)
print(gt)