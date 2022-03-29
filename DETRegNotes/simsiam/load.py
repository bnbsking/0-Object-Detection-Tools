import os; os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torchvision
import torch; device = torch.device('cuda'); #torch.manual_seed(0)

class SimSiam(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)

def load_model():
    resnet = torchvision.models.resnet18()
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    model = SimSiam(backbone).to(device)
    model.load_state_dict( torch.load("/home/jovyan/data-vol-1/detreg/_simsiam/model_state_dict_simsiam.pt"), strict=False )
    return model

"""
model.eval()
with torch.no_grad():
    x = torch.rand(2,3,256,256,device=device)
    y = model(x)
    print(y.shape) # 2,512 
    print(y)
"""
