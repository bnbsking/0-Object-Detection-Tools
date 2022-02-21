import os; os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torchvision
import torch; device = torch.device('cuda'); torch.manual_seed(0)

class SimSiam(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        pass

resnet = torchvision.models.resnet18()
backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
model = SimSiam(backbone).to(device)
model.load_state_dict( torch.load("model_state_dict_simsiam.pt"), strict=False )

model.eval()
with torch.no_grad():
    x = torch.rand(2,3,256,256,device=device)
    y = model.backbone(x).flatten(start_dim=1) 
    print(y.shape) # 2,512 
    print(y)